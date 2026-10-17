"""Scripted WebRTC caller: drive a voice call and measure latency from the caller's side of the wire.

    uv run python -m agent.voice_server --data-dir <scratch>        # in one terminal
    uv run python -m eval.voice_probe --script booking               # in another
    uv run python -m eval.voice_probe --script barge-in

The probe connects the way the browser page does (POST /api/offer, audio over WebRTC via aiortc).
Caller lines are synthesized ahead of time with Kokoro in a voice different from the agent's, then
streamed at real-time pace. For each caller turn it records:

    latency = first agent audio received  -  end of the caller's speech on the wire

"End of speech" is the scheduled wall-clock time of the last voiced sample the probe sends.
"First agent audio" is the arrival of the first received frame above -50 dBFS, which excludes the
playout buffer a browser adds on top. Results go to logs/probe/ and include the server's own
per-turn breakdown from logs/calls/.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import fractions
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import soxr
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from agent.config import REPO_ROOT

RATE = 48000
FRAME = 960  # 20 ms
LOUD_DB = -50.0
CALLER_VOICE = "am_michael"

SCRIPTS = {
    "booking": [
        "Hi, I'd like to book a furnace tune-up for Wednesday at 2 p.m.",
        "My name is Dana Reyes, and my number is 555-0134.",
        "Yes, that's right.",
        "No, that's all. Thanks, bye.",
    ],
    "latency": [
        "What are your hours on Saturday?",
        "Do you work on heat pumps?",
        "How much is the diagnostic fee?",
        "Okay, thanks. That's all.",
    ],
    "barge-in": [
        "Sorry to cut in. What are your hours on Saturday?",
        "Okay, thanks. That's all.",
    ],
}


def db(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768.0
    return 20 * np.log10(rms + 1e-9)


def synthesize(lines: list[str]) -> list[np.ndarray]:
    """Caller audio as int16 mono at 48 kHz, leading and trailing silence trimmed."""
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR

    kokoro = Kokoro(str(KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"), str(KOKORO_CACHE_DIR / "voices-v1.0.bin"))
    out = []
    for line in lines:
        samples, rate = kokoro.create(line, voice=CALLER_VOICE, speed=1.0, lang="en-us")
        audio = soxr.resample(samples, rate, RATE)
        loud = np.flatnonzero(np.abs(audio) > 0.01)
        audio = audio[loud[0] : loud[-1] + 1]
        out.append((np.clip(audio, -1, 1) * 32767).astype(np.int16))
    return out


class CallerTrack(MediaStreamTrack):
    """Outgoing mic: silence, except while an utterance is queued. Paced at real time."""

    kind = "audio"

    def __init__(self):
        super().__init__()
        self._start: float | None = None
        self._pts = 0
        self._pending: collections.deque[tuple[np.ndarray, asyncio.Future]] = collections.deque()
        self._current: np.ndarray | None = None
        self._pos = 0
        self._current_done: asyncio.Future | None = None
        self.speech_started_at: float | None = None

    def say(self, audio: np.ndarray) -> asyncio.Future:
        done = asyncio.get_running_loop().create_future()
        self._pending.append((audio, done))
        return done

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.monotonic()
        due = self._start + self._pts / RATE
        if (wait := due - time.monotonic()) > 0:
            await asyncio.sleep(wait)
        chunk = np.zeros(FRAME, dtype=np.int16)
        if self._current is None and self._pending:
            self._current, self._current_done = self._pending.popleft()
            self._pos = 0
            self.speech_started_at = due
        if self._current is not None:
            part = self._current[self._pos : self._pos + FRAME]
            chunk[: len(part)] = part
            self._pos += len(part)
            if self._pos >= len(self._current):
                # Wall-clock time at which the last voiced sample is due on the wire.
                self._current_done.set_result(due + len(part) / RATE)
                self._current = None
        frame = AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, RATE)
        self._pts += FRAME
        return frame


class AgentEar:
    """Records (arrival time, level) for every frame of the agent's audio."""

    def __init__(self):
        self.frames: list[tuple[float, float]] = []

    async def listen(self, track) -> None:
        try:
            while True:
                frame = await track.recv()
                self.frames.append((time.monotonic(), db(frame.to_ndarray())))
        except MediaStreamError:
            return

    def first_loud_after(self, t: float) -> float | None:
        return next((ts for ts, level in self.frames if ts > t and level > LOUD_DB), None)

    def last_loud(self) -> float | None:
        return next((ts for ts, level in reversed(self.frames) if level > LOUD_DB), None)

    async def wait_for_audio(self, after: float, timeout: float) -> float | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (t := self.first_loud_after(after)) is not None:
                return t
            await asyncio.sleep(0.01)
        return None

    async def wait_for_silence(self, gap: float, timeout: float) -> float | None:
        """Wait until the agent has been quiet for ``gap`` seconds; return when it went quiet."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            last = self.last_loud()
            if last is not None and time.monotonic() - last >= gap:
                return last
            await asyncio.sleep(0.05)
        return None

    def stopped_after(self, t: float, gap: float = 0.5) -> float | None:
        """Last loud frame before the first quiet stretch of ``gap`` seconds that starts after ``t``."""
        loud = [ts for ts, level in self.frames if level > LOUD_DB and ts > t - 1.0]
        for a, b in zip(loud, loud[1:]):
            if a > t and b - a >= gap:
                return a
        return loud[-1] if loud and loud[-1] > t else None


def hardware() -> dict:
    def ps(cmd: str) -> str:
        try:
            return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "cpu": ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"),
        "ram_gb": ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB)"),
        "gpu": ps("(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ', '"),
        "os": platform.platform(),
    }


async def run(script: str, url: str, log_dir: Path) -> dict:
    lines = SCRIPTS[script]
    print(f"synthesizing {len(lines)} caller lines ({CALLER_VOICE})...")
    audio = synthesize(lines)
    started = time.time()

    pc = RTCPeerConnection()
    caller = CallerTrack()
    ear = AgentEar()
    listen_tasks = []
    pc.addTrack(caller)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            listen_tasks.append(asyncio.ensure_future(ear.listen(track)))

    await pc.setLocalDescription(await pc.createOffer())
    async with aiohttp.ClientSession() as http:
        async with http.post(f"{url}/api/offer", json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}) as resp:
            resp.raise_for_status()
            answer = await resp.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    connected = time.monotonic()

    turns = []
    greeting_start = await ear.wait_for_audio(connected, timeout=60)
    result = {"script": script, "greeting_first_audio_s": None if greeting_start is None else round(greeting_start - connected, 3)}

    if script == "barge-in":
        # Cut into the greeting 1.5 s after it starts.
        await asyncio.sleep(max(0.0, greeting_start + 1.5 - time.monotonic()))
        done = caller.say(audio[0])
        await asyncio.sleep(0.05)
        speech_start = caller.speech_started_at
        speech_end = await done
        stopped = ear.stopped_after(speech_start)
        await asyncio.sleep(1.5)
        stopped = ear.stopped_after(speech_start)
        reply = await ear.wait_for_audio(speech_end, timeout=30)
        result["barge_in"] = {
            "caller_speech_start_into_greeting_s": round(speech_start - greeting_start, 3),
            "agent_audio_stopped_after_caller_started_s": None if stopped is None else round(stopped - speech_start, 3),
            "reply_latency_s": None if reply is None else round(reply - speech_end, 3),
        }
        await ear.wait_for_silence(gap=1.5, timeout=60)
        remaining = audio[1:]
    else:
        await ear.wait_for_silence(gap=1.2, timeout=60)
        remaining = audio

    for text, clip in zip(lines[len(lines) - len(remaining):], remaining):
        end = await caller.say(clip)
        first = await ear.wait_for_audio(end, timeout=45)
        latency = None if first is None else round(first - end, 3)
        turns.append({"caller": text, "latency_s": latency})
        print(f"  {'-' if latency is None else f'{latency:5.2f}s'}  <- {text}")
        if first is None:
            break
        await ear.wait_for_silence(gap=1.5, timeout=60)

    await pc.close()
    for task in listen_tasks:
        task.cancel()
    await asyncio.sleep(3)  # let the server write its call record

    measured = [t["latency_s"] for t in turns if t["latency_s"] is not None]
    result.update(
        {
            "turns": turns,
            "latency_s": {
                "min": min(measured) if measured else None,
                "median": float(np.median(measured)) if measured else None,
                "max": max(measured) if measured else None,
            },
            "hardware": hardware(),
        }
    )
    records = sorted((REPO_ROOT / "logs" / "calls").glob("*.json"), key=lambda p: p.stat().st_mtime)
    fresh = [p for p in records if p.stat().st_mtime >= started]
    if fresh:
        server = json.loads(fresh[-1].read_text(encoding="utf-8"))
        result["server"] = {k: server.get(k) for k in ("providers", "ended_by", "latency", "transcript", "tool_calls")}
        result["server_record"] = str(fresh[-1])

    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{script}.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["file"] = str(out)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.voice_probe", description=__doc__.splitlines()[0])
    parser.add_argument("--script", choices=sorted(SCRIPTS), default="booking")
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs" / "probe")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    result = asyncio.run(run(args.script, args.url, args.log_dir))
    print(json.dumps({k: result.get(k) for k in ("greeting_first_audio_s", "barge_in", "latency_s", "file")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
