#!/usr/bin/env python3
"""Speech-to-text for Wayland desktops using Groq-hosted transcription."""

from __future__ import annotations

import argparse
import importlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from groq import Groq


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
TMP_DIR = Path(tempfile.gettempdir())
DEFAULT_STATE_FILE = TMP_DIR / "local-stt-wayland-state.json"

load_dotenv(SCRIPT_DIR / ".env")


@dataclass
class AppConfig:
    transcription_model: str = "whisper-large-v3-turbo"
    language: str = "en"
    prompt: str | None = None
    sample_rate: int = 16_000
    max_duration: int = 60
    state_file: Path = DEFAULT_STATE_FILE


class AppError(RuntimeError):
    """Raised for expected application-level failures."""


class AudioRecorder:
    """Record mono microphone audio using sounddevice."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.is_recording = False
        self.stream: Any | None = None
        self.audio_data: list[np.ndarray] = []

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        del frames, time_info
        if status:
            print(f"⚠️ Audio status: {status}", file=sys.stderr)
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start(self) -> None:
        self.is_recording = True
        self.audio_data = []
        sounddevice = get_sounddevice()
        self.stream = sounddevice.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self.stream.start()
        print("🎤 Recording...")

    def stop(self) -> np.ndarray:
        self.is_recording = False
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0).flatten()
        else:
            audio = np.array([], dtype="float32")

        print("⏹️ Recording stopped.")
        return audio

    def record_until_enter(self) -> np.ndarray:
        self.start()
        try:
            input()
        except KeyboardInterrupt:
            pass
        return self.stop()


class GroqRecognizer:
    """Transcribe recorded audio through Groq's hosted speech-to-text API."""

    _client: Groq | None = None

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._get_client()

    @classmethod
    def _get_client(cls) -> Groq:
        if cls._client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise AppError("GROQ_API_KEY is not set. Add it to .env or your shell environment.")

            print("☁️ Connecting to Groq...")
            cls._client = Groq(api_key=api_key)
            print("✅ Groq client ready")

        return cls._client

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) == 0:
            return ""

        wav_bytes = audio_to_wav_bytes(audio, self.config.sample_rate)
        request_args: dict[str, Any] = {
            "file": ("speech.wav", wav_bytes),
            "model": self.config.transcription_model,
            "language": self.config.language,
            "response_format": "json",
            "temperature": 0.0,
        }
        if self.config.prompt:
            request_args["prompt"] = self.config.prompt

        try:
            transcription = self._get_client().audio.transcriptions.create(**request_args)
        except Exception as exc:
            raise AppError(f"Groq transcription failed: {exc}") from exc

        return transcription.text.strip()


class WaylandTyper:
    """Paste transcribed text using Wayland clipboard and wtype."""

    @staticmethod
    def sanitize_text(text: str) -> str:
        return " ".join(text.replace("\n", " ").replace("\r", " ").split())

    @staticmethod
    def copy_to_clipboard(text: str) -> bool:
        if not command_exists("wl-copy"):
            print(f"📝 Transcribed text: {text}")
            print("⚠️ wl-copy is not installed, so the text could not be copied automatically.")
            return False

        subprocess.run(["wl-copy", "--", text], check=True, capture_output=True)
        print(f"📋 Text copied to clipboard: {text}")
        print("   Press Ctrl+V to paste it.")
        notify("Speech-to-Text", preview_text("Text copied: ", text), timeout_ms=3000)
        return True

    @classmethod
    def type_text(cls, text: str) -> bool:
        cleaned = cls.sanitize_text(text)
        if not cleaned:
            print("⚠️ No text to type.")
            return False

        if not command_exists("wl-copy"):
            print(f"📝 Transcribed text: {cleaned}")
            print("⚠️ wl-copy is not installed; printing text instead of typing it.")
            return False

        try:
            subprocess.run(["wl-copy", "--", cleaned], check=True, capture_output=True)
            if command_exists("wtype"):
                time.sleep(0.1)
                subprocess.run(
                    ["wtype", "-M", "ctrl", "-P", "v", "-p", "v", "-m", "ctrl"],
                    check=True,
                    capture_output=True,
                )
                print(f"⌨️ Typed at cursor: {cleaned}")
                return True

            print("⚠️ wtype is not installed; text was copied to the clipboard instead.")
            print(f"📋 Copied text: {cleaned}")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"❌ Error typing text: {exc}")
            return cls.copy_to_clipboard(cleaned)


class ToggleController:
    """Coordinate start/stop behavior across shortcut invocations."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.state_file = config.state_file

    def toggle(self, args: argparse.Namespace) -> int:
        state = load_state(self.state_file)
        if state and is_process_running(state.get("recorder_pid", -1)):
            return self.stop()
        if state:
            cleanup_stale_state(self.state_file)
        return self.start(args)

    def start(self, args: argparse.Namespace) -> int:
        audio_file = TMP_DIR / f"local-stt-wayland-{os.getpid()}-{int(time.time())}.npy"
        worker_args = build_worker_command(args, self.config, audio_file)

        process = subprocess.Popen(  # noqa: S603
            worker_args,
            cwd=str(SCRIPT_DIR),
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        save_state(
            self.state_file,
            {
                "recorder_pid": process.pid,
                "audio_file": str(audio_file),
                "started_at": time.time(),
            },
        )
        notify("🎤 Recording Started", "Speak now. Press the shortcut again to stop.", 2000)
        print("🎤 Started background recording")
        return 0

    def stop(self) -> int:
        state = load_state(self.state_file)
        if not state:
            raise AppError("No active recording state found.")

        pid = state["recorder_pid"]
        audio_file = Path(state["audio_file"])

        if not is_process_running(pid):
            cleanup_state_files(self.state_file, audio_file)
            raise AppError("The recording process is no longer running.")

        print("⏹️ Stopping recording...")
        os.kill(pid, signal.SIGTERM)
        wait_for_process_exit(pid, timeout=5.0)
        wait_for_file(audio_file, timeout=5.0)

        if not audio_file.exists():
            cleanup_state_files(self.state_file, audio_file)
            notify("Speech-to-Text", "Recording failed - no audio captured", 3000)
            raise AppError("No audio file found after stopping recording.")

        audio = np.load(str(audio_file))
        cleanup_state_files(self.state_file, audio_file)

        if len(audio) < self.config.sample_rate * 0.5:
            notify("Speech-to-Text", "Recording too short", 2000)
            print("⚠️ Recording too short")
            return 1

        print("🔄 Transcribing with Groq...")
        notify("🔄 Processing", "Transcribing your speech...", 2000)
        recognizer = GroqRecognizer(self.config)
        text = recognizer.transcribe(audio)
        if not text:
            notify("Speech-to-Text", "No speech detected", 2000)
            print("⚠️ No speech detected")
            return 1

        time.sleep(0.3)
        WaylandTyper.type_text(text)
        notify("✅ Speech-to-Text", preview_text("Typed: ", text), 3000)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speech-to-text for Wayland desktops using Groq-hosted transcription"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="interactive",
        choices=["interactive", "toggle", "record-worker"],
        help="Run interactively, as a keyboard shortcut toggle, or as the internal recorder worker.",
    )
    parser.add_argument(
        "--model",
        default="whisper-large-v3-turbo",
        help="Groq-hosted transcription model, e.g. whisper-large-v3-turbo or whisper-large-v3.",
    )
    parser.add_argument("--language", default="en", help="Language code to pass to Groq transcription.")
    parser.add_argument("--prompt", help="Optional transcription prompt to improve domain-specific spelling.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Microphone sample rate in Hz")
    parser.add_argument("--max-duration", type=int, default=60, help="Maximum recording length in seconds")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help="State file used by toggle mode.",
    )
    parser.add_argument("--audio-file", type=Path, help=argparse.SUPPRESS)
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        transcription_model=args.model,
        language=args.language,
        prompt=args.prompt,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        state_file=args.state_file,
    )


def get_sounddevice() -> Any:
    return importlib.import_module("sounddevice")


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    clipped = np.clip(audio, -1.0, 1.0)
    pcm_audio = (clipped * np.iinfo(np.int16).max).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio.tobytes())

    return buffer.getvalue()


def command_exists(command: str) -> bool:
    return subprocess.run(
        ["which", command], capture_output=True, check=False, text=True
    ).returncode == 0


def notify(title: str, body: str, timeout_ms: int = 2000) -> None:
    if command_exists("notify-send"):
        subprocess.run(
            ["notify-send", title, body, "-t", str(timeout_ms)],
            check=False,
            capture_output=True,
        )


def preview_text(prefix: str, text: str, limit: int = 50) -> str:
    clipped = text[:limit] + ("..." if len(text) > limit else "")
    return f"{prefix}{clipped}"


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state))


def cleanup_state_files(state_file: Path, audio_file: Path) -> None:
    if state_file.exists():
        state_file.unlink()
    if audio_file.exists():
        audio_file.unlink()


def cleanup_stale_state(state_file: Path) -> None:
    state = load_state(state_file)
    if not state:
        return
    audio_file = Path(state.get("audio_file", "")) if state.get("audio_file") else None
    if state_file.exists():
        state_file.unlink()
    if audio_file and audio_file.exists():
        audio_file.unlink()


def is_process_running(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def wait_for_process_exit(pid: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not is_process_running(pid):
            return
        time.sleep(0.1)


def wait_for_file(path: Path, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return
        time.sleep(0.1)


def build_worker_command(args: argparse.Namespace, config: AppConfig, audio_file: Path) -> list[str]:
    del args

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "record-worker",
        "--model",
        config.transcription_model,
        "--language",
        config.language,
        "--sample-rate",
        str(config.sample_rate),
        "--max-duration",
        str(config.max_duration),
        "--state-file",
        str(config.state_file),
        "--audio-file",
        str(audio_file),
    ]

    if config.prompt:
        command.extend(["--prompt", config.prompt])

    return command


def check_dependencies(mode: str) -> bool:
    issues: list[str] = []

    try:
        sounddevice = get_sounddevice()
        devices = sounddevice.query_devices()
        input_devices = [device for device in devices if device["max_input_channels"] > 0]
        if not input_devices:
            issues.append("No microphone found.")
        else:
            default_index = sounddevice.default.device[0]
            if default_index is not None and default_index >= 0:
                print(f"🎙️ Microphone: {sounddevice.query_devices(default_index)['name']}")
    except Exception as exc:
        issues.append(f"Error checking audio devices: {exc}")

    if mode != "record-worker":
        if not os.environ.get("GROQ_API_KEY"):
            issues.append("GROQ_API_KEY is not set. Add it to .env or your shell environment.")

        if not command_exists("wl-copy"):
            issues.append("wl-copy is not installed. Run: sudo apt install wl-clipboard")

        if not command_exists("wtype"):
            print("⚠️ wtype is not installed; text will be copied to the clipboard instead of pasted.")

    if mode != "interactive" and not command_exists("notify-send"):
        print("⚠️ notify-send is not installed; desktop notifications will be skipped.")

    if issues:
        print("❌ Dependency check failed:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    return True


def interactive_mode(config: AppConfig) -> int:
    print("\n" + "=" * 50)
    print("🎤 Speech-to-Text - Groq Cloud Edition")
    print("=" * 50)
    print(f"Model: {config.transcription_model} (hosted by Groq)")
    print("Recording starts immediately")
    print("Press Enter to stop and paste")
    print("Press Ctrl+C to quit")
    print("=" * 50 + "\n")

    recognizer = GroqRecognizer(config)
    recorder = AudioRecorder(config.sample_rate)

    try:
        while True:
            audio = recorder.record_until_enter()
            if len(audio) == 0:
                print("⚠️ No audio recorded.")
                continue

            print("🔄 Transcribing with Groq...")
            text = recognizer.transcribe(audio)
            if text:
                WaylandTyper.type_text(text)
            else:
                print("⚠️ No speech detected.")
            print("\nStarting a new recording. Press Enter to stop...")
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0


def record_worker_mode(config: AppConfig, audio_file: Path | None) -> int:
    if audio_file is None:
        raise AppError("record-worker mode requires --audio-file")

    recorder = AudioRecorder(config.sample_rate)
    stop_requested = False

    def handle_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)

    recorder.start()
    start_time = time.time()
    try:
        while not stop_requested and (time.time() - start_time) < config.max_duration:
            time.sleep(0.1)
    finally:
        audio = recorder.stop()
        np.save(str(audio_file), audio)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    print("🔍 Checking dependencies...")
    if not check_dependencies(args.mode):
        return 1
    print("✅ Dependencies OK\n")

    if args.mode == "interactive":
        return interactive_mode(config)
    if args.mode == "toggle":
        return ToggleController(config).toggle(args)
    if args.mode == "record-worker":
        return record_worker_mode(config, args.audio_file)
    raise AppError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AppError as exc:
        print(f"❌ {exc}")
        raise SystemExit(1)
