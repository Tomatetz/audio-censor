from __future__ import annotations

import queue
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
    return f"{minutes:02d}:{seconds:05.2f}"


class TranscriptRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None
        self._lock = threading.Lock()

    def start(self, *, mode: str, delay: float, model: str, words: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._file.write("STREAM CENSOR — ЖУРНАЛ СЕССИИ\n")
        self._file.write(f"Начало: {datetime.now().isoformat(timespec='seconds')}\n")
        self._file.write(f"Модель: {model}; режим: {mode}; задержка: {delay:.1f} с\n")
        self._file.write(f"Целевые основы: {words or '(нет)'}\n\n")
        self._file.write(
            "Время относится к итоговому WAV после добавления задержки.\n"
            "[ASR:STABLE] — новые подтверждённые слова без повторов окон.\n"
            "[CENSOR:*] — участок, который реально был заменён.\n"
            "[MISS] — цель видна в строке, но у неё нет временных границ слова.\n"
            "[RISK]/[LATE] — мало запаса или обработка опоздала.\n\n"
        )
        self._file.flush()

    def event(
        self,
        kind: str,
        start_seconds: float,
        end_seconds: float,
        message: str,
    ) -> None:
        if not self._file:
            return
        line = (
            f"{format_timestamp(start_seconds)}–{format_timestamp(end_seconds)} "
            f"[{kind}] {message}\n"
        )
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        if not self._file:
            return
        with self._lock:
            self._file.write(
                f"\nКонец: {datetime.now().isoformat(timespec='seconds')}\n"
            )
            self._file.close()
            self._file = None


class WavRecorder:
    def __init__(self, path: str | Path, sample_rate: int):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._write_loop,
            name="wav-recorder",
            daemon=True,
        )
        self._thread.start()

    def add(self, samples: np.ndarray) -> None:
        # Copy because PortAudio reuses callback buffers after callback returns.
        self._queue.put_nowait(np.asarray(samples, dtype=np.float32).copy())

    def close(self) -> None:
        if not self._thread:
            return
        self._queue.put(None)
        self._thread.join(timeout=10)
        self._thread = None

    def _write_loop(self) -> None:
        with wave.open(str(self.path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            while True:
                samples = self._queue.get()
                if samples is None:
                    break
                pcm = (
                    np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max
                ).astype("<i2")
                output.writeframes(pcm.tobytes())
