from __future__ import annotations

import wave
from pathlib import Path
from typing import Dict, List

import numpy as np


class SoundLibrary:
    def __init__(self, root: str | Path, sample_rate: int):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.sounds: Dict[str, List[np.ndarray]] = {
            "bark": self._load_kind("bark"),
            "meow": self._load_kind("meow"),
        }

    def _load_kind(self, kind: str) -> List[np.ndarray]:
        return [self._load_wav(path) for path in sorted((self.root / kind).glob("*.wav"))]

    def _load_wav(self, path: Path) -> np.ndarray:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            source_rate = audio.getframerate()
            width = audio.getsampwidth()
            frames = audio.readframes(audio.getnframes())
        if width != 2:
            raise ValueError(f"{path}: поддерживается только 16-bit PCM WAV")
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        samples /= np.iinfo(np.int16).max
        if source_rate != self.sample_rate:
            output_length = round(len(samples) * self.sample_rate / source_rate)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, output_length),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        return samples

    def count(self, kind: str) -> int:
        return len(self.sounds.get(kind, ()))

    def part(
        self,
        kind: str,
        variant: int,
        offset: int,
        count: int,
        total_samples: int,
    ) -> np.ndarray:
        choices = self.sounds.get(kind, ())
        if not choices:
            return np.zeros(count, dtype=np.float32)
        source = choices[variant % len(choices)]
        if len(source) >= total_samples:
            fitted = source[:total_samples]
        else:
            repeats = (total_samples + len(source) - 1) // len(source)
            fitted = np.tile(source, repeats)[:total_samples]

        result = fitted[offset : offset + count].copy()
        positions = np.arange(offset, offset + count)
        fade = max(1, min(round(self.sample_rate * 0.015), total_samples // 4))
        gain = np.ones(count, dtype=np.float32)
        gain = np.minimum(gain, np.clip(positions / fade, 0.0, 1.0))
        gain = np.minimum(
            gain,
            np.clip((total_samples - 1 - positions) / fade, 0.0, 1.0),
        )
        return result * gain
