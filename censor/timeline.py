from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import List

import numpy as np

from .matcher import normalize_word


@dataclass(frozen=True)
class CensorEvent:
    start_sample: int
    end_sample: int
    word: str
    variant: int = 0
    mode: str = "reverse"


class CensorTimeline:
    def __init__(self, sample_rate: int, padding_ms: int = 90):
        self.sample_rate = sample_rate
        self.padding_samples = round(sample_rate * padding_ms / 1000)
        self._events: List[CensorEvent] = []
        self._lock = Lock()

    def add(
        self,
        start_sample: int,
        end_sample: int,
        word: str,
        variant: int = 0,
        mode: str = "reverse",
    ) -> bool:
        start = max(0, start_sample - self.padding_samples)
        end = max(start + 1, end_sample + self.padding_samples)
        event = CensorEvent(start, end, word, variant, mode)
        normalized = normalize_word(word)
        with self._lock:
            # Sliding transcription windows can report the same word repeatedly.
            for old in self._events:
                overlap = min(old.end_sample, end) - max(old.start_sample, start)
                nearby = (
                    abs(old.start_sample - start) <= self.sample_rate
                    and abs(old.end_sample - end) <= self.sample_rate
                )
                if (
                    normalize_word(old.word) == normalized
                    and (overlap > 0 or nearby)
                ):
                    return False
            self._events.append(event)
            self._events.sort(key=lambda item: item.start_sample)
        return True

    def mask_for(self, start_sample: int, frame_count: int) -> np.ndarray:
        end_sample = start_sample + frame_count
        mask = np.zeros(frame_count, dtype=bool)
        events = self.events_for(start_sample, frame_count)
        for event in events:
            left = max(start_sample, event.start_sample)
            right = min(end_sample, event.end_sample)
            if right > left:
                mask[left - start_sample : right - start_sample] = True
        return mask

    def events_for(self, start_sample: int, frame_count: int) -> List[CensorEvent]:
        end_sample = start_sample + frame_count
        with self._lock:
            self._events = [
                event for event in self._events if event.end_sample >= start_sample - self.sample_rate
            ]
            return [
                event
                for event in self._events
                if event.end_sample > start_sample and event.start_sample < end_sample
            ]
