from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from .matcher import normalize_word


@dataclass(frozen=True)
class WordObservation:
    text: str
    start_sample: int
    end_sample: int
    probability: float = 1.0


@dataclass
class _Candidate:
    observation: WordObservation
    hits: int
    last_scan: int


class StreamingWordStabilizer:
    """Commit new words from overlapping ASR windows exactly once."""

    def __init__(
        self,
        sample_rate: int,
        confirmation_count: int = 2,
        stability_delay: float = 0.7,
        time_tolerance: float = 0.4,
    ):
        self.sample_rate = sample_rate
        self.confirmation_count = max(1, confirmation_count)
        self.stability_samples = round(stability_delay * sample_rate)
        self.tolerance_samples = round(time_tolerance * sample_rate)
        self._scan = 0
        self._candidates: List[_Candidate] = []
        self._committed: List[WordObservation] = []

    def ingest(
        self,
        observations: Iterable[WordObservation],
        endpoint_sample: int,
    ) -> List[WordObservation]:
        self._scan += 1
        for observation in sorted(observations, key=lambda item: item.start_sample):
            normalized = normalize_word(observation.text)
            if not normalized or self._already_committed(observation, normalized):
                continue
            # A newer window can revise a word at the same timestamp. Drop the
            # unconfirmed old spelling instead of eventually committing both.
            self._candidates = [
                candidate
                for candidate in self._candidates
                if not (
                    candidate.hits < self.confirmation_count
                    and candidate.last_scan < self._scan
                    and normalize_word(candidate.observation.text) != normalized
                    and abs(
                        candidate.observation.start_sample
                        - observation.start_sample
                    )
                    <= self.tolerance_samples
                )
            ]
            candidate = self._find_candidate(observation, normalized)
            if candidate is None:
                self._candidates.append(_Candidate(observation, 1, self._scan))
            elif candidate.last_scan != self._scan:
                candidate.hits += 1
                candidate.last_scan = self._scan
                # Prefer the latest timestamps; Whisper usually refines them.
                candidate.observation = observation

        ready = []
        remaining = []
        for candidate in self._candidates:
            age = endpoint_sample - candidate.observation.end_sample
            confirmed = candidate.hits >= self.confirmation_count
            mature = age >= self.stability_samples
            if confirmed or mature:
                normalized = normalize_word(candidate.observation.text)
                if not self._already_committed(candidate.observation, normalized):
                    ready.append(candidate.observation)
                    self._committed.append(candidate.observation)
            elif age < self.stability_samples * 4:
                remaining.append(candidate)
        self._candidates = remaining
        self._prune_committed(endpoint_sample)
        return sorted(ready, key=lambda item: item.start_sample)

    def _find_candidate(
        self,
        observation: WordObservation,
        normalized: str,
    ) -> _Candidate | None:
        matches = [
            candidate
            for candidate in self._candidates
            if normalize_word(candidate.observation.text) == normalized
            and abs(candidate.observation.start_sample - observation.start_sample)
            <= self.tolerance_samples
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda candidate: abs(
                candidate.observation.start_sample - observation.start_sample
            ),
        )

    def _already_committed(
        self,
        observation: WordObservation,
        normalized: str,
    ) -> bool:
        return any(
            normalize_word(committed.text) == normalized
            and abs(committed.start_sample - observation.start_sample)
            <= self.tolerance_samples
            for committed in self._committed
        )

    def _prune_committed(self, endpoint_sample: int) -> None:
        keep_after = endpoint_sample - self.sample_rate * 30
        self._committed = [
            word for word in self._committed if word.end_sample >= keep_after
        ]
