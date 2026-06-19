from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from .matcher import WordMatcher
from .recorder import TranscriptRecorder, WavRecorder
from .samples import SoundLibrary
from .timeline import CensorTimeline


@dataclass
class EngineConfig:
    input_device: Optional[int]
    output_device: Optional[int]
    sample_rate: int = 48000
    block_size: int = 960
    delay_seconds: float = 5.0
    chunk_seconds: float = 3.0
    scan_every: float = 0.8
    language: str = "ru"
    model: str = "small"
    compute_type: str = "int8"
    mode: str = "reverse"
    beep_frequency: float = 880.0
    beam_size: int = 3
    debug_transcript: bool = False
    debug_words: bool = False
    safety_margin: float = 0.8
    record_output: bool = True
    record_transcript: bool = True
    recordings_directory: str = "recordings"


class CensorEngine:
    recognition_sample_rate = 16000

    def __init__(self, config: EngineConfig, matcher: WordMatcher):
        self.config = config
        self.matcher = matcher
        self.timeline = CensorTimeline(config.sample_rate)
        sounds_path = Path(__file__).resolve().parent.parent / "assets" / "sounds"
        self.sound_library = SoundLibrary(sounds_path, config.sample_rate)
        capacity_seconds = config.delay_seconds + config.chunk_seconds + 5
        self.capacity = round(capacity_seconds * config.sample_rate)
        self.audio = np.zeros(self.capacity, dtype=np.float32)
        self.write_sample = 0
        self.audio_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.model: Optional[WhisperModel] = None
        self._recognizer: Optional[threading.Thread] = None
        self.recorder: Optional[WavRecorder] = None
        self.transcript: Optional[TranscriptRecorder] = None
        self._last_sound_variant = {"bark": -1, "meow": -1}

    def _choose_sound_variant(self, kind: str) -> int:
        count = self.sound_library.count(kind)
        if count <= 1:
            return 0
        previous = self._last_sound_variant.get(kind, -1)
        variant = random.randrange(count - 1)
        if variant >= previous:
            variant += 1
        self._last_sound_variant[kind] = variant
        return variant

    def _output_seconds(self, source_sample: int) -> float:
        return source_sample / self.config.sample_rate + self.config.delay_seconds

    def _transcript_event(
        self,
        kind: str,
        start_sample: int,
        end_sample: int,
        message: str,
    ) -> None:
        if self.transcript:
            self.transcript.event(
                kind,
                self._output_seconds(start_sample),
                self._output_seconds(end_sample),
                message,
            )

    def _write_audio(self, samples: np.ndarray) -> int:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        with self.audio_lock:
            start = self.write_sample
            indices = np.arange(start, start + len(samples)) % self.capacity
            self.audio[indices] = samples
            self.write_sample += len(samples)
            return start

    def _read_audio(self, start: int, count: int) -> np.ndarray:
        result = np.zeros(count, dtype=np.float32)
        if start < 0:
            skip = min(count, -start)
            start = 0
        else:
            skip = 0
        if skip >= count:
            return result
        with self.audio_lock:
            available_end = self.write_sample
            oldest = max(0, available_end - self.capacity)
            read_start = max(start, oldest)
            read_end = min(start + count, available_end)
            if read_end <= read_start:
                return result
            destination = skip + (read_start - start)
            indices = np.arange(read_start, read_end) % self.capacity
            result[destination : destination + len(indices)] = self.audio[indices]
        return result

    def _apply_censor(self, samples: np.ndarray, timeline_start: int) -> np.ndarray:
        mask = self.timeline.mask_for(timeline_start, len(samples))
        if not mask.any():
            return samples
        output = samples.copy()
        if self.config.mode == "mute":
            output[mask] = 0
        elif self.config.mode == "beep":
            positions = np.arange(len(samples), dtype=np.float32) + timeline_start
            tone = 0.18 * np.sin(
                2 * math.pi * self.config.beep_frequency * positions / self.config.sample_rate
            )
            output[mask] = tone[mask]
        else:
            block_end = timeline_start + len(samples)
            for event in self.timeline.events_for(timeline_start, len(samples)):
                overlap_start = max(timeline_start, event.start_sample)
                overlap_end = min(block_end, event.end_sample)
                destination_start = overlap_start - timeline_start
                destination_end = overlap_end - timeline_start
                event_offset = overlap_start - event.start_sample
                count = overlap_end - overlap_start

                if self.config.mode == "reverse":
                    # Read corresponding positions from the complete word in
                    # reverse order, so callback boundaries stay inaudible.
                    reverse_start = event.end_sample - overlap_end
                    original_start = event.start_sample + reverse_start
                    replacement = self._read_audio(original_start, count)[::-1]
                else:
                    replacement = self.sound_library.part(
                        self.config.mode,
                        event.variant,
                        event_offset,
                        count,
                        event.end_sample - event.start_sample,
                    )
                output[destination_start:destination_end] = replacement
        return output

    def _audio_callback(self, indata, outdata, frames, _time_info, status):
        if status:
            print(f"[audio] {status}", flush=True)
        outdata[:, 0] = self._process_input(indata[:, 0], frames)

    def _input_callback(self, indata, frames, _time_info, status):
        if status:
            print(f"[audio] {status}", flush=True)
        self._process_input(indata[:, 0], frames)

    def _process_input(self, input_samples: np.ndarray, frames: int) -> np.ndarray:
        block_start = self._write_audio(input_samples)
        delay = round(self.config.delay_seconds * self.config.sample_rate)
        playback_start = block_start - delay
        delayed = self._read_audio(playback_start, frames)
        processed = self._apply_censor(delayed, playback_start)
        if self.recorder:
            self.recorder.add(processed)
        return processed

    def _recognition_loop(self):
        chunk_samples = round(self.config.chunk_seconds * self.config.sample_rate)
        # Do not transcribe the very same endpoint repeatedly.
        last_endpoint = 0
        while not self.stop_event.wait(self.config.scan_every):
            with self.audio_lock:
                endpoint = self.write_sample
            if endpoint - last_endpoint < round(self.config.scan_every * self.config.sample_rate * 0.6):
                continue
            start = max(0, endpoint - chunk_samples)
            chunk = self._read_audio(start, endpoint - start)
            last_endpoint = endpoint
            if len(chunk) < self.config.sample_rate // 2 or float(np.max(np.abs(chunk))) < 0.005:
                continue
            if self.config.sample_rate != self.recognition_sample_rate:
                output_length = round(
                    len(chunk) * self.recognition_sample_rate / self.config.sample_rate
                )
                source_positions = np.linspace(0, len(chunk) - 1, output_length)
                chunk_for_recognition = np.interp(
                    source_positions,
                    np.arange(len(chunk)),
                    chunk,
                ).astype(np.float32)
            else:
                chunk_for_recognition = chunk
            try:
                segments, _ = self.model.transcribe(
                    chunk_for_recognition,
                    language=self.config.language,
                    beam_size=self.config.beam_size,
                    best_of=self.config.beam_size,
                    word_timestamps=True,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.35,
                        "min_speech_duration_ms": 80,
                        "min_silence_duration_ms": 300,
                        "speech_pad_ms": 300,
                    },
                    no_speech_threshold=0.4,
                    condition_on_previous_text=False,
                    hotwords=self.matcher.hotwords or None,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                    hallucination_silence_threshold=1.0,
                )
                for segment in segments:
                    if self.config.debug_transcript and segment.text.strip():
                        print(f"[heard] {segment.text.strip()}", flush=True)
                    segment_start = start + round(
                        segment.start * self.config.sample_rate
                    )
                    segment_end = start + round(
                        segment.end * self.config.sample_rate
                    )
                    if segment.text.strip():
                        self._transcript_event(
                            "ASR",
                            segment_start,
                            segment_end,
                            segment.text.strip(),
                        )
                    words = tuple(segment.words or ())
                    matched_segment = False
                    for word in words:
                        if self.config.debug_words:
                            matched = "MATCH" if self.matcher.matches(word.word) else "-"
                            print(
                                f"[word] {word.start:4.2f}–{word.end:4.2f} "
                                f"{word.word!r} {matched}",
                                flush=True,
                            )
                        if not self.matcher.matches(word.word):
                            continue
                        matched_segment = True
                        word_start = start + round(word.start * self.config.sample_rate)
                        word_end = start + round(word.end * self.config.sample_rate)
                        with self.audio_lock:
                            current_endpoint = self.write_sample
                        playback_position = current_endpoint - round(
                            self.config.delay_seconds * self.config.sample_rate
                        )
                        seconds_until_output = (
                            word_start - playback_position
                        ) / self.config.sample_rate
                        if word_start <= playback_position:
                            self._transcript_event(
                                "LATE",
                                word_start,
                                word_end,
                                f"{word.word.strip()!r}; не заменено",
                            )
                            print(
                                f"[late] {word.word.strip()!r}; "
                                f"опоздание {-seconds_until_output:.1f} с — "
                                "увеличьте --delay",
                                flush=True,
                            )
                            continue
                        if seconds_until_output < self.config.safety_margin:
                            self._transcript_event(
                                "RISK",
                                word_start,
                                word_end,
                                f"{word.word.strip()!r}; "
                                f"запас {seconds_until_output:.1f} с",
                            )
                            print(
                                f"[risk] {word.word.strip()!r}; до выхода только "
                                f"{seconds_until_output:.1f} с. Увеличьте --delay "
                                f"минимум на "
                                f"{self.config.safety_margin - seconds_until_output:.1f} с",
                                flush=True,
                            )
                        variant = self._choose_sound_variant(self.config.mode)
                        if self.timeline.add(
                            word_start,
                            word_end,
                            word.word,
                            variant=variant,
                        ):
                            self._transcript_event(
                                f"CENSOR:{self.config.mode}",
                                word_start,
                                word_end,
                                f"{word.word.strip()!r}; "
                                f"вариант {variant + 1}; "
                                f"запас {max(0.0, seconds_until_output):.1f} с",
                            )
                            print(
                                f"[censor] {word.word.strip()!r}; "
                                f"до выхода {max(0.0, seconds_until_output):.1f} с",
                                flush=True,
                            )
                    if (
                        self.config.debug_transcript
                        and not matched_segment
                        and self.matcher.matches_text(segment.text)
                    ):
                        tokens = " | ".join(repr(word.word) for word in words)
                        self._transcript_event(
                            "MISS",
                            segment_start,
                            segment_end,
                            f"{segment.text.strip()} | токены: "
                            f"{tokens or '(пусто)'}",
                        )
                        print(
                            "[miss] Целевое слово видно в тексте сегмента, "
                            f"но отсутствует среди word timestamps: {tokens or '(пусто)'}",
                            flush=True,
                        )
            except Exception as error:
                print(f"[recognizer] {error}", flush=True)

    def run(self):
        print(f"Загрузка модели {self.config.model!r}…", flush=True)
        self.model = WhisperModel(
            self.config.model,
            device="auto",
            compute_type=self.config.compute_type,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        recording_directory = Path(self.config.recordings_directory)
        recording_path = recording_directory / f"processed_{timestamp}.wav"
        transcript_path = recording_directory / f"processed_{timestamp}.txt"
        if self.config.record_transcript:
            self.transcript = TranscriptRecorder(transcript_path)
            self.transcript.start(
                mode=self.config.mode,
                delay=self.config.delay_seconds,
                model=self.config.model,
                words=self.matcher.hotwords,
            )
            print(f"Журнал расшифровки: {transcript_path}", flush=True)
        self._recognizer = threading.Thread(
            target=self._recognition_loop, name="recognizer", daemon=True
        )
        self._recognizer.start()
        if self.config.record_output:
            self.recorder = WavRecorder(recording_path, self.config.sample_rate)
            self.recorder.start()
            print(f"Запись обработанного звука: {recording_path}", flush=True)
        print(
            f"Фильтр запущен: задержка {self.config.delay_seconds:.1f} с, "
            f"режим {self.config.mode}, "
            f"вывод {'отключён' if self.config.output_device is None else 'включён'}. "
            "Ctrl+C — остановить.",
            flush=True,
        )
        try:
            if self.config.output_device is None:
                stream = sd.InputStream(
                    device=self.config.input_device,
                    samplerate=self.config.sample_rate,
                    blocksize=self.config.block_size,
                    channels=1,
                    dtype="float32",
                    callback=self._input_callback,
                )
            else:
                stream = sd.Stream(
                    device=(self.config.input_device, self.config.output_device),
                    samplerate=self.config.sample_rate,
                    blocksize=self.config.block_size,
                    channels=1,
                    dtype="float32",
                    callback=self._audio_callback,
                )
            with stream:
                while not self.stop_event.wait(0.5):
                    pass
        finally:
            self.stop_event.set()
            if self._recognizer:
                self._recognizer.join(timeout=2)
            if self.recorder:
                self.recorder.close()
                print(f"Запись сохранена: {self.recorder.path}", flush=True)
            if self.transcript:
                self.transcript.close()
                print(f"Расшифровка сохранена: {self.transcript.path}", flush=True)
