from __future__ import annotations

import math
import os
import random
import json
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
from .matcher import normalize_word
from .paths import resource_root
from .recorder import TranscriptRecorder, WavRecorder
from .samples import SoundLibrary
from .streaming import StreamingWordStabilizer, WordObservation
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
    debug_hypotheses: bool = False
    debug_words: bool = False
    safety_margin: float = 0.8
    record_output: bool = True
    record_transcript: bool = True
    recordings_directory: str = "recordings"
    runtime_control_file: str = ".runtime-control.json"
    runtime_status_file: str = ".runtime-status.json"
    effect_volume: float = 1.0
    confirmation_count: int = 2
    stability_delay: float = 0.7
    word_time_tolerance: float = 0.4
    censor_lead_ms: int = 20
    censor_tail_ms: int = 80
    crossfade_ms: int = 8
    input_activity_threshold: float = 0.003
    custom_sounds_directory: str = "custom_sounds"


class CensorEngine:
    recognition_sample_rate = 16000

    def __init__(self, config: EngineConfig, matcher: WordMatcher):
        self.config = config
        self.matcher = matcher
        self.timeline = CensorTimeline(
            config.sample_rate,
            lead_padding_ms=config.censor_lead_ms,
            tail_padding_ms=config.censor_tail_ms,
        )
        self.stabilizer = StreamingWordStabilizer(
            sample_rate=config.sample_rate,
            confirmation_count=config.confirmation_count,
            stability_delay=config.stability_delay,
            time_tolerance=config.word_time_tolerance,
        )
        sounds_path = resource_root() / "assets" / "sounds"
        self.sound_library = SoundLibrary(
            sounds_path,
            config.sample_rate,
            custom_root=config.custom_sounds_directory,
        )
        capacity_seconds = config.delay_seconds + config.chunk_seconds + 5
        self.capacity = round(capacity_seconds * config.sample_rate)
        self.audio = np.zeros(self.capacity, dtype=np.float32)
        self.write_sample = 0
        self.audio_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.model: Optional[WhisperModel] = None
        self._recognizer: Optional[threading.Thread] = None
        self._control_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None
        self._mode_lock = threading.Lock()
        self._settings_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._status_write_lock = threading.Lock()
        self._last_status_wall = time.monotonic()
        self._last_status_cpu = time.process_time()
        self._runtime_status = {
            "phase": "starting",
            "model_state": "waiting",
            "audio_state": "waiting",
            "mic_rms": 0.0,
            "mic_peak": 0.0,
            "clipping": False,
            "cpu_percent": 0.0,
            "asr_state": "waiting",
            "asr_duration": None,
            "last_error": None,
        }
        self.recorder: Optional[WavRecorder] = None
        self.transcript: Optional[TranscriptRecorder] = None
        self._last_sound_variant = {"bark": -1, "meow": -1, "custom": -1}
        self._last_stable_word_end = 0
        self.stats = {
            "censored": 0,
            "miss": 0,
            "risk": 0,
            "late": 0,
            "min_margin": None,
            "modes": {},
        }

    def current_mode(self) -> str:
        with self._mode_lock:
            return self.config.mode

    def set_mode(self, mode: str) -> None:
        if mode not in {"reverse", "beep", "bark", "meow", "custom", "mute"}:
            return
        with self._mode_lock:
            old_mode = self.config.mode
            self.config.mode = mode
        if old_mode != mode:
            print(f"[mode] {old_mode} → {mode}", flush=True)
            with self.audio_lock:
                sample = self.write_sample
            self._transcript_event("MODE", sample, sample, f"{old_mode} → {mode}")

    def effect_volume(self) -> float:
        with self._settings_lock:
            return self.config.effect_volume

    def set_effect_volume(self, volume: float) -> None:
        volume = max(0.0, min(2.0, float(volume)))
        with self._settings_lock:
            self.config.effect_volume = volume

    def _control_loop(self) -> None:
        path = Path(self.config.runtime_control_file)
        last_mtime = 0
        while not self.stop_event.wait(0.15):
            try:
                mtime = path.stat().st_mtime_ns
                if mtime == last_mtime:
                    continue
                last_mtime = mtime
                data = json.loads(path.read_text(encoding="utf-8"))
                self.set_mode(str(data.get("mode", "")))
                if "effect_volume" in data:
                    self.set_effect_volume(float(data["effect_volume"]))
            except (OSError, json.JSONDecodeError):
                continue

    def _set_runtime_status(self, **values) -> None:
        with self._status_lock:
            self._runtime_status.update(values)

    def _update_audio_metrics(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(samples):
            return
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        with self._status_lock:
            # A little smoothing keeps the meter readable without hiding peaks.
            previous_rms = float(self._runtime_status["mic_rms"])
            previous_peak = float(self._runtime_status["mic_peak"])
            self._runtime_status["mic_rms"] = previous_rms * 0.72 + rms * 0.28
            self._runtime_status["mic_peak"] = max(peak, previous_peak * 0.78)
            self._runtime_status["clipping"] = peak >= 0.98 or (
                bool(self._runtime_status["clipping"]) and previous_peak >= 0.9
            )

    def runtime_status(self) -> dict:
        with self._status_lock:
            status = dict(self._runtime_status)
        minimum = self.stats["min_margin"]
        if status["phase"] == "error" or status["audio_state"] == "error":
            overall = "red"
        elif status["phase"] in {"starting", "loading"}:
            overall = "yellow"
        elif status["phase"] == "running":
            if self.stats["late"]:
                overall = "red"
            elif (
                self.stats["risk"]
                or status["clipping"]
                or (minimum is not None and minimum < self.config.safety_margin)
            ):
                overall = "yellow"
            else:
                overall = "green"
        else:
            overall = "idle"
        return {
            "updated_at": time.time(),
            "overall": overall,
            **status,
            "mode": self.current_mode(),
            "censored": self.stats["censored"],
            "risk": self.stats["risk"],
            "late": self.stats["late"],
            "min_margin": minimum,
            "delay_seconds": self.config.delay_seconds,
            "chunk_seconds": self.config.chunk_seconds,
            "scan_every": self.config.scan_every,
        }

    def _write_runtime_status(self) -> None:
        path = Path(self.config.runtime_status_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with self._status_write_lock:
            temporary.write_text(
                json.dumps(self.runtime_status(), ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(path)

    def _status_loop(self) -> None:
        while not self.stop_event.wait(0.25):
            try:
                now = time.monotonic()
                cpu_now = time.process_time()
                wall_delta = max(0.001, now - self._last_status_wall)
                cpu_cores = max(
                    0.0,
                    min(999.0, (cpu_now - self._last_status_cpu) / wall_delta * 100),
                )
                cpu_percent = min(100.0, cpu_cores / max(1, os.cpu_count() or 1))
                self._last_status_wall = now
                self._last_status_cpu = cpu_now
                self._set_runtime_status(
                    cpu_percent=cpu_percent,
                    cpu_cores_used=cpu_cores / 100,
                )
                self._write_runtime_status()
            except OSError:
                pass

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
        block_end = timeline_start + len(samples)
        for event in self.timeline.events_for(timeline_start, len(samples)):
            overlap_start = max(timeline_start, event.start_sample)
            overlap_end = min(block_end, event.end_sample)
            destination_start = overlap_start - timeline_start
            destination_end = overlap_end - timeline_start
            event_offset = overlap_start - event.start_sample
            count = overlap_end - overlap_start

            if event.mode == "mute":
                replacement = np.zeros(count, dtype=np.float32)
            elif event.mode == "beep":
                positions = np.arange(overlap_start, overlap_end, dtype=np.float32)
                replacement = event.volume * 0.18 * np.sin(
                    2
                    * math.pi
                    * self.config.beep_frequency
                    * positions
                    / self.config.sample_rate
                )
            elif event.mode == "reverse":
                reverse_start = event.end_sample - overlap_end
                original_start = event.start_sample + reverse_start
                replacement = self._read_audio(original_start, count)[::-1]
            else:
                replacement = self.sound_library.part(
                    event.mode,
                    event.variant,
                    event_offset,
                    count,
                    event.end_sample - event.start_sample,
                )
                replacement *= event.volume
            fade_samples = min(
                round(self.config.crossfade_ms * self.config.sample_rate / 1000),
                count // 2,
            )
            if fade_samples > 0:
                relative = np.arange(event_offset, event_offset + count)
                event_length = event.end_sample - event.start_sample
                blend = np.ones(count, dtype=np.float32)
                blend = np.minimum(
                    blend, np.clip(relative / fade_samples, 0.0, 1.0)
                )
                blend = np.minimum(
                    blend,
                    np.clip(
                        (event_length - 1 - relative) / fade_samples,
                        0.0,
                        1.0,
                    ),
                )
                original = samples[destination_start:destination_end]
                replacement = original * (1.0 - blend) + replacement * blend
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
        self._update_audio_metrics(input_samples)
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
            if (
                len(chunk) < self.config.sample_rate // 2
                or float(np.sqrt(np.mean(np.square(chunk))))
                < self.config.input_activity_threshold
            ):
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
            recognition_started = time.monotonic()
            self._set_runtime_status(asr_state="transcribing")
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
                segment_list = list(segments)
                observations = []
                hypotheses = []
                for segment in segment_list:
                    if segment.text.strip():
                        hypotheses.append(segment.text.strip())
                    for word in segment.words or ():
                        observation = WordObservation(
                            text=word.word,
                            start_sample=start
                            + round(word.start * self.config.sample_rate),
                            end_sample=start
                            + round(word.end * self.config.sample_rate),
                            probability=float(getattr(word, "probability", 1.0)),
                        )
                        observations.append(observation)
                        if self.config.debug_words:
                            matched = (
                                "MATCH" if self.matcher.matches(word.word) else "-"
                            )
                            print(
                                f"[word] {observation.start_sample / self.config.sample_rate:6.2f}–"
                                f"{observation.end_sample / self.config.sample_rate:6.2f} "
                                f"{word.word!r} {matched}",
                                flush=True,
                            )
                if self.config.debug_hypotheses and hypotheses:
                    print(f"[hypothesis] {' '.join(hypotheses)}", flush=True)

                stable_words = self.stabilizer.ingest(observations, endpoint)
                if stable_words:
                    stable_text = "".join(word.text for word in stable_words).strip()
                    if self.config.debug_transcript:
                        print(f"[stable] {stable_text}", flush=True)
                    self._transcript_event(
                        "ASR:STABLE",
                        stable_words[0].start_sample,
                        stable_words[-1].end_sample,
                        stable_text,
                    )
                for word in stable_words:
                    previous_end = self._last_stable_word_end
                    self._handle_stable_word(word, previous_end)
                    self._last_stable_word_end = max(
                        self._last_stable_word_end,
                        word.end_sample,
                    )
            except Exception as error:
                print(f"[recognizer] {error}", flush=True)
            finally:
                self._set_runtime_status(
                    asr_state="idle",
                    asr_duration=time.monotonic() - recognition_started,
                )

    def _adjusted_word_start(
        self,
        word: WordObservation,
        previous_word_end: int = 0,
    ) -> int:
        characters = len(normalize_word(word.text))
        if not characters:
            return word.start_sample
        # Whisper occasionally places the beginning of a long word too late.
        # Estimate a conservative minimum duration, but never cross the end
        # of the preceding confirmed word.
        estimated_ms = min(520, max(180, characters * 55))
        estimated_samples = round(
            estimated_ms * self.config.sample_rate / 1000
        )
        desired_start = word.end_sample - estimated_samples
        maximum_extension = round(0.3 * self.config.sample_rate)
        adjusted = max(
            word.start_sample - maximum_extension,
            min(word.start_sample, desired_start),
        )
        return max(previous_word_end, adjusted, 0)

    def _handle_stable_word(
        self,
        word: WordObservation,
        previous_word_end: int = 0,
    ) -> None:
        if not self.matcher.matches(word.text):
            return
        word_start = self._adjusted_word_start(word, previous_word_end)
        word_end = word.end_sample
        with self.audio_lock:
            current_endpoint = self.write_sample
        playback_position = current_endpoint - round(
            self.config.delay_seconds * self.config.sample_rate
        )
        seconds_until_output = (
            word_start - playback_position
        ) / self.config.sample_rate
        if word_start <= playback_position:
            self.stats["late"] += 1
            self._transcript_event(
                "LATE", word_start, word_end, f"{word.text.strip()!r}; не заменено"
            )
            print(
                f"[late] {word.text.strip()!r}; "
                f"опоздание {-seconds_until_output:.1f} с — увеличьте --delay",
                flush=True,
            )
            return
        if seconds_until_output < self.config.safety_margin:
            self.stats["risk"] += 1
            self._transcript_event(
                "RISK",
                word_start,
                word_end,
                f"{word.text.strip()!r}; запас {seconds_until_output:.1f} с",
            )
            print(
                f"[risk] {word.text.strip()!r}; до выхода только "
                f"{seconds_until_output:.1f} с",
                flush=True,
            )
        event_mode = self.current_mode()
        event_volume = self.effect_volume()
        variant = self._choose_sound_variant(event_mode)
        if self.timeline.add(
            word_start,
            word_end,
            word.text,
            variant=variant,
            mode=event_mode,
            volume=event_volume,
        ):
            self.stats["censored"] += 1
            modes = self.stats["modes"]
            modes[event_mode] = modes.get(event_mode, 0) + 1
            minimum = self.stats["min_margin"]
            self.stats["min_margin"] = (
                seconds_until_output
                if minimum is None
                else min(minimum, seconds_until_output)
            )
            self._transcript_event(
                f"CENSOR:{event_mode}",
                word_start,
                word_end,
                f"{word.text.strip()!r}; вариант {variant + 1}; "
                f"запас {max(0.0, seconds_until_output):.1f} с",
            )
            print(
                f"[censor] {word.text.strip()!r}; "
                f"до выхода {max(0.0, seconds_until_output):.1f} с",
                flush=True,
            )

    def run(self):
        self._set_runtime_status(phase="loading", model_state="loading")
        self._write_runtime_status()
        self._status_thread = threading.Thread(
            target=self._status_loop, name="runtime-status", daemon=True
        )
        self._status_thread.start()
        print(f"Загрузка модели {self.config.model!r}…", flush=True)
        try:
            self.model = WhisperModel(
                self.config.model,
                device="auto",
                compute_type=self.config.compute_type,
            )
            self._set_runtime_status(
                model_state="ready",
                audio_state="starting",
                asr_state="idle",
            )
        except Exception as error:
            self._set_runtime_status(
                phase="error",
                model_state="error",
                last_error=str(error),
            )
            self._write_runtime_status()
            self.stop_event.set()
            raise
        control_path = Path(self.config.runtime_control_file)
        control_path.write_text(
            json.dumps(
                {
                    "mode": self.current_mode(),
                    "effect_volume": self.effect_volume(),
                }
            ),
            encoding="utf-8",
        )
        self._control_thread = threading.Thread(
            target=self._control_loop, name="runtime-control", daemon=True
        )
        self._control_thread.start()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        recording_directory = Path(self.config.recordings_directory)
        recording_path = recording_directory / f"processed_{timestamp}.wav"
        transcript_path = recording_directory / f"processed_{timestamp}.txt"
        if self.config.record_transcript:
            self.transcript = TranscriptRecorder(transcript_path)
            self.transcript.start(
                mode=self.current_mode(),
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
            f"режим {self.current_mode()}, "
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
                self._set_runtime_status(phase="running", audio_state="running")
                while not self.stop_event.wait(0.5):
                    pass
        except Exception as error:
            self._set_runtime_status(
                phase="error",
                audio_state="error",
                last_error=str(error),
            )
            raise
        finally:
            self.stop_event.set()
            if self._recognizer:
                self._recognizer.join(timeout=2)
            if self._control_thread:
                self._control_thread.join(timeout=1)
            with self._status_lock:
                failed = self._runtime_status["phase"] == "error"
            if not failed:
                self._set_runtime_status(
                    phase="stopped",
                    audio_state="stopped",
                    model_state="stopped",
                )
            self._write_runtime_status()
            if self._status_thread:
                self._status_thread.join(timeout=1)
            if self.recorder:
                self.recorder.close()
                print(f"Запись сохранена: {self.recorder.path}", flush=True)
            if self.transcript:
                self.transcript.close()
                print(f"Расшифровка сохранена: {self.transcript.path}", flush=True)
