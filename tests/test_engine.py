import unittest
from unittest.mock import patch

import numpy as np

from censor.engine import CensorEngine, EngineConfig
from censor.matcher import WordMatcher
from censor.streaming import WordObservation


class EngineTests(unittest.TestCase):
    def test_reverses_complete_event_across_callback_blocks(self):
        config = EngineConfig(
            input_device=None,
            output_device=None,
            sample_rate=1000,
            mode="reverse",
            crossfade_ms=0,
        )
        engine = CensorEngine(config, WordMatcher([]))
        source = np.arange(300, dtype=np.float32)
        engine._write_audio(source)
        engine.timeline.lead_padding_samples = 0
        engine.timeline.tail_padding_samples = 0
        engine.timeline.add(100, 200, "word")

        first = engine._apply_censor(source[100:150], 100)
        second = engine._apply_censor(source[150:200], 150)

        np.testing.assert_array_equal(
            np.concatenate((first, second)),
            source[100:200][::-1],
        )

    def test_processes_input_without_output_stream(self):
        config = EngineConfig(
            input_device=0,
            output_device=None,
            sample_rate=1000,
            delay_seconds=0.1,
        )
        engine = CensorEngine(config, WordMatcher([]))
        source = np.arange(200, dtype=np.float32)

        first = engine._process_input(source[:100], 100)
        second = engine._process_input(source[100:], 100)

        np.testing.assert_array_equal(first, np.zeros(100, dtype=np.float32))
        np.testing.assert_array_equal(second, source[:100])

    def test_animal_replacement_is_continuous_across_blocks(self):
        for mode in ("bark", "meow"):
            config = EngineConfig(
                input_device=None,
                output_device=None,
                sample_rate=1000,
                mode=mode,
                crossfade_ms=0,
            )
            engine = CensorEngine(config, WordMatcher([]))
            source = np.zeros(300, dtype=np.float32)
            engine._write_audio(source)
            engine.timeline.lead_padding_samples = 0
            engine.timeline.tail_padding_samples = 0
            engine.timeline.add(100, 200, "word", variant=1, mode=mode, volume=1.0)

            first = engine._apply_censor(source[100:150], 100)
            second = engine._apply_censor(source[150:200], 150)
            complete = engine.sound_library.part(mode, 1, 0, 100, 100)

            np.testing.assert_allclose(
                np.concatenate((first, second)),
                complete,
                atol=1e-6,
            )
            self.assertGreater(float(np.max(np.abs(complete))), 0.001)

    def test_does_not_repeat_same_animal_variant_consecutively(self):
        engine = CensorEngine(
            EngineConfig(input_device=None, output_device=None),
            WordMatcher([]),
        )
        for mode in ("bark", "meow"):
            variants = [engine._choose_sound_variant(mode) for _ in range(20)]
            self.assertTrue(all(a != b for a, b in zip(variants, variants[1:])))

    def test_existing_event_keeps_mode_after_runtime_change(self):
        engine = CensorEngine(
            EngineConfig(
                input_device=None,
                output_device=None,
                sample_rate=1000,
                mode="bark",
                crossfade_ms=0,
            ),
            WordMatcher([]),
        )
        source = np.zeros(300, dtype=np.float32)
        engine._write_audio(source)
        engine.timeline.lead_padding_samples = 0
        engine.timeline.tail_padding_samples = 0
        engine.timeline.add(100, 200, "word", variant=0, mode="bark", volume=1.0)
        engine.set_mode("meow")

        actual = engine._apply_censor(source[100:200], 100)
        expected = engine.sound_library.part("bark", 0, 0, 100, 100)
        np.testing.assert_allclose(actual, expected)

    def test_existing_event_keeps_volume_after_runtime_change(self):
        engine = CensorEngine(
            EngineConfig(
                input_device=None,
                output_device=None,
                sample_rate=1000,
                mode="bark",
                effect_volume=0.5,
                crossfade_ms=0,
            ),
            WordMatcher([]),
        )
        source = np.zeros(300, dtype=np.float32)
        engine._write_audio(source)
        engine.timeline.lead_padding_samples = 0
        engine.timeline.tail_padding_samples = 0
        engine.timeline.add(100, 200, "word", variant=0, mode="bark", volume=0.5)
        engine.set_effect_volume(2.0)
        actual = engine._apply_censor(source[100:200], 100)
        expected = engine.sound_library.part("bark", 0, 0, 100, 100) * 0.5
        np.testing.assert_allclose(actual, expected)

    def test_runtime_status_contains_live_audio_and_censor_metrics(self):
        engine = CensorEngine(
            EngineConfig(input_device=None, output_device=None, sample_rate=1000),
            WordMatcher([]),
        )
        engine._set_runtime_status(
            phase="running",
            model_state="ready",
            audio_state="running",
        )
        engine._update_audio_metrics(np.full(100, 0.25, dtype=np.float32))
        engine.stats["censored"] = 3
        engine.stats["min_margin"] = 1.4

        status = engine.runtime_status()

        self.assertEqual(status["overall"], "green")
        self.assertEqual(status["censored"], 3)
        self.assertEqual(status["min_margin"], 1.4)
        self.assertGreater(status["mic_rms"], 0)
        self.assertEqual(status["delay_seconds"], 5.0)
        self.assertEqual(status["chunk_seconds"], 3.0)
        self.assertIn("cpu_percent", status)

    def test_runtime_status_warns_about_clipping(self):
        engine = CensorEngine(
            EngineConfig(input_device=None, output_device=None),
            WordMatcher([]),
        )
        engine._set_runtime_status(
            phase="running",
            model_state="ready",
            audio_state="running",
        )
        engine._update_audio_metrics(np.ones(100, dtype=np.float32))

        self.assertEqual(engine.runtime_status()["overall"], "yellow")
        self.assertTrue(engine.runtime_status()["clipping"])

    def test_cpu_load_is_normalized_to_machine_capacity(self):
        engine = CensorEngine(
            EngineConfig(input_device=None, output_device=None),
            WordMatcher([]),
        )
        engine._last_status_wall = 10.0
        engine._last_status_cpu = 4.0
        engine.stop_event.wait = lambda _timeout: engine.stop_event.is_set()
        engine._write_runtime_status = lambda: engine.stop_event.set()

        with (
            patch("censor.engine.time.monotonic", return_value=11.0),
            patch("censor.engine.time.process_time", return_value=8.0),
            patch("censor.engine.os.cpu_count", return_value=8),
        ):
            engine._status_loop()

        status = engine.runtime_status()
        self.assertEqual(status["cpu_percent"], 50.0)
        self.assertEqual(status["cpu_cores_used"], 4.0)

    def test_extends_late_start_of_long_word_without_touching_previous_word(self):
        engine = CensorEngine(
            EngineConfig(
                input_device=None,
                output_device=None,
                sample_rate=1000,
            ),
            WordMatcher([]),
        )
        word = WordObservation(
            text=" карандаш",
            start_sample=1000,
            end_sample=1200,
            probability=0.9,
        )

        self.assertEqual(engine._adjusted_word_start(word, 700), 760)
        self.assertEqual(engine._adjusted_word_start(word, 900), 900)

    def test_does_not_extend_normal_long_word_timestamp(self):
        engine = CensorEngine(
            EngineConfig(
                input_device=None,
                output_device=None,
                sample_rate=1000,
            ),
            WordMatcher([]),
        )
        word = WordObservation(
            text=" карандаш",
            start_sample=700,
            end_sample=1200,
            probability=0.9,
        )

        self.assertEqual(engine._adjusted_word_start(word, 600), 700)


if __name__ == "__main__":
    unittest.main()
