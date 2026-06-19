import unittest

import numpy as np

from censor.engine import CensorEngine, EngineConfig
from censor.matcher import WordMatcher


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
            self.assertGreater(float(np.max(np.abs(complete))), 0.01)

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


if __name__ == "__main__":
    unittest.main()
