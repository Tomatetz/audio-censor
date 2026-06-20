import unittest
import tempfile
import shutil
from pathlib import Path

import numpy as np

from censor.samples import SoundLibrary


class SoundLibraryTests(unittest.TestCase):
    def test_loads_multiple_real_samples_for_each_mode(self):
        library = SoundLibrary(Path("assets/sounds"), sample_rate=48000)
        self.assertGreaterEqual(library.count("bark"), 4)
        self.assertGreaterEqual(library.count("meow"), 4)

    def test_returns_requested_part_of_same_variant(self):
        library = SoundLibrary(Path("assets/sounds"), sample_rate=48000)
        first = library.part("bark", 2, 0, 1000, 2000)
        second = library.part("bark", 2, 1000, 1000, 2000)
        complete = library.part("bark", 2, 0, 2000, 2000)
        np.testing.assert_allclose(np.concatenate((first, second)), complete)

    def test_loads_wav_from_user_custom_folder(self):
        bundled_count = SoundLibrary(
            Path("assets/sounds"),
            sample_rate=48000,
        ).count("custom")
        with tempfile.TemporaryDirectory() as directory:
            custom = Path(directory)
            shutil.copy("assets/sounds/meow/meow_1.wav", custom / "anything.wav")
            library = SoundLibrary(
                Path("assets/sounds"),
                sample_rate=48000,
                custom_root=custom,
            )
            self.assertEqual(library.count("custom"), bundled_count + 1)

    def test_custom_sound_plays_once_and_then_becomes_silent(self):
        library = SoundLibrary(Path("assets/sounds"), sample_rate=48000)
        library.sounds["custom"] = [np.ones(100, dtype=np.float32)]

        result = library.part("custom", 0, 0, 300, 300)

        self.assertGreater(float(np.max(result[:100])), 0)
        np.testing.assert_array_equal(result[100:], np.zeros(200, dtype=np.float32))

    def test_normalizes_effect_loudness_and_limits_peak(self):
        library = SoundLibrary(Path("assets/sounds"), sample_rate=48000)
        quiet = np.concatenate(
            (np.zeros(100), np.full(1000, 0.01), np.zeros(100))
        ).astype(np.float32)
        loud = np.concatenate(
            (np.zeros(100), np.full(1000, 0.8), np.zeros(100))
        ).astype(np.float32)

        normalized_quiet = library._normalize(quiet)
        normalized_loud = library._normalize(loud)
        quiet_rms = float(np.sqrt(np.mean(normalized_quiet[100:1100] ** 2)))
        loud_rms = float(np.sqrt(np.mean(normalized_loud[100:1100] ** 2)))

        self.assertAlmostEqual(quiet_rms, loud_rms, places=5)
        self.assertLessEqual(
            float(np.max(np.abs(normalized_loud))),
            library.peak_limit + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
