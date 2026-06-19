import unittest
from pathlib import Path

import numpy as np

from censor.samples import SoundLibrary


class SoundLibraryTests(unittest.TestCase):
    def test_loads_four_real_samples_for_each_mode(self):
        library = SoundLibrary(Path("assets/sounds"), sample_rate=48000)
        self.assertEqual(library.count("bark"), 4)
        self.assertEqual(library.count("meow"), 4)

    def test_returns_requested_part_of_same_variant(self):
        library = SoundLibrary(Path("assets/sounds"), sample_rate=48000)
        first = library.part("bark", 2, 0, 1000, 2000)
        second = library.part("bark", 2, 1000, 1000, 2000)
        complete = library.part("bark", 2, 0, 2000, 2000)
        np.testing.assert_allclose(np.concatenate((first, second)), complete)


if __name__ == "__main__":
    unittest.main()
