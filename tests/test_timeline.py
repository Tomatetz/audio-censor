import unittest

import numpy as np

from censor.timeline import CensorTimeline


class TimelineTests(unittest.TestCase):
    def test_masks_intersection_with_output_block(self):
        timeline = CensorTimeline(sample_rate=1000, padding_ms=0)
        timeline.add(1050, 1100, "word")
        expected = np.zeros(100, dtype=bool)
        expected[50:] = True
        np.testing.assert_array_equal(timeline.mask_for(1000, 100), expected)

    def test_deduplicates_overlapping_word(self):
        timeline = CensorTimeline(sample_rate=1000, padding_ms=0)
        self.assertTrue(timeline.add(100, 200, "word"))
        self.assertFalse(timeline.add(120, 220, "WORD"))

    def test_deduplicates_punctuation_and_timestamp_drift(self):
        timeline = CensorTimeline(sample_rate=1000, padding_ms=0)
        self.assertTrue(timeline.add(100, 200, "трактор..."))
        self.assertFalse(timeline.add(700, 800, "Трактор"))

    def test_returns_events_intersecting_block(self):
        timeline = CensorTimeline(sample_rate=1000, padding_ms=0)
        timeline.add(100, 200, "word")
        timeline.add(400, 500, "later")
        events = timeline.events_for(150, 100)
        self.assertEqual([event.word for event in events], ["word"])


if __name__ == "__main__":
    unittest.main()
