import unittest

from censor.streaming import StreamingWordStabilizer, WordObservation


def word(text, start, end):
    return WordObservation(text, start, end)


class StreamingWordStabilizerTests(unittest.TestCase):
    def test_commits_repeated_hypothesis_only_once(self):
        stabilizer = StreamingWordStabilizer(
            sample_rate=1000,
            confirmation_count=2,
            stability_delay=1.0,
            time_tolerance=0.2,
        )
        self.assertEqual(stabilizer.ingest([word("четыре", 100, 400)], 500), [])
        committed = stabilizer.ingest([word("четыре,", 120, 420)], 700)
        self.assertEqual([item.text for item in committed], ["четыре,"])
        self.assertEqual(
            stabilizer.ingest([word("четыре", 110, 410)], 900),
            [],
        )

    def test_commits_mature_word_without_second_hypothesis(self):
        stabilizer = StreamingWordStabilizer(
            sample_rate=1000,
            confirmation_count=2,
            stability_delay=0.5,
        )
        committed = stabilizer.ingest([word("лимон", 100, 300)], 900)
        self.assertEqual([item.text for item in committed], ["лимон"])

    def test_keeps_two_real_repetitions(self):
        stabilizer = StreamingWordStabilizer(
            sample_rate=1000,
            confirmation_count=1,
            time_tolerance=0.2,
        )
        committed = stabilizer.ingest(
            [word("раз", 100, 250), word("раз", 600, 750)],
            1000,
        )
        self.assertEqual(len(committed), 2)

    def test_replaces_unconfirmed_changed_hypothesis(self):
        stabilizer = StreamingWordStabilizer(
            sample_rate=1000,
            confirmation_count=2,
            stability_delay=0.7,
            time_tolerance=0.3,
        )
        self.assertEqual(stabilizer.ingest([word("булки", 100, 400)], 500), [])
        self.assertEqual(stabilizer.ingest([word("булочка", 100, 450)], 700), [])
        committed = stabilizer.ingest([word("булочка", 120, 460)], 900)
        self.assertEqual([item.text for item in committed], ["булочка"])


if __name__ == "__main__":
    unittest.main()
