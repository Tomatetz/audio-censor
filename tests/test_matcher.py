import unittest

from censor.matcher import WordMatcher, normalize_word


class MatcherTests(unittest.TestCase):
    def test_normalizes_punctuation_and_case(self):
        self.assertEqual(normalize_word("«СЛОВО!»"), "слово")

    def test_exact_pattern(self):
        matcher = WordMatcher(["слово"])
        self.assertTrue(matcher.matches("Слово,"))
        self.assertFalse(matcher.matches("словом"))

    def test_prefix_pattern(self):
        matcher = WordMatcher(["корень*"])
        self.assertTrue(matcher.matches("корень"))
        self.assertTrue(matcher.matches("кореньями"))
        self.assertFalse(matcher.matches("другой"))

    def test_regex_pattern(self):
        matcher = WordMatcher([r"re:сл[оа]во"])
        self.assertTrue(matcher.matches("славо"))

    def test_builds_hotwords_from_literal_patterns(self):
        matcher = WordMatcher(["четыре*", "горошек", r"re:сл[оа]во"])
        self.assertEqual(matcher.hotwords, "четыре, горошек")

    def test_finds_target_inside_segment_text(self):
        matcher = WordMatcher(["булочк*"])
        self.assertTrue(matcher.matches_text("Я купил свежую булочку."))
        self.assertFalse(matcher.matches_text("Я купил свежий хлеб."))


if __name__ == "__main__":
    unittest.main()
