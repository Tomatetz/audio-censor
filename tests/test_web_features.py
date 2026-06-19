import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import web_gui
from censor.engine import CensorEngine, EngineConfig
from censor.matcher import WordMatcher


class WebFeaturesTests(unittest.TestCase):
    def test_validates_dictionary_regex(self):
        self.assertEqual(web_gui.validate_words_text("булочк*\nre:сл[оа]во\n"), [])
        errors = web_gui.validate_words_text("re:[\n")
        self.assertEqual(len(errors), 1)
        self.assertIn("Строка 1", errors[0])

    def test_preview_is_valid_wav(self):
        data = web_gui.preview_wav("meow", 0.75)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.wav"
            path.write_bytes(data)
            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getframerate(), 48000)
                self.assertGreater(audio.getnframes(), 1000)

    def test_report_recommends_more_delay_after_late_event(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = CensorEngine(
                EngineConfig(
                    input_device=None,
                    output_device=None,
                    delay_seconds=5,
                ),
                WordMatcher([]),
            )
            engine.report_path = Path(directory) / "report.json"
            engine.stats["late"] = 1
            engine._save_report()
            report = json.loads(engine.report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["recommended_delay"], 7.0)

    def test_reads_latest_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "a.report.json"
            newer = root / "b.report.json"
            older.write_text('{"censored": 1}', encoding="utf-8")
            newer.write_text('{"censored": 2}', encoding="utf-8")
            older.touch()
            newer.touch()
            with patch.object(web_gui, "RECORDINGS", root):
                self.assertEqual(web_gui.latest_report()["censored"], 2)


if __name__ == "__main__":
    unittest.main()
