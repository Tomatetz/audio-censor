import tempfile
import unittest
from pathlib import Path

from app import load_config
from web_gui import highlight_rules, update_jsonc


class GUIConfigTests(unittest.TestCase):
    def test_updates_values_without_removing_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.jsonc"
            path.write_text(
                '{\n  // Задержка\n  "delay": 5.0,\n  "output_device": 1\n}\n',
                encoding="utf-8",
            )
            update_jsonc(path, {"delay": 7.0, "output_device": None})
            text = path.read_text(encoding="utf-8")
            self.assertIn("// Задержка", text)
            self.assertEqual(load_config(path)["delay"], 7.0)
            self.assertIsNone(load_config(path)["output_device"])

    def test_builds_highlight_rules_from_words_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.txt"
            path.write_text("# comment\nбулочк*\nчетыре\n", encoding="utf-8")
            self.assertEqual(
                highlight_rules(path),
                [
                    {"type": "prefix", "value": "булочк"},
                    {"type": "exact", "value": "четыре"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
