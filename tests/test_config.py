import json
import tempfile
import unittest
from pathlib import Path

from app import build_parser, load_config, strip_json_comments


class ConfigTests(unittest.TestCase):
    def test_loads_json_config_as_parser_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps({"delay": 7, "beam_size": 5, "input_activity_threshold": 0.004}),
                encoding="utf-8",
            )
            args = build_parser(load_config(path)).parse_args([])
            self.assertEqual(args.delay, 7)
            self.assertEqual(args.beam_size, 5)
            self.assertEqual(args.input_activity_threshold, 0.004)

    def test_command_line_overrides_config(self):
        args = build_parser({"delay": 7}).parse_args(["--delay", "9"])
        self.assertEqual(args.delay, 9)

    def test_jsonc_comments_do_not_break_strings(self):
        text = """
        {
          // Обычный комментарий
          "words": "https://example.test/words.txt",
          /* Многострочный комментарий */
          "delay": 5
        }
        """
        data = json.loads(strip_json_comments(text))
        self.assertEqual(data["words"], "https://example.test/words.txt")
        self.assertEqual(data["delay"], 5)


if __name__ == "__main__":
    unittest.main()
