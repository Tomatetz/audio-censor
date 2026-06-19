import ast
import unittest
from pathlib import Path


class TranscriptionOptionsTests(unittest.TestCase):
    def test_does_not_feed_word_list_as_initial_prompt(self):
        source = Path("censor/engine.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        transcribe_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "transcribe"
        ]
        self.assertEqual(len(transcribe_calls), 1)
        keyword_names = {item.arg for item in transcribe_calls[0].keywords}
        self.assertNotIn("initial_prompt", keyword_names)
        self.assertIn("hotwords", keyword_names)


if __name__ == "__main__":
    unittest.main()
