import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import web_gui
from app import load_config


class RuntimeModeTests(unittest.TestCase):
    def test_writes_runtime_mode_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.jsonc"
            config.write_text(
                '{"runtime_control_file": ".runtime-control.json"}',
                encoding="utf-8",
            )
            with patch.object(web_gui, "ROOT", root):
                with patch.object(web_gui, "DEFAULT_CONFIG", config):
                    web_gui.write_runtime_settings("meow", 1.25)
            self.assertEqual(
                load_config(root / ".runtime-control.json")["mode"],
                "meow",
            )
            self.assertEqual(
                load_config(root / ".runtime-control.json")["effect_volume"],
                1.25,
            )


if __name__ == "__main__":
    unittest.main()
