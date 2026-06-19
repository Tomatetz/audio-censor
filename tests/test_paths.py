import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from censor import paths


class PathsTests(unittest.TestCase):
    def test_data_directory_override(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                "os.environ",
                {"STREAM_CENSOR_DATA_DIR": directory},
            ):
                self.assertEqual(paths.data_root(), Path(directory))


if __name__ == "__main__":
    unittest.main()
