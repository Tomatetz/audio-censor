import tempfile
import unittest
from pathlib import Path

from censor.recorder import TranscriptRecorder, format_timestamp


class TranscriptRecorderTests(unittest.TestCase):
    def test_writes_timestamped_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.txt"
            recorder = TranscriptRecorder(path)
            recorder.start(mode="bark", delay=7, model="small", words="булочк")
            recorder.event("ASR", 7.0, 8.5, "свежая булочка")
            recorder.event("CENSOR:bark", 8.0, 8.5, "'булочка'")
            recorder.close()

            text = path.read_text(encoding="utf-8")
            self.assertIn("[ASR] свежая булочка", text)
            self.assertIn("[CENSOR:bark] 'булочка'", text)
            self.assertIn("00:07.00–00:08.50", text)

    def test_formats_long_timestamps(self):
        self.assertEqual(format_timestamp(3661.25), "01:01:01.25")


if __name__ == "__main__":
    unittest.main()
