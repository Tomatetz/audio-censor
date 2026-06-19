import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from censor.recorder import WavRecorder


class RecorderTests(unittest.TestCase):
    def test_writes_mono_pcm_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            recorder = WavRecorder(path, sample_rate=16000)
            recorder.start()
            recorder.add(np.array([0.0, 0.5, -0.5], dtype=np.float32))
            recorder.close()

            with wave.open(str(path), "rb") as audio:
                self.assertEqual(audio.getnchannels(), 1)
                self.assertEqual(audio.getsampwidth(), 2)
                self.assertEqual(audio.getframerate(), 16000)
                self.assertEqual(audio.getnframes(), 3)


if __name__ == "__main__":
    unittest.main()
