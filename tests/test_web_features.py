import json
import tempfile
import time
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

    def test_marks_stale_runtime_status_as_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.jsonc"
            config.write_text(
                '{"runtime_status_file": ".runtime-status.json"}',
                encoding="utf-8",
            )
            (root / ".runtime-status.json").write_text(
                json.dumps(
                    {
                        "updated_at": time.time() - 10,
                        "overall": "green",
                        "audio_state": "running",
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.object(web_gui, "ROOT", root),
                patch.object(web_gui, "DEFAULT_CONFIG", config),
            ):
                status = web_gui.runtime_status()

            self.assertEqual(status["overall"], "red")
            self.assertEqual(status["audio_state"], "error")

    def test_dashboard_replaces_large_log_panel(self):
        self.assertIn('class="panel cluster-panel"', web_gui.HTML)
        self.assertIn('id="mic_segments"', web_gui.HTML)
        self.assertIn('id="risk_lamp"', web_gui.HTML)
        self.assertIn('id="delay_track"', web_gui.HTML)
        self.assertIn('id="cpu_meter"', web_gui.HTML)
        self.assertIn('@keyframes modelBoot', web_gui.HTML)
        self.assertIn('.cluster-side { display:grid; grid-template-columns:1fr 1fr;', web_gui.HTML)
        self.assertIn('class="telemetry-cell"><div class="system-label">WHISPER CPU', web_gui.HTML)
        self.assertIn('.system-cell { min-height:45px;', web_gui.HTML)
        self.assertIn('id="diagnostics_dialog"', web_gui.HTML)
        self.assertIn('async function startCalibration()', web_gui.HTML)
        self.assertIn('async function openCustomSounds()', web_gui.HTML)
        self.assertIn('value="custom"', web_gui.HTML)
        self.assertNotIn('id="report"', web_gui.HTML)
        self.assertNotIn('<h2 data-i18n="advanced">', web_gui.HTML)
        self.assertNotIn('data-i18n="save_settings"', web_gui.HTML)
        self.assertIn('function enableAutosave()', web_gui.HTML)
        self.assertIn('class="help-tip"', web_gui.HTML)
        self.assertIn('panel display-panel', web_gui.HTML)
        self.assertIn('class="cluster-log"><pre id="log"', web_gui.HTML)
        self.assertIn('class="cluster-log-title" data-i18n="log">ЖУРНАЛ', web_gui.HTML)
        self.assertIn('<select id="language">', web_gui.HTML)
        self.assertIn('value="en">EN — English', web_gui.HTML)
        self.assertIn('h1::before { content:"SC-86 // "', web_gui.HTML)
        self.assertIn('.check input:checked', web_gui.HTML)
        self.assertIn('function enhanceSelect(select)', web_gui.HTML)
        self.assertIn('className="custom-select-option"', web_gui.HTML)
        self.assertIn("onclick=\"setUILanguage('ru')\"", web_gui.HTML)
        self.assertIn("localStorage.setItem(\"stream-censor-ui-language\"", web_gui.HTML)
        self.assertIn('document.querySelector("#censored_count").textContent="0"', web_gui.HTML)
        self.assertLess(
            web_gui.HTML.index('STREAM CENSOR / DIGITAL'),
            web_gui.HTML.index('class="cluster-log-title" data-i18n="log">ЖУРНАЛ'),
        )

    def test_analyzes_healthy_microphone_calibration(self):
        result = web_gui.analyze_calibration(
            [0.0005] * 20,
            [0.08] * 20,
            [0.3] * 20,
        )
        self.assertEqual(result["rating"], "green")
        self.assertGreater(result["snr_db"], 20)
        self.assertGreater(result["threshold"], 0.001)

    def test_calibration_detects_clipping(self):
        result = web_gui.analyze_calibration(
            [0.001] * 20,
            [0.2] * 20,
            [1.0],
        )
        self.assertEqual(result["rating"], "red")


if __name__ == "__main__":
    unittest.main()
