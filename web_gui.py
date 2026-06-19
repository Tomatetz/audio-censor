from __future__ import annotations

import json
import io
import errno
import multiprocessing
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
import wave
from urllib.error import URLError
from urllib.request import urlopen
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import sounddevice as sd
import numpy as np

from app import DEFAULT_CONFIG, load_config
from censor.paths import data_root, ensure_user_data, resource_root
from censor.samples import SoundLibrary


RESOURCE_ROOT = resource_root()
ROOT = ensure_user_data()
APP_PATH = RESOURCE_ROOT / "app.py"
TEST_SCRIPT = ROOT / "test_script.txt"
WORDS_PATH = ROOT / "words.txt"
RECORDINGS = ROOT / "recordings"
HOST = "127.0.0.1"
PORT = 8765
MAX_PORT_ATTEMPTS = 10


def highlight_rules(path: Path) -> list[dict[str, str]]:
    rules = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip().casefold()
        if not value or value.startswith("#"):
            continue
        if value.startswith("re:"):
            rules.append({"type": "regex", "value": value[3:]})
        elif value.endswith("*"):
            rules.append({"type": "prefix", "value": value[:-1]})
        else:
            rules.append({"type": "exact", "value": value})
    return rules


def json_value(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def update_jsonc(path: Path, values: dict) -> None:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        pattern = re.compile(
            rf'(^\s*"{re.escape(key)}"\s*:\s*)(.*?)(\s*,?\s*$)',
            re.MULTILINE,
        )
        text, count = pattern.subn(
            rf"\g<1>{json_value(value)}\g<3>", text, count=1
        )
        if count == 0:
            raise ValueError(f"Параметр {key!r} не найден в {path.name}")
    path.write_text(text, encoding="utf-8")


def write_runtime_settings(mode: str, effect_volume: float) -> None:
    if mode not in {"reverse", "beep", "bark", "meow", "mute"}:
        raise ValueError("Неизвестный режим обработки.")
    config = load_config(DEFAULT_CONFIG)
    path = ROOT / config.get("runtime_control_file", ".runtime-control.json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "mode": mode,
                "effect_volume": max(0.0, min(2.0, float(effect_volume))),
            }
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_words_text(text: str) -> list[str]:
    errors = []
    for number, raw in enumerate(text.splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if value == "*" or value == "re:":
            errors.append(f"Строка {number}: пустой шаблон.")
        elif value.startswith("re:"):
            try:
                re.compile(value[3:])
            except re.error as error:
                errors.append(f"Строка {number}: {error}.")
    return errors


def preview_wav(mode: str, volume: float, sample_rate: int = 48000) -> bytes:
    duration = 0.9
    count = round(sample_rate * duration)
    if mode == "beep":
        positions = np.arange(count, dtype=np.float32)
        samples = 0.18 * np.sin(2 * np.pi * 880 * positions / sample_rate)
    elif mode in {"bark", "meow"}:
        library = SoundLibrary(RESOURCE_ROOT / "assets" / "sounds", sample_rate)
        samples = library.part(mode, 0, 0, count, count)
    elif mode == "mute":
        samples = np.zeros(count, dtype=np.float32)
    else:
        # A short reversed spoken-like sweep demonstrates the transformation.
        positions = np.arange(count, dtype=np.float32)
        samples = (
            0.14
            * np.sin(2 * np.pi * (180 + 220 * positions / count) * positions / sample_rate)
        )[::-1]
    samples = np.clip(samples * max(0.0, min(2.0, volume)), -1.0, 1.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def latest_report() -> dict | None:
    reports = sorted(RECORDINGS.glob("*.report.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        return None
    return json.loads(reports[-1].read_text(encoding="utf-8"))


def runtime_status() -> dict | None:
    config = load_config(DEFAULT_CONFIG)
    path = ROOT / config.get("runtime_status_file", ".runtime-status.json")
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(status, dict):
            return None
        if time.time() - float(status.get("updated_at", 0)) > 2:
            status["overall"] = "red"
            status["audio_state"] = "error"
            status["last_error"] = "Движок перестал обновлять состояние."
        return status
    except (OSError, ValueError, json.JSONDecodeError):
        return None


class AppState:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.logs: list[str] = []
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-1000:]

    def start(self) -> None:
        if self.running():
            return
        config = load_config(DEFAULT_CONFIG)
        status_path = ROOT / config.get("runtime_status_file", ".runtime-status.json")
        status_path.unlink(missing_ok=True)
        self.append_log("\n=== Новый запуск ===\n")
        command = (
            [sys.executable, "--engine"]
            if getattr(sys, "frozen", False)
            else [sys.executable, "-u", str(APP_PATH)]
        )
        self.process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            self.append_log(line)
        code = self.process.wait()
        self.append_log(f"\n=== Процесс завершён, код {code} ===\n")

    def stop(self) -> None:
        if self.running():
            assert self.process
            self.process.send_signal(signal.SIGINT)

    def log_text(self) -> str:
        with self.lock:
            return "".join(self.logs)


STATE = AppState()
SERVER: ThreadingHTTPServer | None = None


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def is_stream_censor_server(port: int) -> bool:
    try:
        with urlopen(f"http://{HOST}:{port}/api/health", timeout=0.5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("app") == "stream-censor"
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False


def find_or_create_server(
    start_port: int = PORT,
    attempts: int = MAX_PORT_ATTEMPTS,
) -> tuple[ThreadingHTTPServer | None, int, bool]:
    for port in range(start_port, start_port + attempts):
        try:
            return ReusableThreadingHTTPServer((HOST, port), Handler), port, False
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
            if is_stream_censor_server(port):
                return None, port, True
    raise OSError(
        f"Не удалось найти свободный порт в диапазоне "
        f"{start_port}–{start_port + attempts - 1}"
    )


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stream Censor</title>
<style>
:root { color-scheme:dark; --bg:#080d0b; --panel:#101815; --line:#315544;
  --text:#8ff3ca; --muted:#57967c; --accent:#49d99b; --green:#55efaa;
  --yellow:#ffc84d; --red:#ff6257; --screen:#06110d; --metal:#222b26; }
* { box-sizing:border-box } body { margin:0; color:var(--text);
  background:
    linear-gradient(rgba(66,255,166,.018) 1px,transparent 1px),
    linear-gradient(90deg,rgba(66,255,166,.018) 1px,transparent 1px),
    radial-gradient(circle at 50% -20%,#233a30 0,#0a110e 40%,#050806 82%);
  background-size:24px 24px,24px 24px,auto;
  font:13px Menlo,Monaco,monospace; }
header { height:60px; display:flex; align-items:center; justify-content:space-between;
  padding:0 22px; border-bottom:2px solid #315b4b; background:
  linear-gradient(180deg,#202b26,#111814); box-shadow:0 4px 14px #000 }
h1 { margin:0; color:#8fffd2; font-size:18px; letter-spacing:.12em;
  text-shadow:0 0 9px rgba(74,255,181,.42) }
h1::before { content:"SC-86 // "; color:#ffcc61; font-size:10px; vertical-align:2px }
#status { padding:6px 10px; border:1px solid #315b4b; color:#6ccba5;
  background:#07110d; font-size:10px; letter-spacing:.08em; text-transform:uppercase;
  box-shadow:inset 0 0 10px #000 }
main { display:grid; grid-template-columns:330px 1fr; gap:14px; padding:14px;
  height:calc(100vh - 60px); }
.panel { min-height:0; padding:16px; border:1px solid #3c584c; border-radius:5px;
  background:linear-gradient(135deg,#17211d,#0d1411);
  box-shadow:inset 0 0 0 2px #090e0c,0 5px 14px rgba(0,0,0,.45) }
.controls { overflow:auto; border-radius:5px 5px 16px 5px }
.right { display:grid; grid-template-rows:1fr 1fr; gap:14px; min-height:0 }
h2 { margin:0 0 14px; color:#ffcc61; font-size:11px; letter-spacing:.18em;
  text-transform:uppercase; text-shadow:0 0 7px rgba(255,190,50,.28) }
h2::before { content:"[ "; color:#4edf9f } h2::after { content:" ]"; color:#4edf9f }
label { display:block; margin:11px 0 5px; color:#5fa789; font-size:10px;
  letter-spacing:.09em; text-transform:uppercase }
input,select { width:100%; padding:9px 10px; border:1px solid #37634f;
  border-radius:2px; outline:none; color:#8ff3ca; background:#07110d;
  font:12px Menlo,Monaco,monospace; box-shadow:inset 0 0 10px rgba(0,0,0,.8) }
input:focus,select:focus { border-color:#63e9ad; box-shadow:inset 0 0 10px #000,0 0 7px rgba(74,255,181,.3) }
.native-select-hidden { position:absolute!important; width:1px!important; height:1px!important;
  padding:0!important; opacity:0!important; pointer-events:none!important }
.custom-select { position:relative; width:100%; min-width:0 }
.custom-select-trigger { display:flex; width:100%; align-items:center; justify-content:space-between;
  gap:10px; padding:9px 10px; border:1px solid #37634f; border-radius:2px;
  color:#8ff3ca; background:#07110d; font:12px Menlo,Monaco,monospace;
  text-align:left; text-transform:none; letter-spacing:0;
  box-shadow:inset 0 0 10px rgba(0,0,0,.8) }
.custom-select-trigger::after { content:""; flex:0 0 auto; width:8px; height:8px;
  border-right:2px solid #ffc84d; border-bottom:2px solid #ffc84d;
  transform:rotate(45deg) translateY(-2px); filter:drop-shadow(0 0 3px #d99724) }
.custom-select.open .custom-select-trigger { border-color:#63e9ad;
  box-shadow:inset 0 0 10px #000,0 0 7px rgba(74,255,181,.3) }
.custom-select.open .custom-select-trigger::after { transform:rotate(225deg) translate(-2px,-2px) }
.custom-select-menu { position:absolute; z-index:50; top:calc(100% + 4px); left:0; right:0;
  display:none; max-height:240px; overflow:auto; padding:4px; border:1px solid #4b8068;
  border-radius:2px; background:#06110d;
  box-shadow:inset 0 0 18px #000,0 8px 20px rgba(0,0,0,.8),0 0 8px rgba(74,255,181,.18) }
.custom-select.open .custom-select-menu { display:block }
.custom-select-option { width:100%; padding:9px 10px; border:0; color:#68bd9a;
  background:transparent; box-shadow:none; font:11px Menlo,Monaco,monospace;
  text-align:left; text-transform:none; letter-spacing:0 }
.custom-select-option:hover,.custom-select-option.focused { color:#b0ffdf;
  background:#123527; text-shadow:0 0 6px #42e69b }
.custom-select-option.selected { color:#ffd46e; background:#392d14;
  text-shadow:0 0 6px rgba(255,190,50,.7) }
.custom-select-option.selected::before { content:"> "; color:#55efaa }
.custom-select-menu::-webkit-scrollbar { width:8px }
.custom-select-menu::-webkit-scrollbar-track { background:#07110d }
.custom-select-menu::-webkit-scrollbar-thumb { border:2px solid #07110d; background:#396b55 }
.check { display:flex; gap:9px; align-items:center; margin:12px 0; color:#79caa9;
  font-size:11px; text-transform:uppercase }
.check input { width:14px; height:14px; appearance:none; border:1px solid #477661;
  background:#07110d; padding:0 }
.check input:checked { background:#55efaa; box-shadow:inset 0 0 0 3px #092117,0 0 7px #35dc8d }
.buttons { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:16px }
button { padding:10px; cursor:pointer; border:1px solid #426d59; border-radius:2px;
  color:#83dfba; background:linear-gradient(#263d33,#14241d);
  font:700 10px Menlo,Monaco,monospace; letter-spacing:.06em; text-transform:uppercase;
  box-shadow:inset 0 0 0 1px #0c1511,0 2px 4px #000 }
button:hover { color:#b2ffe0; border-color:#6ce9b2; text-shadow:0 0 6px #4cffaa }
button:active { transform:translateY(1px); box-shadow:inset 0 0 8px #000 }
button.primary { color:#07150f; border-color:#72f1b8; background:#51d99d;
  box-shadow:inset 0 0 8px #94ffd1,0 0 8px rgba(70,239,165,.25) }
button.stop { color:#ffcf77; border-color:#906923; background:#392b13 }
button:disabled { opacity:.35; cursor:default; filter:grayscale(.5) }
.danger { width:100%; margin-top:16px; color:#ff7d72; border-color:#7f3832;
  background:#2d1715 }
.full { width:100%; margin-top:8px }
.inline { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center }
.inline button { padding:9px 12px }
.volume { display:flex; gap:10px; align-items:center }.volume input { flex:1 }
.volume input[type=range] { appearance:none; height:8px; padding:0; border:1px solid #315b4b;
  background:#07110d }
.volume input[type=range]::-webkit-slider-thumb { appearance:none; width:12px; height:18px;
  border:1px solid #b08a3e; background:#ffca58; box-shadow:0 0 6px #ffba32 }
.volume output { min-width:42px; text-align:right; color:#ffcc61; text-shadow:0 0 5px #ca8d24 }
.report { margin-top:12px; padding:10px; border:1px solid #315b4b; border-radius:2px;
  color:#63ae90; background:#07110d; line-height:1.5; font-size:10px;
  box-shadow:inset 0 0 12px #000 }
.mini-log { height:190px; margin-top:14px }
.mini-log pre { height:calc(100% - 28px); padding:10px; font-size:10px }
.cluster-panel { padding:12px; overflow:hidden }
.cluster { position:relative; height:100%; min-height:260px; padding:20px 22px 18px;
  overflow:hidden; border:2px solid #4b514f; border-radius:8px 8px 26px 26px;
  color:#8dffd1; background:
    repeating-linear-gradient(0deg,rgba(255,255,255,.025) 0 1px,transparent 1px 4px),
    radial-gradient(ellipse at 50% 38%,#14382f 0,#081713 47%,#030706 75%);
  box-shadow:inset 0 0 0 4px #0a0d0c,inset 0 0 38px #000,0 0 0 1px #101311 }
.cluster::after { content:""; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(105deg,transparent 0 42%,rgba(185,255,225,.055) 47%,transparent 53%) }
.cluster-top { position:relative; z-index:1; display:flex; align-items:center;
  justify-content:space-between; padding-bottom:11px; border-bottom:1px solid #315b4b }
.cluster-title { color:#67cda8; font:11px Menlo,monospace; letter-spacing:.24em }
.health { display:inline-flex; align-items:center; gap:8px; color:#749589;
  font:700 12px Menlo,monospace; letter-spacing:.08em; text-transform:uppercase }
.health::before { content:""; width:10px; height:10px; border-radius:2px;
  background:#34433e; box-shadow:0 0 0 2px #111b18 }
.health.green { color:#7effc9 }.health.green::before { background:#66ffb8;box-shadow:0 0 9px #42ff9d }
.health.yellow { color:#ffd36d }.health.yellow::before { background:#ffc64a;box-shadow:0 0 9px #ffb82e }
.health.red { color:#ff7b72 }.health.red::before { background:#ff5148;box-shadow:0 0 10px #ff3b30 }
.cluster-main { position:relative; z-index:1; display:grid;
  grid-template-columns:minmax(0,1.25fr) minmax(170px,.75fr); gap:22px;
  align-items:stretch; height:calc(100% - 44px); padding-top:15px }
.voice-gauge { display:flex; flex-direction:column; justify-content:space-between; min-width:0 }
.gauge-caption { display:flex; justify-content:space-between; color:#4e9279;
  font:10px Menlo,monospace; letter-spacing:.16em }
.segment-track { display:grid; grid-template-columns:repeat(20,1fr); gap:4px;
  height:38px; margin:8px 0 5px }
.segment { align-self:end; height:18px; clip-path:polygon(12% 0,88% 0,100% 20%,100% 80%,88% 100%,12% 100%,0 80%,0 20%);
  background:#17382e; box-shadow:inset 0 0 0 1px #285446 }
.segment.lit { background:#55efaa; box-shadow:0 0 8px rgba(77,255,176,.72) }
.segment.warn.lit { background:#ffc84d; box-shadow:0 0 8px rgba(255,190,50,.8) }
.segment.danger.lit { background:#ff5d50; box-shadow:0 0 9px rgba(255,60,45,.85) }
.db-row { display:flex; align-items:baseline; gap:10px }
.digital { color:#91ffd4; font:700 42px "Arial Narrow",Menlo,monospace;
  letter-spacing:.04em; line-height:1; text-shadow:0 0 9px rgba(74,255,181,.6) }
.digital small { font-size:12px; color:#4e9279; letter-spacing:.12em }
.system-grid { display:grid; grid-template-columns:1fr 1fr; gap:9px; margin-top:14px }
.system-cell { padding:8px 10px; border:1px solid #244b3d; background:rgba(5,24,18,.65) }
.system-label { color:#4d8a73; font:9px Menlo,monospace; letter-spacing:.13em }
.system-value { margin-top:3px; overflow:hidden; color:#76dcb5; font:12px Menlo,monospace;
  text-overflow:ellipsis; white-space:nowrap }
.cluster-side { display:grid; grid-template-rows:1fr 1fr auto; gap:10px }
.readout { display:flex; flex-direction:column; justify-content:center; padding:10px;
  border:1px solid #315b4b; background:rgba(3,18,13,.72); text-align:center }
.readout-label { color:#4d8a73; font:9px Menlo,monospace; letter-spacing:.16em }
.readout-value { margin-top:5px; color:#ffce62; font:700 34px Menlo,monospace;
  line-height:1; text-shadow:0 0 9px rgba(255,190,50,.5) }
.readout-value small { font-size:11px; color:#907b48 }
.warning-lamps { display:grid; grid-template-columns:repeat(3,1fr); gap:6px }
.lamp { padding:7px 4px; border:1px solid #293b35; color:#455b54;
  background:#101713; font:700 9px Menlo,monospace; text-align:center }
.lamp.on-yellow { color:#ffd15e; border-color:#8a6922; box-shadow:inset 0 0 12px #5e430f }
.lamp.on-red { color:#ff7168; border-color:#8d302c; box-shadow:inset 0 0 12px #5f1714 }
dialog { width:min(720px,90vw); padding:18px; border:2px solid #426d59;
  border-radius:4px 4px 18px 4px; color:var(--text);
  background:linear-gradient(135deg,#1b2822,#0a100d);
  box-shadow:inset 0 0 0 3px #080d0b,0 15px 50px #000 }
dialog::backdrop { background:rgba(1,7,4,.82); backdrop-filter:blur(2px) }
dialog textarea { width:100%; min-height:340px; resize:vertical; padding:12px;
  background:#06110d; color:#8ff3ca; border:1px solid #37634f; border-radius:2px;
  font:12px Menlo,Monaco,monospace; text-shadow:0 0 5px rgba(83,255,181,.3) }
.advanced-grid { display:grid; grid-template-columns:1fr 1fr; gap:0 18px }
.advanced-grid .wide { grid-column:1 / -1 }
.display-panel { position:relative; overflow:hidden; padding:12px;
  border-color:#344b43; background:#111715 }
.display-panel::after { content:""; position:absolute; inset:40px 12px 12px;
  z-index:2; pointer-events:none; border-radius:7px;
  background:repeating-linear-gradient(0deg,transparent 0 3px,rgba(94,255,177,.025) 3px 4px) }
.display-panel h2 { position:relative; z-index:3; margin:0 0 10px; color:#67cda8;
  font:11px Menlo,monospace; letter-spacing:.18em; text-transform:uppercase;
  text-shadow:0 0 7px rgba(75,255,177,.28) }
.display-panel h2::before { content:"■"; margin-right:8px; color:#4cffaa;
  text-shadow:0 0 8px #3cff9c }
.display-panel h2::after { content:"" }
.script,pre { position:relative; z-index:1; width:100%; height:calc(100% - 24px);
  margin:0; padding:16px; overflow:auto; border:1px solid #315b4b;
  border-radius:7px; color:#84efc4; background:
    radial-gradient(ellipse at 50% 20%,rgba(19,71,54,.65),transparent 75%),
    #06110d; line-height:1.55; text-shadow:0 0 6px rgba(83,255,181,.34);
  box-shadow:inset 0 0 24px rgba(0,0,0,.8),inset 0 0 1px #78dcb6 }
.script { font:15px Menlo,Monaco,monospace; white-space:pre-wrap; letter-spacing:.015em }
mark { padding:1px 3px; border:1px solid #d89b32; border-radius:2px;
  color:#ffe29a; background:#4b3310; text-shadow:0 0 7px rgba(255,201,76,.85);
  box-shadow:inset 0 0 8px rgba(255,181,49,.25),0 0 5px rgba(255,177,34,.18) }
pre { white-space:pre-wrap; font:11px Menlo,Monaco,monospace }
pre::-webkit-scrollbar,.script::-webkit-scrollbar { width:9px }
pre::-webkit-scrollbar-track,.script::-webkit-scrollbar-track { background:#07110d }
pre::-webkit-scrollbar-thumb,.script::-webkit-scrollbar-thumb {
  border:2px solid #07110d; border-radius:8px; background:#2b6650 }
.controls::-webkit-scrollbar,dialog textarea::-webkit-scrollbar { width:9px }
.controls::-webkit-scrollbar-track,dialog textarea::-webkit-scrollbar-track { background:#09100d }
.controls::-webkit-scrollbar-thumb,dialog textarea::-webkit-scrollbar-thumb {
  border:2px solid #09100d; background:#396b55 }
@media(max-width:950px){.cluster-main{grid-template-columns:1fr}.cluster-side{grid-template-columns:1fr 1fr;grid-template-rows:auto}.warning-lamps{grid-column:1/-1}}
@media(max-width:800px){ main{grid-template-columns:1fr;height:auto}.right{height:900px}.cluster{min-height:430px} }
</style>
</head>
<body>
<header><h1>Stream Censor</h1><div id="status">Загрузка…</div></header>
<main>
  <section class="panel controls">
    <h2>Настройки</h2>
    <label>Микрофон</label><select id="input_device"></select>
    <label>Вывод</label><select id="output_device"></select>
    <label>Обработка — меняется на лету</label><div class="inline"><select id="mode" onchange="changeRuntimeSettings()">
      <option value="reverse">Проиграть наоборот</option>
      <option value="beep">ПИП</option>
      <option value="bark">Гавканье</option>
      <option value="meow">Мяуканье</option>
      <option value="mute">Заглушить</option>
    </select><button onclick="previewEffect()">▶</button></div>
    <label>Громкость эффекта</label><div class="volume"><input id="effect_volume" type="range" min="0" max="2" step="0.05" oninput="volume_value.value=this.value" onchange="changeRuntimeSettings()"><output id="volume_value">1.0</output></div>
    <div class="buttons">
      <button id="start" class="primary" onclick="startApp()">▶ Запустить</button>
      <button id="stop" class="stop" onclick="stopApp()">■ Остановить</button>
    </div>
    <div class="mini-log display-panel"><h2>Журнал</h2><pre id="log"></pre></div>
    <button class="full" onclick="save()">Сохранить настройки</button>
    <button class="full" onclick="document.querySelector('#advanced_dialog').showModal()">Расширенные настройки</button>
    <button class="full" onclick="openWords()">Редактировать словарь</button>
    <button class="full" onclick="openRecordings()">Открыть папку записей</button>
    <div id="report" class="report">Отчётов пока нет.</div>
    <button class="danger" onclick="closeApp()">Закрыть приложение</button>
  </section>
  <section class="right">
    <div class="panel display-panel"><h2>Текст для проверки</h2><div id="script" class="script"></div></div>
    <div class="panel cluster-panel">
      <div class="cluster">
        <div class="cluster-top">
          <span class="cluster-title">STREAM CENSOR / DIGITAL</span>
          <span id="health" class="health">Ожидание</span>
        </div>
        <div class="cluster-main">
          <div class="voice-gauge">
            <div>
              <div class="gauge-caption"><span>MIC INPUT</span><span>-60 · -30 · -12 · 0 dBFS</span></div>
              <div id="mic_segments" class="segment-track"></div>
              <div class="db-row"><span id="mic_db" class="digital">—<small> dBFS</small></span></div>
            </div>
            <div class="system-grid">
              <div class="system-cell"><div class="system-label">ASR MODEL</div><div id="model_state" class="system-value">—</div></div>
              <div class="system-cell"><div class="system-label">AUDIO LINK</div><div id="audio_state" class="system-value">—</div></div>
              <div class="system-cell"><div class="system-label">EFFECT MODE</div><div id="mode_state" class="system-value">—</div></div>
              <div class="system-cell"><div class="system-label">MIC STATUS</div><div id="mic_state" class="system-value">—</div></div>
            </div>
          </div>
          <div class="cluster-side">
            <div class="readout"><div class="readout-label">WORDS CENSORED</div><div id="censored_count" class="readout-value">0</div></div>
            <div class="readout"><div class="readout-label">MINIMUM MARGIN</div><div id="margin_state" class="readout-value">—</div></div>
            <div class="warning-lamps">
              <div id="clip_lamp" class="lamp">CLIP</div>
              <div id="risk_lamp" class="lamp">RISK</div>
              <div id="late_lamp" class="lamp">LATE</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</main>
<dialog id="advanced_dialog">
  <h2>Расширенные настройки</h2>
  <div class="advanced-grid">
    <div><label>Задержка, сек</label><input id="delay" type="number" step=".1"></div>
    <div><label>Окно распознавания, сек</label><input id="chunk" type="number" step=".1"></div>
    <div><label>Период распознавания, сек</label><input id="scan_every" type="number" step=".1"></div>
    <div><label>Подтверждений слова</label><input id="confirmation_count" type="number" min="1" max="4"></div>
    <div><label>Стабилизация, сек</label><input id="stability_delay" type="number" min="0" max="3" step=".1"></div>
    <div><label>Модель</label><select id="model">
      <option>tiny</option><option>base</option><option>small</option>
      <option>medium</option><option>large-v3</option></select></div>
    <div><label>Beam size</label><input id="beam_size" type="number" min="1" max="10"></div>
    <div class="wide">
      <label class="check"><input id="debug_transcript" type="checkbox">Показывать распознанный текст</label>
      <label class="check"><input id="debug_hypotheses" type="checkbox">Показывать сырые гипотезы</label>
      <label class="check"><input id="record_output" type="checkbox">Записывать WAV</label>
      <label class="check"><input id="record_transcript" type="checkbox">Сохранять журнал TXT</label>
    </div>
  </div>
  <div class="buttons">
    <button onclick="document.querySelector('#advanced_dialog').close()">Закрыть</button>
    <button class="primary" onclick="saveAdvanced()">Сохранить</button>
  </div>
</dialog>
<dialog id="words_dialog">
  <h2>Словарь замены</h2>
  <p>Одна запись на строку: слово, основа со звёздочкой или <code>re:выражение</code>.</p>
  <textarea id="words_editor"></textarea>
  <div class="buttons"><button onclick="document.querySelector('#words_dialog').close()">Отмена</button><button class="primary" onclick="saveWords()">Сохранить</button></div>
</dialog>
<script>
const ids=["delay","chunk","scan_every","confirmation_count","stability_delay","model","beam_size","mode","effect_volume","debug_transcript","debug_hypotheses","record_output","record_transcript"];
document.querySelector("#mic_segments").innerHTML=Array.from({length:20},(_,i)=>`<i class="segment${i>=17?" danger":i>=14?" warn":""}"></i>`).join("");
function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function matchesRule(word,rules){const w=word.toLocaleLowerCase("ru");return rules.some(r=>{if(r.type==="prefix")return w.startsWith(r.value);if(r.type==="exact")return w===r.value;if(r.type==="regex"){try{return new RegExp("^(?:"+r.value+")$","iu").test(w)}catch(e){return false}}return false})}
function highlightScript(text,rules){let out="",last=0;const rx=/[\p{L}\p{N}_ё]+/giu;for(const m of text.matchAll(rx)){out+=escapeHtml(text.slice(last,m.index));const word=m[0];out+=matchesRule(word,rules)?`<mark>${escapeHtml(word)}</mark>`:escapeHtml(word);last=m.index+word.length}return out+escapeHtml(text.slice(last))}
async function api(path, options={}) {
  const r=await fetch(path,{headers:{"Content-Type":"application/json"},...options});
  const data=await r.json(); if(!r.ok) throw new Error(data.error||"Ошибка"); return data;
}
function option(select,value,label){const o=document.createElement("option");o.value=value;o.textContent=label;select.appendChild(o)}
function closeSelects(except=null){document.querySelectorAll(".custom-select.open").forEach(x=>{if(x!==except)x.classList.remove("open")})}
function enhanceSelect(select){
  if(select.dataset.enhanced)return;
  select.dataset.enhanced="1";select.classList.add("native-select-hidden");
  const wrap=document.createElement("div");wrap.className="custom-select";
  const trigger=document.createElement("button");trigger.type="button";trigger.className="custom-select-trigger";
  trigger.setAttribute("aria-haspopup","listbox");trigger.setAttribute("aria-expanded","false");
  const menu=document.createElement("div");menu.className="custom-select-menu";menu.setAttribute("role","listbox");
  select.parentNode.insertBefore(wrap,select);wrap.append(select,trigger,menu);
  function rebuild(){
    menu.innerHTML="";
    [...select.options].forEach((o,i)=>{
      const item=document.createElement("button");item.type="button";item.className="custom-select-option";
      item.dataset.index=String(i);item.textContent=o.textContent;item.setAttribute("role","option");
      item.onclick=()=>choose(i);
      menu.appendChild(item);
    });sync();
  }
  function sync(){
    const chosen=select.options[select.selectedIndex];
    trigger.textContent=chosen?chosen.textContent:"—";
    menu.querySelectorAll(".custom-select-option").forEach((item,i)=>{
      item.classList.toggle("selected",i===select.selectedIndex);
      item.setAttribute("aria-selected",i===select.selectedIndex?"true":"false");
    });
  }
  function choose(index){
    select.selectedIndex=index;sync();wrap.classList.remove("open");trigger.setAttribute("aria-expanded","false");
    select.dispatchEvent(new Event("change",{bubbles:true}));trigger.focus();
  }
  function toggle(){
    const opening=!wrap.classList.contains("open");closeSelects(wrap);wrap.classList.toggle("open",opening);
    trigger.setAttribute("aria-expanded",opening?"true":"false");
  }
  trigger.onclick=toggle;
  trigger.onkeydown=e=>{
    if(e.key==="Escape"){wrap.classList.remove("open");trigger.setAttribute("aria-expanded","false");return}
    if(!["ArrowDown","ArrowUp","Enter"," "].includes(e.key))return;
    e.preventDefault();
    if(!wrap.classList.contains("open")){toggle();return}
    let i=select.selectedIndex;
    if(e.key==="ArrowDown")i=Math.min(select.options.length-1,i+1);
    else if(e.key==="ArrowUp")i=Math.max(0,i-1);
    else {choose(i);return}
    select.selectedIndex=i;sync();
  };
  select.addEventListener("change",sync);rebuild();
}
document.addEventListener("click",e=>{if(!e.target.closest(".custom-select"))closeSelects()});
async function load(){
  try {
    const d=await api("/api/state"), c=d.config;
    const inp=document.querySelector("#input_device"),out=document.querySelector("#output_device");
    d.inputs.forEach(x=>option(inp,x.id,x.label)); option(out,"null","Не выводить звук (только запись)");
    d.outputs.forEach(x=>option(out,x.id,x.label));
    inp.value=String(c.input_device); out.value=c.output_device===null?"null":String(c.output_device);
    ids.forEach(id=>{const e=document.querySelector("#"+id);e.type==="checkbox"?e.checked=!!c[id]:e.value=c[id]});
    document.querySelector("#script").innerHTML=highlightScript(d.script,d.highlight_rules);
    document.querySelector("#volume_value").value=Number(c.effect_volume||1).toFixed(2);
    document.querySelectorAll("select").forEach(enhanceSelect);
    renderReport(d.report); updateState(d.running);
  } catch(e){document.querySelector("#status").textContent=e.message}
}
function values(){const input=document.querySelector("#input_device"),output=document.querySelector("#output_device");const v={input_device:Number(input.value),output_device:output.value==="null"?null:Number(output.value)};
  ids.forEach(id=>{const e=document.querySelector("#"+id);v[id]=e.type==="checkbox"?e.checked:(e.type==="number"?Number(e.value):e.value)});return v}
async function save(){try{await api("/api/config",{method:"POST",body:JSON.stringify(values())});document.querySelector("#status").textContent="Настройки сохранены";return true}catch(e){alert(e.message);return false}}
async function saveAdvanced(){if(await save())document.querySelector("#advanced_dialog").close()}
async function changeRuntimeSettings(){const mode=document.querySelector("#mode").value,effect_volume=Number(document.querySelector("#effect_volume").value);document.querySelector("#volume_value").value=effect_volume.toFixed(2);document.querySelector("#mode_state").textContent=modeLabel(mode);try{await api("/api/runtime",{method:"POST",body:JSON.stringify({mode,effect_volume})});document.querySelector("#status").textContent="Настройки эффекта применены"}catch(e){alert(e.message)}}
async function previewEffect(){const mode=document.querySelector("#mode").value,volume=document.querySelector("#effect_volume").value;try{await new Audio(`/api/preview?mode=${encodeURIComponent(mode)}&volume=${encodeURIComponent(volume)}&t=${Date.now()}`).play()}catch(e){document.querySelector("#status").textContent="Браузер заблокировал звук — нажми Preview ещё раз"}}
async function openWords(){const d=await api("/api/words");document.querySelector("#words_editor").value=d.text;document.querySelector("#words_dialog").showModal()}
async function saveWords(){try{const text=document.querySelector("#words_editor").value;const d=await api("/api/words",{method:"POST",body:JSON.stringify({text})});document.querySelector("#script").innerHTML=highlightScript(d.script,d.highlight_rules);document.querySelector("#words_dialog").close();document.querySelector("#status").textContent=d.restart_required?"Словарь сохранён — перезапусти фильтр для применения":"Словарь сохранён"}catch(e){alert(e.message)}}
function renderReport(r){const el=document.querySelector("#report");if(!r){el.textContent="Отчётов пока нет.";return}const min=r.min_margin===null?"—":Number(r.min_margin).toFixed(1)+" с";el.innerHTML=`<b>Последняя сессия</b><br>Заменено: ${r.censored}, MISS: ${r.miss}, RISK: ${r.risk}, LATE: ${r.late}<br>Минимальный запас: ${min}<br>Рекомендуемая задержка: <b>${r.recommended_delay} с</b>`}
function stateLabel(value){return ({waiting:"ожидание",loading:"загрузка…",starting:"запуск…",ready:"готова",running:"работает",stopped:"остановлен",error:"ошибка"})[value]||"—"}
function modeLabel(value){return ({reverse:"REVERSE",beep:"BEEP",bark:"BARK",meow:"MEOW",mute:"MUTE"})[value]||"—"}
function lightSegments(percent){
  const lit=Math.round(Math.max(0,Math.min(100,percent))/100*20);
  document.querySelectorAll("#mic_segments .segment").forEach((el,i)=>el.classList.toggle("lit",i<lit));
}
function setLamp(id,on,kind){const el=document.querySelector(id);el.className="lamp"+(on?` on-${kind}`:"")}
function renderMetrics(m,running){
  const health=document.querySelector("#health");
  health.className="health";
  if(!running){
    health.textContent="Ожидание";
    document.querySelector("#model_state").textContent="—";
    document.querySelector("#audio_state").textContent="—";
    document.querySelector("#mic_state").textContent="—";
    document.querySelector("#mic_db").innerHTML="—<small> dBFS</small>";
    document.querySelector("#mode_state").textContent=modeLabel(document.querySelector("#mode").value);
    lightSegments(0);setLamp("#clip_lamp",false,"red");setLamp("#risk_lamp",false,"yellow");setLamp("#late_lamp",false,"red");
    document.querySelector("#censored_count").textContent=m?.censored||0;
    document.querySelector("#margin_state").textContent="—";
    return;
  }
  if(!m){
    health.classList.add("yellow"); health.textContent="Запуск…";
    document.querySelector("#model_state").textContent="ожидание данных";
    document.querySelector("#audio_state").textContent="ожидание данных";
    return;
  }
  const names={green:"Всё хорошо",yellow:"Внимание",red:"Проблема",idle:"Ожидание"};
  health.classList.add(m.overall||"yellow"); health.textContent=names[m.overall]||"Запуск…";
  document.querySelector("#model_state").textContent=stateLabel(m.model_state);
  document.querySelector("#audio_state").textContent=stateLabel(m.audio_state);
  document.querySelector("#mode_state").textContent=modeLabel(m.mode);
  const rms=Math.max(Number(m.mic_rms)||0,1e-6),db=20*Math.log10(rms);
  const percent=Math.max(0,Math.min(100,(db+60)/60*100));
  document.querySelector("#mic_state").textContent=m.clipping?"ПЕРЕГРУЗ":"НОРМА";
  document.querySelector("#mic_db").innerHTML=`${db.toFixed(0)}<small> dBFS</small>`;
  lightSegments(percent);
  document.querySelector("#censored_count").textContent=m.censored||0;
  document.querySelector("#margin_state").innerHTML=m.min_margin===null?"—":`${Number(m.min_margin).toFixed(1)}<small> SEC</small>`;
  setLamp("#clip_lamp",!!m.clipping,"red");
  setLamp("#risk_lamp",Number(m.risk)>0,"yellow");
  setLamp("#late_lamp",Number(m.late)>0,"red");
  if(m.last_error)document.querySelector("#audio_state").title=m.last_error;
}
async function startApp(){if(!await save())return;try{await api("/api/start",{method:"POST",body:"{}"});updateState(true)}catch(e){alert(e.message)}}
async function stopApp(){await api("/api/stop",{method:"POST",body:"{}"})}
async function openRecordings(){await api("/api/open-recordings",{method:"POST",body:"{}"})}
async function closeApp(){
  if(!confirm("Остановить фильтр и закрыть Stream Censor?"))return;
  document.querySelector("#status").textContent="Закрытие…";
  try{await api("/api/shutdown",{method:"POST",body:"{}"})}catch(e){}
  setTimeout(()=>{
    window.open("","_self");
    window.close();
    document.body.innerHTML="<main style='display:block;height:auto;max-width:620px;margin:80px auto'><section class='panel'><h1>Stream Censor закрыт</h1><p>Сервер и консоль остановлены. Эту вкладку можно закрыть.</p></section></main>";
  },300);
}
function updateState(r){document.querySelector("#start").disabled=r;document.querySelector("#stop").disabled=!r;document.querySelector("#status").textContent=r?"Фильтр работает":"Готов к запуску"}
async function poll(){try{const d=await api("/api/log");const p=document.querySelector("#log");if(p.textContent!==d.log){p.textContent=d.log;p.scrollTop=p.scrollHeight}updateState(d.running);renderMetrics(d.metrics,d.running);if(!d.running&&d.report)renderReport(d.report)}catch(e){}setTimeout(poll,350)}
load();poll();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _json(self, data, status=HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/state":
            try:
                devices = list(sd.query_devices())
                inputs = [
                    {"id": i, "label": f"{i}: {d['name']}"}
                    for i, d in enumerate(devices) if d["max_input_channels"] > 0
                ]
                outputs = [
                    {"id": i, "label": f"{i}: {d['name']}"}
                    for i, d in enumerate(devices) if d["max_output_channels"] > 0
                ]
                self._json({
                    "config": load_config(DEFAULT_CONFIG),
                    "inputs": inputs,
                    "outputs": outputs,
                    "script": TEST_SCRIPT.read_text(encoding="utf-8"),
                    "highlight_rules": highlight_rules(WORDS_PATH),
                    "running": STATE.running(),
                    "report": latest_report(),
                })
            except Exception as error:
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/log":
            self._json(
                {
                    "log": STATE.log_text(),
                    "running": STATE.running(),
                    "metrics": runtime_status(),
                    "report": None if STATE.running() else latest_report(),
                }
            )
        elif path == "/api/health":
            self._json({"app": "stream-censor", "running": STATE.running()})
        elif path == "/api/words":
            self._json({"text": WORDS_PATH.read_text(encoding="utf-8")})
        elif path == "/api/preview":
            parameters = parse_qs(urlparse(self.path).query)
            mode = parameters.get("mode", ["beep"])[0]
            volume = float(parameters.get("volume", ["1"])[0])
            if mode not in {"reverse", "beep", "bark", "meow", "mute"}:
                mode = "beep"
            body = preview_wav(mode, volume)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            self._json({"error": "Не найдено"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/config":
                values = self._body()
                if values["delay"] < values["chunk"] + 2:
                    raise ValueError("Задержка должна быть минимум на 2 секунды больше окна.")
                allowed = {
                    "input_device", "output_device", "delay", "chunk", "scan_every",
                    "model", "beam_size", "mode", "debug_transcript", "record_output",
                    "record_transcript", "effect_volume", "confirmation_count",
                    "stability_delay", "debug_hypotheses",
                }
                update_jsonc(DEFAULT_CONFIG, {k: v for k, v in values.items() if k in allowed})
                self._json({"ok": True})
            elif path == "/api/start":
                STATE.start()
                self._json({"ok": True})
            elif path == "/api/stop":
                STATE.stop()
                self._json({"ok": True})
            elif path == "/api/runtime":
                data = self._body()
                mode = str(data.get("mode", ""))
                effect_volume = float(data.get("effect_volume", 1.0))
                if mode not in {"reverse", "beep", "bark", "meow", "mute"}:
                    raise ValueError("Неизвестный режим обработки.")
                if not 0 <= effect_volume <= 2:
                    raise ValueError("Громкость должна быть от 0 до 2.")
                update_jsonc(
                    DEFAULT_CONFIG,
                    {"mode": mode, "effect_volume": effect_volume},
                )
                if STATE.running():
                    write_runtime_settings(mode, effect_volume)
                self._json(
                    {
                        "ok": True,
                        "mode": mode,
                        "effect_volume": effect_volume,
                    }
                )
            elif path == "/api/words":
                text = str(self._body().get("text", ""))
                errors = validate_words_text(text)
                if errors:
                    raise ValueError("\n".join(errors))
                WORDS_PATH.write_text(text.rstrip() + "\n", encoding="utf-8")
                self._json(
                    {
                        "ok": True,
                        "script": TEST_SCRIPT.read_text(encoding="utf-8"),
                        "highlight_rules": highlight_rules(WORDS_PATH),
                        "restart_required": STATE.running(),
                    }
                )
            elif path == "/api/open-recordings":
                RECORDINGS.mkdir(exist_ok=True)
                subprocess.Popen(["open", str(RECORDINGS)])
                self._json({"ok": True})
            elif path == "/api/shutdown":
                self._json({"ok": True})
                threading.Thread(target=shutdown_application, daemon=True).start()
            else:
                self._json({"error": "Не найдено"}, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def shutdown_application() -> None:
    STATE.stop()
    process = STATE.process
    if process and process.poll() is None:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
    if SERVER:
        SERVER.shutdown()


def main() -> None:
    if "--engine" in sys.argv:
        from app import main as engine_main

        sys.argv = [argument for argument in sys.argv if argument != "--engine"]
        engine_main()
        return
    global SERVER
    try:
        server, port, already_running = find_or_create_server()
    except OSError as error:
        print(f"Не удалось запустить Stream Censor: {error}", flush=True)
        return
    url = f"http://{HOST}:{port}"
    if already_running:
        print(f"Stream Censor уже запущен: {url}", flush=True)
        webbrowser.open(url)
        return
    assert server is not None
    SERVER = server
    print(f"Stream Censor: {url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        STATE.stop()
    finally:
        STATE.stop()
        server.server_close()
        SERVER = None


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
