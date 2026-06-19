from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import sounddevice as sd

from app import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "app.py"
TEST_SCRIPT = ROOT / "test_script.txt"
WORDS_PATH = ROOT / "words.txt"
RECORDINGS = ROOT / "recordings"
HOST = "127.0.0.1"
PORT = 8765


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
        self.append_log("\n=== Новый запуск ===\n")
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(APP_PATH)],
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


HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stream Censor</title>
<style>
:root { color-scheme: dark; --bg:#111318; --panel:#1b1e25; --line:#303541;
  --text:#f1f3f7; --muted:#9da5b4; --accent:#725cff; --green:#34c98f; }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--text);
  font:14px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
header { height:60px; display:flex; align-items:center; justify-content:space-between;
  padding:0 22px; border-bottom:1px solid var(--line); }
h1 { font-size:20px; margin:0 } #status { color:var(--muted) }
main { display:grid; grid-template-columns:330px 1fr; gap:14px; padding:14px;
  height:calc(100vh - 60px); }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:16px; min-height:0; } .controls { overflow:auto }
.right { display:grid; grid-template-rows:1fr 1fr; gap:14px; min-height:0 }
h2 { font-size:14px; margin:0 0 14px; color:#cbd0db }
label { display:block; color:var(--muted); margin:11px 0 5px }
input,select { width:100%; padding:9px 10px; border-radius:7px; border:1px solid #414755;
  background:#11141a; color:var(--text); }
.check { display:flex; gap:8px; align-items:center; color:var(--text); margin:12px 0 }
.check input { width:auto }
.buttons { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:16px }
button { border:0; border-radius:8px; padding:10px; cursor:pointer; font-weight:600;
  color:white; background:#343946 } button.primary { background:var(--accent) }
button.stop { background:#b64252 } button:disabled { opacity:.45; cursor:default }
.danger { width:100%; margin-top:16px; background:#8f2f3c }
.full { width:100%; margin-top:8px }
.script,pre { width:100%; height:calc(100% - 28px); margin:0; padding:14px;
  border:1px solid var(--line); border-radius:8px; background:#101218; color:#e5e8ef;
  overflow:auto; line-height:1.45 }
.script { font:16px -apple-system,BlinkMacSystemFont,sans-serif; white-space:pre-wrap }
mark { background:#725cff; color:white; border-radius:4px; padding:1px 3px;
  box-shadow:0 0 0 1px rgba(255,255,255,.12) inset }
pre { white-space:pre-wrap; font:12px Menlo,monospace }
@media(max-width:800px){ main{grid-template-columns:1fr;height:auto}.right{height:900px} }
</style>
</head>
<body>
<header><h1>Stream Censor</h1><div id="status">Загрузка…</div></header>
<main>
  <section class="panel controls">
    <h2>Настройки</h2>
    <label>Микрофон</label><select id="input_device"></select>
    <label>Вывод</label><select id="output_device"></select>
    <label>Задержка, сек</label><input id="delay" type="number" step=".1">
    <label>Окно распознавания, сек</label><input id="chunk" type="number" step=".1">
    <label>Период распознавания, сек</label><input id="scan_every" type="number" step=".1">
    <label>Модель</label><select id="model">
      <option>tiny</option><option>base</option><option>small</option>
      <option>medium</option><option>large-v3</option></select>
    <label>Beam size</label><input id="beam_size" type="number" min="1" max="10">
    <label>Обработка</label><select id="mode">
      <option value="reverse">Проиграть наоборот</option>
      <option value="beep">ПИП</option>
      <option value="bark">Гавканье</option>
      <option value="meow">Мяуканье</option>
      <option value="mute">Заглушить</option>
    </select>
    <label class="check"><input id="debug_transcript" type="checkbox">Показывать распознанный текст</label>
    <label class="check"><input id="record_output" type="checkbox">Записывать WAV</label>
    <label class="check"><input id="record_transcript" type="checkbox">Сохранять журнал TXT</label>
    <div class="buttons">
      <button id="start" class="primary" onclick="startApp()">▶ Запустить</button>
      <button id="stop" class="stop" onclick="stopApp()">■ Остановить</button>
    </div>
    <button class="full" onclick="save()">Сохранить настройки</button>
    <button class="full" onclick="openRecordings()">Открыть папку записей</button>
    <button class="danger" onclick="closeApp()">Закрыть приложение</button>
  </section>
  <section class="right">
    <div class="panel"><h2>Текст для проверки</h2><div id="script" class="script"></div></div>
    <div class="panel"><h2>Журнал</h2><pre id="log"></pre></div>
  </section>
</main>
<script>
const ids=["delay","chunk","scan_every","model","beam_size","mode","debug_transcript","record_output","record_transcript"];
function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]))}
function matchesRule(word,rules){const w=word.toLocaleLowerCase("ru");return rules.some(r=>{if(r.type==="prefix")return w.startsWith(r.value);if(r.type==="exact")return w===r.value;if(r.type==="regex"){try{return new RegExp("^(?:"+r.value+")$","iu").test(w)}catch(e){return false}}return false})}
function highlightScript(text,rules){let out="",last=0;const rx=/[\p{L}\p{N}_ё]+/giu;for(const m of text.matchAll(rx)){out+=escapeHtml(text.slice(last,m.index));const word=m[0];out+=matchesRule(word,rules)?`<mark>${escapeHtml(word)}</mark>`:escapeHtml(word);last=m.index+word.length}return out+escapeHtml(text.slice(last))}
async function api(path, options={}) {
  const r=await fetch(path,{headers:{"Content-Type":"application/json"},...options});
  const data=await r.json(); if(!r.ok) throw new Error(data.error||"Ошибка"); return data;
}
function option(select,value,label){const o=document.createElement("option");o.value=value;o.textContent=label;select.appendChild(o)}
async function load(){
  try {
    const d=await api("/api/state"), c=d.config;
    const inp=document.querySelector("#input_device"),out=document.querySelector("#output_device");
    d.inputs.forEach(x=>option(inp,x.id,x.label)); option(out,"null","Не выводить звук (только запись)");
    d.outputs.forEach(x=>option(out,x.id,x.label));
    inp.value=String(c.input_device); out.value=c.output_device===null?"null":String(c.output_device);
    ids.forEach(id=>{const e=document.querySelector("#"+id);e.type==="checkbox"?e.checked=!!c[id]:e.value=c[id]});
    document.querySelector("#script").innerHTML=highlightScript(d.script,d.highlight_rules); updateState(d.running);
  } catch(e){document.querySelector("#status").textContent=e.message}
}
function values(){const input=document.querySelector("#input_device"),output=document.querySelector("#output_device");const v={input_device:Number(input.value),output_device:output.value==="null"?null:Number(output.value)};
  ids.forEach(id=>{const e=document.querySelector("#"+id);v[id]=e.type==="checkbox"?e.checked:(e.type==="number"?Number(e.value):e.value)});return v}
async function save(){try{await api("/api/config",{method:"POST",body:JSON.stringify(values())});document.querySelector("#status").textContent="Настройки сохранены";return true}catch(e){alert(e.message);return false}}
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
async function poll(){try{const d=await api("/api/log");const p=document.querySelector("#log");if(p.textContent!==d.log){p.textContent=d.log;p.scrollTop=p.scrollHeight}updateState(d.running)}catch(e){}setTimeout(poll,700)}
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
                })
            except Exception as error:
                self._json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        elif path == "/api/log":
            self._json({"log": STATE.log_text(), "running": STATE.running()})
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
                    "record_transcript",
                }
                update_jsonc(DEFAULT_CONFIG, {k: v for k, v in values.items() if k in allowed})
                self._json({"ok": True})
            elif path == "/api/start":
                STATE.start()
                self._json({"ok": True})
            elif path == "/api/stop":
                STATE.stop()
                self._json({"ok": True})
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
    global SERVER
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    SERVER = server
    url = f"http://{HOST}:{PORT}"
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
    main()
