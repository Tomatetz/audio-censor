from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading

# macOS ships an old but still usable system Tk. Hide its deprecation notice
# before importing tkinter; it otherwise appears every time Finder launches us.
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import sounddevice as sd

from app import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "app.py"
TEST_SCRIPT = ROOT / "test_script.txt"
RECORDINGS = ROOT / "recordings"


def json_value(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def update_jsonc(path: Path, values: dict) -> None:
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        pattern = re.compile(
            rf'(^\s*"{re.escape(key)}"\s*:\s*)(.*?)(\s*,?\s*$)',
            re.MULTILINE,
        )
        replacement = rf"\g<1>{json_value(value)}\g<3>"
        text, count = pattern.subn(replacement, text, count=1)
        if count == 0:
            raise ValueError(f"Параметр {key!r} не найден в {path.name}")
    path.write_text(text, encoding="utf-8")


class StreamCensorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Stream Censor")
        self.root.geometry("1120x760")
        self.root.minsize(900, 620)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.input_devices: dict[str, int] = {}
        self.output_devices: dict[str, int | None] = {}
        self.config = load_config(DEFAULT_CONFIG)

        self._build()
        self._load_non_device_values()
        self._load_script()
        self.root.after(100, self._drain_log_queue)
        self.root.after(50, self._start_device_scan)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(outer, text="Настройки", padding=12)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        controls.columnconfigure(1, weight=1)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.delay_var = tk.StringVar()
        self.chunk_var = tk.StringVar()
        self.scan_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.beam_var = tk.StringVar()
        self.mode_var = tk.StringVar()
        self.debug_var = tk.BooleanVar()
        self.record_var = tk.BooleanVar()

        row = 0
        row = self._combo_row(
            controls, row, "Микрофон", self.input_var, (), width=31
        )
        self.input_combo = controls.grid_slaves(row=row - 1, column=1)[0]
        row = self._combo_row(
            controls, row, "Вывод", self.output_var, (), width=31
        )
        self.output_combo = controls.grid_slaves(row=row - 1, column=1)[0]
        row = self._entry_row(controls, row, "Задержка, сек", self.delay_var)
        row = self._entry_row(controls, row, "Окно ASR, сек", self.chunk_var)
        row = self._entry_row(controls, row, "Период ASR, сек", self.scan_var)
        row = self._combo_row(
            controls,
            row,
            "Модель",
            self.model_var,
            ("tiny", "base", "small", "medium", "large-v3"),
        )
        row = self._entry_row(controls, row, "Beam size", self.beam_var)
        row = self._combo_row(
            controls,
            row,
            "Обработка",
            self.mode_var,
            ("reverse", "beep", "bark", "meow", "mute"),
        )

        ttk.Checkbutton(
            controls, text="Показывать распознанный текст", variable=self.debug_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 2))
        row += 1
        ttk.Checkbutton(
            controls, text="Записывать WAV", variable=self.record_var
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        buttons = ttk.Frame(controls)
        buttons.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(14, 4))
        buttons.columnconfigure((0, 1), weight=1)
        self.start_button = ttk.Button(
            buttons, text="▶ Запустить", command=self.start
        )
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = ttk.Button(
            buttons, text="■ Остановить", command=self.stop, state="disabled"
        )
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        row += 1

        ttk.Button(
            controls, text="Сохранить настройки", command=self.save
        ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Button(
            controls, text="Открыть папку записей", command=self.open_recordings
        ).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)

        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=3)

        script_frame = ttk.LabelFrame(right, text="Текст для проверки", padding=6)
        log_frame = ttk.LabelFrame(right, text="Журнал", padding=6)
        script_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

        self.script_text = tk.Text(
            script_frame, wrap="word", font=("Helvetica", 14), height=12
        )
        script_scroll = ttk.Scrollbar(
            script_frame, orient="vertical", command=self.script_text.yview
        )
        self.script_text.configure(yscrollcommand=script_scroll.set)
        self.script_text.pack(side="left", fill="both", expand=True)
        script_scroll.pack(side="right", fill="y")

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            background="#151515",
            foreground="#e8e8e8",
            insertbackground="white",
            font=("Menlo", 11),
        )
        log_scroll = ttk.Scrollbar(
            log_frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

        self.status_var = tk.StringVar(value="Готов к запуску")
        ttk.Label(
            self.root, textvariable=self.status_var, anchor="w", padding=(12, 4)
        ).pack(fill="x")

    def _entry_row(self, parent, row, label, variable) -> int:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Entry(parent, textvariable=variable, width=14).grid(
            row=row, column=1, sticky="ew", pady=4
        )
        return row + 1

    def _combo_row(self, parent, row, label, variable, values, width=18) -> int:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            width=width,
        ).grid(row=row, column=1, sticky="ew", pady=4)
        return row + 1

    def _start_device_scan(self) -> None:
        self.status_var.set("Поиск аудиоустройств…")
        self.input_combo.configure(state="disabled")
        self.output_combo.configure(state="disabled")
        threading.Thread(target=self._query_devices, daemon=True).start()

    def _query_devices(self) -> None:
        try:
            devices = list(sd.query_devices())
        except Exception as error:
            error_message = str(error)
            self.root.after(0, lambda: self._device_scan_failed(error_message))
            return
        self.root.after(0, lambda: self._apply_devices(devices))

    def _apply_devices(self, devices) -> None:
        self.input_devices.clear()
        self.output_devices = {"Не выводить звук (только запись)": None}
        for index, device in enumerate(devices):
            name = f"{index}: {device['name']}"
            if device["max_input_channels"] > 0:
                self.input_devices[name] = index
            if device["max_output_channels"] > 0:
                self.output_devices[name] = index

        self.input_combo.configure(values=tuple(self.input_devices))
        self.output_combo.configure(values=tuple(self.output_devices))
        self.input_combo.configure(state="readonly")
        self.output_combo.configure(state="readonly")
        self.input_var.set(
            self._select_device(self.input_devices, self.config.get("input_device"))
        )
        self.output_var.set(
            self._select_device(self.output_devices, self.config.get("output_device"))
        )
        self.status_var.set(
            f"Найдено устройств: входов {len(self.input_devices)}, "
            f"выходов {max(0, len(self.output_devices) - 1)}"
        )

    def _device_scan_failed(self, error: str) -> None:
        self.input_combo.configure(state="readonly")
        self.output_combo.configure(state="readonly")
        self.output_combo.configure(values=tuple(self.output_devices))
        self.output_var.set("Не выводить звук (только запись)")
        self.status_var.set("Не удалось получить список аудиоустройств")
        messagebox.showerror("Аудиоустройства", error)

    def _select_device(self, mapping, desired):
        for label, index in mapping.items():
            if index == desired:
                return label
        return next(iter(mapping), "")

    def _load_non_device_values(self) -> None:
        self.delay_var.set(str(self.config.get("delay", 7.0)))
        self.chunk_var.set(str(self.config.get("chunk", 3.0)))
        self.scan_var.set(str(self.config.get("scan_every", 0.5)))
        self.model_var.set(self.config.get("model", "small"))
        self.beam_var.set(str(self.config.get("beam_size", 5)))
        self.mode_var.set(self.config.get("mode", "reverse"))
        self.debug_var.set(bool(self.config.get("debug_transcript", True)))
        self.record_var.set(bool(self.config.get("record_output", True)))

    def _load_script(self) -> None:
        if TEST_SCRIPT.exists():
            self.script_text.insert("1.0", TEST_SCRIPT.read_text(encoding="utf-8"))

    def _values(self) -> dict:
        if self.input_var.get() not in self.input_devices:
            raise ValueError("Выберите микрофон.")
        return {
            "input_device": self.input_devices[self.input_var.get()],
            "output_device": self.output_devices.get(self.output_var.get()),
            "delay": float(self.delay_var.get().replace(",", ".")),
            "chunk": float(self.chunk_var.get().replace(",", ".")),
            "scan_every": float(self.scan_var.get().replace(",", ".")),
            "model": self.model_var.get(),
            "beam_size": int(self.beam_var.get()),
            "mode": self.mode_var.get(),
            "debug_transcript": self.debug_var.get(),
            "record_output": self.record_var.get(),
        }

    def save(self, quiet=False) -> bool:
        try:
            values = self._values()
            if values["delay"] < values["chunk"] + 2:
                raise ValueError(
                    "Задержка должна быть минимум на 2 секунды больше окна ASR."
                )
            update_jsonc(DEFAULT_CONFIG, values)
            self.config.update(values)
            self.status_var.set("Настройки сохранены")
            if not quiet:
                messagebox.showinfo("Stream Censor", "Настройки сохранены.")
            return True
        except (ValueError, OSError) as error:
            messagebox.showerror("Ошибка настроек", str(error))
            return False

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not self.save(quiet=True):
            return
        self._append_log("\n=== Новый запуск ===\n")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", str(APP_PATH)],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            messagebox.showerror("Не удалось запустить", str(error))
            return
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set("Фильтр работает")
        threading.Thread(target=self._read_process, daemon=True).start()

    def _read_process(self) -> None:
        assert self.process and self.process.stdout
        for line in self.process.stdout:
            self.log_queue.put(line)
        code = self.process.wait()
        self.log_queue.put(f"\n=== Процесс завершён, код {code} ===\n")
        self.root.after(0, self._process_finished)

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.status_var.set("Остановка…")
        self.process.send_signal(signal.SIGINT)

    def _process_finished(self) -> None:
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Остановлено")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                self._append_log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def open_recordings(self) -> None:
        RECORDINGS.mkdir(exist_ok=True)
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(RECORDINGS)])
        elif os.name == "nt":
            os.startfile(RECORDINGS)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(RECORDINGS)])

    def _on_close(self) -> None:
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno(
                "Stream Censor", "Фильтр работает. Остановить и закрыть?"
            ):
                return
            self.process.send_signal(signal.SIGINT)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    StreamCensorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
