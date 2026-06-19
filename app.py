from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from censor.paths import ensure_user_data


DEFAULT_CONFIG = ensure_user_data() / "config.jsonc"


def strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments without touching comment-like text in strings."""
    result = []
    index = 0
    in_string = False
    escaped = False
    while index < len(text):
        char = text[index]
        following = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and following == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and following == "*":
            index += 2
            while index + 1 < len(text) and text[index : index + 2] != "*/":
                if text[index] in "\r\n":
                    result.append(text[index])
                index += 1
            index += 2
            continue
        result.append(char)
        index += 1
    return "".join(result)


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        data = json.loads(
            strip_json_comments(config_path.read_text(encoding="utf-8"))
        )
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"Ошибка в {config_path}: строка {error.lineno}, столбец "
            f"{error.colno}: {error.msg}"
        )
    if not isinstance(data, dict):
        raise SystemExit(f"Ошибка в {config_path}: ожидается JSON-объект.")
    return data


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description="Локальный фильтр ненормативной речи для OBS"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="путь к JSON-файлу настроек",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device", type=int, default=defaults.get("input_device"))
    parser.add_argument("--output-device", type=int, default=defaults.get("output_device"))
    parser.add_argument("--delay", type=float, default=defaults.get("delay", 5.0))
    parser.add_argument("--chunk", type=float, default=defaults.get("chunk", 3.0))
    parser.add_argument(
        "--scan-every", type=float, default=defaults.get("scan_every", 0.8)
    )
    parser.add_argument(
        "--sample-rate", type=int, default=defaults.get("sample_rate", 48000)
    )
    parser.add_argument("--model", default=defaults.get("model", "small"))
    parser.add_argument(
        "--compute-type", default=defaults.get("compute_type", "int8")
    )
    parser.add_argument("--language", default=defaults.get("language", "ru"))
    parser.add_argument(
        "--mode",
        choices=("reverse", "beep", "bark", "meow", "mute"),
        default=defaults.get("mode", "reverse"),
    )
    parser.add_argument(
        "--beep-frequency",
        type=float,
        default=defaults.get("beep_frequency", 880.0),
    )
    parser.add_argument("--beam-size", type=int, default=defaults.get("beam_size", 3))
    parser.add_argument(
        "--debug-transcript",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("debug_transcript", False),
    )
    parser.add_argument(
        "--debug-hypotheses",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("debug_hypotheses", False),
    )
    parser.add_argument(
        "--debug-words",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("debug_words", False),
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=defaults.get("safety_margin", 0.8),
    )
    parser.add_argument(
        "--record-output",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("record_output", True),
    )
    parser.add_argument(
        "--record-transcript",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("record_transcript", True),
    )
    parser.add_argument(
        "--recordings-directory",
        default=defaults.get("recordings_directory", "recordings"),
    )
    parser.add_argument(
        "--runtime-control-file",
        default=defaults.get("runtime_control_file", ".runtime-control.json"),
    )
    parser.add_argument(
        "--effect-volume",
        type=float,
        default=defaults.get("effect_volume", 1.0),
    )
    parser.add_argument(
        "--confirmation-count",
        type=int,
        default=defaults.get("confirmation_count", 2),
    )
    parser.add_argument(
        "--stability-delay",
        type=float,
        default=defaults.get("stability_delay", 0.7),
    )
    parser.add_argument(
        "--word-time-tolerance",
        type=float,
        default=defaults.get("word_time_tolerance", 0.4),
    )
    parser.add_argument(
        "--censor-lead-ms",
        type=int,
        default=defaults.get("censor_lead_ms", 20),
    )
    parser.add_argument(
        "--censor-tail-ms",
        type=int,
        default=defaults.get("censor_tail_ms", 80),
    )
    parser.add_argument(
        "--crossfade-ms",
        type=int,
        default=defaults.get("crossfade_ms", 8),
    )
    parser.add_argument("--words", default=defaults.get("words", "words.txt"))
    return parser


def main() -> None:
    user_root = ensure_user_data()
    os.chdir(user_root)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    config_args, _ = config_parser.parse_known_args()
    defaults = load_config(config_args.config)
    args = build_parser(defaults).parse_args()
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit(
            "Зависимости не установлены. Выполните: "
            "python3 -m venv .venv && source .venv/bin/activate && "
            "python -m pip install -r requirements.txt"
        )

    if args.list_devices:
        print(sd.query_devices())
        return

    from censor.engine import CensorEngine, EngineConfig
    from censor.matcher import WordMatcher

    recommended_delay = args.chunk + 2.0
    if args.delay < recommended_delay:
        print(
            f"Предупреждение: задержка {args.delay:.1f} с слишком мала для окна "
            f"{args.chunk:.1f} с. Рекомендуется минимум {recommended_delay:.1f} с, "
            "иначе Whisper может завершить распознавание после выхода слова."
        )

    config = EngineConfig(
        input_device=args.input_device,
        output_device=args.output_device,
        sample_rate=args.sample_rate,
        delay_seconds=args.delay,
        chunk_seconds=args.chunk,
        scan_every=args.scan_every,
        language=args.language,
        model=args.model,
        compute_type=args.compute_type,
        mode=args.mode,
        beep_frequency=args.beep_frequency,
        beam_size=args.beam_size,
        debug_transcript=args.debug_transcript,
        debug_hypotheses=args.debug_hypotheses,
        debug_words=args.debug_words,
        safety_margin=args.safety_margin,
        record_output=args.record_output,
        record_transcript=args.record_transcript,
        recordings_directory=args.recordings_directory,
        runtime_control_file=args.runtime_control_file,
        effect_volume=args.effect_volume,
        confirmation_count=args.confirmation_count,
        stability_delay=args.stability_delay,
        word_time_tolerance=args.word_time_tolerance,
        censor_lead_ms=args.censor_lead_ms,
        censor_tail_ms=args.censor_tail_ms,
        crossfade_ms=args.crossfade_ms,
    )
    print(f"Настройки загружены из: {args.config}")
    engine = CensorEngine(config, WordMatcher.from_file(args.words))
    try:
        engine.run()
    except KeyboardInterrupt:
        print("\nОстановлено.")


if __name__ == "__main__":
    main()
