from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Pattern


_TRIM_RE = re.compile(r"(^[^\wё]+|[^\wё]+$)", re.IGNORECASE)


def normalize_word(word: str) -> str:
    return _TRIM_RE.sub("", word.casefold().replace("ë", "ё"))


class WordMatcher:
    def __init__(self, patterns: Iterable[str]):
        self._patterns: List[Pattern[str]] = []
        self._hotwords: List[str] = []
        for raw in patterns:
            value = raw.strip().casefold()
            if not value or value.startswith("#"):
                continue
            if value.startswith("re:"):
                expression = value[3:]
            elif value.endswith("*"):
                literal = value[:-1]
                expression = re.escape(literal) + r"\w*"
                self._hotwords.append(literal)
            else:
                expression = re.escape(value)
                self._hotwords.append(value)
            self._patterns.append(re.compile(rf"^(?:{expression})$", re.IGNORECASE))

    @classmethod
    def from_file(cls, path: str | Path) -> "WordMatcher":
        return cls(Path(path).read_text(encoding="utf-8").splitlines())

    def matches(self, word: str) -> bool:
        normalized = normalize_word(word)
        return bool(normalized) and any(p.fullmatch(normalized) for p in self._patterns)

    def matches_text(self, text: str) -> bool:
        return any(self.matches(word) for word in re.findall(r"[\wё]+", text.casefold()))

    @property
    def hotwords(self) -> str:
        return ", ".join(dict.fromkeys(self._hotwords))
