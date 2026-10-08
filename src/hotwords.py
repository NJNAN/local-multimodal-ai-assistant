"""可持久化的专业词纠正表；只替换用户明确填写的别名，不猜测相似词。"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

LOGGER = logging.getLogger(__name__)
DEFAULT_ENTRIES = [
    {'term': '通义千问', 'aliases': ['同义千问']},
    {'term': 'SenseVoice', 'aliases': ['sense voice']},
    {'term': 'FAISS', 'aliases': ['faiss', '费斯']},
    {'term': 'PyQt', 'aliases': ['py qt']},
]


class HotwordStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.enabled = True
        self._entries = []
        self._rules = ()
        self._install(DEFAULT_ENTRIES, True)
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding='utf-8'))
                self._install(payload['entries'], payload.get('enabled', True))
            except (ValueError, TypeError, KeyError, OSError):
                LOGGER.exception('热词配置无法读取，使用默认词表: %s', self.path)

    @property
    def entries(self):
        with self._lock:
            return [{'term': e['term'], 'aliases': e['aliases'][:]} for e in self._entries]

    @staticmethod
    def _prepare(entries):
        normalized, replacements = [], {}
        for entry in entries:
            term = str(entry['term']).strip()
            if not term:
                raise ValueError('标准词不能为空，请填写后再保存。')
            aliases = entry.get('aliases', [])
            if isinstance(aliases, str):
                aliases = re.split(r'[,，;；\n]', aliases)
            aliases = list(dict.fromkeys(str(a).strip() for a in aliases if str(a).strip()))
            normalized.append({'term': term, 'aliases': aliases})
            for alias in [term, *aliases]:
                key = alias.casefold()
                if key in replacements and replacements[key] != term:
                    raise ValueError(f'“{alias}”对应多个标准词，请保留一个。')
                replacements[key] = term
        rules = []
        for alias, term in sorted(replacements.items(), key=lambda p: len(p[0]), reverse=True):
            pattern = re.escape(alias)
            if re.match(r'[a-z0-9_]', alias[0]):
                pattern = r'(?<![A-Za-z0-9_])' + pattern
            if re.match(r'[a-z0-9_]', alias[-1]):
                pattern += r'(?![A-Za-z0-9_])'
            rules.append((re.compile(pattern, re.IGNORECASE), term))
        return normalized, tuple(rules)

    def _install(self, entries, enabled):
        normalized, rules = self._prepare(entries)
        with self._lock:
            self._entries, self._rules, self.enabled = normalized, rules, bool(enabled)

    def update(self, entries, enabled=True):
        normalized, rules = self._prepare(entries)
        payload = {'enabled': bool(enabled), 'entries': normalized}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        with self._lock:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(self.path)
            self._entries, self._rules, self.enabled = normalized, rules, bool(enabled)
        LOGGER.info('热词已保存: %s 条，enabled=%s', len(normalized), enabled)

    def correct(self, text: str) -> str:
        with self._lock:
            enabled, rules = self.enabled, self._rules
        if not enabled or not rules:
            return text
        # 匹配原始文本后一次性替换，避免一个标准词被下一条规则二次改写。
        matches = []
        for pattern, term in rules:
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), term))
        matches.sort(key=lambda m: (m[0], -(m[1] - m[0])))
        parts, end = [], 0
        for start, stop, term in matches:
            if start < end:
                continue
            parts.extend([text[end:start], term])
            end = stop
        parts.append(text[end:])
        return ''.join(parts)
