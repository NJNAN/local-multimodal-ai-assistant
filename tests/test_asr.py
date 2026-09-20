from __future__ import annotations

from src.asr import ASREngine


class FakeModel:
    def generate(self, **kwargs):
        return [{"text": "<|zh|><|NEUTRAL|><|WITHBG|>你好， AI 助手"}]


def test_asr_cleans_sensevoice_tags(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(b"test")
    engine = ASREngine(model=FakeModel())
    assert engine.transcribe(path) == "你好， AI 助手"


def test_asr_extracts_nested_result():
    assert ASREngine._extract_text([[{"text": "ok"}]]) == "ok"
