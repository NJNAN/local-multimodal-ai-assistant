import json

import pytest

from src.conversation import export_conversation


def test_export_preserves_all_turns_sources_metrics_and_cancelled_text(tmp_path):
    records = [{"query": f"问题{i}", "answer": "回答", "status": "completed"} for i in range(9)]
    records[-1].update(answer="部分回答", status="cancelled", details={
        "mode": "text", "elapsed_seconds": 1.5, "first_chunk_seconds": 0.2,
        "sources": [{"source": "课程.pdf", "page": 3}],
    })
    json_path = export_conversation(records, tmp_path / "中文目录" / "对话.json")
    assert json.loads(json_path.read_text(encoding="utf-8"))["turns"] == records
    md_path = export_conversation(records, tmp_path / "对话.md")
    text = md_path.read_text(encoding="utf-8")
    assert "第 9 轮" in text and "部分回答" in text and "cancelled" in text
    assert "课程.pdf，第 3 页" in text
    assert "总耗时：1.50s" in text and "首段输出：0.20s" in text


def test_export_rejects_unknown_format_without_creating_file(tmp_path):
    path = tmp_path / "x.exe"
    with pytest.raises(ValueError):
        export_conversation([], path)
    assert not path.exists()
