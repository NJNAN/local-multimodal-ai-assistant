"""Export a complete session, independently of the model's limited context window."""
from __future__ import annotations

import json
from pathlib import Path


def export_conversation(records: list[dict], destination: str | Path) -> Path:
    path = Path(destination)
    if path.suffix.lower() == ".json":
        content = json.dumps({"title": "多模态 AI 助手对话", "turns": records}, ensure_ascii=False, indent=2)
    elif path.suffix.lower() == ".md":
        parts = ["# 多模态 AI 助手对话\n"]
        for number, record in enumerate(records, 1):
            parts.append(f"## 第 {number} 轮 · {record.get('timestamp', '')}\n")
            parts.append(f"**你：**\n\n{record['query']}\n")
            parts.append(f"**助手：**\n\n{record.get('answer') or '（没有回答内容）'}\n")
            parts.append(f"状态：{record.get('status', 'generating')}\n")
            if record.get("error"):
                parts.append(f"错误：{record['error']}\n")
            details = record.get("details") or {}
            if details:
                parts.append(f"模式：{details['mode']} · 总耗时：{details['elapsed_seconds']:.2f}s\n")
                first = details.get("first_chunk_seconds")
                if first is not None:
                    parts.append(f"首段输出：{first:.2f}s\n")
                inference = details.get("inference") or {}
                if inference.get("output_tokens") is not None:
                    parts.append(f"输出 token：{inference['output_tokens']}\n")
                if inference.get("tpot_ms") is not None:
                    parts.append(f"SSE 观测 TPOT：{inference['tpot_ms']:.2f}ms\n")
                for index, source in enumerate(details.get("sources", []), 1):
                    page = f"，第 {source['page']} 页" if source.get("page") else ""
                    parts.append(f"- 资料 {index}：{source.get('source', '未知来源')}{page}\n")
            parts.append("---\n")
        content = "\n".join(parts)
    else:
        raise ValueError("对话导出仅支持 .md 或 .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
