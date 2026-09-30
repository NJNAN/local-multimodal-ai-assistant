"""Read-only desktop observability. Refresh never starts/unloads any model."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout


class InferenceMonitor(QDialog):
    snapshot_ready = pyqtSignal(object)

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.snapshot = {}
        self.setWindowTitle("推理监控 · 8GB 实验台")
        self.resize(780, 600)
        self.setStyleSheet("""
            QDialog { background: #0c1421; }
            QLabel { color: #cfdcec; }
            QTextEdit { background: #101b2b; color: #dce7f5; border: 1px solid #33445b; border-radius: 8px; padding: 10px; }
            QPushButton { background: #182338; color: #cfdcec; border: 1px solid #33445b; border-radius: 8px; padding: 8px; }
            QPushButton:hover { background: #24344b; }
        """)
        layout = QVBoxLayout(self)
        self.summary = QLabel("查看模型驻留、上下文、KV 类型、投机解码与原生指标")
        self.summary.setWordWrap(True)
        self.summary.setMinimumHeight(44)
        layout.addWidget(self.summary)
        layout.addWidget(QLabel("刷新只读取正在运行的服务；不加载模型。未运行和缺失指标保留未知状态。"))
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details)
        actions = QHBoxLayout()
        self.refresh_button = QPushButton("刷新指标")
        self.refresh_button.clicked.connect(self.refresh)
        actions.addWidget(self.refresh_button)
        export = QPushButton("导出监控 JSON")
        export.clicked.connect(self.export)
        actions.addWidget(export)
        layout.addLayout(actions)
        self.snapshot_ready.connect(self.show_snapshot)
        self.refresh()

    def refresh(self):
        self.refresh_button.setEnabled(False)
        def worker():
            try:
                snapshot = {"timestamp": datetime.now().isoformat(), "models": self.manager.describe() if self.manager else {},
                            "metrics": {mode: server.metrics() for mode, server in self.manager.servers.items()} if self.manager else {},
                            "note": "仅观测；实验矩阵请运行 scripts/run_all_bench.ps1"}
                for mode, state in snapshot["models"].items():
                    state["context_tokens"] = getattr(self.manager.servers[mode], "n_ctx", None)
            except Exception as exc:
                snapshot = {"error": str(exc)}
            self.snapshot_ready.emit(snapshot)
        threading.Thread(target=worker, daemon=True, name="inference-monitor").start()

    def show_snapshot(self, snapshot):
        self.snapshot = snapshot
        self.refresh_button.setEnabled(True)
        states = []
        for mode, model in snapshot.get("models", {}).items():
            tuning = model.get("inference", {})
            state = "运行中" if model["running"] else "已卸载"
            states.append(f"{mode}: {state} · 上下文 {model.get('context_tokens') or '未知'} tok · KV {tuning.get('cache_type_k', 'f16')}/{tuning.get('cache_type_v', 'f16')} · 投机 {tuning.get('spec_type', 'none')} · 活动请求 {model.get('inflight', 0)}")
        self.summary.setText("\n".join(states) or "暂无模型服务；发送问题后可查看推理指标。")
        self.details.setPlainText(json.dumps(snapshot, ensure_ascii=False, indent=2))

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出推理监控", "inference_monitor.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
