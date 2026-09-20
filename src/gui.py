from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .bridge import Bridge


THEME_QSS = """
QMainWindow { background: #0c1421; }
QWidget#root { background: #0c1421; }
QLabel { color: #dbe4f0; font-family: "Microsoft YaHei UI","Microsoft YaHei"; background: transparent; }
QLabel#headerTitle { font-size: 19px; font-weight: 700; color: #eaf1fa; }
QLabel#headerSub { color: #7e93aa; font-size: 12px; }
QLabel#caption { color: #8fa3b8; font-size: 12px; }
QLabel#videoLabel { background: #070d17; color: #7e93aa; border: 1px solid #23304a; border-radius: 12px; font-size: 16px; }
QTextEdit#chat { background: #101a2c; color: #e6edf6; border: 1px solid #23304a; border-radius: 12px; padding: 10px; font-size: 15px; }
QLineEdit#input { background: #101a2c; color: #e6edf6; border: 1px solid #23304a; border-radius: 9px; padding: 9px 12px; font-size: 14px; selection-background-color: #2dd4bf; selection-color: #06211c; }
QLineEdit#input:focus { border: 1px solid #2dd4bf; }
QPushButton { background: #182338; color: #cfdcec; border: 1px solid #2a3a55; border-radius: 9px; padding: 8px 14px; font-size: 13px; }
QPushButton:hover { background: #1f2d47; border-color: #3b527a; }
QPushButton:pressed { background: #14203a; }
QPushButton:disabled { color: #5c6b80; background: #131c2e; border-color: #1d2839; }
QPushButton#sendButton { background: #2dd4bf; color: #06211c; font-weight: 700; border: none; padding: 8px 24px; }
QPushButton#sendButton:hover { background: #4ce0cc; }
QPushButton#sendButton:disabled { background: #1c4a43; color: #3f6f68; }
QPushButton#stopButton { border: 1px solid #7f3b47; color: #f0b4bd; }
QPushButton#stopButton:hover { background: #3a1f27; }
QPushButton#listenButton { border: 1px solid #8a6d2f; color: #f0d08a; }
QPushButton#listenButton:hover { background: #372f1c; }
QSplitter::handle { background: #0c1421; }
QStatusBar { background: #0a111d; color: #8fa3b8; border-top: 1px solid #1c2740; font-size: 12px; }
QStatusBar::item { border: none; }
QScrollBar:vertical { background: #0e1726; width: 10px; border-radius: 5px; margin: 0; }
QScrollBar::handle:vertical { background: #2a3a55; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #3b527a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #0e1726; height: 10px; border-radius: 5px; }
QScrollBar::handle:horizontal { background: #2a3a55; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""

class MainWindow(QMainWindow):
    def __init__(
        self,
        service,
        video_capture,
        *,
        audio_thread=None,
        video_thread=None,
        detector=None,
        gesture_recognizer=None,
        action_mapper=None,
        tts_player=None,
        output_dir: str | Path = "outputs",
        bridge: Bridge | None = None,
    ):
        super().__init__()
        self.service = service
        self.video_capture = video_capture
        self.audio_thread = audio_thread
        self.video_thread = video_thread
        self.detector = detector
        self.gesture_recognizer = gesture_recognizer
        self.action_mapper = action_mapper
        self.tts_player = tts_player
        self.output_dir = Path(output_dir)
        self.bridge = bridge or Bridge()
        self.latest_frame = None
        self._frame_counter = 0
        self._display_busy = False
        self._answer_buffer = ""
        self._pending_query: str | None = None
        self._query_lock = threading.Lock()
        self._query_wake = threading.Event()
        self._closing = False
        threading.Thread(
            target=self._query_loop, daemon=True, name="assistant-query"
        ).start()
        self._setup_ui()
        self._connect_signals()
        if self.video_thread is not None:
            # 检测与标注在采集线程执行，界面线程只负责显示（修复视频卡顿）
            self.video_thread.processor = self._process_camera_frame
            self.video_thread.should_emit = lambda: not self._display_busy

    def _setup_ui(self) -> None:
        self.setWindowTitle("多模态AI助手")
        self.resize(1300, 800)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("🤖 多模态 AI 助手")
        title.setObjectName("headerTitle")
        header.addWidget(title)
        header.addStretch(1)
        sub = QLabel("Qwen3.5-4B · Qwen3-VL-4B · SenseVoice · 本地推理")
        sub.setObjectName("headerSub")
        header.addWidget(sub)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        video_caption = QLabel("📷 实时画面 · 手势识别")
        video_caption.setObjectName("caption")
        left_layout.addWidget(video_caption)
        self.video_label = QLabel("等待摄像头画面")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label, 1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        chat_caption = QLabel("💬 对话")
        chat_caption.setObjectName("caption")
        right_layout.addWidget(chat_caption)
        self.chat = QTextEdit()
        self.chat.setObjectName("chat")
        self.chat.setReadOnly(True)
        self.chat.document().setDefaultStyleSheet("p { margin: 4px 0; }")
        right_layout.addWidget(self.chat)
        splitter.addWidget(right)
        splitter.setSizes([720, 560])
        layout.addWidget(splitter, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.input = QLineEdit()
        self.input.setObjectName("input")
        self.input.setPlaceholderText("输入问题，或说出问题；含‘看看、画面、摄像头’会进入视觉模式")
        self.input.returnPressed.connect(self._submit_query)
        input_row.addWidget(self.input, 1)
        self.send_button = QPushButton("发送")
        self.send_button.setObjectName("sendButton")
        self.send_button.clicked.connect(self._submit_query)
        input_row.addWidget(self.send_button)
        layout.addLayout(input_row)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self.import_button = QPushButton("📚 导入知识库")
        self.import_button.clicked.connect(self._import_document)
        actions.addWidget(self.import_button)
        self.vision_button = QPushButton("👁 分析画面")
        self.vision_button.clicked.connect(lambda: self._submit_query("/vision 描述当前画面"))
        actions.addWidget(self.vision_button)
        self.listen_button = QPushButton("🎤 暂停聆听")
        self.listen_button.setObjectName("listenButton")
        self.listen_button.clicked.connect(self._toggle_listening)
        actions.addWidget(self.listen_button)
        self.stop_button = QPushButton("⏹ 停止生成与播报")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self._stop_activity)
        actions.addWidget(self.stop_button)
        self.auto_button = QPushButton("🔁 恢复自动路由")
        self.auto_button.clicked.connect(self._enable_auto_route)
        actions.addWidget(self.auto_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.setStyleSheet(THEME_QSS)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("系统就绪")

    def _connect_signals(self) -> None:
        self.bridge.frame_ready.connect(self._update_frame)
        self.bridge.asr_result.connect(self._process_query)
        self.bridge.user_query.connect(self._show_user_query)
        self.bridge.assistant_chunk.connect(self._append_assistant_chunk)
        self.bridge.assistant_done.connect(self._assistant_done)
        self.bridge.status.connect(self.statusBar().showMessage)
        self.bridge.error.connect(self._show_error)
        self.bridge.gesture.connect(self._on_gesture_detected)
        self.bridge.mode_changed.connect(
            lambda mode: self.statusBar().showMessage(f"当前模式: {mode}")
        )
        self.bridge.knowledge_imported.connect(self._knowledge_imported)

    def _submit_query(self, text: str | None = None) -> None:
        query = text if isinstance(text, str) else self.input.text()
        query = query.strip()
        if not query:
            return
        self.input.clear()
        self._process_query(query)

    def _process_query(self, query: str) -> None:
        """单槽排队：忙时保留最新一条提问，避免语音连发把请求堆爆。"""
        if self._closing:
            return
        with self._query_lock:
            replaced = self._pending_query
            self._pending_query = query
        if replaced:
            self.bridge.status.emit(f"已跳过更早的提问：{replaced[:14]}")
        else:
            self.bridge.status.emit("已收到，正在处理…")
        self._query_wake.set()

    def _query_loop(self) -> None:
        while not self._closing:
            self._query_wake.wait(timeout=0.5)
            if self._closing:
                break
            with self._query_lock:
                query = self._pending_query
                self._pending_query = None
            self._query_wake.clear()
            if not query:
                continue
            self.bridge.user_query.emit(query)
            try:
                mode = self.service.router.route(query)
                self.bridge.mode_changed.emit(mode)
                answer = self.service.process(query, self.bridge.assistant_chunk.emit)
                self.bridge.assistant_done.emit(answer)
            except Exception as exc:
                self.bridge.error.emit(str(exc))
                self.bridge.assistant_done.emit("")

    def _show_user_query(self, query: str) -> None:
        self.chat.append(f"<p><span style='color:#22d3ee;font-weight:700'>你：</span>{self._escape(query)}</p>")
        self.chat.append("<p><span style='color:#6ee7b7;font-weight:700'>助手：</span><span id='answer'></span></p>")
        self._answer_buffer = ""
        self.send_button.setEnabled(False)

    def _append_assistant_chunk(self, chunk: str) -> None:
        self._answer_buffer += chunk
        cursor = self.chat.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(chunk)
        self.chat.setTextCursor(cursor)
        self.chat.ensureCursorVisible()

    def _assistant_done(self, answer: str) -> None:
        self.send_button.setEnabled(True)
        self.chat.append("")
        if answer and self.tts_player:
            self.tts_player.enqueue(answer)
        self.statusBar().showMessage("回答完成")

    def _process_camera_frame(self, frame):
        """在采集线程中做人体检测与标注，保持界面线程轻量（修复视频卡顿）。"""
        self.latest_frame = frame.copy()
        display = frame
        self._frame_counter += 1
        if self.detector and self._frame_counter % 3 == 0:
            try:
                results = self.detector.process_frame(frame)
                display = self.detector.draw_landmarks(frame, results)
                hands = getattr(results.get("hands"), "multi_hand_landmarks", None)
                if hands and self.gesture_recognizer:
                    gesture = self.gesture_recognizer.recognize(hands[0])
                    if gesture:
                        self.bridge.gesture.emit(gesture)
            except Exception as exc:
                self.bridge.status.emit(f"检测暂不可用: {exc}")
        return display

    def _update_frame(self, frame) -> None:
        """只做显示；配合采集线程的背压丢帧，画面不滞后累积。"""
        self._display_busy = True
        try:
            import cv2

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(pixmap)
        except Exception as exc:
            self.statusBar().showMessage(f"画面显示失败: {exc}")
        finally:
            self._display_busy = False

    def _on_gesture_detected(self, gesture: str) -> None:
        action = self.action_mapper.map(gesture) if self.action_mapper else "unknown"
        self.chat.append(f"<p><span style='color:#f0d08a;font-weight:700'>手势：</span>{gesture} → {action}</p>")
        if not self.action_mapper:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        callbacks = {
            "start_listening": lambda: self.statusBar().showMessage("正在聆听"),
            "stop": self._stop_activity,
        }
        self.action_mapper.execute(
            action,
            callbacks=callbacks,
            tts_player=self.tts_player,
            frame=self.latest_frame,
            output_path=self.output_dir / f"gesture_test_{stamp}.jpg",
        )

    def _import_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入知识库", "", "文档 (*.pdf *.txt *.md *.docx)"
        )
        if not path:
            return
        self.import_button.setEnabled(False)
        self.statusBar().showMessage("正在解析并建立索引")

        def worker():
            try:
                count = self.service.rag.import_document(path)
                self.bridge.knowledge_imported.emit(
                    f"已导入 {count} 个知识块: {Path(path).name}"
                )
            except Exception as exc:
                self.bridge.error.emit(str(exc))
                self.bridge.knowledge_imported.emit("知识库导入失败")

        threading.Thread(target=worker, daemon=True, name="knowledge-import").start()

    def _knowledge_imported(self, message: str) -> None:
        self.import_button.setEnabled(True)
        self.statusBar().showMessage(message)

    def _stop_activity(self) -> bool:
        self.service.cancel()
        if self.tts_player:
            self.tts_player.stop()
        self.statusBar().showMessage("已停止生成与播报")
        return True

    def _toggle_listening(self) -> None:
        thread = self.audio_thread
        if thread is None or not hasattr(thread, "set_enabled"):
            self.statusBar().showMessage("当前没有可用的麦克风监听")
            return
        enabled = not thread.is_enabled()
        thread.set_enabled(enabled)
        self.listen_button.setText("🎤 暂停聆听" if enabled else "🎤 开始聆听")
        if enabled:
            self.statusBar().showMessage("语音监听已开启")
        else:
            self.statusBar().showMessage("语音监听已暂停，可以安静打字了")

    def _enable_auto_route(self) -> None:
        self.service.router.enable_auto()
        self.statusBar().showMessage("已恢复自动模态路由")

    def _show_error(self, message: str) -> None:
        self.chat.append(f"<p style='color:#f87171'><b>错误：</b>{self._escape(message)}</p>")
        self.statusBar().showMessage(message)

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def closeEvent(self, event) -> None:
        self._closing = True
        self._query_wake.set()
        self._stop_activity()
        for thread in (self.audio_thread, self.video_thread):
            if thread:
                thread.stop()
                thread.join(timeout=3)
        if self.tts_player:
            self.tts_player.shutdown(timeout=3)
        self.video_capture.stop()
        capture = getattr(self.audio_thread, "capture", None)
        if capture:
            capture.stop()
        if self.detector:
            self.detector.close()
        event.accept()
