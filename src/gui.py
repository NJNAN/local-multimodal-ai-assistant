from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QShortcut,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .bridge import Bridge
from .gesture import GestureRecognizer


GESTURE_LABELS = {
    "thumbs_up": "点赞", "thumbs_down": "倒赞", "open_palm": "张开手掌",
    "index_up": "食指向上", "index_down": "食指向下", "victory": "胜利 V",
}
ACTION_LABELS = {
    "volume_up": "增大音量", "volume_down": "减小音量", "stop": "停止生成与播报",
    "start_listening": "开启语音监听", "mute": "切换静音", "take_snapshot": "拍照保存",
    "unknown": "无对应动作",
}


@dataclass
class QueryRequest:
    query: str
    mode: str
    frames: list | None = None
    image_name: str = ""


THEME_QSS = """
QMainWindow { background: #0c1421; }
QWidget#root { background: #0c1421; }
QLabel { color: #dbe4f0; font-family: "Microsoft YaHei UI","Microsoft YaHei"; background: transparent; }
QLabel#headerTitle { font-size: 19px; font-weight: 700; color: #eaf1fa; }
QLabel#headerSub { color: #7e93aa; font-size: 12px; }
QLabel#caption { color: #8fa3b8; font-size: 12px; }
QLabel#badge { background: #162b32; color: #85dccc; border: 1px solid #254b50; border-radius: 7px; padding: 6px 10px; font-size: 12px; }
QLabel#metrics { color: #8fa3b8; font-size: 12px; padding: 5px 0; }
QComboBox { background: #101a2c; color: #e6edf6; border: 1px solid #2a3a55; border-radius: 7px; padding: 7px 10px; min-width: 125px; }
QComboBox QAbstractItemView { background: #182338; color: #e6edf6; selection-background-color: #254b50; }
QPushButton:checked { background: #163b38; border-color: #2dd4bf; color: #b3f5e9; }
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
        detector_error: str = "",
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
        self.detector_error = detector_error
        self.gesture_recognizer = gesture_recognizer or GestureRecognizer()
        self.action_mapper = action_mapper
        self.tts_player = tts_player
        self.output_dir = Path(output_dir)
        self.bridge = bridge or Bridge()
        self.latest_frame = None
        self._frame_counter = 0
        self._gestures_enabled = threading.Event()
        self._gestures_enabled.set()
        self._last_detection_status = 0.0
        self._display_busy = False
        self._answer_buffer = ""
        self._answer_cursor = None
        self._pending_query: QueryRequest | None = None
        self._active_cancel: threading.Event | None = None
        self._display_cancel: threading.Event | None = None
        self._request_active = False
        self._busy = False
        self._suppress_tts = False
        self._response_details = None
        self._attached_frame = None
        self._attached_name = ""
        self.session_records: list[dict] = []
        self._response_started = 0.0
        self._query_lock = threading.Lock()
        self._query_wake = threading.Event()
        self._closing = False
        self._setup_ui()
        self._connect_signals()
        self._query_thread = threading.Thread(
            target=self._query_loop, daemon=True, name="assistant-query"
        )
        self._query_thread.start()
        if self.video_thread is not None:
            # 检测与标注在采集线程执行，界面线程只负责显示（修复视频卡顿）
            self.video_thread.processor = self._process_camera_frame
            self.video_thread.should_emit = lambda: not self._display_busy

    def _setup_ui(self) -> None:
        self.setWindowTitle("LocalInferLab · 多模态 AI 助手")
        self.resize(1300, 800)
        self.setMinimumSize(960, 640)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("多模态 AI 助手  /  8GB 推理实验台")
        title.setObjectName("headerTitle")
        header.addWidget(title)
        header.addStretch(1)
        sub = QLabel("Qwen3.5-4B · Qwen3-VL-4B · SenseVoice · 本地推理")
        sub.setObjectName("headerSub")
        header.addWidget(sub)
        layout.addLayout(header)

        indicators = QHBoxLayout()
        camera = "摄像头已连接" if self.video_thread is not None else "无摄像头 · 可上传图片"
        self.camera_badge = QLabel(camera)
        self.camera_badge.setObjectName("badge")
        indicators.addWidget(self.camera_badge)
        self.audio_badge = QLabel("语音监听已开启" if self.audio_thread is not None else "麦克风未连接")
        self.audio_badge.setObjectName("badge")
        indicators.addWidget(self.audio_badge)
        self.knowledge_badge = QLabel()
        self.knowledge_badge.setObjectName("badge")
        indicators.addWidget(self.knowledge_badge)
        self._refresh_knowledge_count()
        indicators.addStretch(1)
        self.monitor_button = QPushButton("推理监控")
        self.monitor_button.clicked.connect(self._open_inference_monitor)
        indicators.addWidget(self.monitor_button)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("自动识别模式", "auto")
        self.mode_combo.addItem("文字 / 知识库", "text")
        self.mode_combo.addItem("视觉问答", "vision")
        self.mode_combo.currentIndexChanged.connect(self._select_mode)
        indicators.addWidget(self.mode_combo)
        layout.addLayout(indicators)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(10)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)
        self.video_caption = QLabel("实时画面 · 人体姿态与手势识别")
        self.video_caption.setObjectName("caption")
        left_layout.addWidget(self.video_caption)
        self.video_label = QLabel("等待摄像头画面" if self.video_thread else "上传一张图片，开始视觉问答\n也可以直接输入文字进行对话")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(360, 270)
        left_layout.addWidget(self.video_label, 1)
        media_actions = QHBoxLayout()
        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self._upload_image)
        media_actions.addWidget(self.upload_button)
        self.camera_source_button = QPushButton("使用摄像头画面")
        self.camera_source_button.clicked.connect(self._use_camera)
        media_actions.addWidget(self.camera_source_button)
        left_layout.addLayout(media_actions)
        hint = QLabel("手势：张开手掌停止 · V 手势拍照 · 食指向上开启监听")
        hint.setObjectName("caption")
        hint.setWordWrap(True)
        left_layout.addWidget(hint)
        gesture_actions = QHBoxLayout()
        self.gesture_analyze_button = QPushButton("分析手势")
        self.gesture_analyze_button.setToolTip("分析当前图片或最新摄像头帧，只显示结果，不执行系统控制动作")
        self.gesture_analyze_button.setEnabled(self.detector is not None)
        self.gesture_analyze_button.clicked.connect(self._analyze_gestures)
        gesture_actions.addWidget(self.gesture_analyze_button)
        control_text = "实时手势控制：开启" if self.video_thread is not None else "实时控制：无摄像头"
        self.gesture_toggle_button = QPushButton(control_text)
        self.gesture_toggle_button.setCheckable(True)
        self.gesture_toggle_button.setChecked(True)
        self.gesture_toggle_button.setEnabled(self.detector is not None and self.video_thread is not None)
        self.gesture_toggle_button.setToolTip("保持手势至少 0.35 秒并连续确认后执行一次；松手或移出画面 0.6 秒后可再次触发，动作间隔至少 2 秒")
        self.gesture_toggle_button.toggled.connect(self._toggle_gestures)
        gesture_actions.addWidget(self.gesture_toggle_button)
        left_layout.addLayout(gesture_actions)
        self.gesture_status_label = QLabel()
        self.gesture_status_label.setObjectName("caption")
        self.gesture_status_label.setWordWrap(True)
        if self.detector is None:
            self.gesture_status_label.setText(f"手势检测不可用：{self.detector_error or '检测模块未初始化'}")
        elif self.video_thread is None:
            self.gesture_status_label.setText("手势检测已就绪 · 可上传图片后点击“分析手势”")
        else:
            self.gesture_status_label.setText("实时手势检测已就绪 · 请将手放入摄像头画面")
        left_layout.addWidget(self.gesture_status_label)
        self.gesture_analysis_label = QLabel("六种手势：点赞 / 倒赞 / 张开手掌 / 食指向上 / 食指向下 / 胜利 V")
        self.gesture_analysis_label.setObjectName("caption")
        self.gesture_analysis_label.setWordWrap(True)
        left_layout.addWidget(self.gesture_analysis_label)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)
        chat_caption = QLabel("对话记录 · 支持多轮追问")
        chat_caption.setObjectName("caption")
        right_layout.addWidget(chat_caption)
        self.chat = QTextEdit()
        self.chat.setObjectName("chat")
        self.chat.setReadOnly(True)
        self.chat.document().setDefaultStyleSheet("p { margin: 4px 0; }")
        right_layout.addWidget(self.chat)
        self.metrics_label = QLabel("输入问题开始体验，回答完成后显示耗时和资料来源")
        self.metrics_label.setObjectName("metrics")
        self.metrics_label.setWordWrap(True)
        right_layout.addWidget(self.metrics_label)
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
        self.vision_button.clicked.connect(self._analyze_camera)
        actions.addWidget(self.vision_button)
        self.listen_button = QPushButton("🎤 暂停聆听")
        self.listen_button.setObjectName("listenButton")
        self.listen_button.clicked.connect(self._toggle_listening)
        self.listen_button.setEnabled(self.audio_thread is not None)
        if self.audio_thread is None:
            self.listen_button.setText("🎤 麦克风未连接")
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

        session_actions = QHBoxLayout()
        self.new_button = QPushButton("新对话")
        self.new_button.clicked.connect(self._new_conversation)
        session_actions.addWidget(self.new_button)
        self.export_button = QPushButton("导出对话")
        self.export_button.clicked.connect(self._export_conversation)
        session_actions.addWidget(self.export_button)
        self.speech_button = QPushButton("自动播报")
        self.speech_button.setCheckable(True)
        self.speech_button.setChecked(bool(self.tts_player and getattr(self.tts_player, "playback", True)))
        self.speech_button.setEnabled(self.tts_player is not None)
        self.speech_button.toggled.connect(self._toggle_speech)
        session_actions.addWidget(self.speech_button)
        session_actions.addStretch(1)
        shortcuts = QLabel("Enter 发送   ·   Esc 停止   ·   Ctrl+L 输入   ·   Ctrl+S 导出")
        shortcuts.setObjectName("caption")
        session_actions.addWidget(shortcuts)
        layout.addLayout(session_actions)

        self.setStyleSheet(THEME_QSS)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("系统就绪")
        self._shortcuts = []
        for key, callback in (("Escape", self._stop_activity), ("Ctrl+L", self.input.setFocus), ("Ctrl+S", self._export_conversation)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(200)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._show_welcome()
        self.input.setFocus()

    def _connect_signals(self) -> None:
        self.bridge.frame_ready.connect(self._update_frame)
        self.bridge.asr_result.connect(self._process_query)
        self.bridge.user_query.connect(self._show_user_query)
        self.bridge.assistant_chunk.connect(self._append_assistant_chunk)
        self.bridge.assistant_done.connect(self._assistant_done)
        self.bridge.response_details.connect(self._show_response_details)
        self.bridge.request_started.connect(self._request_started)
        self.bridge.status.connect(self.statusBar().showMessage)
        self.bridge.error.connect(self._show_error)
        self.bridge.gesture.connect(self._on_gesture_detected)
        self.bridge.detection_status.connect(self._show_detection_status)
        self.bridge.gesture_analysis.connect(self._gesture_analysis_done)
        self.bridge.mode_changed.connect(self._mode_changed)
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
        query = query.strip()
        if not query:
            return
        mode = self.service.router.route(query)
        frames = None
        image_name = ""
        if self._attached_frame is not None and not query.lower().startswith("/text"):
            if self.mode_combo.currentData() != "text" or query.lower().startswith("/vision"):
                mode = "vision"
                frames = [self._attached_frame.copy()]
                image_name = self._attached_name
        request = QueryRequest(query, mode, frames, image_name)
        with self._query_lock:
            replaced = self._pending_query
            self._pending_query = request
            self._query_wake.set()
        self._set_busy(True)
        if replaced:
            self.bridge.status.emit(f"已替换排队中的提问：{replaced.query[:14]}")
        else:
            self.bridge.status.emit("已收到，正在处理…")

    def _query_loop(self) -> None:
        while not self._closing:
            self._query_wake.wait(timeout=0.5)
            if self._closing:
                break
            with self._query_lock:
                request = self._pending_query
                self._pending_query = None
                self._query_wake.clear()
                if request is not None:
                    cancel_event = threading.Event()
                    self._active_cancel = cancel_event
                    self._request_active = True
            if request is None:
                continue
            display_query = request.query
            if request.image_name:
                display_query = f"[图片：{request.image_name}] {display_query}"
            self.bridge.request_started.emit(cancel_event)
            self.bridge.user_query.emit(display_query)
            try:
                self.bridge.mode_changed.emit(request.mode)
                answer = self.service.process(
                    request.query, self.bridge.assistant_chunk.emit,
                    frames=request.frames, mode=request.mode, cancel_event=cancel_event,
                    on_details=self.bridge.response_details.emit,
                )
            except Exception as exc:
                self.bridge.error.emit(str(exc))
                answer = ""
            finally:
                with self._query_lock:
                    self._request_active = False
                    self._active_cancel = None
            self.bridge.assistant_done.emit(answer)

    def _request_started(self, event) -> None:
        self._display_cancel = event
        self._suppress_tts = event.is_set()

    def _open_inference_monitor(self) -> None:
        from .inference_monitor import InferenceMonitor
        client = getattr(self.service.text_llm, "client", None)
        self._monitor = InferenceMonitor(getattr(client, "manager", None), self)
        self._monitor.show()

    def _show_user_query(self, query: str) -> None:
        self.chat.append(f"<p><span style='color:#22d3ee;font-weight:700'>你：</span>{self._escape(query)}</p>")
        self.chat.append("<p><span style='color:#6ee7b7;font-weight:700'>助手：</span><span id='answer'></span></p>")
        self._answer_cursor = self.chat.textCursor()
        self._answer_cursor.movePosition(self._answer_cursor.End)
        self._answer_cursor.setKeepPositionOnInsert(True)
        self._answer_buffer = ""
        self._response_details = None
        self.session_records.append({"query": query, "answer": "", "timestamp": datetime.now().isoformat(timespec="seconds")})
        self._response_started = time.perf_counter()
        self._elapsed_timer.start()
        self._set_busy(True)

    def _append_assistant_chunk(self, chunk: str) -> None:
        if self._closing:
            return
        self._answer_buffer += chunk
        if self.session_records:
            self.session_records[-1]["answer"] = self._answer_buffer
        cursor = self._answer_cursor or self.chat.textCursor()
        # Appending an unrelated message can extend this cursor's selection.
        # Keep the insertion point, but never replace that appended message.
        cursor.clearSelection()
        cursor.setKeepPositionOnInsert(False)
        cursor.insertText(chunk)
        cursor.setKeepPositionOnInsert(True)
        self.chat.setTextCursor(cursor)
        self.chat.ensureCursorVisible()

    def _assistant_done(self, answer: str) -> None:
        self._elapsed_timer.stop()
        with self._query_lock:
            pending = self._pending_query is not None or self._request_active
        self._set_busy(pending)
        self.chat.append("")
        cancelled = self._suppress_tts or bool(self._display_cancel and self._display_cancel.is_set()) or bool(self._response_details and self._response_details.cancelled)
        if self.session_records:
            record = self.session_records[-1]
            record["answer"] = answer or self._answer_buffer
            record["status"] = "cancelled" if cancelled else ("completed" if answer else "error")
        if answer and self.tts_player and self.speech_button.isChecked() and not cancelled and not self._closing:
            self.tts_player.enqueue(answer)
        if cancelled:
            self.statusBar().showMessage("已停止生成与播报")
        elif answer:
            self.statusBar().showMessage("回答完成，可以继续追问")
        elif not self._closing:
            self.statusBar().showMessage("本次请求未完成，请查看错误信息后重试")
            self.metrics_label.setText(f"本次请求未完成 · 耗时 {time.perf_counter() - self._response_started:.2f}s · 可以重试")

    def _process_camera_frame(self, frame):
        """在采集线程中做人体检测与标注，保持界面线程轻量（修复视频卡顿）。"""
        self.latest_frame = frame.copy()
        display = frame
        self._frame_counter += 1
        if self.detector and self._frame_counter % 3 == 0:
            try:
                started = time.perf_counter()
                results = self.detector.process_frame(frame)
                display = self.detector.draw_landmarks(frame, results)
                aspect_ratio = frame.shape[1] / frame.shape[0]
                hands = getattr(results.get("hands"), "multi_hand_landmarks", None)
                if self.gesture_recognizer and self._gestures_enabled.is_set():
                    gesture = self.gesture_recognizer.recognize(
                        hands[0] if hands else None, aspect_ratio=aspect_ratio)
                    if gesture:
                        self.bridge.gesture.emit(gesture)
                report = self._describe_detection(results, (time.perf_counter() - started) * 1000, aspect_ratio)
                report["confirmed"] = self.gesture_recognizer.confirmed_gesture
                if time.monotonic() - self._last_detection_status >= 0.3:
                    self.bridge.detection_status.emit(report)
                    self._last_detection_status = time.monotonic()
            except Exception as exc:
                if self._gestures_enabled.is_set():
                    self.gesture_recognizer.recognize(None)
                if time.monotonic() - self._last_detection_status >= 1.0:
                    self.bridge.detection_status.emit({"error": str(exc)})
                    self._last_detection_status = time.monotonic()
        return display

    def _describe_detection(self, results, elapsed_ms: float, aspect_ratio: float = 1.0) -> dict:
        hands = getattr(results.get("hands"), "multi_hand_landmarks", None) or []
        gestures = [self.gesture_recognizer.classify(hand, aspect_ratio=aspect_ratio) for hand in hands]
        pose = getattr(results.get("pose"), "pose_landmarks", None)
        full_body = bool(pose and self.detector.is_full_body_visible(pose))
        return {"hands": len(hands), "gestures": gestures, "pose": pose is not None,
                "full_body": full_body, "elapsed_ms": elapsed_ms}

    @staticmethod
    def _detection_text(report) -> str:
        if report.get("error"):
            return f"检测失败：{report['error']}"
        gestures = "、".join(GESTURE_LABELS.get(name, "未匹配六种手势") for name in report["gestures"])
        hand_text = f"{report['hands']} 只手 · {gestures}" if report["hands"] else "未检测到手，请调整距离和光照"
        pose_text = "全身可见" if report["full_body"] else ("人体部分可见" if report["pose"] else "未检测到人体")
        return f"{hand_text} · {pose_text} · {report['elapsed_ms']:.0f}ms"

    def _show_detection_status(self, report) -> None:
        state = "实时手势" if self._gestures_enabled.is_set() else "实时手势控制已暂停"
        confirmation = ""
        if self._gestures_enabled.is_set() and not report.get("error"):
            confirmation = (" · 已确认，请松手后重试" if report.get("confirmed")
                            else " · 保持手势等待确认" if any(report["gestures"]) else "")
        self.gesture_status_label.setText(f"{state}：{self._detection_text(report)}{confirmation}")

    def _toggle_gestures(self, enabled: bool) -> None:
        if enabled:
            self._gestures_enabled.set()
        else:
            self._gestures_enabled.clear()
        self.gesture_recognizer.reset()
        self.gesture_toggle_button.setText("实时手势控制：开启" if enabled else "实时手势控制：暂停")
        self.gesture_status_label.setText("实时手势控制已开启" if enabled else "实时手势控制已暂停 · 仍可手动分析图片或画面")

    def _analyze_gestures(self) -> None:
        if self.detector is None:
            self.statusBar().showMessage(f"手势检测不可用：{self.detector_error or '检测模块未初始化'}")
            return
        frame = self._attached_frame if self._attached_frame is not None else self.latest_frame
        source = f"图片 {self._attached_name}" if self._attached_frame is not None else "当前摄像头画面"
        if frame is None:
            frames = self.video_capture.recent_frames(1)
            frame = frames[-1] if frames else None
        if frame is None:
            self.statusBar().showMessage("没有可分析的画面，请打开摄像头或上传包含手部的图片")
            return
        frame = frame.copy()
        self.gesture_analyze_button.setEnabled(False)
        self.gesture_analysis_label.setText(f"正在分析{source}…")

        def worker():
            try:
                started = time.perf_counter()
                results = self.detector.process_frame(frame)
                report = self._describe_detection(results, (time.perf_counter() - started) * 1000,
                                                  frame.shape[1] / frame.shape[0])
                report["source"] = source
            except Exception as exc:
                report = {"error": str(exc), "source": source}
            self.bridge.gesture_analysis.emit(report)

        threading.Thread(target=worker, daemon=True, name="gesture-analysis").start()

    def _gesture_analysis_done(self, report) -> None:
        if self._closing:
            return
        self.gesture_analyze_button.setEnabled(self.detector is not None)
        text = f"{report['source']}：{self._detection_text(report)}"
        self.gesture_analysis_label.setText(text)
        self.chat.append(f"<p style='color:#f0d08a'><b>手势分析：</b>{self._escape(text)}（仅分析，未执行控制动作）</p>")
        self.statusBar().showMessage("手势分析失败" if report.get("error") else "手势分析完成")

    def _update_frame(self, frame) -> None:
        """只做显示；配合采集线程的背压丢帧，画面不滞后累积。"""
        self._display_busy = True
        try:
            if self._attached_frame is not None:
                return
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
        if not self._gestures_enabled.is_set() or self._closing:
            return
        action = self.action_mapper.map(gesture) if self.action_mapper else "unknown"
        name = GESTURE_LABELS.get(gesture, gesture)
        action_label = ACTION_LABELS.get(action, action)
        self.chat.append(f"<p><span style='color:#f0d08a;font-weight:700'>手势：</span>{name}（{gesture}） → {action_label}</p>")
        if not self.action_mapper:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        callbacks = {
            "start_listening": self._start_listening,
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
        self._refresh_knowledge_count()
        self.statusBar().showMessage(message)

    def _stop_activity(self) -> bool:
        with self._query_lock:
            self._pending_query = None
            self._query_wake.clear()
            if self._active_cancel:
                self._active_cancel.set()
            active = self._request_active
        self._suppress_tts = True
        if self._display_cancel:
            self._display_cancel.set()
        self.service.cancel()
        if self.tts_player:
            self.tts_player.stop()
        self.statusBar().showMessage("已停止生成与播报")
        self._set_busy(active)
        return True

    def _toggle_listening(self) -> None:
        thread = self.audio_thread
        if thread is None or not hasattr(thread, "set_enabled"):
            self.statusBar().showMessage("当前没有可用的麦克风监听")
            return
        enabled = not thread.is_enabled()
        thread.set_enabled(enabled)
        self.audio_badge.setText("语音监听已开启" if enabled else "语音监听已暂停")
        self.listen_button.setText("🎤 暂停聆听" if enabled else "🎤 开始聆听")
        if enabled:
            self.statusBar().showMessage("语音监听已开启")
        else:
            self.statusBar().showMessage("语音监听已暂停，可以安静打字了")

    def _enable_auto_route(self) -> None:
        self.service.router.enable_auto()
        self.mode_combo.setCurrentIndex(0)
        self.statusBar().showMessage("已恢复自动模态路由")

    def _select_mode(self) -> None:
        mode = self.mode_combo.currentData()
        if mode == "auto":
            self.service.router.enable_auto()
        else:
            self.service.router.set_mode(mode, auto=False)
        self.statusBar().showMessage(f"模式已切换：{self.mode_combo.currentText()}")

    def _mode_changed(self, mode: str) -> None:
        router = self.service.router
        index = 0 if router.auto_mode else (1 if router.current_mode == "text" else 2)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(index)
        self.mode_combo.blockSignals(False)
        self.statusBar().showMessage("正在进行视觉问答…" if mode == "vision" else "正在进行文字 / 知识库问答…")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.new_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        self.send_button.setText("排队发送" if busy else "发送")

    def _update_elapsed(self) -> None:
        elapsed = time.perf_counter() - self._response_started
        self.metrics_label.setText(f"{'正在生成回答' if self._answer_buffer else '正在检索 / 加载模型'} · 已等待 {elapsed:.1f}s")

    def _show_response_details(self, details) -> None:
        self._response_details = details
        self._elapsed_timer.stop()
        if self.session_records:
            self.session_records[-1]["details"] = asdict(details)
        mode = "视觉问答" if details.mode == "vision" else "文字 / 知识库"
        first = f" · 首段 {details.first_chunk_seconds:.2f}s" if details.first_chunk_seconds is not None else ""
        status = "已停止" if details.cancelled else "完成"
        inference = getattr(details, "inference", {})
        tokens = inference.get("output_tokens")
        rate = inference.get("server_decode_tokens_s")
        performance = f" · 输出 {tokens} tok" if tokens is not None else ""
        if rate is not None:
            performance += f" · 引擎解码 {rate:.1f} tok/s"
        self.metrics_label.setText(f"{mode} · {status} · 总耗时 {details.elapsed_seconds:.2f}s{first}{performance} · 检索 {len(details.sources)} 条资料")
        if details.sources:
            references = []
            for number, item in enumerate(details.sources, 1):
                page = f" · 第 {item['page']} 页" if item.get("page") else ""
                references.append(f"资料 {number}：{self._escape(str(item.get('source', '未知来源')))}{page}")
            self.chat.append("<p style='color:#8fa3b8'>本次检索资料（供核对）：<br>" + "<br>".join(references) + "</p>")

    def _refresh_knowledge_count(self) -> None:
        store = self.service.rag.vector_store
        count = len(store.documents)
        self.knowledge_badge.setText(f"知识库 · {count} 个知识块")

    def _show_welcome(self) -> None:
        self.chat.setHtml(
            "<p style='color:#6ee7b7;font-size:18px'><b>从一个问题开始</b></p>"
            "<p>文字交流、语音提问、画面理解，都在同一个工作台。</p>"
            "<p style='color:#8fa3b8'>试试：<br>• 用三句话介绍你能做什么<br>"
            "• 导入课程资料后，询问其中的内容<br>• 上传图片，询问图中的文字或物体</p>"
        )

    def _upload_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择用于视觉问答的图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not filename:
            return
        try:
            from .image_utils import read_image

            frame = read_image(filename)
            if frame is None:
                raise ValueError("图片无法解码，请选择有效的图片文件")
            self._attached_frame = frame
            self._attached_name = Path(filename).name
            self.video_caption.setText(f"图片问答 · {self._attached_name}")
            self.camera_badge.setText("视觉来源 · 上传图片")
            self.mode_combo.setCurrentIndex(2)
            self._display_attached_image()
            self.input.setPlaceholderText("输入关于这张图片的问题，例如：图中写了什么？")
            self.input.setFocus()
            self.statusBar().showMessage("图片已就绪，输入问题后发送；可连续追问")
        except Exception as exc:
            self._show_error(f"图片加载失败：{exc}")

    def _display_attached_image(self) -> None:
        if self._attached_frame is None:
            return
        # Use the same display path as camera frames without replacing the source.
        attached = self._attached_frame
        self._attached_frame = None
        try:
            self._update_frame(attached)
        finally:
            self._attached_frame = attached

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if getattr(self, "_attached_frame", None) is not None:
            self._display_attached_image()

    def _use_camera(self) -> None:
        self._attached_frame = None
        self._attached_name = ""
        self.video_caption.setText("实时画面 · 人体姿态与手势识别")
        self.camera_badge.setText("摄像头已连接" if self.video_thread else "无摄像头 · 可上传图片")
        self.video_label.clear()
        if self.video_thread is None:
            self.video_label.setText("摄像头未连接，可继续上传图片或文字交流")
        self.input.setPlaceholderText("输入问题，或说出问题；含‘看看、画面、摄像头’会进入视觉模式")
        self._enable_auto_route()

    def _analyze_camera(self) -> None:
        if not self.video_capture.recent_frames(1):
            self.statusBar().showMessage("没有可分析的摄像头画面，请连接摄像头或使用上传图片")
            return
        self._use_camera()
        self._submit_query("/vision 描述当前画面")

    def _start_listening(self) -> bool:
        if self.audio_thread is None:
            self.statusBar().showMessage("麦克风未连接，仍可使用文字和图片问答")
            return False
        self.audio_thread.set_enabled(True)
        self.audio_badge.setText("语音监听已开启")
        self.listen_button.setText("🎤 暂停聆听")
        self.statusBar().showMessage("语音监听已开启")
        return True

    def _toggle_speech(self, enabled: bool) -> None:
        if not enabled and self.tts_player:
            self.tts_player.stop()
        self.statusBar().showMessage("自动播报已开启" if enabled else "自动播报已关闭")

    def _new_conversation(self) -> None:
        # The keyboard or direct calls must follow the same busy check as the button.
        with self._query_lock:
            if self._busy or self._request_active or self._pending_query is not None:
                self.statusBar().showMessage("请先停止并等待当前请求结束，再开始新对话")
                return
        if self.tts_player:
            self.tts_player.stop()
        self.service.reset_history()
        self.session_records.clear()
        self._answer_buffer = ""
        self._answer_cursor = None
        self._response_details = None
        self._show_welcome()
        self.metrics_label.setText("新对话已开始 · 知识库和当前图片仍可使用")
        self.statusBar().showMessage("已开始新对话")
        self.input.setFocus()

    def _export_conversation(self) -> None:
        if self._busy:
            self.statusBar().showMessage("请等待当前回答结束后导出对话")
            return
        if not self.session_records:
            self.statusBar().showMessage("还没有可以导出的对话记录")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "导出完整对话", str(self.output_dir / f"conversation_{stamp}.md"),
            "Markdown 文档 (*.md);;JSON 数据 (*.json)",
        )
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".json" if "JSON" in selected_filter else ".md")
        try:
            from .conversation import export_conversation

            export_conversation(self.session_records, path)
            self.statusBar().showMessage(f"已导出 {len(self.session_records)} 轮对话：{path}")
        except Exception as exc:
            self._show_error(f"对话导出失败：{exc}")

    def _show_error(self, message: str) -> None:
        if self.session_records and self._busy:
            self.session_records[-1]["error"] = message
        self.chat.append(f"<p style='color:#f87171'><b>错误：</b>{self._escape(message)}</p>")
        self.statusBar().showMessage(message)

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def closeEvent(self, event) -> None:
        self._closing = True
        self._query_wake.set()
        self._stop_activity()
        self._elapsed_timer.stop()
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
