"""PyQt5 主窗口：视频区 + 对话区 + 手势 / 语音 / 知识库的全部界面逻辑。

说明：界面线程只负责显示与交互；采集、检测、推理都在各自后台线程执行，
跨线程更新一律经 src/bridge.py 的信号传递。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QScrollArea,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from .bridge import Bridge
from .gesture import GestureRecognizer


# 六种手势的界面显示名
GESTURE_LABELS = {
    "thumbs_up": "点赞", "thumbs_down": "倒赞", "open_palm": "张开手掌",
    "index_up": "食指向上", "index_down": "食指向下", "victory": "胜利 V",
}
# 手势动作的界面显示名（unknown = 无对应动作）
ACTION_LABELS = {
    "volume_up": "增大音量", "volume_down": "减小音量", "stop": "停止生成与播报",
    "start_listening": "开启语音监听", "mute": "切换静音", "take_snapshot": "拍照保存",
    "unknown": "无对应动作",
}


@dataclass
class QueryRequest:
    """一次待处理的用户请求：问题文本 + 模式 + （可选）上传图片帧。"""

    query: str
    mode: str
    frames: list | None = None
    image_name: str = ""


from .chat_widgets import ChatView, MessageInput
from .gui_theme import THEME_QSS
from .hotwords import HotwordStore
from .video_capture import VideoThread
from .tts import SentenceBuffer

LOGGER = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """应用主窗口：装配全部界面元素并托管交互状态。

    关键状态：单槽请求队列（_pending_query）、独立回答气泡（_answer_message）、
    手势开关（_gestures_enabled）、已上传图片（_attached_frame）。
    """

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
        hotwords=None,
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
        self.hotwords = hotwords or getattr(getattr(audio_thread, "asr", None), "hotwords", None) or HotwordStore(self.output_dir / "hotwords.json")
        if audio_thread is not None and hasattr(audio_thread, "asr"):
            audio_thread.asr.hotwords = self.hotwords
        self._camera_enabled = video_thread is not None
        self._answer_message = None
        self._speech_buffer = SentenceBuffer()
        # ---- 运行状态：最新帧 / 手势开关 / 请求槽位 / 流式回答缓冲 ----
        self.latest_frame = None
        self._frame_counter = 0
        self._gestures_enabled = threading.Event()
        self._gestures_enabled.set()
        self._last_detection_status = 0.0
        self._display_busy = False
        self._answer_buffer = ""
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
            self.video_thread.should_emit = lambda: not self._display_busy and self._camera_enabled
            self.video_thread.on_state = self.bridge.camera_state.emit

    def _setup_ui(self) -> None:
        self.setWindowTitle("LocalInferLab · 多模态 AI 助手")
        self.resize(1320, 860)
        self.setMinimumSize(1000, 700)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 18, 24, 8)
        layout.setSpacing(16)
        header = QHBoxLayout()
        title = QLabel("LocalInferLab")
        title.setObjectName("headerTitle")
        header.addWidget(title)
        sub = QLabel("你的多模态 AI 助手")
        sub.setObjectName("headerSub")
        header.addWidget(sub)
        header.addStretch()
        self.new_button = QPushButton("＋ 新对话")
        self.new_button.clicked.connect(self._new_conversation)
        header.addWidget(self.new_button)
        self.export_button = QPushButton("导出对话")
        self.export_button.clicked.connect(self._export_conversation)
        header.addWidget(self.export_button)
        layout.addLayout(header)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(16)
        splitter.setChildrenCollapsible(False)

        side = QFrame()
        side.setObjectName("sidebar")
        side.setMinimumWidth(310)
        side.setMaximumWidth(420)
        side_outer = QVBoxLayout(side)
        side_outer.setContentsMargins(0, 0, 0, 0)
        side_scroll = QScrollArea()
        side_scroll.setObjectName("sidebarScroll")
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setFrameShape(QFrame.NoFrame)
        side_content = QWidget()
        side_content.setObjectName('sidebarContent')
        left = QVBoxLayout(side_content)
        left.setContentsMargins(18, 20, 18, 18)
        left.setSpacing(10)

        def caption(text, section=False):
            label = QLabel(text)
            label.setObjectName("sectionTitle" if section else "caption")
            label.setWordWrap(True)
            left.addWidget(label)
            return label

        caption("视觉与图片", True)
        self.camera_badge = QLabel("摄像头已开启" if self._camera_enabled else "摄像头已关闭")
        self.camera_badge.setObjectName("badge")
        left.addWidget(self.camera_badge)
        self.video_caption = caption("实时画面" if self._camera_enabled else "上传图片或开启摄像头")
        self.video_label = QLabel("等待摄像头画面" if self._camera_enabled else "暂无画面\n上传图片，让助手帮你看看")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(180)
        self.video_label.setMaximumHeight(230)
        left.addWidget(self.video_label)
        media = QHBoxLayout()
        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self._upload_image)
        media.addWidget(self.upload_button)
        self.camera_toggle_button = QPushButton("关闭摄像头" if self._camera_enabled else "开启摄像头")
        self.camera_toggle_button.clicked.connect(self._toggle_camera)
        media.addWidget(self.camera_toggle_button)
        left.addLayout(media)
        source = QHBoxLayout()
        self.camera_source_button = QPushButton("使用摄像头画面")
        self.camera_source_button.clicked.connect(self._use_camera)
        source.addWidget(self.camera_source_button)
        self.vision_button = QPushButton("分析画面")
        self.vision_button.clicked.connect(self._analyze_camera)
        source.addWidget(self.vision_button)
        left.addLayout(source)
        caption("手势互动", True)
        gestures = QHBoxLayout()
        self.gesture_analyze_button = QPushButton("分析手势")
        self.gesture_analyze_button.setEnabled(self.detector is not None)
        self.gesture_analyze_button.clicked.connect(self._analyze_gestures)
        gestures.addWidget(self.gesture_analyze_button)
        self.gesture_toggle_button = QPushButton("实时手势控制：开启")
        self.gesture_toggle_button.setCheckable(True)
        self.gesture_toggle_button.setChecked(True)
        self.gesture_toggle_button.setEnabled(self.detector is not None and self._camera_enabled)
        self.gesture_toggle_button.toggled.connect(self._toggle_gestures)
        gestures.addWidget(self.gesture_toggle_button)
        left.addLayout(gestures)
        self.gesture_toggle_button.setToolTip("保持手势约 0.35 秒触发；松手后可再次使用。")
        self.gesture_status_label = caption("请将手放入画面" if self.detector and self._camera_enabled else "上传图片后可分析手势" if self.detector else "手势检测暂不可用")
        self.gesture_analysis_label = caption("点赞 / 倒赞：音量  ·  张掌：停止\n食指向上：聆听  ·  食指向下：静音  ·  V：拍照")
        caption("我的知识库", True)
        self.knowledge_badge = QLabel()
        self.knowledge_badge.setObjectName("badge")
        left.addWidget(self.knowledge_badge)
        self._refresh_knowledge_count()
        self.import_button = QPushButton("＋ 导入资料")
        self.import_button.clicked.connect(self._import_document)
        left.addWidget(self.import_button)
        caption("支持 PDF、Word、TXT、Markdown")
        caption("语音偏好", True)
        ready = getattr(getattr(self.audio_thread, 'asr', None), 'ready', None)
        self.audio_badge = QLabel("正在准备语音识别…" if ready is not None and not ready.is_set() else "语音监听已开启" if self.audio_thread else "麦克风未连接")
        self.audio_badge.setObjectName("badge")
        left.addWidget(self.audio_badge)
        voice = QHBoxLayout()
        self.listen_button = QPushButton("🎤 暂停聆听" if self.audio_thread else "🎤 开始聆听")
        self.listen_button.setObjectName("listenButton")
        self.listen_button.setEnabled(self.audio_thread is not None)
        self.listen_button.clicked.connect(self._toggle_listening)
        voice.addWidget(self.listen_button)
        self.speech_button = QPushButton("自动播报")
        self.speech_button.setCheckable(True)
        self.speech_button.setChecked(bool(self.tts_player and getattr(self.tts_player, "playback", True)))
        self.speech_button.setEnabled(self.tts_player is not None)
        self.speech_button.toggled.connect(self._toggle_speech)
        voice.addWidget(self.speech_button)
        left.addLayout(voice)
        self.finish_speech_button = QPushButton('立即识别')
        self.finish_speech_button.setEnabled(self.audio_thread is not None)
        self.finish_speech_button.setToolTip('说完后点击，立即结束当前录音并识别；嘈杂环境无需等待自动断句')
        self.finish_speech_button.clicked.connect(self._finish_speech)
        self.voice_mode_combo = QComboBox()
        self.voice_mode_combo.addItem('快速音色（离线）', 'auto')
        self.voice_mode_combo.addItem('自然音色（联网）', 'online')
        engine = getattr(self.tts_player, 'tts', None)
        self.voice_mode_combo.setCurrentIndex(1 if getattr(engine, 'backend', 'auto') == 'online' else 0)
        self.voice_mode_combo.setEnabled(engine is not None)
        self.voice_mode_combo.currentIndexChanged.connect(self._select_voice_mode)
        left.addWidget(self.voice_mode_combo)
        self.hotword_button = QPushButton("自定义语音热词")
        self.hotword_button.clicked.connect(self._open_hotwords)
        self.hotword_label = QLabel("")
        self.hotword_label.setObjectName('caption')
        self._refresh_hotwords()
        left.addStretch()
        self.monitor_button = QPushButton("推理监控")
        self.monitor_button.clicked.connect(self._open_inference_monitor)
        side_scroll.setWidget(side_content)
        side_outer.addWidget(side_scroll)
        preferences = QVBoxLayout()
        preferences.setContentsMargins(18, 6, 18, 16)
        preferences.setSpacing(8)
        preferences.addWidget(self.hotword_button)
        preferences.addWidget(self.hotword_label)
        preferences.addWidget(self.monitor_button)
        side_outer.addLayout(preferences)
        splitter.addWidget(side)

        panel = QFrame()
        panel.setObjectName("chatPanel")
        right = QVBoxLayout(panel)
        right.setContentsMargins(18, 18, 18, 16)
        right.setSpacing(12)
        chat_header = QHBoxLayout()
        label = QLabel("和助手聊聊")
        label.setObjectName("sectionTitle")
        chat_header.addWidget(label)
        chat_header.addStretch()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("自动识别模式", "auto")
        self.mode_combo.addItem("文字 / 知识库", "text")
        self.mode_combo.addItem("视觉问答", "vision")
        self.mode_combo.currentIndexChanged.connect(self._select_mode)
        chat_header.addWidget(self.mode_combo)
        self.auto_button = QPushButton("恢复自动路由")
        self.auto_button.clicked.connect(self._enable_auto_route)
        chat_header.addWidget(self.auto_button)
        right.addLayout(chat_header)
        self.chat = ChatView()
        self.chat.setObjectName("chat")
        right.addWidget(self.chat, 1)
        self.metrics_label = QLabel("准备好了，随时开始提问")
        self.metrics_label.setObjectName("metrics")
        self.metrics_label.setWordWrap(True)
        right.addWidget(self.metrics_label)
        self.attachment_label = QLabel()
        self.attachment_label.setObjectName("caption")
        self.attachment_label.hide()
        right.addWidget(self.attachment_label)
        composer = QFrame()
        composer.setObjectName("composer")
        compose = QVBoxLayout(composer)
        compose.setContentsMargins(10, 6, 10, 10)
        self.input = MessageInput()
        self.input.setObjectName("input")
        self.input.setFixedHeight(72)
        self.input.setPlaceholderText("发送消息，或说出你的问题…")
        self.input.returnPressed.connect(self._submit_query)
        compose.addWidget(self.input)
        input_actions = QHBoxLayout()
        input_actions.addWidget(self.finish_speech_button)
        hint = QLabel("Enter 发送  ·  Shift+Enter 换行")
        hint.setObjectName("caption")
        input_actions.addWidget(hint)
        input_actions.addStretch()
        self.stop_button = QPushButton("停止生成与播报")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self._stop_activity)
        input_actions.addWidget(self.stop_button)
        self.send_button = QPushButton("发送  →")
        self.send_button.setObjectName("sendButton")
        self.send_button.clicked.connect(self._submit_query)
        input_actions.addWidget(self.send_button)
        compose.addLayout(input_actions)
        right.addWidget(composer)
        splitter.addWidget(panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 920])
        layout.addWidget(splitter, 1)
        self.setStyleSheet(THEME_QSS)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("已就绪  ·  Esc 停止  ·  Ctrl+L 输入  ·  Ctrl+S 导出")
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
        """把所有后台线程信号接到界面槽函数（跨线程安全）。"""
        self.bridge.frame_ready.connect(self._update_frame)
        self.bridge.camera_state.connect(self._camera_state_changed)
        self.bridge.asr_result.connect(self._process_query)
        self.bridge.audio_status.connect(self.audio_badge.setText)
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
        if mode == 'vision' and frames is None:
            # 在提交时冻结视觉来源，关闭后的请求不能继续使用缓存里的旧画面。
            frames = self.video_capture.recent_frames(2) if self._camera_enabled else []
            if not frames:
                self.statusBar().showMessage('请先上传图片或开启摄像头，再提问画面内容')
                return
            request.frames = frames
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
        """请求处理循环：从单槽取最新请求 → 交给服务处理 → 回传结果。"""
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
                LOGGER.exception('用户请求处理失败')
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
        """打开只读的推理监控窗口（不加载 / 卸载模型）。"""
        from .inference_monitor import InferenceMonitor
        client = getattr(self.service.text_llm, "client", None)
        self._monitor = InferenceMonitor(getattr(client, "manager", None), self)
        self._monitor.show()

    def _show_user_query(self, query: str) -> None:
        if self.tts_player:
            self.tts_player.stop()
        self._speech_buffer = SentenceBuffer()
        self.chat.add_message("user", query)
        self._answer_message = self.chat.add_message("assistant", "")
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
        if self._answer_message:
            self.chat.update_message(self._answer_message, self._answer_buffer)
        sentences = self._speech_buffer.feed(chunk)
        if self._can_speak():
            for sentence in sentences:
                self.tts_player.enqueue(sentence)

    def _can_speak(self):
        return self.tts_player and self.speech_button.isChecked() and not self._suppress_tts and not self._closing and not (self._display_cancel and self._display_cancel.is_set())

    def _assistant_done(self, answer: str) -> None:
        """回答结束的收尾：更新忙闲、写会话记录、（未取消时）排队播报。"""
        self._elapsed_timer.stop()
        with self._query_lock:
            pending = self._pending_query is not None or self._request_active
        self._set_busy(pending)
        cancelled = self._suppress_tts or bool(self._display_cancel and self._display_cancel.is_set()) or bool(self._response_details and self._response_details.cancelled)
        if self.session_records:
            record = self.session_records[-1]
            record["answer"] = answer or self._answer_buffer
            record["status"] = "cancelled" if cancelled else ("completed" if answer else "error")
        if self._answer_message:
            self.chat.update_message(self._answer_message, answer or self._answer_buffer or ("已停止回答。" if cancelled else "这次没能完成回答，请稍后重试。"))
            if cancelled:
                self._answer_message.set_meta("已停止")
        if not pending:
            self.metrics_label.setText("已停止，可以继续提问" if cancelled else "回答完成，可以继续追问" if answer else "暂时无法回答，请重试")
        if answer and self.tts_player and self.speech_button.isChecked() and not cancelled and not self._closing:
            if not self._answer_buffer:
                self.tts_player.enqueue(answer)
            else:
                for sentence in self._speech_buffer.flush():
                    self.tts_player.enqueue(sentence)
        else:
            self._speech_buffer.flush()
        if cancelled:
            self.statusBar().showMessage("已停止生成与播报")
        elif answer:
            self.statusBar().showMessage("回答完成，可以继续追问")
        elif not self._closing:
            self.statusBar().showMessage("本次请求未完成，请查看错误信息后重试")
            self.metrics_label.setText("本次请求未完成，可以重试")

    def _process_camera_frame(self, frame):
        """在采集线程中做人体检测与标注，保持界面线程轻量（修复视频卡顿）。"""
        if self.video_thread is not None and not self._camera_enabled:
            return frame
        self.latest_frame = frame.copy()
        display = frame
        self._frame_counter += 1
        # 每 3 帧检测一次：单次检测约 46ms（complexity=1），兼顾流畅与跟手
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
            LOGGER.error('手势检测失败: %s', report['error'])
            return "暂时无法检测，请稍后重试"
        gestures = "、".join(GESTURE_LABELS.get(name, "未匹配六种手势") for name in report["gestures"])
        hand_text = f"{report['hands']} 只手 · {gestures}" if report["hands"] else "未检测到手，请调整距离和光照"
        pose_text = "全身可见" if report["full_body"] else ("人体部分可见" if report["pose"] else "未检测到人体")
        return f"{hand_text} · {pose_text}"

    def _show_detection_status(self, report) -> None:
        if self.video_thread is not None and not self._camera_enabled:
            return
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
        """手动分析当前图片 / 画面中的手势（只显示结果，不执行系统动作）。"""
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
        LOGGER.info("手势分析: %s", text)
        if not report.get("error"):
            names = "、".join(GESTURE_LABELS.get(name, "暂未识别") for name in report["gestures"])
            self.chat.add_message("assistant", f"{report['source']}：识别到{names}。" if names else f"{report['source']}：没有检测到手，请调整距离和光照后重试。")
        self.statusBar().showMessage("手势分析失败" if report.get("error") else "手势分析完成")

    def _update_frame(self, frame) -> None:
        """只做显示；配合采集线程的背压丢帧，画面不滞后累积。"""
        self._display_busy = True
        try:
            if self._attached_frame is not None or (not self._camera_enabled and not getattr(self, "_rendering_attachment", False)):
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
        """手势确认后的动作派发（名称映射 + 执行；冷却由识别器负责）。"""
        if not self._gestures_enabled.is_set() or self._closing or (self.video_thread is not None and not self._camera_enabled):
            return
        action = self.action_mapper.map(gesture) if self.action_mapper else "unknown"
        name = GESTURE_LABELS.get(gesture, gesture)
        action_label = ACTION_LABELS.get(action, action)
        LOGGER.info("手势动作: %s (%s) -> %s", name, gesture, action_label)
        self.gesture_status_label.setText(f"{name} · {action_label}")
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
        """选择文档并在后台线程导入知识库（解析 + 建索引 + 保存）。"""
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
        """停止当前生成 / 播报并清空排队请求（Esc、停止按钮、张掌手势共用）。"""
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

    def _finish_speech(self):
        if self.audio_thread and hasattr(self.audio_thread, 'request_flush'):
            self.audio_thread.request_flush()
            self.statusBar().showMessage('已结束说话，正在提交识别')

    def _select_voice_mode(self):
        if self.tts_player and hasattr(self.tts_player, 'tts'):
            self.tts_player.stop()
            self.tts_player.tts.set_backend(self.voice_mode_combo.currentData())
            self.statusBar().showMessage(f'朗读音色已切换：{self.voice_mode_combo.currentText()}')

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
        """统一忙闲状态：控制按钮可用性与“排队发送”文案。"""
        self._busy = busy
        self.new_button.setEnabled(not busy)
        self.export_button.setEnabled(not busy)
        self.send_button.setText("排队发送" if busy else "发送")

    def _update_elapsed(self) -> None:
        elapsed = time.perf_counter() - self._response_started
        self.metrics_label.setText("正在回答…" if self._answer_buffer else "正在思考…" if elapsed < 10 else "正在准备回答，请稍候…")

    def _show_response_details(self, details) -> None:
        self._response_details = details
        self._elapsed_timer.stop()
        if self.session_records:
            self.session_records[-1]["details"] = asdict(details)
        LOGGER.info("回答指标: %s", asdict(details))
        if details.sources and self._answer_message:
            references = []
            for item in details.sources:
                page = f" · 第 {item['page']} 页" if item.get("page") else ""
                references.append(f"{Path(str(item.get('source', '资料'))).name}{page}")
            self._answer_message.set_meta("参考资料：" + "；".join(references))

    def _refresh_knowledge_count(self) -> None:
        store = self.service.rag.vector_store
        count = len(store.documents)
        self.knowledge_badge.setText(f"知识库 · {count} 个知识块")

    def _show_welcome(self) -> None:
        self._answer_message = None
        self.chat.show_welcome(self._fill_prompt, self._upload_image, self._import_document)

    def _fill_prompt(self, text: str) -> None:
        self.input.setPlainText(text)
        self.input.setFocus()

    def _upload_image(self) -> None:
        """选择本地图片作为视觉输入（无摄像头也能用），支持连续追问。"""
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
            self.attachment_label.setText(f"已附图片：{self._attached_name}  ·  点击“使用摄像头画面”可切回")
            self.attachment_label.show()
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
        """把上传的图片按与摄像头帧相同的路径渲染到视频区（不替换图片源）。"""
        if self._attached_frame is None:
            return
        # 复用摄像头帧的显示路径；临时清空 _attached_frame，渲染完再放回。
        # Use the same display path as camera frames without replacing the source.
        attached = self._attached_frame
        self._attached_frame = None
        self._rendering_attachment = True
        try:
            self._update_frame(attached)
        finally:
            self._attached_frame = attached
            self._rendering_attachment = False

    def resizeEvent(self, event) -> None:
        """窗口缩放时重绘已上传的图片。"""
        super().resizeEvent(event)
        if getattr(self, "_attached_frame", None) is not None:
            self._display_attached_image()

    def _use_camera(self) -> None:
        self._attached_frame = None
        self._attached_name = ""
        self.attachment_label.hide()
        self.video_caption.setText("实时画面")
        self.camera_badge.setText("摄像头已开启" if self._camera_enabled else "摄像头已关闭")
        self.video_label.clear()
        self.video_label.setText("等待摄像头画面" if self._camera_enabled else "摄像头已关闭，点击“开启摄像头”继续")
        self.input.setPlaceholderText("发送消息，或说出你的问题…")
        self._enable_auto_route()

    def _toggle_camera(self) -> None:
        self._camera_enabled = not self._camera_enabled
        enabled = self._camera_enabled
        self.camera_toggle_button.setEnabled(False)
        self.camera_toggle_button.setText("正在开启…" if enabled else "正在关闭…")
        if not enabled:
            self.latest_frame = None
            self.gesture_recognizer.reset()
            self.camera_badge.setText("正在关闭摄像头…")
            if self._attached_frame is None:
                self.video_label.clear()
                self.video_label.setText("摄像头已关闭\n仍可上传图片或继续聊天")
        else:
            self.camera_badge.setText("正在开启摄像头…")
        self.gesture_toggle_button.setEnabled(enabled and self.detector is not None)
        if self.video_thread is None:
            self.video_thread = VideoThread(
                self.video_capture, self.bridge.frame_ready.emit,
                processor=self._process_camera_frame,
                should_emit=lambda: not self._display_busy and self._camera_enabled,
                on_state=self.bridge.camera_state.emit)
            self.video_thread.start()
        else:
            self.video_thread.set_enabled(enabled)

    def _camera_state_changed(self, state) -> None:
        if self._closing:
            return
        self._camera_enabled = bool(state['enabled'])
        self.camera_toggle_button.setEnabled(True)
        self.camera_toggle_button.setText("关闭摄像头" if self._camera_enabled else "开启摄像头")
        self.camera_badge.setText("摄像头已开启" if self._camera_enabled else "摄像头已关闭")
        self.gesture_toggle_button.setEnabled(self._camera_enabled and self.detector is not None)
        if not self._camera_enabled:
            self.latest_frame = None
            self.gesture_status_label.setText("摄像头已关闭 · 可手动分析上传图片")
        if state.get('error'):
            LOGGER.error("摄像头操作失败: %s", state['error'])
            self.statusBar().showMessage("摄像头无法开启，请检查设备连接或是否被其他应用占用")
        else:
            self.statusBar().showMessage("摄像头已开启" if self._camera_enabled else "摄像头已关闭，设备已释放")

    def _open_hotwords(self) -> None:
        from .hotword_dialog import HotwordDialog
        self._hotword_dialog = HotwordDialog(self.hotwords, self)
        self._hotword_dialog.accepted.connect(self._refresh_hotwords)
        self._hotword_dialog.show()

    def _refresh_hotwords(self) -> None:
        self.hotword_label.setText(f"{'已启用' if self.hotwords.enabled else '已暂停'} · {len(self.hotwords.entries)} 个专业词")

    def _analyze_camera(self) -> None:
        if self._attached_frame is not None:
            self._submit_query("/vision 描述这张图片")
            return
        if not self._camera_enabled or not self.video_capture.recent_frames(1):
            self.statusBar().showMessage("请先开启摄像头或上传图片，再分析画面")
            return
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
        # 快捷键与直接调用也必须走和按钮相同的忙闲检查：
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
        self._response_details = None
        self._show_welcome()
        self.metrics_label.setText("新对话已开始 · 知识库和当前图片仍可使用")
        self.statusBar().showMessage("已开始新对话")
        self.input.setFocus()

    def _export_conversation(self) -> None:
        """导出整段会话（Markdown / JSON，含耗时与检索来源）。"""
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
        LOGGER.error("后台错误: %s", message)
        self.statusBar().showMessage("操作暂未完成，请检查设备或稍后重试。详细信息已记录到后台日志。")

    @staticmethod
    def _escape(text: str) -> str:
        """转义 HTML 特殊字符（聊天区按富文本渲染，防注入）。"""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def closeEvent(self, event) -> None:
        """窗口关闭：封口（拒绝新请求）→ 停线程 → 收播放器 → 释放硬件。"""
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
