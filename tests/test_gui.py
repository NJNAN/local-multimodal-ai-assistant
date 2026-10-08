"""用真实的 Qt 信号与后台请求验证界面行为（不下载任何模型）。

外部依赖（文本模型 / RAG / 视觉模型 / 播放器）全部用替身对象注入，
配合 offscreen 平台跑通真实信号回路。守护：单槽排队、取消、上传图片、
多轮视觉、/text 命令、对话导出、手势只报告不执行、监控只读等验收点。
"""
import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


from src.assistant_service import AssistantService
from src.modal_router import ModalRouter
from src.rag import RAGPipeline


class FakeText:
    """替身文本模型：可挂“闸门”让生成卡住，便于测试排队与取消。"""

    def __init__(self):
        self.gate = None
        self.entered = threading.Event()
        self.queries = []

    def stream_generate(self, query, **kwargs):
        self.queries.append(query)
        self.entered.set()
        if self.gate:
            assert self.gate.wait(3), "test model was not released"
        yield "测试回答"


class FakePlayer:
    """替身 TTS 播放器：记录被播报的回答；停止/关闭为空实现。"""

    playback = True

    def __init__(self):
        self.answers = []

    def enqueue(self, answer):
        self.answers.append(answer)

    def stop(self):
        pass

    def shutdown(self, **kwargs):
        pass


@pytest.fixture(scope="module")
def qt_app():
    """整个模块共用一个离屏 QApplication；Qt 延迟到 fixture 内才导入。

    （收集阶段导入 Qt 会破坏 Windows 上 PyTorch 的 DLL 加载。）
    """
    # Import Qt when the fixture runs, after the ASR tests initialized PyTorch.
    # Importing Qt during collection breaks PyTorch DLL loading on Windows.
    global QFileDialog, MainWindow
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication, QFileDialog
    from src.gui import MainWindow
    from src.qt_runtime import configure_qt_plugins

    configure_qt_plugins()
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qt_app, tmp_path):
    """搭一个「假后端 + 真界面」的主窗口；结束时清理并断言查询线程已退出。"""
    text = FakeText()
    sources = [{"source": "课程.pdf", "page": 1, "text": "课程内容"}]
    rag = SimpleNamespace(
        vector_store=SimpleNamespace(documents=sources),
        retrieve=lambda query, k: sources, build_context=RAGPipeline.build_context,
    )
    video = SimpleNamespace(recent_frames=lambda count: [], stop=lambda: None)
    vision_calls = []

    def analyze_frames(frames, query, **kwargs):
        vision_calls.append((frames, query, kwargs))
        return "图片回答"

    vision = SimpleNamespace(analyze_frames=analyze_frames, calls=vision_calls)
    service = AssistantService(text, rag, vision, ModalRouter(), video)
    view = MainWindow(service, video, tts_player=FakePlayer(), output_dir=tmp_path)
    yield view
    if text.gate:
        text.gate.set()
    view.close()
    view._query_thread.join(timeout=3)
    qt_app.processEvents()
    assert not view._query_thread.is_alive()


def pump_until(app, condition):
    """手动转 Qt 事件循环直到条件满足；4 秒超时直接判失败。"""
    deadline = time.perf_counter() + 4
    while time.perf_counter() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.005)
    pytest.fail("Qt background request did not reach the expected state")


def test_qt_uses_real_plugin_path_and_can_render_text(qt_app):
    """守护：Qt 插件路径指向真实目录、离屏环境字体可渲染（中文路径修复回归）。"""
    from pathlib import Path
    from PyQt5.QtCore import QCoreApplication
    from PyQt5.QtGui import QFontDatabase

    assert any((Path(directory) / "platforms").is_dir() for directory in QCoreApplication.libraryPaths())
    assert QFontDatabase().families(), "offscreen preview must contain readable text"


def test_text_turn_sources_speech_and_new_session(window, qt_app):
    """守护：一轮文字问答（回答/来源/耗时/自动播报）以及新对话的清空行为。"""
    assert not window.listen_button.isEnabled()
    window._submit_query("你好")
    pump_until(qt_app, lambda: len(window.session_records) == 1 and not window._busy)
    assert window.session_records[0]["answer"] == "测试回答"
    assert window.session_records[0]["status"] == "completed"
    assert "课程.pdf" in window.chat.toPlainText()
    assert "回答完成" in window.metrics_label.text()
    assert window.tts_player.answers == ["测试回答"]
    window._new_conversation()
    assert not window.session_records and not window.service.history
    assert len(window.service.rag.vector_store.documents) == 1


def test_stop_clears_pending_request_and_never_replays_cancelled_answer(window, qt_app):
    """守护：停止会丢弃排队请求；被取消的回答不进历史、不播报。"""
    text = window.service.text_llm
    text.gate = threading.Event()
    window._submit_query("当前问题")
    pump_until(qt_app, lambda: text.entered.is_set() and len(window.session_records) == 1)
    window._submit_query("排队问题")
    window._stop_activity()
    text.gate.set()
    pump_until(qt_app, lambda: not window._busy)
    assert text.queries == ["当前问题"]
    assert window.session_records[0]["status"] == "cancelled"
    assert not window.tts_player.answers and not window.service.history
    assert window.new_button.isEnabled()


def test_busy_queue_keeps_latest_query_and_its_submitted_mode(window, qt_app):
    """守护：忙时单槽只保留最新一条；已排队请求的模式不受之后切换影响。"""
    text = window.service.text_llm
    text.gate = threading.Event()
    window._submit_query("当前")
    pump_until(qt_app, lambda: text.entered.is_set() and len(window.session_records) == 1)
    window._submit_query("旧提问")
    window._submit_query("最新提问")
    # 之后切换模式不得改变已排队请求的路由：
    # A later mode selection must not reroute the already queued text request.
    window.mode_combo.setCurrentIndex(2)
    text.gate.set()
    pump_until(qt_app, lambda: len(window.session_records) == 2 and not window._busy)
    assert text.queries == ["当前", "最新提问"]
    assert not window.service.vl_model.calls


def test_upload_unicode_image_multiturn_text_override_and_export(window, qt_app, tmp_path, monkeypatch):
    """守护：中文名图片上传、多轮视觉历史、/text 覆盖、对话导出 3 轮。"""
    import numpy as np
    from src.image_utils import write_image

    path = write_image(tmp_path / "中文图片.png", np.zeros((40, 60, 3), dtype="uint8"))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
    window._upload_image()
    assert window._attached_frame is not None
    assert window.video_label.pixmap() is not None
    window._submit_query("这是什么")
    pump_until(qt_app, lambda: len(window.session_records) == 1 and not window._busy)
    window._submit_query("刚才说了什么")
    pump_until(qt_app, lambda: len(window.session_records) == 2 and not window._busy)
    assert len(window.service.vl_model.calls[1][2]["history"]) == 2
    assert "中文图片.png" in window.session_records[0]["query"]
    window._submit_query("/text 继续文字交流")
    pump_until(qt_app, lambda: len(window.session_records) == 3 and not window._busy)
    assert window.service.text_llm.queries == ["继续文字交流"]
    destination = tmp_path / "导出.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination), "JSON 数据 (*.json)"))
    window._export_conversation()
    import json

    assert len(json.loads(destination.read_text(encoding="utf-8"))["turns"]) == 3


def test_gesture_starts_paused_microphone(window):
    """守护：手势“食指向上”等价于把暂停的语音监听重新打开。"""
    states = []
    window.audio_thread = SimpleNamespace(set_enabled=states.append)
    assert window._start_listening()
    assert states == [True]
    assert "暂停聆听" in window.listen_button.text()
    window.audio_thread = None


def test_stream_text_stays_with_its_answer_when_gesture_messages_arrive(window):
    """守护：流式文本落点保持——中途插入手势消息后，回答仍拼回原位置。"""
    window._show_user_query("问题")
    window._append_assistant_chunk("前半段")
    window._on_gesture_detected("victory")
    window._append_assistant_chunk("后半段")
    text = window.chat.toPlainText()
    assert "前半段后半段" in text
    assert "victory" not in text, repr(text)
    assert "胜利 V" in window.gesture_status_label.text()


def test_uploaded_gesture_analysis_reports_result_without_executing_action(window, qt_app):
    """守护：“分析手势”只显示结果，绝不执行系统动作。"""
    import numpy as np
    from src.gesture_actions import GestureActionMapper
    from test_gesture import hand_for_gesture

    calls = []
    window.detector = SimpleNamespace(
        process_frame=lambda frame: {"hands": SimpleNamespace(multi_hand_landmarks=[hand_for_gesture("victory")]),
                                      "pose": SimpleNamespace(pose_landmarks=None)},
        close=lambda: None,
    )
    window.action_mapper = GestureActionMapper()
    window.action_mapper.execute = lambda *args, **kwargs: calls.append(args)
    window._attached_frame = np.zeros((40, 60, 3), dtype="uint8")
    window._attached_name = "手势图片.png"
    window._analyze_gestures()
    pump_until(qt_app, lambda: window.gesture_analyze_button.isEnabled())
    assert "胜利 V" in window.gesture_analysis_label.text()
    assert "手势图片.png" in window.chat.toPlainText()
    assert not calls


def test_live_gestures_can_pause_and_all_six_actions_still_work(window):
    """守护：六种手势都能派发动作；暂停开关生效后不再执行。"""
    from src.gesture import GestureRecognizer
    from src.gesture_actions import GestureActionMapper

    calls = []
    window.action_mapper = GestureActionMapper()
    window.action_mapper.execute = lambda action, **kwargs: calls.append(action)
    for gesture in GestureRecognizer.GESTURES:
        window._on_gesture_detected(gesture)
    assert set(calls) == set(window.action_mapper.action_map.values())
    window.gesture_toggle_button.setChecked(False)
    window._on_gesture_detected("open_palm")
    assert len(calls) == 6
    assert "暂停" in window.gesture_status_label.text()


def test_camera_processing_publishes_status_and_dispatches_gesture(window, qt_app):
    """守护：采集线程路径——状态节流上报，连续确认后派发手势。"""
    import numpy as np
    from test_gesture import hand_for_gesture

    report = []
    gestures = []
    now = [0.0]
    window.gesture_recognizer.clock = lambda: now[0]
    window.bridge.detection_status.connect(report.append)
    window.bridge.gesture.connect(gestures.append)
    window.detector = SimpleNamespace(
        process_frame=lambda frame: {"hands": SimpleNamespace(multi_hand_landmarks=[hand_for_gesture("index_up")]),
                                      "pose": SimpleNamespace(pose_landmarks=None)},
        draw_landmarks=lambda frame, results: frame,
        close=lambda: None,
    )
    frame = np.zeros((40, 60, 3), dtype="uint8")
    for _ in range(9):
        now[0] += 0.04
        window._process_camera_frame(frame)
    assert not gestures
    for _ in range(21):
        now[0] += 0.04
        window._last_detection_status = float("-inf")
        window._process_camera_frame(frame)
    assert gestures == ["index_up"]
    assert report[0]["hands"] == 1
    assert "食指向上" in window.gesture_status_label.text()
    assert "已确认" in window.gesture_status_label.text()


def test_camera_real_thumbs_down_is_one_action_and_missing_hand_rearms(window, qt_app):
    """守护：真实拇指下压连续帧只触发一次动作；手离开画面后能重新激活。"""
    import numpy as np
    from src.gesture_actions import GestureActionMapper
    from test_gesture import REGRESSION_SAMPLES, point, hand_for_gesture

    samples = [s for s in REGRESSION_SAMPLES if s["expected"] == "thumbs_down"]
    hand = [None]
    now = [0.0]
    gestures, actions = [], []
    window.gesture_recognizer.clock = lambda: now[0]
    window.bridge.gesture.connect(gestures.append)
    window.action_mapper = GestureActionMapper()
    window.action_mapper.execute = lambda action, **kwargs: actions.append(action)
    window.detector = SimpleNamespace(
        process_frame=lambda frame: {"hands": SimpleNamespace(multi_hand_landmarks=[hand[0]] if hand[0] else None),
                                      "pose": SimpleNamespace(pose_landmarks=None)},
        draw_landmarks=lambda frame, results: frame,
        close=lambda: None,
    )
    frame = np.zeros((240, 320, 3), dtype="uint8")
    for i in range(90):
        now[0] += 0.04
        hand[0] = [point(*p) for p in samples[(i // 3) % len(samples)]["landmarks"]]
        window._process_camera_frame(frame)
    assert gestures == ["thumbs_down"]
    assert actions == [window.action_mapper.map("thumbs_down")]
    # 连续交替标签的误检流不得打开麦克风或拍照：
    # An unbroken stream of alternate labels must not open the microphone or take photos.
    hand[0] = hand_for_gesture("victory")
    for _ in range(30):
        now[0] += 0.04
        window._process_camera_frame(frame)
    assert gestures == ["thumbs_down"]
    # 空帧必须到达识别器，否则下一次意图手势无法生效：
    # Empty frames must reach the recognizer so the next intentional gesture can work.
    hand[0] = None
    for _ in range(30):
        now[0] += 0.04
        window._process_camera_frame(frame)
    assert window.gesture_recognizer.confirmed_gesture is None
    hand[0] = hand_for_gesture("victory")
    for _ in range(30):
        now[0] += 0.04
        window._process_camera_frame(frame)
    assert gestures == ["thumbs_down", "victory"]
    assert actions == [window.action_mapper.map("thumbs_down"), window.action_mapper.map("victory")]


def test_monitor_is_read_only_and_works_without_loading_models(window, qt_app):
    """守护：推理监控只读——没有模型服务时也能打开，且不加载任何模型。"""
    window.monitor_button.click()
    pump_until(qt_app, lambda: window._monitor.refresh_button.isEnabled())
    assert "暂无模型服务" in window._monitor.summary.text()
    assert window._monitor.snapshot["models"] == {}
    window._monitor.close()


def test_chat_bubbles_align_and_logs_never_enter_conversation(window, qt_app):
    window.show()
    window._show_user_query('<用户问题>\n第二行')
    window._append_assistant_chunk('前半')
    window._on_gesture_detected('victory')
    window._show_error('backend-secret /server/traceback')
    window._append_assistant_chunk('后半')
    qt_app.processEvents()
    user, assistant = window.chat.messages
    assert user.role == 'user' and assistant.role == 'assistant'
    assert user.card.geometry().right() > assistant.card.geometry().right()
    assert '<用户问题>\n第二行' in window.chat.toPlainText()
    assert assistant.text == '前半后半'
    assert 'backend-secret' not in window.chat.toPlainText()


def test_hotword_editor_add_save_reload_and_live_correction(window):
    from PyQt5.QtWidgets import QTableWidgetItem
    from src.hotword_dialog import HotwordDialog
    from src.hotwords import HotwordStore
    dialog = HotwordDialog(window.hotwords, window)
    dialog.add_button.click()
    row = dialog.table.rowCount() - 1
    dialog.table.setItem(row, 0, QTableWidgetItem('自定义课程术语'))
    dialog.table.setItem(row, 1, QTableWidgetItem('自定义课程数语，课程错词'))
    dialog.save_button.click()
    assert dialog.result() == dialog.Accepted
    assert window.hotwords.correct('解释课程错词') == '解释自定义课程术语'
    assert HotwordStore(window.hotwords.path).correct('自定义课程数语') == '自定义课程术语'


def test_camera_toggle_releases_device_ignores_late_frames_and_reopens(window, qt_app):
    from src.video_capture import VideoCapture
    from test_video_capture import Device
    capture = VideoCapture(capture_factory=lambda _: Device(), fps=60)
    window.video_capture = window.service.video_capture = capture
    window.camera_toggle_button.click()
    pump_until(qt_app, lambda: capture.is_opened and bool(capture.recent_frames())
               and window.camera_toggle_button.isEnabled())
    window.camera_toggle_button.click()
    pump_until(qt_app, lambda: not capture.is_opened and window.camera_toggle_button.isEnabled())
    assert capture.recent_frames() == [] and window.latest_frame is None
    import numpy as np
    window._update_frame(np.zeros((20, 30, 3), dtype='uint8'))
    assert window.video_label.pixmap() is None
    assert '关闭' in window.camera_badge.text()
    window.camera_toggle_button.click()
    pump_until(qt_app, lambda: capture.is_opened and window.camera_toggle_button.isEnabled())


def test_closed_camera_cannot_submit_stale_frames(window):
    import numpy as np
    window.video_capture.recent_frames = lambda count: [np.zeros((10, 10, 3), dtype='uint8')]
    window._submit_query('/vision 看看画面')
    assert window._pending_query is None and not window._busy
    assert '开启摄像头' in window.statusBar().currentMessage()


def test_analyze_button_uses_uploaded_image_when_camera_is_closed(window, qt_app):
    import numpy as np
    window._attached_frame = np.zeros((20, 30, 3), dtype='uint8')
    window._attached_name = '图片.png'
    window.vision_button.click()
    pump_until(qt_app, lambda: len(window.session_records) == 1 and not window._busy)
    assert window.session_records[0]['answer'] == '图片回答'
    assert window._attached_frame is not None


def test_hotword_editor_rejects_blank_and_allows_deleting_rules(window):
    from src.hotword_dialog import HotwordDialog
    dialog = HotwordDialog(window.hotwords, window)
    dialog.add_button.click()
    dialog.save_button.click()
    assert dialog.result() != dialog.Accepted
    assert '不能为空' in dialog.feedback.text()
    dialog.table.selectRow(dialog.table.rowCount() - 1)
    dialog.remove_button.click()
    dialog.table.selectRow(0)
    dialog.remove_button.click()
    dialog.save_button.click()
    assert dialog.result() == dialog.Accepted
    assert window.hotwords.correct('同义千问') == '同义千问'


def test_first_sentence_is_sent_to_speech_before_answer_finishes(window):
    window._show_user_query('一个问题')
    window._append_assistant_chunk('这是首句。后面还在生成')
    assert window.tts_player.answers == ['这是首句。']
    assert window._busy
    window._assistant_done('这是首句。后面还在生成')
    assert window.tts_player.answers == ['这是首句。', '后面还在生成']


def test_cancel_does_not_speak_unfinished_stream_tail(window):
    window._show_user_query('问题')
    window._append_assistant_chunk('已完成。还未完成')
    window._stop_activity()
    window._assistant_done('已完成。还未完成')
    assert window.tts_player.answers == ['已完成。']
