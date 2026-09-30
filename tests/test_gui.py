"""Exercise real Qt signal delivery and background requests without model downloads."""
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
    deadline = time.perf_counter() + 4
    while time.perf_counter() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.005)
    pytest.fail("Qt background request did not reach the expected state")


def test_qt_uses_real_plugin_path_and_can_render_text(qt_app):
    from pathlib import Path
    from PyQt5.QtCore import QCoreApplication
    from PyQt5.QtGui import QFontDatabase

    assert any((Path(directory) / "platforms").is_dir() for directory in QCoreApplication.libraryPaths())
    assert QFontDatabase().families(), "offscreen preview must contain readable text"


def test_text_turn_sources_speech_and_new_session(window, qt_app):
    assert not window.listen_button.isEnabled()
    window._submit_query("你好")
    pump_until(qt_app, lambda: len(window.session_records) == 1 and not window._busy)
    assert window.session_records[0]["answer"] == "测试回答"
    assert window.session_records[0]["status"] == "completed"
    assert "课程.pdf" in window.chat.toPlainText()
    assert "总耗时" in window.metrics_label.text()
    assert window.tts_player.answers == ["测试回答"]
    window._new_conversation()
    assert not window.session_records and not window.service.history
    assert len(window.service.rag.vector_store.documents) == 1


def test_stop_clears_pending_request_and_never_replays_cancelled_answer(window, qt_app):
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
    text = window.service.text_llm
    text.gate = threading.Event()
    window._submit_query("当前")
    pump_until(qt_app, lambda: text.entered.is_set() and len(window.session_records) == 1)
    window._submit_query("旧提问")
    window._submit_query("最新提问")
    # A later mode selection must not reroute the already queued text request.
    window.mode_combo.setCurrentIndex(2)
    text.gate.set()
    pump_until(qt_app, lambda: len(window.session_records) == 2 and not window._busy)
    assert text.queries == ["当前", "最新提问"]
    assert not window.service.vl_model.calls


def test_upload_unicode_image_multiturn_text_override_and_export(window, qt_app, tmp_path, monkeypatch):
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
    states = []
    window.audio_thread = SimpleNamespace(set_enabled=states.append)
    assert window._start_listening()
    assert states == [True]
    assert "暂停聆听" in window.listen_button.text()
    window.audio_thread = None


def test_stream_text_stays_with_its_answer_when_gesture_messages_arrive(window):
    window._show_user_query("问题")
    window._append_assistant_chunk("前半段")
    window._on_gesture_detected("victory")
    window._append_assistant_chunk("后半段")
    text = window.chat.toPlainText()
    assert "前半段后半段" in text
    assert "victory" in text, repr(text)
    assert text.index("前半段后半段") < text.index("victory")


def test_uploaded_gesture_analysis_reports_result_without_executing_action(window, qt_app):
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
    # An unbroken stream of alternate labels must not open the microphone or take photos.
    hand[0] = hand_for_gesture("victory")
    for _ in range(30):
        now[0] += 0.04
        window._process_camera_frame(frame)
    assert gestures == ["thumbs_down"]
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
    window.monitor_button.click()
    pump_until(qt_app, lambda: window._monitor.refresh_button.isEnabled())
    assert "暂无模型服务" in window._monitor.summary.text()
    assert window._monitor.snapshot["models"] == {}
    window._monitor.close()
