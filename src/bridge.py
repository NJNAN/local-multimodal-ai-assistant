from __future__ import annotations

try:
    from PyQt5.QtCore import QObject, pyqtSignal

    class Bridge(QObject):
        frame_ready = pyqtSignal(object)
        asr_result = pyqtSignal(str)
        assistant_chunk = pyqtSignal(str)
        assistant_done = pyqtSignal(str)
        status = pyqtSignal(str)
        error = pyqtSignal(str)
        gesture = pyqtSignal(str)
        mode_changed = pyqtSignal(str)
        knowledge_imported = pyqtSignal(str)
        user_query = pyqtSignal(str)
        response_details = pyqtSignal(object)
        request_started = pyqtSignal(object)
        detection_status = pyqtSignal(object)
        gesture_analysis = pyqtSignal(object)

except ImportError:

    class _Signal:
        def __init__(self):
            self.callbacks = []

        def connect(self, callback):
            self.callbacks.append(callback)

        def emit(self, *args):
            for callback in list(self.callbacks):
                callback(*args)

    class Bridge:
        def __init__(self):
            for name in (
                "frame_ready",
                "asr_result",
                "assistant_chunk",
                "assistant_done",
                "status",
                "error",
                "gesture",
                "mode_changed",
                "knowledge_imported",
                "user_query",
                "response_details",
                "request_started",
                "detection_status",
                "gesture_analysis",
            ):
                setattr(self, name, _Signal())
