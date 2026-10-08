import threading
import time
import numpy as np
from src.video_capture import VideoCapture, VideoThread


class Device:
    def __init__(self):
        self.opened = True
    def isOpened(self):
        return self.opened
    def set(self, *args):
        pass
    def read(self):
        return True, np.zeros((10, 10, 3), dtype='uint8')
    def release(self):
        self.opened = False


def wait_for(condition):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(.01)
    assert condition()


def test_stop_discards_stale_camera_frames():
    capture = VideoCapture(capture_factory=lambda _: Device()).start()
    capture.read_frame()
    capture.stop()
    assert capture.recent_frames() == []


def test_camera_pause_releases_device_and_resume_reopens_same_worker():
    capture = VideoCapture(capture_factory=lambda _: Device(), fps=100).start()
    worker = VideoThread(capture)
    try:
        worker.start()
        wait_for(lambda: bool(capture.recent_frames()))
        worker.set_enabled(False)
        wait_for(lambda: not capture.is_opened)
        assert capture.recent_frames() == []
        worker.set_enabled(True)
        wait_for(lambda: capture.is_opened and bool(capture.recent_frames()))
        assert worker.is_alive()
    finally:
        worker.stop()
        worker.join(3)
    assert not capture.is_opened
