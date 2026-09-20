from __future__ import annotations

import logging
import threading
import time
from collections import deque

LOGGER = logging.getLogger(__name__)


class VideoCapture:
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        frame_buffer_size: int = 60,
        capture_factory=None,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self._capture_factory = capture_factory
        self._frames = deque(maxlen=frame_buffer_size)
        self._lock = threading.Lock()

    @property
    def is_opened(self) -> bool:
        return bool(self.cap and self.cap.isOpened())

    def start(self) -> "VideoCapture":
        if self.is_opened:
            return self
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("缺少 opencv-python") from exc
        factory = self._capture_factory or cv2.VideoCapture
        self.cap = factory(self.camera_id)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}，请检查占用和权限")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        LOGGER.info("摄像头已启动: id=%s, %sx%s", self.camera_id, self.width, self.height)
        return self

    def read_frame(self):
        if not self.is_opened:
            raise RuntimeError("摄像头尚未启动")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        with self._lock:
            self._frames.append((time.time(), frame.copy()))
        return frame

    def recent_frames(self, count: int = 2):
        if count <= 0:
            return []
        with self._lock:
            return [frame.copy() for _, frame in list(self._frames)[-count:]]

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        LOGGER.info("摄像头已停止")

    def __enter__(self) -> "VideoCapture":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


class VideoThread(threading.Thread):
    def __init__(self, capture: VideoCapture, on_frame=None, processor=None, should_emit=None):
        super().__init__(daemon=True, name="video-capture")
        self.capture = capture
        self.on_frame = on_frame
        self.processor = processor      # 在采集线程执行的帧处理（检测+标注），避免阻塞界面
        self.should_emit = should_emit  # 返回 False 时丢帧（背压：只显示最新画面）
        self._stop_event = threading.Event()

    def run(self) -> None:
        delay = 1.0 / max(1, self.capture.fps)
        while not self._stop_event.is_set():
            started = time.perf_counter()
            try:
                frame = self.capture.read_frame()
                if frame is not None:
                    display = frame
                    if self.processor is not None:
                        try:
                            processed = self.processor(frame)
                            if processed is not None:
                                display = processed
                        except Exception as exc:
                            LOGGER.exception("帧处理异常: %s", exc)
                    if self.on_frame and (self.should_emit is None or self.should_emit()):
                        self.on_frame(display)
            except Exception as exc:
                LOGGER.exception("视频采集异常: %s", exc)
                break
            self._stop_event.wait(max(0.0, delay - (time.perf_counter() - started)))

    def stop(self) -> None:
        self._stop_event.set()
