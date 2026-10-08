"""摄像头视频采集模块。

所属链路：视觉输入链路（视频采集 → 帧处理/检测 → 主界面显示）。
职责：通过 OpenCV 的 ``VideoCapture`` 抽象拉取摄像头帧，并可选地在
    后台线程中按固定 FPS 节拍持续读取。
    - 输入：摄像头索引、目标分辨率、FPS、最近帧缓冲长度；
    - 输出：每次 ``read_frame()`` 返回 BGR ndarray；``recent_frames()``
      返回最近若干帧的拷贝列表（线程安全）；
    - 调用方：``scripts/02_capture_video.py``、主界面启动 ``VideoThread``
      后通过 ``on_frame`` 回调把画面推送给 UI。

设计要点：
    1. 默认 640x480 / 30 FPS，覆盖大多数笔记本摄像头的常见分辨率；
    2. ``_frames`` 用 ``deque(maxlen=...)`` 做环形缓冲，内存上限可控；
    3. ``VideoThread`` 提供生产者线程；``should_emit`` 是 UI 侧背压信号，
       当 UI 处理不过来时返回 ``False`` 直接丢帧。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

LOGGER = logging.getLogger(__name__)


class VideoCapture:
    """摄像头采集与最近帧回放封装，基于 ``cv2.VideoCapture``。

    通过 ``start`` 打开设备、``read_frame`` 拉帧、``stop`` 释放；``with``
    语法同样适用。``capture_factory`` 便于在测试中注入假工厂。
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        frame_buffer_size: int = 60,
        capture_factory=None,
    ):
        """初始化摄像头参数；并不立即打开设备。

        :param camera_id: OpenCV 摄像头索引；默认 0（系统默认摄像头）。
        :param width: 请求的帧宽度（最终值以驱动实际生效为准）。
        :param height: 请求的帧高度。
        :param fps: 期望 FPS；``VideoThread`` 会按该值节拍轮询。
        :param frame_buffer_size: 最近帧环形缓冲长度，按帧数计。
        :param capture_factory: 可选工厂函数，签名 ``(camera_id) -> cv2.VideoCapture``。
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self._capture_factory = capture_factory
        # 环形缓冲保存最近若干帧，供 ``recent_frames`` 回放；maxlen 控制上限避免内存膨胀。
        self._frames = deque(maxlen=frame_buffer_size)
        self._lock = threading.Lock()

    @property
    def is_opened(self) -> bool:
        """摄像头是否已成功打开（供其他模块判断是否需要 ``start``）。"""
        return bool(self.cap and self.cap.isOpened())

    def start(self) -> "VideoCapture":
        """打开摄像头并应用请求的分辨率 / FPS；幂等。"""
        if self.is_opened:
            return self
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("缺少 opencv-python") from exc
        factory = self._capture_factory or cv2.VideoCapture
        self.cap = factory(self.camera_id)
        if not self.cap.isOpened():
            # 打开失败时主动 release 并清空引用，避免半死状态被后续误用。
            self.cap.release()
            self.cap = None
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}，请检查占用和权限")
        # 这三个 set 只是向驱动"建议"参数，实际生效值要靠驱动；调用方不应假设绝对一致。
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        LOGGER.info("摄像头已启动: id=%s, %sx%s", self.camera_id, self.width, self.height)
        return self

    def read_frame(self):
        """读取一帧 BGR ndarray，失败返回 ``None``。

        成功读取时会把 ``(时间戳, 帧拷贝)`` 追加进环形缓冲，供
        ``recent_frames`` 回放使用；拷贝避免外部修改影响缓冲。
        """
        if not self.is_opened:
            raise RuntimeError("摄像头尚未启动")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        with self._lock:
            self._frames.append((time.time(), frame.copy()))
        return frame

    def recent_frames(self, count: int = 2):
        """返回最近 ``count`` 帧的浅拷贝列表（线程安全）。

        ``count <= 0`` 时返回空列表。返回的每帧都是 ``frame.copy()``，
        调用方可以安全就地修改（例如画标注）而不会污染缓冲。
        """
        if count <= 0:
            return []
        with self._lock:
            return [frame.copy() for _, frame in list(self._frames)[-count:]]

    def stop(self) -> None:
        """释放摄像头资源；幂等。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        with self._lock:
            self._frames.clear()
        LOGGER.info("摄像头已停止")

    def __enter__(self) -> "VideoCapture":
        """进入 with 块时自动打开摄像头。"""
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """退出 with 块时自动释放。"""
        self.stop()


class VideoThread(threading.Thread):
    """按固定 FPS 拉帧并把帧推给回调的后台线程。

    把"读帧 + 帧处理"放在采集线程中，避免阻塞 UI 主线程：
        - ``processor`` 同步执行（如检测 + 标注），保证画面送出时已带标注；
        - ``on_frame`` 把画面交给 UI；
        - ``should_emit`` 是 UI 侧背压闸门，返回 ``False`` 时本线程直接
          丢帧，避免队列堆积。
    """

    def __init__(self, capture: VideoCapture, on_frame=None, processor=None, should_emit=None, on_state=None):
        """构造视频采集线程（守护线程，名称 ``video-capture``）。

        :param capture: 已 start 过的 :class:`VideoCapture` 实例。
        :param on_frame: 接收到帧后的回调，签名 ``(frame) -> None``。
        :param processor: 在采集线程内执行的帧处理函数；返回新帧或 ``None``。
        :param should_emit: 背压判断，``False`` 时直接丢帧。
        """
        super().__init__(daemon=True, name="video-capture")
        self.capture = capture
        self.on_frame = on_frame
        self.processor = processor      # 在采集线程执行的帧处理（检测+标注），避免阻塞界面
        self.should_emit = should_emit  # 返回 False 时丢帧（背压：只显示最新画面）
        self._stop_event = threading.Event()
        self._enabled = threading.Event()
        self._enabled.set()
        self._wake = threading.Event()
        self.on_state = on_state

    def run(self) -> None:
        """线程主循环：拉帧 → 可选处理 → 可选背压丢弃 → 回调。"""
        # 按目标 FPS 计算每帧之间的最小间隔；max(1, fps) 防御 fps=0。
        delay = 1.0 / max(1, self.capture.fps)
        while not self._stop_event.is_set():
            if not self._enabled.is_set():
                if self.capture.is_opened:
                    self.capture.stop()
                    self._publish_state(False)
                self._wake.wait(.1)
                self._wake.clear()
                continue
            started = time.perf_counter()
            try:
                if not self.capture.is_opened:
                    self.capture.start()
                    self._publish_state(True)
                frame = self.capture.read_frame()
                if frame is not None:
                    display = frame
                    if self.processor is not None:
                        try:
                            # 处理函数返回 None 时仍回退到原帧，保证 UI 仍能拿到画面。
                            processed = self.processor(frame)
                            if processed is not None:
                                display = processed
                        except Exception as exc:
                            LOGGER.exception("帧处理异常: %s", exc)
                    # 背压：UI 处理不过来时 should_emit() 返回 False，直接丢帧。
                    if self._enabled.is_set() and self.on_frame and (self.should_emit is None or self.should_emit()):
                        self.on_frame(display)
            except Exception as exc:
                LOGGER.exception("视频采集异常: %s", exc)
                self.capture.stop()
                self._enabled.clear()
                self._publish_state(False, str(exc))
            # 用 Event.wait 做"剩余时间等待"，比 sleep 更易被 stop() 立即打断。
            self._stop_event.wait(max(0.0, delay - (time.perf_counter() - started)))
        self.capture.stop()

    def _publish_state(self, enabled, error=''):
        if self.on_state:
            self.on_state({'enabled': enabled, 'error': error})

    def set_enabled(self, enabled: bool) -> None:
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()
        self._wake.set()

    def stop(self) -> None:
        """请求线程退出（通过 ``_stop_event`` 通知主循环）。"""
        self._stop_event.set()
        self._wake.set()
