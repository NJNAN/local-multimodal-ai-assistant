from __future__ import annotations

import logging
import threading

LOGGER = logging.getLogger(__name__)


class AudioCapture:
    """16 kHz mono PCM microphone capture with deterministic lifecycle."""

    def __init__(
        self,
        rate: int = 16_000,
        chunk: int = 1_024,
        channels: int = 1,
        input_device_index: int | None = None,
        backend=None,
    ):
        self.rate = rate
        self.chunk = chunk
        self.channels = channels
        self.input_device_index = input_device_index
        self.pa = backend
        self.stream = None
        self._owns_backend = backend is None
        self._lock = threading.Lock()

    @property
    def is_active(self) -> bool:
        return bool(self.stream and self.stream.is_active())

    def start(self) -> "AudioCapture":
        with self._lock:
            if self.is_active:
                return self
            try:
                import pyaudio
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 PyAudio。请在项目虚拟环境运行 pip install PyAudio。"
                ) from exc
            if self.pa is None:
                self.pa = pyaudio.PyAudio()
            try:
                self.stream = self.pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    frames_per_buffer=self.chunk,
                    input_device_index=self.input_device_index,
                )
            except Exception:
                if self._owns_backend and self.pa is not None:
                    self.pa.terminate()
                    self.pa = None
                raise
            LOGGER.info("麦克风已启动: %s Hz, %s 声道", self.rate, self.channels)
            return self

    def read_chunk(self) -> bytes:
        if not self.is_active:
            raise RuntimeError("音频流尚未启动")
        return self.stream.read(self.chunk, exception_on_overflow=False)

    def stop(self) -> None:
        with self._lock:
            if self.stream is not None:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                finally:
                    self.stream.close()
                    self.stream = None
            if self._owns_backend and self.pa is not None:
                self.pa.terminate()
                self.pa = None
            LOGGER.info("麦克风已停止")

    def __enter__(self) -> "AudioCapture":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()
