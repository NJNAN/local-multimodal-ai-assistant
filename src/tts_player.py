from __future__ import annotations

import logging
import queue
import threading
import time
import uuid

LOGGER = logging.getLogger(__name__)


class StreamingTTSPlayer:
    def __init__(self, tts_engine, on_status=None, playback: bool = True):
        self.tts = tts_engine
        self.on_status = on_status
        self.playback = playback
        self.sentence_queue: queue.Queue[str | None] = queue.Queue()
        self._shutdown_event = threading.Event()
        self._cancel_event = threading.Event()
        self._player_thread: threading.Thread | None = None

    def start(self) -> None:
        if self._player_thread and self._player_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._cancel_event.clear()
        self._cleanup_leftovers()
        self._player_thread = threading.Thread(
            target=self._play_loop, daemon=True, name="streaming-tts"
        )
        self._player_thread.start()

    def _cleanup_leftovers(self) -> None:
        """删除上次会话残留的临时音频（正在使用的文件会自动跳过）。"""
        try:
            for leftover in self.tts.output_dir.glob("tts_*.mp3"):
                try:
                    leftover.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    def enqueue(self, text: str) -> None:
        self._cancel_event.clear()
        for sentence in self.tts.split_sentences(text):
            self.sentence_queue.put(sentence)

    def stop(self) -> None:
        self._cancel_event.set()
        self._clear_queue()
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except (ImportError, RuntimeError):
            pass
        self._emit("已停止")

    def shutdown(self, timeout: float = 3.0) -> None:
        self._shutdown_event.set()
        self.stop()
        self.sentence_queue.put(None)
        if self._player_thread:
            self._player_thread.join(timeout=timeout)

    def _clear_queue(self) -> None:
        while True:
            try:
                self.sentence_queue.get_nowait()
                self.sentence_queue.task_done()
            except queue.Empty:
                return

    def _play_loop(self) -> None:
        mixer_ready = False
        if self.playback:
            try:
                import pygame

                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=24_000)
                mixer_ready = True
            except Exception as exc:
                LOGGER.warning("音频播放器初始化失败，将只合成不播放: %s", exc)
        while not self._shutdown_event.is_set():
            try:
                sentence = self.sentence_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if sentence is None:
                self.sentence_queue.task_done()
                break
            path = self.tts.output_dir / f"tts_{uuid.uuid4().hex}.mp3"
            try:
                if self._cancel_event.is_set():
                    continue
                self._emit("正在合成")
                self.tts.synthesize_sync(sentence, path)
                if mixer_ready and not self._cancel_event.is_set():
                    import pygame

                    self._emit("正在播放")
                    pygame.mixer.music.load(str(path))
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        if self._shutdown_event.is_set() or self._cancel_event.is_set():
                            pygame.mixer.music.stop()
                            break
                        time.sleep(0.05)
                self._emit("播放完成")
            except Exception as exc:
                LOGGER.exception("TTS 处理失败: %s", exc)
                self._emit(f"TTS 错误: {exc}")
            finally:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    # Windows 上 pygame 可能仍持有文件句柄，留待下次启动清理
                    pass
                self.sentence_queue.task_done()

    def _emit(self, status: str) -> None:
        if self.on_status:
            self.on_status(status)
