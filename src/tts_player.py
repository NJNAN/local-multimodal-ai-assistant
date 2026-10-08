"""分句预合成与播放两条流水线；每轮取消事件独立，旧音频不能复活。"""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from .tts import SpeechCancelled

LOGGER = logging.getLogger(__name__)


class _PygameOutput:
    def __init__(self):
        import pygame
        self.pygame = pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=24_000)

    def play(self, path, cancelled):
        music = self.pygame.mixer.music
        try:
            music.load(str(path))
            music.play()
            while music.get_busy():
                if cancelled():
                    music.stop()
                    break
                time.sleep(.02)
        finally:
            music.unload()

    def stop(self):
        if self.pygame.mixer.get_init():
            self.pygame.mixer.music.stop()


class StreamingTTSPlayer:
    def __init__(self, tts_engine, on_status=None, playback=True, *, audio_output=None, on_activity=None):
        self.tts = tts_engine
        self.on_status = on_status
        self.playback = playback
        self.audio_output = audio_output
        self.on_activity = on_activity
        self.sentence_queue = queue.Queue()
        self._ready_queue = queue.Queue(maxsize=2)
        self._shutdown_event = threading.Event()
        self._cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._player_thread = None
        self._synthesis_thread = None

    def start(self):
        if self._player_thread and self._player_thread.is_alive():
            return
        self._shutdown_event.clear()
        self._cleanup_leftovers()
        self._synthesis_thread = threading.Thread(target=self._synthesis_loop, daemon=True, name='tts-synthesis')
        self._player_thread = threading.Thread(target=self._play_loop, daemon=True, name='tts-playback')
        self._synthesis_thread.start()
        self._player_thread.start()

    def _cleanup_leftovers(self):
        for path in self.tts.output_dir.glob('tts_*'):
            if path.suffix in {'.mp3', '.wav'}:
                self._remove(path)

    @staticmethod
    def _remove(path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def enqueue(self, text):
        if self._shutdown_event.is_set():
            return
        with self._state_lock:
            if self._cancel_event.is_set():
                self._cancel_event = threading.Event()
            event = self._cancel_event
            for sentence in self.tts.split_sentences(text):
                self.sentence_queue.put((sentence, event))

    def stop(self):
        with self._state_lock:
            self._cancel_event.set()
            self._drain(self.sentence_queue)
            self._drain(self._ready_queue, remove_files=True)
        if self.audio_output:
            try:
                self.audio_output.stop()
            except Exception:
                LOGGER.exception('停止音频播放失败')
        self._emit('已停止播报')

    def shutdown(self, timeout=3.0):
        self._shutdown_event.set()
        self.stop()
        self.sentence_queue.put(None)
        for worker in (self._synthesis_thread, self._player_thread):
            if worker:
                worker.join(timeout)
        self._drain(self._ready_queue, remove_files=True)
        if hasattr(self.tts, 'close'):
            self.tts.close()

    def _drain(self, work_queue, remove_files=False):
        while True:
            try:
                item = work_queue.get_nowait()
                if remove_files and item:
                    self._remove(item[0])
                work_queue.task_done()
            except queue.Empty:
                return

    def _synthesis_loop(self):
        if hasattr(self.tts, 'warmup'):
            try:
                self.tts.warmup()
            except Exception:
                LOGGER.exception('离线朗读预热失败')
        while not self._shutdown_event.is_set():
            try:
                item = self.sentence_queue.get(timeout=.1)
            except queue.Empty:
                continue
            path = None
            handed_off = False
            try:
                if item is None:
                    return
                sentence, event = item
                if event.is_set():
                    continue
                path = self.tts.output_dir / f'tts_{uuid.uuid4().hex}{getattr(self.tts, "audio_suffix", ".mp3")}'
                started = time.perf_counter()
                self._emit('正在准备朗读…')
                path = self.tts.synthesize_sync(sentence, path, cancel_event=event)
                LOGGER.info('TTS 合成 %.3fs, chars=%d', time.perf_counter() - started, len(sentence))
                while not event.is_set() and not self._shutdown_event.is_set():
                    try:
                        self._ready_queue.put((path, event), timeout=.05)
                        handed_off = True
                        break
                    except queue.Full:
                        pass
            except SpeechCancelled:
                pass
            except Exception:
                LOGGER.exception('TTS 合成失败')
                self._emit('暂时无法朗读，请稍后重试')
            finally:
                if path and not handed_off:
                    self._remove(path)
                self.sentence_queue.task_done()

    def _play_loop(self):
        if self.playback and self.audio_output is None:
            try:
                self.audio_output = _PygameOutput()
            except Exception:
                LOGGER.exception('音频播放器初始化失败')
                self._emit('音频输出不可用，请检查扬声器')
        while not self._shutdown_event.is_set():
            try:
                path, event = self._ready_queue.get(timeout=.1)
            except queue.Empty:
                continue
            active = False
            try:
                if event.is_set():
                    continue
                if self.playback and self.audio_output:
                    if self.on_activity:
                        self.on_activity(True)
                    active = True
                    self._emit('正在朗读…')
                    started = time.perf_counter()
                    self.audio_output.play(path, lambda: event.is_set() or self._shutdown_event.is_set())
                    LOGGER.info('TTS 播放 %.3fs', time.perf_counter() - started)
                if not event.is_set():
                    self._emit('朗读完成')
            except Exception:
                LOGGER.exception('TTS 播放失败')
                self._emit('音频播放失败，请检查扬声器')
            finally:
                if active and self.on_activity:
                    self.on_activity(False)
                self._remove(path)
                self._ready_queue.task_done()

    def _emit(self, status):
        if self.on_status:
            self.on_status(status)
