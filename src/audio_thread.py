"""音频流水线线程：采集 → VAD 逐帧断句 → ASR 识别 → 回调/落盘。

作为独立线程持续读麦克风数据，把断出的每句话保存为 WAV 并识别；
识别记录追加写入 outputs/asr_test_results.txt，结果经回调通知界面。
有效字数不足 2 的识别文本（“嗯”“啊”等）不会向上触发回答。
"""
from __future__ import annotations

import csv
import logging
import queue
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class AudioStreamThread(threading.Thread):
    """采集→VAD→ASR 的后台流水线（daemon 线程）。

    keep_audio=True 时保留每句的 WAV（演示/验收留证据）；False 时识别后即删。
    set_enabled(False) 可暂停监听：音频段直接丢弃、不做识别。
    """

    def __init__(
        self,
        audio_capture,
        vad_processor,
        asr_engine,
        on_result=None,
        on_error=None,
        output_dir: str | Path | None = None,
        keep_audio: bool = True,
        on_status=None,
    ):
        super().__init__(daemon=True, name="audio-vad-asr")
        self.capture = audio_capture
        self.vad = vad_processor
        self.asr = asr_engine
        self.on_result = on_result
        self.on_error = on_error
        self.on_status = on_status
        self._last_status = None
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_audio = keep_audio
        self._stop_event = threading.Event()
        self._enabled = threading.Event()  # 暂停聆听开关
        self._enabled.set()
        self._pcm_buffer = bytearray()  # 跨读取块的半帧缓存
        self.records: list[dict] = []
        self._segments = queue.Queue(maxsize=2)
        self._flush_requested = threading.Event()
        self._reset_requested = threading.Event()
        self._suppressed = threading.Event()
        self._discard_until = 0.0
        self._generation = 0
        self._state_lock = threading.RLock()
        self._recognizer = None

    def _accepting_audio(self) -> bool:
        return self._enabled.is_set() and not self._suppressed.is_set() and time.monotonic() >= self._discard_until and not self._stop_event.is_set()

    def _report(self, status):
        if status != self._last_status:
            self._last_status = status
            if self.on_status:
                self.on_status(status)

    def run(self) -> None:
        """采集和断句持续运行；独立识别线程消费有限长度的语音队列。"""
        self._recognizer = threading.Thread(target=self._recognize_loop, daemon=True, name="asr-recognition")
        self._recognizer.start()
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = self.capture.read_chunk()
                except EOFError:
                    break
                self._pcm_buffer.extend(chunk)
                self._consume_frames()
                if self._flush_requested.is_set():
                    self._flush_requested.clear()
                    if self._accepting_audio() and self.vad.flush():
                        self._queue_segment(self.vad.get_segment(), 'manual')
        except Exception as exc:
            LOGGER.exception("音频流水线异常: %s", exc)
            if self.on_error:
                self.on_error(exc)
        finally:
            if self._accepting_audio() and self.vad.flush():
                self._queue_segment(self.vad.get_segment(), 'end_of_input')
            if self._stop_event.is_set():
                self._clear_segments()
            self._segments.put(None)
            self._recognizer.join(timeout=3)

    def _consume_frames(self) -> None:
        if self._reset_requested.is_set() or not self._accepting_audio():
            self._reset_requested.clear()
            self._pcm_buffer.clear()
            if hasattr(self.vad, 'reset'):
                self.vad.reset()
            return
        frame_bytes = self.vad.frame_bytes
        while len(self._pcm_buffer) >= frame_bytes:
            frame = bytes(self._pcm_buffer[:frame_bytes])
            del self._pcm_buffer[:frame_bytes]
            if self.vad.process_frame(frame):
                reason = getattr(self.vad, 'end_reason', 'silence')
                self._queue_segment(self.vad.get_segment(), reason)
            elif getattr(self.vad, 'in_speech', False):
                self._report('正在听你说话…')

    def _queue_segment(self, pcm, reason):
        if not pcm or not self._accepting_audio():
            return
        item = (pcm, self._generation, time.perf_counter(), reason)
        try:
            self._segments.put_nowait(item)
        except queue.Full:
            try:
                old = self._segments.get_nowait()
                self._segments.task_done()
                if old is None:
                    self._segments.put_nowait(None)
                    return
            except queue.Empty:
                pass
            self._segments.put_nowait(item)
            LOGGER.warning('识别队列已满，丢弃最旧的待处理语音')
        LOGGER.info('语音断句: %.2fs, reason=%s', len(pcm) / (self.vad.sample_rate * 2), reason)

    def _recognize_loop(self):
        while True:
            item = self._segments.get()
            try:
                if item is None:
                    return
                pcm, generation, queued, reason = item
                if generation == self._generation and self._accepting_audio():
                    self._process_segment(pcm, generation, queued, reason)
            finally:
                self._segments.task_done()

    def _clear_segments(self):
        while True:
            try:
                item = self._segments.get_nowait()
                self._segments.task_done()
                if item is None:
                    self._segments.put_nowait(None)
                    return
            except queue.Empty:
                return

    def request_flush(self):
        """由界面通知采集线程立即结束当前句话，不等待静音。"""
        self._flush_requested.set()

    def set_suppressed(self, speaking: bool):
        """朗读时继续读麦克风但丢弃数据，避免自己的回答再触发提问。"""
        with self._state_lock:
            if speaking:
                self._suppressed.set()
                self._generation += 1
                self._clear_segments()
                self._report('正在播报，聆听暂缓')
            else:
                self._suppressed.clear()
                self._discard_until = time.monotonic() + .25
                self._report('语音监听已开启' if self.is_enabled() else '语音监听已暂停')
            self._reset_requested.set()

    def _process_segment(self, pcm: bytes, generation=None, queued=None, reason="silence") -> None:
        """处理一个语音段：写 WAV → ASR → 记录/落盘/回调。"""
        if not pcm:
            LOGGER.debug("短语音段已丢弃")
            return
        if not self._accepting_audio():
            return  # 监听已暂停：丢弃音频，不做识别
        generation = self._generation if generation is None else generation
        self._report("正在识别…")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_path = self.output_dir / f"test_asr_{stamp}.wav"
        self._write_wav(wav_path, pcm)
        started = time.perf_counter()
        try:
            text = self.asr.transcribe(wav_path)
            elapsed = time.perf_counter() - started
            record = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "audio": str(wav_path),
                "text": text,
                "elapsed_seconds": round(elapsed, 4),
                "audio_seconds": round(len(pcm) / (self.vad.sample_rate * 2), 3),
                "queue_seconds": round(started - queued, 4) if queued else 0.0,
                "endpoint_reason": reason,
            }
            self.records.append(record)
            self._append_log(record)
            LOGGER.info("ASR %.3fs (audio=%.2fs, queue=%.3fs, endpoint=%s): %s", elapsed, record["audio_seconds"], record["queue_seconds"], reason, text or "<空>")
            # 去掉纯标点后不足 2 个字符的（“嗯”“啊”）不触发回答
            meaningful = (text or "").strip("。！？?!，,、… .;；:：~～-—")
            with self._state_lock:
                if len(meaningful) >= 2 and self.on_result and generation == self._generation and self._accepting_audio():
                    self.on_result(text)
        except Exception as exc:
            LOGGER.exception("ASR 识别失败: %s", exc)
            if self.on_error:
                self.on_error(exc)
        finally:
            if self._accepting_audio():
                self._report("语音监听已开启")
            if not self.keep_audio:
                wav_path.unlink(missing_ok=True)

    def _write_wav(self, path: Path, pcm: bytes) -> None:
        """把 PCM 段写成 16kHz / 单声道 / 16-bit 的 WAV 文件。"""
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.vad.sample_rate)
            wav.writeframes(pcm)

    def _append_log(self, record: dict) -> None:
        """把识别记录以 TSV 追加到 asr_test_results.txt（首行写表头）。"""
        path = self.output_dir / "asr_test_results.txt"
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "audio", "text", "elapsed_seconds"], delimiter="\t", extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(record)

    def stop(self) -> None:
        """请求停止（线程会在当前一次读取结束后退出）。"""
        self._stop_event.set()

    def set_enabled(self, enabled: bool) -> None:
        """暂停/恢复语音监听：暂停时直接丢弃音频段，不做识别。"""
        with self._state_lock:
            self._generation += 1
            self._reset_requested.set()
            self._clear_segments()
            if enabled:
                self._enabled.set()
            else:
                self._enabled.clear()
            self._report("语音监听已开启" if enabled else "语音监听已暂停")

    def is_enabled(self) -> bool:
        """当前是否在监听（界面按钮状态用）。"""
        return self._enabled.is_set()
