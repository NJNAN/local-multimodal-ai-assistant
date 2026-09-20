from __future__ import annotations

import csv
import logging
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class AudioStreamThread(threading.Thread):
    def __init__(
        self,
        audio_capture,
        vad_processor,
        asr_engine,
        on_result=None,
        on_error=None,
        output_dir: str | Path | None = None,
        keep_audio: bool = True,
    ):
        super().__init__(daemon=True, name="audio-vad-asr")
        self.capture = audio_capture
        self.vad = vad_processor
        self.asr = asr_engine
        self.on_result = on_result
        self.on_error = on_error
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_audio = keep_audio
        self._stop_event = threading.Event()
        self._enabled = threading.Event()
        self._enabled.set()
        self._pcm_buffer = bytearray()
        self.records: list[dict] = []

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = self.capture.read_chunk()
                except EOFError:
                    break
                self._pcm_buffer.extend(chunk)
                self._consume_frames()
        except Exception as exc:
            LOGGER.exception("音频流水线异常: %s", exc)
            if self.on_error:
                self.on_error(exc)
        finally:
            if self.vad.flush():
                self._process_segment(self.vad.get_segment())

    def _consume_frames(self) -> None:
        frame_bytes = self.vad.frame_bytes
        while len(self._pcm_buffer) >= frame_bytes:
            frame = bytes(self._pcm_buffer[:frame_bytes])
            del self._pcm_buffer[:frame_bytes]
            if self.vad.process_frame(frame):
                self._process_segment(self.vad.get_segment())

    def _process_segment(self, pcm: bytes) -> None:
        if not pcm:
            LOGGER.debug("短语音段已丢弃")
            return
        if not self._enabled.is_set():
            return  # 监听已暂停：丢弃音频，不做识别
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
            }
            self.records.append(record)
            self._append_log(record)
            LOGGER.info("ASR %.3fs: %s", elapsed, text or "<空>")
            meaningful = (text or "").strip("。！？?!，,、… .;；:：~～-—")
            if len(meaningful) >= 2 and self.on_result:
                self.on_result(text)
        except Exception as exc:
            LOGGER.exception("ASR 识别失败: %s", exc)
            if self.on_error:
                self.on_error(exc)
        finally:
            if not self.keep_audio:
                wav_path.unlink(missing_ok=True)

    def _write_wav(self, path: Path, pcm: bytes) -> None:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.vad.sample_rate)
            wav.writeframes(pcm)

    def _append_log(self, record: dict) -> None:
        path = self.output_dir / "asr_test_results.txt"
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record), delimiter="\t")
            if new_file:
                writer.writeheader()
            writer.writerow(record)

    def stop(self) -> None:
        self._stop_event.set()

    def set_enabled(self, enabled: bool) -> None:
        """暂停/恢复语音监听：暂停时直接丢弃音频段，不做识别。"""
        if enabled:
            self._enabled.set()
        else:
            self._enabled.clear()

    def is_enabled(self) -> bool:
        return self._enabled.is_set()
