from __future__ import annotations

import math
from collections import deque


class VADProcessor:
    """Streaming WebRTC VAD with pre-roll, silence endpointing and filtering."""

    FRAME_MS = 30
    PRE_ROLL_MS = 240
    TAIL_PADDING_MS = 120
    VALID_SAMPLE_RATES = {8_000, 16_000, 32_000, 48_000}

    def __init__(
        self,
        sample_rate: int = 16_000,
        silence_duration_ms: int = 600,
        min_speech_duration_ms: int = 500,
        aggressiveness: int = 2,
        vad=None,
    ):
        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(f"WebRTC VAD 不支持采样率 {sample_rate}")
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness 必须为 0..3")
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * self.FRAME_MS / 1000)
        self.frame_bytes = self.frame_size * 2
        self.silence_threshold = math.ceil(silence_duration_ms / self.FRAME_MS)
        self.min_speech_frames = math.ceil(min_speech_duration_ms / self.FRAME_MS)
        self.pre_roll_frames = self.PRE_ROLL_MS // self.FRAME_MS
        self.tail_frames = self.TAIL_PADDING_MS // self.FRAME_MS
        if vad is None:
            try:
                import webrtcvad
            except ImportError as exc:
                raise RuntimeError("缺少 webrtcvad-wheels") from exc
            vad = webrtcvad.Vad(aggressiveness)
        self.vad = vad
        self.audio_buffer: list[bytes] = []
        self.rolling_buffer: deque[bytes] = deque(maxlen=self.pre_roll_frames)
        self.pending_silence: list[bytes] = []
        self.silence_count = 0
        self.speech_frame_count = 0
        self.in_speech = False
        self._boundary_ready = False

    def process_frame(self, frame: bytes) -> bool:
        """Consume exactly one 30 ms, 16-bit mono PCM frame.

        Returns True when a segment boundary is available. Call get_segment()
        before feeding the next frame.
        """
        if len(frame) != self.frame_bytes:
            raise ValueError(
                f"VAD 帧长度应为 {self.frame_bytes} 字节，实际为 {len(frame)}"
            )
        if self._boundary_ready:
            raise RuntimeError("检测到断句后必须先调用 get_segment()")

        is_speech = bool(self.vad.is_speech(frame, self.sample_rate))
        if not self.in_speech:
            if is_speech:
                self.audio_buffer.extend(self.rolling_buffer)
                self.audio_buffer.append(frame)
                self.in_speech = True
                self.speech_frame_count = 1
                self.silence_count = 0
            else:
                self.rolling_buffer.append(frame)
            return False

        if is_speech:
            if self.pending_silence:
                self.audio_buffer.extend(self.pending_silence)
                self.pending_silence.clear()
            self.audio_buffer.append(frame)
            self.speech_frame_count += 1
            self.silence_count = 0
            return False

        self.pending_silence.append(frame)
        self.silence_count += 1
        if self.silence_count >= self.silence_threshold:
            self.audio_buffer.extend(self.pending_silence[: self.tail_frames])
            self._boundary_ready = True
            return True
        return False

    def flush(self) -> bool:
        """Finalize an unfinished segment when an input stream ends."""
        if not self.in_speech or self._boundary_ready:
            return self._boundary_ready
        self.audio_buffer.extend(self.pending_silence[: self.tail_frames])
        self._boundary_ready = True
        return True

    def get_segment(self) -> bytes:
        segment = (
            b"".join(self.audio_buffer)
            if self.speech_frame_count >= self.min_speech_frames
            else b""
        )
        trailing = self.pending_silence[-self.pre_roll_frames :]
        self.audio_buffer.clear()
        self.rolling_buffer.clear()
        self.rolling_buffer.extend(trailing)
        self.pending_silence.clear()
        self.silence_count = 0
        self.speech_frame_count = 0
        self.in_speech = False
        self._boundary_ready = False
        return segment

    def reset(self) -> None:
        self.audio_buffer.clear()
        self.rolling_buffer.clear()
        self.pending_silence.clear()
        self.silence_count = 0
        self.speech_frame_count = 0
        self.in_speech = False
        self._boundary_ready = False
