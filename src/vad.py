"""流式语音活动检测（WebRTC VAD）：断句、防丢字、短句过滤。

以 30ms 帧为单位持续消费音频，当检测到「连续静音达到阈值」时输出一句完整的
PCM 段（同时补回首部前滚与尾部余音）。VADProcessor 是纯状态机、不依赖音频设备，
由 AudioStreamThread 逐帧喂入；测试可注入假 VAD 构造。
"""
from __future__ import annotations

import math
from collections import deque


class VADProcessor:
    """流式 WebRTC VAD 处理：前滚补偿、静音断句与最短语音过滤。

    断句逻辑（默认参数）：
      - 每帧 30ms；连续静音 ≥600ms（20 帧）判定一句话结束；
      - 帧数不足 500ms 的碎片直接丢弃（防咳嗽/键盘声误触发）；
      - 判定「开始说话」有 1-2 帧延迟，故保留最近 240ms 的前滚补回句首；
      - 句尾额外保留 120ms，防止尾音被切掉。
    """

    FRAME_MS = 30  # 每帧时长（WebRTC VAD 的标准帧长之一）
    PRE_ROLL_MS = 240  # 句首前滚：补回 VAD 起判延迟吃掉的音频
    TAIL_PADDING_MS = 120  # 句尾余音
    VALID_SAMPLE_RATES = {8_000, 16_000, 32_000, 48_000}

    def __init__(
        self,
        sample_rate: int = 16_000,
        silence_duration_ms: int = 600,
        min_speech_duration_ms: int = 500,
        aggressiveness: int = 2,
        vad=None,
        max_segment_duration_ms: int = 8000,
    ):
        if sample_rate not in self.VALID_SAMPLE_RATES:
            raise ValueError(f"WebRTC VAD 不支持采样率 {sample_rate}")
        if aggressiveness not in range(4):
            raise ValueError("aggressiveness 必须为 0..3")
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * self.FRAME_MS / 1000)  # 480 个采样点 @16kHz
        self.frame_bytes = self.frame_size * 2  # 16-bit PCM = 960 字节/帧
        self.silence_threshold = math.ceil(silence_duration_ms / self.FRAME_MS)  # 600ms → 20 帧
        self.min_speech_frames = math.ceil(min_speech_duration_ms / self.FRAME_MS)  # 500ms → 17 帧
        if max_segment_duration_ms < min_speech_duration_ms:
            raise ValueError('最长语音段不能短于最短语音时长')
        self.max_segment_frames = math.ceil(max_segment_duration_ms / self.FRAME_MS)
        self._segment_frames = 0
        self.end_reason = ''
        self.pre_roll_frames = self.PRE_ROLL_MS // self.FRAME_MS  # 8 帧 = 240ms
        self.tail_frames = self.TAIL_PADDING_MS // self.FRAME_MS  # 4 帧 = 120ms
        if vad is None:
            try:
                import webrtcvad
            except ImportError as exc:
                raise RuntimeError("缺少 webrtcvad-wheels") from exc
            vad = webrtcvad.Vad(aggressiveness)
        self.vad = vad
        self.audio_buffer: list[bytes] = []  # 本句已收集的帧
        self.rolling_buffer: deque[bytes] = deque(maxlen=self.pre_roll_frames)  # 最近 8 帧（前滚窗）
        self.pending_silence: list[bytes] = []  # 句尾待定静音帧（断句未确认前不算句尾）
        self.silence_count = 0
        self.speech_frame_count = 0
        self.in_speech = False
        self._boundary_ready = False  # True 时须先 get_segment() 再喂下一帧

    def process_frame(self, frame: bytes) -> bool:
        """消费恰好一帧 30ms、16-bit 单声道 PCM。

        返回 True 表示已产生一个完整语音段；必须先调用 get_segment() 取走，
        再喂下一帧（否则抛错，防止无意丢段）。
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
                # 起说：把前滚缓冲里的 240ms 补回句首，避免吃掉开头的字
                self.audio_buffer.extend(self.rolling_buffer)
                self.audio_buffer.append(frame)
                self.in_speech = True
                self.speech_frame_count = 1
                self.silence_count = 0
                self._segment_frames = 1
            else:
                self.rolling_buffer.append(frame)  # 静音期持续维护前滚窗口
            return False

        self._segment_frames += 1
        if is_speech:
            # 说话中：若刚才有暂挂的静音帧，说明只是句中停顿，全部并回
            if self.pending_silence:
                self.audio_buffer.extend(self.pending_silence)
                self.pending_silence.clear()
            self.audio_buffer.append(frame)
            self.speech_frame_count += 1
            self.silence_count = 0
            if self._segment_frames >= self.max_segment_frames:
                self.end_reason = 'max_duration'
                self._boundary_ready = True
                return True
            return False

        self.pending_silence.append(frame)
        self.silence_count += 1
        if self.silence_count >= self.silence_threshold or self._segment_frames >= self.max_segment_frames:
            # 连续静音达标：断句；静音的前 120ms 作为句尾保留
            self.audio_buffer.extend(self.pending_silence[: self.tail_frames])
            self._boundary_ready = True
            self.end_reason = 'silence' if self.silence_count >= self.silence_threshold else 'max_duration'
            return True
        return False

    def flush(self) -> bool:
        """输入流结束时收尾：把未完成的语音段按当前内容封口。"""
        if not self.in_speech or self._boundary_ready:
            return self._boundary_ready
        self.audio_buffer.extend(self.pending_silence[: self.tail_frames])
        self._boundary_ready = True
        self.end_reason = 'manual'
        return True

    def get_segment(self) -> bytes:
        """取走当前语音段并复位状态；不足最短时长的段返回空 bytes（被过滤）。"""
        segment = (
            b"".join(self.audio_buffer)
            if self.speech_frame_count >= self.min_speech_frames
            else b""
        )
        # 句尾的静音帧转为下一句的前滚候选，避免两句衔接处丢字
        trailing = self.pending_silence[-self.pre_roll_frames :]
        self.audio_buffer.clear()
        self.rolling_buffer.clear()
        self.rolling_buffer.extend(trailing)
        self.pending_silence.clear()
        self.silence_count = 0
        self.speech_frame_count = 0
        self.in_speech = False
        self._boundary_ready = False
        self._segment_frames = 0
        return segment

    def reset(self) -> None:
        """清空全部缓冲与状态（重新开始监听时使用）。"""
        self.audio_buffer.clear()
        self.rolling_buffer.clear()
        self.pending_silence.clear()
        self.silence_count = 0
        self.speech_frame_count = 0
        self.in_speech = False
        self._boundary_ready = False
        self._segment_frames = 0
        self.end_reason = ''
