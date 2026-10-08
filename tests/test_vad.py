"""VAD 端点检测的单元测试。

通过伪造一个按预设序列返回 ``is_speech`` 结果的 VAD 后端，验证
``src.vad.VADProcessor`` 在不同语音段长度、尾帧裁剪、帧长校验等场景下的
行为是否符合产品需求。
"""
from __future__ import annotations

import pytest

from src.vad import VADProcessor


class SequenceVad:
    """按预设布尔序列返回 ``is_speech`` 结果的假 VAD。

    用于让 ``VADProcessor`` 在不依赖真实语音检测模型的情况下复现特定输入。
    """

    def __init__(self, states):
        # ``iter`` 让 ``next`` 一次消耗一个；用完即抛 ``StopIteration``。
        self.states = iter(states)

    def is_speech(self, frame, sample_rate):
        # ``sample_rate`` 参数被忽略：测试只关心返回的布尔序列。
        return next(self.states)


def frames(count, value):
    """生成 ``count`` 个等长（960 字节）假帧，每帧内容为 ``bytes([value]) * 960``。"""
    return [bytes([value]) * 960 for _ in range(count)]


def test_vad_endpoint_pre_roll_and_tail():
    """守护：VAD 的「前滚 + 尾帧裁剪」端点检测行为。

    验收点：
    1. 8 帧静音 + 17 帧语音 + 20 帧静音 中，第 4 帧语音（pre_roll=4）处触发
       ``boundary=True``；
    2. ``get_segment`` 返回的语音段包含 8 帧静音 + 17 帧语音 + 4 帧收尾（tail=4）
       共 29 帧，最后 4 帧（20-23）使用收尾阶段采集到的静音帧。
    """
    states = [False] * 8 + [True] * 17 + [False] * 20
    processor = VADProcessor(vad=SequenceVad(states))
    boundary = False
    # 把三段拼接起来连续送入处理器；最后一帧返回时 ``boundary`` 才会变 True。
    for frame in frames(8, 1) + frames(17, 2) + frames(20, 3):
        boundary = processor.process_frame(frame)
    assert boundary
    segment = processor.get_segment()
    # 总帧数 = 前 8 静音 + 17 语音 + 4 收尾（tail 截断），每帧 960 字节。
    assert len(segment) == (8 + 17 + 4) * 960
    assert segment.startswith(bytes([1]) * 960)
    assert segment.endswith(bytes([3]) * 960)


def test_vad_drops_segment_shorter_than_half_second():
    """守护：长度低于 0.5 秒的语音段被丢弃。

    验收点：16 帧语音（≈0.5s，960 字节/帧 = 30ms）+ 后续静音；
    短于阈值的语音段应被丢弃，``get_segment`` 返回空字节串。
    """
    states = [True] * 16 + [False] * 20
    processor = VADProcessor(vad=SequenceVad(states))
    for frame in frames(16, 2) + frames(20, 0):
        processor.process_frame(frame)
    assert processor.get_segment() == b""


def test_vad_rejects_wrong_frame_size():
    """守护：``process_frame`` 对帧长做严格校验。

    验收点：传入 480 字节（不足 960）的帧时，处理器抛出 ``ValueError``，
    且异常消息中含 ``"960"`` 提示期望帧长。
    """
    processor = VADProcessor(vad=SequenceVad([False]))
    with pytest.raises(ValueError, match="960"):
        processor.process_frame(b"x" * 480)


def test_background_marked_as_speech_cannot_delay_endpoint_indefinitely():
    processor = VADProcessor(vad=SequenceVad([True] * 100), max_segment_duration_ms=900)
    for i, frame in enumerate(frames(100, 2), 1):
        if processor.process_frame(frame):
            break
    assert i == 30
    assert len(processor.get_segment()) == 30 * 960


def test_segment_limit_keeps_every_frame_of_continuous_speech():
    processor = VADProcessor(vad=SequenceVad([True] * 60), max_segment_duration_ms=900)
    segments = []
    for frame in frames(60, 2):
        if processor.process_frame(frame):
            segments.append(processor.get_segment())
    assert [len(segment) for segment in segments] == [30 * 960, 30 * 960]
