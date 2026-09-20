from __future__ import annotations

import pytest

from src.vad import VADProcessor


class SequenceVad:
    def __init__(self, states):
        self.states = iter(states)

    def is_speech(self, frame, sample_rate):
        return next(self.states)


def frames(count, value):
    return [bytes([value]) * 960 for _ in range(count)]


def test_vad_endpoint_pre_roll_and_tail():
    states = [False] * 8 + [True] * 17 + [False] * 20
    processor = VADProcessor(vad=SequenceVad(states))
    boundary = False
    for frame in frames(8, 1) + frames(17, 2) + frames(20, 3):
        boundary = processor.process_frame(frame)
    assert boundary
    segment = processor.get_segment()
    assert len(segment) == (8 + 17 + 4) * 960
    assert segment.startswith(bytes([1]) * 960)
    assert segment.endswith(bytes([3]) * 960)


def test_vad_drops_segment_shorter_than_half_second():
    states = [True] * 16 + [False] * 20
    processor = VADProcessor(vad=SequenceVad(states))
    for frame in frames(16, 2) + frames(20, 0):
        processor.process_frame(frame)
    assert processor.get_segment() == b""


def test_vad_rejects_wrong_frame_size():
    processor = VADProcessor(vad=SequenceVad([False]))
    with pytest.raises(ValueError, match="960"):
        processor.process_frame(b"x" * 480)
