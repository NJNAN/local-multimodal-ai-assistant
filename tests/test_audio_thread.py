"""语音监听开关与结果过滤的单元测试。"""
import threading

from src.audio_thread import AudioStreamThread


class FakeCapture:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read_chunk(self):
        if not self._chunks:
            raise EOFError
        return self._chunks.pop(0)


class FakeVAD:
    frame_bytes = 960
    sample_rate = 16_000

    def __init__(self):
        self._frames = 0

    def process_frame(self, frame):
        self._frames += 1
        return self._frames == 2  # 仅在第 2 帧截获一次语音段

    def get_segment(self):
        return b"\x00\x00" * 960

    def flush(self):
        return False


class FakeASR:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, audio_path_or_bytes, **kwargs):
        self.calls += 1
        return self.text


def _build(tmp_path, text):
    results = []
    asr = FakeASR(text)
    thread = AudioStreamThread(
        FakeCapture([b"\x00" * 3840]),
        FakeVAD(),
        asr,
        on_result=results.append,
        output_dir=tmp_path,
    )
    return thread, asr, results


def test_audio_thread_sends_meaningful_result(tmp_path):
    thread, asr, results = _build(tmp_path, "你好，测试。")
    thread.run()
    assert results == ["你好，测试。"]
    assert asr.calls >= 1


def test_audio_thread_filters_punctuation_only(tmp_path):
    thread, _, results = _build(tmp_path, "。")
    thread.run()
    assert results == []


def test_audio_thread_pause_skips_recognition(tmp_path):
    thread, asr, results = _build(tmp_path, "你好。")
    thread.set_enabled(False)
    assert thread.is_enabled() is False
    thread.run()
    assert results == []
    assert asr.calls == 0
    thread.set_enabled(True)
    assert thread.is_enabled() is True
