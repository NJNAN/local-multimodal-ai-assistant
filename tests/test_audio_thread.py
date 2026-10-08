"""语音监听开关与结果过滤的单元测试。

通过伪造麦克风采集、VAD、ASR 三个组件，验证
``src.audio_thread.AudioStreamThread`` 在不同结果（有效/纯标点/暂停态）下
对回调分发的行为。
"""
import threading

from src.audio_thread import AudioStreamThread


class FakeCapture:
    """伪造的麦克风采集器。

    按预存列表顺序返回 PCM 块；取完抛 ``EOFError`` 模拟录音结束。
    """

    def __init__(self, chunks):
        # 用 ``list`` 拷贝避免外部修改影响预定义序列。
        self._chunks = list(chunks)

    def read_chunk(self):
        if not self._chunks:
            raise EOFError
        return self._chunks.pop(0)


class FakeVAD:
    """伪造的 VAD。

    第 2 帧返回 ``True``（表示这一帧是语音段的端点），其他帧返回 ``False``；
    ``get_segment`` 与 ``flush`` 返回固定占位数据。
    """

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
    """伪造的 ASR：始终返回预设文本，并记录调用次数。"""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def transcribe(self, audio_path_or_bytes, **kwargs):
        # ``audio_path_or_bytes`` 与 ``**kwargs`` 被忽略：测试不关心实际传参。
        self.calls += 1
        return self.text


def _build(tmp_path, text):
    """构造测试夹具：返回 ``(thread, asr, results)`` 三元组。

    ``results`` 是 ``on_result`` 回调写入的列表，便于断言分发结果。
    """
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
    """守护：ASR 返回的有效文本会被原样推送到 ``on_result`` 回调。

    验收点：识别到 ``"你好，测试。"`` 时，回调列表恰好收到一次该文本；
    ASR 至少被调用 1 次。
    """
    thread, asr, results = _build(tmp_path, "你好，测试。")
    thread.run()
    assert results == ["你好，测试。"]
    assert asr.calls >= 1


def test_audio_thread_filters_punctuation_only(tmp_path):
    """守护：仅含标点的识别结果被过滤掉，不上抛给回调。

    验收点：ASR 返回 ``"。"`` 时，回调列表为空（不推送无意义结果）。
    """
    thread, _, results = _build(tmp_path, "。")
    thread.run()
    assert results == []


def test_audio_thread_pause_skips_recognition(tmp_path):
    """守护：暂停态下整个识别链路被跳过。

    验收点：
    1. ``set_enabled(False)`` 后 ``is_enabled()`` 返回 ``False``；
    2. ``run`` 过程中不调用 ASR（``asr.calls == 0``）且无结果推送；
    3. ``set_enabled(True)`` 后 ``is_enabled()`` 恢复为 ``True``。
    """
    thread, asr, results = _build(tmp_path, "你好。")
    thread.set_enabled(False)
    assert thread.is_enabled() is False
    thread.run()
    assert results == []
    assert asr.calls == 0
    thread.set_enabled(True)
    assert thread.is_enabled() is True


def test_slow_recognition_does_not_block_microphone_reads(tmp_path):
    entered, release, consumed = threading.Event(), threading.Event(), threading.Event()
    class ASR:
        def transcribe(self, *args):
            entered.set()
            assert release.wait(3)
            return '已识别'
    class Capture(FakeCapture):
        def read_chunk(self):
            result = super().read_chunk()
            if not self._chunks:
                consumed.set()
            return result
    worker = AudioStreamThread(Capture([b'\0' * 1920] * 4), FakeVAD(), ASR(), output_dir=tmp_path)
    worker.start()
    try:
        assert entered.wait(2)
        assert consumed.wait(.5), 'ASR blocked the microphone reader'
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()


def test_manual_finish_delivers_speech_without_waiting_for_silence(tmp_path):
    from src.vad import VADProcessor
    from test_vad import SequenceVad
    vad = VADProcessor(vad=SequenceVad([True] * 20))
    results = []
    worker = AudioStreamThread(FakeCapture([]), vad, FakeASR('手动结束'), on_result=results.append, output_dir=tmp_path)
    for _ in range(20):
        vad.process_frame(b'\0' * 960)
    worker.request_flush()
    worker.run()
    assert results == ['手动结束']


def test_pause_while_asr_runs_does_not_submit_stale_result(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class ASR:
        def transcribe(self, *args):
            entered.set()
            assert release.wait(3)
            return '旧识别'
    results = []
    worker = AudioStreamThread(FakeCapture([b'\0' * 1920]), FakeVAD(), ASR(), on_result=results.append, output_dir=tmp_path)
    worker.start()
    try:
        assert entered.wait(2)
        worker.set_enabled(False)
        release.set()
        worker.join(3)
        assert not worker._recognizer.is_alive()
        assert results == []
    finally:
        release.set()
        worker.stop()
        worker.join(3)


def test_full_recognition_queue_keeps_latest_segments(tmp_path):
    worker, _, _ = _build(tmp_path, '测试')
    worker._queue_segment(b'1' * 960, 'silence')
    worker._queue_segment(b'2' * 960, 'silence')
    worker._queue_segment(b'3' * 960, 'silence')
    assert worker._segments.get_nowait()[0] == b'2' * 960
    assert worker._segments.get_nowait()[0] == b'3' * 960
