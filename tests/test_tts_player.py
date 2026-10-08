import threading
import time
from pathlib import Path
from src.tts import TTSEngine
from src.tts_player import StreamingTTSPlayer


def wait_for(condition):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(.005)
    assert condition()


def test_next_sentence_synthesizes_while_first_sentence_plays(tmp_path):
    second_synthesized, release = threading.Event(), threading.Event()
    class Engine(TTSEngine):
        def synthesize_sync(self, text, path, **kwargs):
            if text == '第二句。':
                second_synthesized.set()
            path.write_bytes(text.encode())
            return path
    class Output:
        def play(self, path, cancelled):
            if path.read_text() == '第一句。':
                assert release.wait(3)
        def stop(self):
            release.set()
    player = StreamingTTSPlayer(Engine(tmp_path), audio_output=Output())
    try:
        player.start()
        player.enqueue('第一句。第二句。')
        assert second_synthesized.wait(1), 'next sentence waits for playback to finish'
    finally:
        release.set()
        player.shutdown()


def test_cancelled_synthesis_cannot_play_after_new_answer(tmp_path):
    entered, release = threading.Event(), threading.Event()
    played = []
    class Engine(TTSEngine):
        def synthesize_sync(self, text, path, **kwargs):
            if text == '旧回答。':
                entered.set()
                assert release.wait(3)
            path.write_text(text)
            return path
    class Output:
        def play(self, path, cancelled):
            played.append(path.read_text())
        def stop(self):
            pass
    player = StreamingTTSPlayer(Engine(tmp_path), audio_output=Output())
    try:
        player.start()
        player.enqueue('旧回答。')
        assert entered.wait(2)
        player.stop()
        player.enqueue('新回答。')
        release.set()
        wait_for(lambda: bool(played))
        assert played == ['新回答。']
    finally:
        release.set()
        player.shutdown()


def test_online_synthesis_has_a_total_deadline(tmp_path, monkeypatch):
    import asyncio
    class Engine(TTSEngine):
        async def synthesize(self, *args):
            await asyncio.sleep(5)
    engine = Engine(tmp_path, timeout_seconds=.05)
    started = time.monotonic()
    import pytest
    with pytest.raises(TimeoutError):
        engine.synthesize_sync('测试', tmp_path / 'a.mp3', retries=0)
    assert time.monotonic() - started < .5
