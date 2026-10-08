import os
import wave
import pytest


@pytest.mark.skipif(os.name != 'nt', reason='Windows native speech')
def test_installed_chinese_voice_produces_playable_wav(tmp_path):
    from src.tts import TTSEngine
    from src.windows_tts import WindowsVoicesUnavailable
    engine = TTSEngine(tmp_path, backend='windows')
    try:
        try:
            path = engine.synthesize_sync('你好，语音测试。', tmp_path / 'native.wav')
        except WindowsVoicesUnavailable:
            pytest.skip('No installed Chinese Windows voice')
        with wave.open(str(path)) as audio:
            assert audio.getnframes() > 1000
            assert audio.getframerate() >= 16000
    finally:
        engine.close()
