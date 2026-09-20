from __future__ import annotations

from _bootstrap import PROJECT_ROOT

from config import PATHS, TTS_CONFIG
from src.tts import TTSEngine


def main() -> int:
    engine = TTSEngine(PATHS["outputs"], TTS_CONFIG["zh_voice"], TTS_CONFIG["en_voice"])
    samples = {
        "zh": "你好，我是多模态AI助手，很高兴为你服务。",
        "en": "Hello, I am a multimodal AI assistant.",
    }
    for language, text in samples.items():
        output = PATHS["outputs"] / f"tts_test_{language}.mp3"
        engine.synthesize_sync(text, output)
        print(f"{language}: {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
