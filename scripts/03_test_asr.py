from __future__ import annotations

import argparse
import time

from _bootstrap import PROJECT_ROOT

from config import AUDIO_CONFIG, DEVICE, MODEL_CONFIG, PATHS
from src.asr import ASREngine
from src.audio_capture import AudioCapture
from src.audio_thread import AudioStreamThread
from src.vad import VADProcessor


def main() -> int:
    parser = argparse.ArgumentParser(description="SenseVoice ASR 测试")
    parser.add_argument("files", nargs="*", help="可选 WAV 文件；不填则从麦克风识别")
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()
    engine = ASREngine(MODEL_CONFIG["asr"], DEVICE)
    if args.files:
        for index, path in enumerate(args.files, 1):
            started = time.perf_counter()
            text = engine.transcribe(path)
            print(f"[{index}] {time.perf_counter()-started:.3f}s | {text}")
        return 0

    capture = AudioCapture(
        AUDIO_CONFIG["rate"],
        AUDIO_CONFIG["chunk"],
        AUDIO_CONFIG["channels"],
        AUDIO_CONFIG["input_device_index"],
    ).start()
    vad = VADProcessor(
        AUDIO_CONFIG["rate"],
        AUDIO_CONFIG["silence_duration_ms"],
        AUDIO_CONFIG["min_speech_duration_ms"],
        AUDIO_CONFIG["vad_aggressiveness"],
    )
    thread = AudioStreamThread(
        capture, vad, engine, lambda text: print(f"识别: {text}"), print, PATHS["outputs"]
    )
    thread.start()
    print(f"请开始说话，{args.duration:.0f} 秒后结束；停顿 0.6 秒自动断句")
    try:
        thread.join(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        thread.stop()
        thread.join(3)
        capture.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
