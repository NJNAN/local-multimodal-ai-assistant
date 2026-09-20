from __future__ import annotations

import argparse
import time
import wave

from _bootstrap import PROJECT_ROOT

from config import AUDIO_CONFIG, PATHS
from src.audio_capture import AudioCapture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    output = PATHS["outputs"] / "test_audio.wav"
    chunks = []
    with AudioCapture(
        AUDIO_CONFIG["rate"],
        AUDIO_CONFIG["chunk"],
        AUDIO_CONFIG["channels"],
        AUDIO_CONFIG["input_device_index"],
    ) as capture:
        print(f"开始录音 {args.seconds:.1f} 秒")
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            chunks.append(capture.read_chunk())
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(AUDIO_CONFIG["channels"])
        wav.setsampwidth(AUDIO_CONFIG["sample_width"])
        wav.setframerate(AUDIO_CONFIG["rate"])
        wav.writeframes(b"".join(chunks))
    print(f"已保存: {output}，{output.stat().st_size} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
