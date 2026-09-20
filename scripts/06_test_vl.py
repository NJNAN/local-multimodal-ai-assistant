from __future__ import annotations

import argparse
import time

from _bootstrap import PROJECT_ROOT

from config import MODEL_CONFIG, PATHS, VIDEO_CONFIG
from src.image_utils import read_image, write_image
from src.video_capture import VideoCapture
from src.vl_model import VLModel


def main() -> int:
    import cv2  # noqa: F401  (确保 opencv 可用)

    parser = argparse.ArgumentParser(description="两帧视觉理解测试")
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--query", default="画面中有什么？有几个人？")
    args = parser.parse_args()
    if args.images:
        frames = [read_image(path) for path in args.images[-2:]]
    else:
        with VideoCapture(**VIDEO_CONFIG) as capture:
            frames = []
            for _ in range(2):
                frame = capture.read_frame()
                if frame is not None:
                    frames.append(frame)
                time.sleep(0.5)
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError("未获得有效测试图像")
    for index, frame in enumerate(frames, 1):
        write_image(PATHS["outputs"] / f"vl_test_{index}.jpg", frame)
    model = VLModel(MODEL_CONFIG["vision"])
    started = time.perf_counter()
    answer = model.analyze_frames(frames, args.query)
    elapsed = time.perf_counter() - started
    print(f"模型: {model.model_name}（llama.cpp / llama-server 后端）")
    print(f"回答: {answer}\n耗时: {elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
