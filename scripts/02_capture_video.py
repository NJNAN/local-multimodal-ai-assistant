from __future__ import annotations

import argparse
import time

from _bootstrap import PROJECT_ROOT

from config import PATHS, VIDEO_CONFIG
from src.image_utils import write_image
from src.video_capture import VideoCapture


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    output = PATHS["outputs"] / "test_frame.jpg"
    first = None
    with VideoCapture(**VIDEO_CONFIG) as capture:
        for index in range(args.frames):
            frame = capture.read_frame()
            if frame is None:
                continue
            if first is None:
                first = frame.copy()
            if args.show:
                cv2.imshow("Video capture test - press Q", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(1 / VIDEO_CONFIG["fps"])
    cv2.destroyAllWindows()
    if first is None:
        raise RuntimeError("未采集到有效画面")
    write_image(output, first)
    print(f"已保存首帧: {output}，尺寸 {first.shape[1]}x{first.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
