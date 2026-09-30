from __future__ import annotations

import argparse
import time

from _bootstrap import PROJECT_ROOT

from config import PATHS, VIDEO_CONFIG
from src.gesture import GestureRecognizer
from src.human_detector import HumanDetector
from src.image_utils import write_image
from src.video_capture import VideoCapture


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description="MediaPipe 姿态与六手势测试")
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    detector = HumanDetector()
    recognizer = GestureRecognizer()
    counts = {name: 0 for name in recognizer.GESTURES}
    deadline = time.monotonic() + args.seconds
    snapshot = 0
    try:
        with VideoCapture(**VIDEO_CONFIG) as capture:
            while time.monotonic() < deadline:
                frame = capture.read_frame()
                if frame is None:
                    continue
                results = detector.process_frame(frame)
                annotated = detector.draw_landmarks(frame, results)
                hands = results["hands"].multi_hand_landmarks
                gesture = recognizer.recognize(hands[0] if hands else None,
                                               aspect_ratio=frame.shape[1] / frame.shape[0])
                if gesture:
                    counts[gesture] += 1
                    snapshot += 1
                    write_image(PATHS["outputs"] / f"gesture_test_{snapshot}_{gesture}.jpg", annotated)
                    print(f"识别手势: {gesture}")
                cv2.imshow("Detection test - Q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        detector.close()
        cv2.destroyAllWindows()
    print("触发计数:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
