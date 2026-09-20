from __future__ import annotations

import logging
from collections import deque

LOGGER = logging.getLogger(__name__)


class HumanDetector:
    MAX_HISTORY = 10

    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mediapipe_module=None,
    ):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("缺少 opencv-python 或 mediapipe") from exc
        self.cv2 = cv2
        self.mp = mediapipe_module or mp
        solutions = self.mp.solutions
        self.pose = solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.face = solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_detection_confidence
        )
        self.history = deque(maxlen=self.MAX_HISTORY)

    def process_frame(self, bgr_image) -> dict:
        if bgr_image is None:
            raise ValueError("输入图像不能为空")
        rgb = self.cv2.cvtColor(bgr_image, self.cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        pose_result = self.pose.process(rgb)
        hands_result = self.hands.process(rgb)
        face_result = self.face.process(rgb)
        pose_landmarks = getattr(pose_result, "pose_landmarks", None)
        if pose_landmarks:
            self._remember_pose(pose_landmarks)
        return {"pose": pose_result, "hands": hands_result, "face": face_result}

    def draw_landmarks(self, image, results: dict):
        output = image.copy()
        drawing = self.mp.solutions.drawing_utils
        styles = self.mp.solutions.drawing_styles
        pose = results.get("pose")
        hands = results.get("hands")
        face = results.get("face")
        if pose and pose.pose_landmarks:
            drawing.draw_landmarks(
                output,
                pose.pose_landmarks,
                self.mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
            )
        if hands and hands.multi_hand_landmarks:
            for hand in hands.multi_hand_landmarks:
                drawing.draw_landmarks(
                    output,
                    hand,
                    self.mp.solutions.hands.HAND_CONNECTIONS,
                    styles.get_default_hand_landmarks_style(),
                    styles.get_default_hand_connections_style(),
                )
        if face and face.detections:
            for detection in face.detections:
                drawing.draw_detection(output, detection)
        return output

    @staticmethod
    def _landmarks(value):
        return value.landmark if hasattr(value, "landmark") else value

    def is_full_body_visible(self, pose_landmarks) -> bool:
        if pose_landmarks is None:
            return False
        landmarks = self._landmarks(pose_landmarks)
        required = (11, 12, 23, 24, 25, 26, 27, 28)
        return len(landmarks) > max(required) and all(
            getattr(landmarks[index], "visibility", 0.0) >= 0.5 for index in required
        )

    def _remember_pose(self, pose_landmarks) -> None:
        landmarks = self._landmarks(pose_landmarks)
        self.history.append([(point.x, point.y, point.z) for point in landmarks])

    def is_pose_stable(self, pose_landmarks=None, threshold: float = 0.1) -> bool:
        if pose_landmarks is not None:
            self._remember_pose(pose_landmarks)
        if len(self.history) < self.MAX_HISTORY:
            return False
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("缺少 numpy") from exc
        positions = np.asarray(self.history, dtype="float32")
        center = positions.mean(axis=0, keepdims=True)
        mean_displacement = np.linalg.norm(positions - center, axis=2).mean()
        return bool(mean_displacement < threshold)

    def close(self) -> None:
        for model in (self.pose, self.hands, self.face):
            close = getattr(model, "close", None)
            if close:
                close()
