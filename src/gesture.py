from __future__ import annotations

import math
import time
import threading


class GestureRecognizer:
    GESTURES = [
        "thumbs_up",
        "thumbs_down",
        "open_palm",
        "index_up",
        "index_down",
        "victory",
    ]
    TIP_PIP = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}

    FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
               "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}

    def __init__(
        self, cooldown_seconds: float = 2.0, clock=time.monotonic, *,
        stable_frames: int = 4, stable_seconds: float = 0.35,
        release_seconds: float = 0.6, max_frame_gap: float = 0.5,
    ):
        self.cooldown = cooldown_seconds
        self.clock = clock
        self.stable_frames = max(1, stable_frames)
        self.stable_seconds = max(0.0, stable_seconds)
        self.release_seconds = max(0.0, release_seconds)
        self.max_frame_gap = max_frame_gap
        self.last_gesture: str | None = None
        self.last_trigger_time = float("-inf")
        self._lock = threading.Lock()
        self._candidate = None
        self._candidate_since = 0.0
        self._candidate_count = 0
        self._last_frame_time = None
        self._release_since = None
        self._latched = None

    @staticmethod
    def _landmarks(value):
        return value.landmark if hasattr(value, "landmark") else value

    def recognize(self, hand_landmarks, *, aspect_ratio: float = 1.0) -> str | None:
        """Confirm a sustained gesture once, then wait for an explicit release.

        Cooldown applies across all actions. A brief alternate label or missing
        hand cannot retrigger a held gesture or launch a different action.
        """
        gesture = self.classify(hand_landmarks, aspect_ratio=aspect_ratio)
        now = self.clock()
        with self._lock:
            gap = self._last_frame_time is not None and now - self._last_frame_time > self.max_frame_gap
            self._last_frame_time = now
            if gesture is None:
                self._candidate = None
                self._candidate_count = 0
                if self._release_since is None or gap:
                    self._release_since = now
                if now - self._release_since >= self.release_seconds:
                    self._latched = None
                return None
            self._release_since = None
            if self._latched is not None:
                return None
            if gesture != self._candidate or gap:
                self._candidate = gesture
                self._candidate_since = now
                self._candidate_count = 1
            else:
                self._candidate_count += 1
            if (self._candidate_count < self.stable_frames
                    or now - self._candidate_since < self.stable_seconds
                    or now - self.last_trigger_time < self.cooldown):
                return None
            self.last_gesture = gesture
            self.last_trigger_time = now
            self._latched = gesture
            return gesture

    def reset(self) -> None:
        """Reset observation state on source/pause changes, preserving cooldown."""
        with self._lock:
            self._candidate = self._latched = self._release_since = None
            self._candidate_count = 0
            self._last_frame_time = None

    @property
    def confirmed_gesture(self) -> str | None:
        with self._lock:
            return self._latched

    def classify(self, hand_landmarks, *, aspect_ratio: float = 1.0) -> str | None:
        """Identify a hand without triggering or modifying the action cooldown."""
        if hand_landmarks is None:
            return None
        lm = self._landmarks(hand_landmarks)
        if len(lm) < 21:
            return None
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
            return None
        try:
            # MediaPipe normalizes x/z by image width and y by image height.
            # Put all axes in the same units before measuring angles/lengths.
            points = [(float(p.x), float(p.y) / aspect_ratio, float(getattr(p, "z", 0.0))) for p in lm[:21]]
        except (TypeError, ValueError, AttributeError):
            return None
        if not all(math.isfinite(value) for point in points for value in point):
            return None
        scale = max(self._distance(points[0], points[9]), self._distance(points[5], points[17]))
        if scale < 0.015:
            return None
        states = {name: self._finger_state(points, joints, scale) for name, joints in self.FINGERS.items()}
        extended = {name for name, state in states.items() if state == "extended"}
        folded = {name for name, state in states.items() if state == "folded"}
        thumb_straight = self._angle(points[2], points[3], points[4]) >= 140
        if len(folded) == 4 and thumb_straight:
            direction = self._vertical_direction(points[2], points[4], scale)
            thumb_reach = self._distance(points[4], points[0]) - self._distance(points[3], points[0])
            if direction and thumb_reach > 0.10 * scale:
                return "thumbs_up" if direction == "up" else "thumbs_down"
        if len(extended) == 4 and thumb_straight:
            # Require the thumb to spread away from the index base, avoiding
            # four extended fingers with a tucked thumb being called a palm.
            spread = self._distance(points[4], points[5]) - self._distance(points[3], points[5])
            if spread > 0.12 * scale:
                return "open_palm"
        if extended == {"index", "middle"} and folded == {"ring", "pinky"}:
            if self._distance(points[8], points[12]) >= 0.30 * scale:
                return "victory"
        if extended == {"index"} and folded == {"middle", "ring", "pinky"}:
            direction = self._vertical_direction(points[5], points[8], scale)
            if direction:
                return "index_up" if direction == "up" else "index_down"
        return None

    @staticmethod
    def _distance(a, b) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _angle(a, b, c) -> float:
        ab = [x - y for x, y in zip(a, b)]
        cb = [x - y for x, y in zip(c, b)]
        denominator = math.sqrt(sum(x * x for x in ab) * sum(x * x for x in cb))
        if denominator < 1e-10:
            return 0.0
        cosine = sum(x * y for x, y in zip(ab, cb)) / denominator
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _finger_state(self, points, joints, scale) -> str:
        mcp, pip, dip, tip = joints
        lengths = [self._distance(points[a], points[b]) for a, b in zip(joints, joints[1:])]
        if min(lengths) < 0.025 * scale:
            return "uncertain"
        pip_angle = self._angle(points[mcp], points[pip], points[dip])
        dip_angle = self._angle(points[pip], points[dip], points[tip])
        reach = self._distance(points[mcp], points[tip]) / sum(lengths)
        outward = self._distance(points[tip], points[0]) - self._distance(points[pip], points[0])
        if pip_angle >= 155 and dip_angle >= 150 and reach >= 0.85 and outward >= 0.08 * scale:
            return "extended"
        if pip_angle < 140 or dip_angle < 140 or reach < 0.70:
            return "folded"
        return "uncertain"

    @staticmethod
    def _vertical_direction(base, tip, scale) -> str | None:
        dx, dy = tip[0] - base[0], tip[1] - base[1]
        if abs(dy) < 0.25 * scale or abs(dy) < 1.15 * abs(dx):
            return None
        return "up" if dy < 0 else "down"

    def _is_within_cooldown(self) -> bool:
        return self.clock() - self.last_trigger_time < self.cooldown
