from __future__ import annotations

import math
import time


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

    def __init__(self, cooldown_seconds: float = 2.0, clock=time.monotonic):
        self.cooldown = cooldown_seconds
        self.clock = clock
        self.last_gesture: str | None = None
        self.last_trigger_time = float("-inf")

    @staticmethod
    def _landmarks(value):
        return value.landmark if hasattr(value, "landmark") else value

    def recognize(self, hand_landmarks) -> str | None:
        if hand_landmarks is None:
            return None
        lm = self._landmarks(hand_landmarks)
        if len(lm) < 21:
            return None
        checks = (
            ("open_palm", self._is_open_palm),
            ("victory", self._is_victory),
            ("thumbs_up", self._is_thumb_up),
            ("thumbs_down", self._is_thumb_down),
            ("index_up", self._is_index_up),
            ("index_down", self._is_index_down),
        )
        gesture = next((name for name, check in checks if check(lm)), None)
        if gesture is None:
            return None
        now = self.clock()
        if gesture == self.last_gesture and now - self.last_trigger_time < self.cooldown:
            return None
        self.last_gesture = gesture
        self.last_trigger_time = now
        return gesture

    @staticmethod
    def _finger_up(lm, tip: int, pip: int, margin: float = 0.025) -> bool:
        return lm[tip].y < lm[pip].y - margin

    @staticmethod
    def _finger_down(lm, tip: int, pip: int, margin: float = 0.025) -> bool:
        return lm[tip].y > lm[pip].y + margin

    def _states(self, lm) -> dict[str, bool]:
        return {
            name: self._finger_up(lm, tip, pip)
            for name, (tip, pip) in self.TIP_PIP.items()
        }

    @staticmethod
    def _thumb_vertical(lm, direction: str) -> bool:
        tip, ip, mcp = lm[4], lm[3], lm[2]
        vertical = abs(tip.y - mcp.y) > abs(tip.x - mcp.x) * 0.7
        if direction == "up":
            return vertical and tip.y < ip.y - 0.025 and ip.y < mcp.y
        return vertical and tip.y > ip.y + 0.025 and ip.y > mcp.y

    @staticmethod
    def _thumb_extended(lm) -> bool:
        tip, mcp = lm[4], lm[2]
        wrist = lm[0]
        tip_distance = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        mcp_distance = math.hypot(mcp.x - wrist.x, mcp.y - wrist.y)
        return tip_distance > mcp_distance + 0.035

    def _is_thumb_up(self, lm) -> bool:
        states = self._states(lm)
        return self._thumb_vertical(lm, "up") and not any(states.values())

    def _is_thumb_down(self, lm) -> bool:
        states = self._states(lm)
        return self._thumb_vertical(lm, "down") and not any(states.values())

    def _is_open_palm(self, lm) -> bool:
        states = self._states(lm)
        return all(states.values()) and self._thumb_extended(lm)

    def _is_index_up(self, lm) -> bool:
        states = self._states(lm)
        return states["index"] and not any(states[name] for name in ("middle", "ring", "pinky"))

    def _is_index_down(self, lm) -> bool:
        states = self._states(lm)
        return self._finger_down(lm, 8, 6) and not any(
            states[name] for name in ("middle", "ring", "pinky")
        )

    def _is_victory(self, lm) -> bool:
        states = self._states(lm)
        return (
            states["index"]
            and states["middle"]
            and not states["ring"]
            and not states["pinky"]
        )

    def _is_within_cooldown(self) -> bool:
        return self.clock() - self.last_trigger_time < self.cooldown
