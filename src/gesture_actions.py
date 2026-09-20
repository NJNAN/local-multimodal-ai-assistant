from __future__ import annotations

import ctypes
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class GestureActionMapper:
    def __init__(self):
        self.action_map = {
            "thumbs_up": "volume_up",
            "thumbs_down": "volume_down",
            "open_palm": "stop",
            "index_up": "start_listening",
            "index_down": "mute",
            "victory": "take_snapshot",
        }

    def map(self, gesture: str) -> str:
        return self.action_map.get(gesture, "unknown")

    def execute(self, action: str, **kwargs):
        callbacks = kwargs.get("callbacks", {})
        if action in callbacks:
            return callbacks[action]()
        if action == "volume_up":
            return self._press_media_key(0xAF)
        if action == "volume_down":
            return self._press_media_key(0xAE)
        if action == "mute":
            return self._press_media_key(0xAD)
        if action == "stop":
            player = kwargs.get("tts_player")
            if player:
                player.stop()
                return True
        if action == "take_snapshot":
            frame = kwargs.get("frame")
            output_path = kwargs.get("output_path")
            if frame is not None and output_path:
                from .image_utils import write_image

                path = Path(output_path)
                write_image(path, frame)
                return True
        LOGGER.info("动作 %s 没有可执行目标", action)
        return False

    @staticmethod
    def _press_media_key(code: int) -> bool:
        try:
            user32 = ctypes.windll.user32
            user32.keybd_event(code, 0, 0, 0)
            user32.keybd_event(code, 0, 2, 0)
            return True
        except (AttributeError, OSError):
            return False
