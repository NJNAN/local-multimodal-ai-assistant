from __future__ import annotations

from datetime import datetime


class ModalRouter:
    VISION_KEYWORDS = [
        "画面",
        "摄像头",
        "镜头",
        "屏幕",
        "图像",
        "图片",
        "照片",
        "看看",
        "看一下",
        "看下",
        "看一眼",
        "看这",
        "看那",
        "看到",
        "看见",
        "瞅瞅",
        "拍照",
        "拍一下",
        "拍到",
        "拍下",
        "拍张",
        "拍个",
        "look",
        "see",
        "camera",
        "view",
        "screenshot",
        "photo",
    ]

    def __init__(self):
        self.current_mode = "text"
        self.auto_mode = True
        self.history: list[dict] = []

    def route(self, user_input: str) -> str:
        text = user_input.strip().lower()
        if text.startswith("/vision"):
            self.set_mode("vision", auto=False)
        elif text.startswith("/text"):
            self.set_mode("text", auto=False)
        elif self.auto_mode:
            mode = "vision" if any(key in text for key in self.VISION_KEYWORDS) else "text"
            self.set_mode(mode, auto=True)
        return self.current_mode

    def set_mode(self, mode: str, auto: bool | None = None) -> None:
        if mode not in {"text", "vision"}:
            raise ValueError("mode 必须是 text 或 vision")
        changed = mode != self.current_mode
        self.current_mode = mode
        if auto is not None:
            self.auto_mode = auto
        if changed:
            self.history.append(
                {"timestamp": datetime.now().isoformat(timespec="seconds"), "mode": mode}
            )

    def enable_auto(self) -> None:
        self.auto_mode = True
