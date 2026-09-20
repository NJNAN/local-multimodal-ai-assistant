"""连续两轮视觉对话检查（Qwen3-VL 多轮历史 + 图片输入）。

第 1 轮：图形场景图（test_shapes.jpg）——识别形状与颜色。
第 2 轮：携带第 1 轮历史，读中文图（test_ocr_cn.jpg）里的文字。

用法:

    D:\\mm_ai_env\\Scripts\\python.exe scripts\\12_vision_multiturn_check.py
"""

from __future__ import annotations

import time

from _bootstrap import PROJECT_ROOT

from config import MODEL_CONFIG, PATHS
from src.image_utils import read_image
from src.vl_model import VLModel


def main() -> int:
    vl = VLModel(MODEL_CONFIG["vision"])
    frame_shapes = read_image(PATHS["outputs"] / "test_shapes.jpg")
    frame_ocr = read_image(PATHS["outputs"] / "test_ocr_cn.jpg")

    history: list[dict] = []
    q1 = "图中有哪些几何图形？分别是什么颜色？"
    started = time.perf_counter()
    a1 = vl.analyze_frames([frame_shapes], q1, max_new_tokens=160)
    t1 = time.perf_counter() - started
    print(f"[第1轮 {t1:.2f}s]\nQ: {q1}\nA: {a1}\n")

    history.extend(
        [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
        ]
    )
    q2 = "请读出第二张图中的所有文字。"
    started = time.perf_counter()
    a2 = vl.analyze_frames([frame_ocr], q2, max_new_tokens=160, history=history)
    t2 = time.perf_counter() - started
    print(f"[第2轮 {t2:.2f}s]\nQ: {q2}\nA: {a2}\n")

    ok = any(key in a2 for key in ("银发课堂", "每周三", "下午两点", "开课", "智能手机", "健康信息"))
    print(f"两轮对话完成；第二轮命中预期文字: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
