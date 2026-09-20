"""生成视觉测试用图片（中文 OCR 图文 / 图形场景图）。

用法:

    D:\\mm_ai_env\\Scripts\\python.exe tools\\make_test_images.py

输出（outputs/）:

* test_ocr_cn.jpg   – 中文文字图片，用于 OCR 测试，文字内容固定：
                      "银发课堂 每周三 下午两点开课"
* test_shapes.jpg   – 图形细节图（蓝圆 / 红三角 / 绿方块），用于图片细节问答
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs"

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",   # 黑体
    "C:/Windows/Fonts/simsun.ttc",   # 宋体
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise FileNotFoundError("未找到中文字体（msyh.ttc / simhei.ttf / simsun.ttc）")


def make_ocr_image() -> Path:
    image = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(64)
    body_font = load_font(44)
    draw.text((60, 50), "银发课堂", fill="black", font=title_font)
    draw.text((60, 170), "每周三 下午两点开课", fill="black", font=body_font)
    draw.text((60, 260), "课程：智能手机使用", fill="black", font=body_font)
    draw.text((60, 350), "课程：健康信息辨识", fill="black", font=body_font)
    draw.text((60, 430), "电话 8888-6666", fill="black", font=body_font)
    path = OUTPUT / "test_ocr_cn.jpg"
    image.save(path, quality=95)
    return path


def make_shapes_image() -> Path:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((80, 80, 320, 320), fill="#2f6fed")      # 蓝色圆
    draw.polygon([(420, 90), (560, 320), (280, 320)], fill="#d64545")  # 红色三角
    draw.rectangle((420, 390, 660, 550), fill="#3fa34d")  # 绿色方块
    path = OUTPUT / "test_shapes.jpg"
    image.save(path, quality=95)
    return path


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ocr = make_ocr_image()
    shapes = make_shapes_image()
    print(f"已生成: {ocr}")
    print(f"已生成: {shapes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
