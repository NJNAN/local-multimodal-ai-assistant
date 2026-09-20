from __future__ import annotations

from pathlib import Path


def write_image(path: str | Path, image, quality: int = 92) -> Path:
    """Write an OpenCV image to Unicode paths reliably on Windows."""
    import cv2

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"图像编码失败: {suffix}")
    output.write_bytes(encoded.tobytes())
    return output


def read_image(path: str | Path):
    """Read an OpenCV image from Unicode paths reliably on Windows."""
    import cv2
    import numpy as np

    data = np.fromfile(Path(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image
