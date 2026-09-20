from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from config import AUDIO_CONFIG, DEVICE, LLAMA_CONFIG, MODEL_CONFIG, VIDEO_CONFIG
from src.llama_backend import detect_auto_device


def check(name: str, action):
    try:
        detail = action()
        print(f"[通过] {name}: {detail}")
        return True
    except Exception as exc:
        print(f"[失败] {name}: {exc}")
        return False


def _human_size(path: Path) -> str:
    size = float(path.stat().st_size)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _audio_probe() -> str:
    from src.audio_capture import AudioCapture

    capture = AudioCapture(
        AUDIO_CONFIG["rate"],
        AUDIO_CONFIG["chunk"],
        AUDIO_CONFIG["channels"],
        AUDIO_CONFIG["input_device_index"],
    )
    try:
        capture.start()
        chunk = capture.read_chunk()
        return f"读取 {len(chunk)} 字节 / {AUDIO_CONFIG['rate']} Hz"
    finally:
        capture.stop()


def _camera_probe() -> str:
    import cv2

    cap = cv2.VideoCapture(VIDEO_CONFIG["camera_id"])
    try:
        if not cap.isOpened():
            raise RuntimeError("摄像头无法打开（可能被其他程序占用）")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("摄像头读取失败")
        return f"{frame.shape[1]}x{frame.shape[0]}"
    finally:
        cap.release()


def main() -> int:
    results = []
    results.append(check("Python", lambda: platform.python_version() if sys.version_info[:2] == (3, 10) else (_ for _ in ()).throw(RuntimeError("需要 Python 3.10"))))
    modules = {
        "NumPy": "numpy",
        "PyAudio": "pyaudio",
        "WebRTC VAD": "webrtcvad",
        "OpenCV": "cv2",
        "FunASR": "funasr",
        "Sentence Transformers": "sentence_transformers",
        "FAISS": "faiss",
        "PyMuPDF": "pymupdf",
        "MediaPipe": "mediapipe",
        "PyQt5": "PyQt5",
        "Edge TTS": "edge_tts",
        "Pygame": "pygame",
    }
    for label, module in modules.items():
        results.append(check(label, lambda module=module: importlib.import_module(module).__name__))

    # --- llama.cpp 运行时与模型文件 ----------------------------------------
    server = LLAMA_CONFIG["server"]
    results.append(
        check(
            "llama-server",
            lambda: server
            if server and Path(server).is_file()
            else (_ for _ in ()).throw(
                RuntimeError("未找到 llama-server（winget install ggml.llamacpp 或设置 MMAI_LLAMA_SERVER）")
            ),
        )
    )

    def file_detail(path: Path) -> str:
        if not path.is_file():
            raise RuntimeError(f"缺失: {path}")
        return f"{path.name}（{_human_size(path)}）"

    results.append(check("文本模型 Qwen3.5-4B IQ4_XS", lambda: file_detail(Path(MODEL_CONFIG["text_gguf"]))))
    results.append(check("视觉模型 Qwen3-VL-4B Q4_K_M", lambda: file_detail(Path(MODEL_CONFIG["vl_gguf"]))))
    results.append(check("视觉 mmproj Q8_0", lambda: file_detail(Path(MODEL_CONFIG["vl_mmproj"]))))

    # --- 音视频硬件 ----------------------------------------------------------
    results.append(check("麦克风采集", _audio_probe))
    results.append(check("摄像头采集", _camera_probe))

    results.append(check("推理设备", lambda: detect_auto_device(server or "") or "自动选择（单设备或 CPU）"))
    results.append(check("项目目录", lambda: PROJECT_ROOT))
    print(f"设备选择: {DEVICE}")
    print(f"总计: {sum(results)}/{len(results)} 项通过")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
