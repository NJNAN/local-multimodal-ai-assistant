"""Central configuration for the multimodal assistant.

Every value can be overridden with an environment variable so the repository
does not contain machine-specific absolute paths.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = int(os.getenv("MMAI_SEED", "42"))
EXTERNAL_MODEL_ROOT = Path(os.getenv("MMAI_EXTERNAL_MODEL_ROOT", "D:/mm_ai_models"))


def _detect_device() -> str:
    requested = os.getenv("MMAI_DEVICE", "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE = _detect_device()

PATHS = {
    "data": PROJECT_ROOT / "data",
    "models": PROJECT_ROOT / "models",
    "outputs": PROJECT_ROOT / "outputs",
    "temp": PROJECT_ROOT / "outputs" / "temp",
    "logs": PROJECT_ROOT / "outputs" / "logs",
    "knowledge": PROJECT_ROOT / "data" / "knowledge",
}
for _path in PATHS.values():
    _path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# llama.cpp (llama-server) runtime — the inference backend of this project.
#
# The text model (Qwen3.5-4B IQ4_XS) and the vision-language model
# (Qwen3-VL-4B-Instruct Q4_K_M + Q8_0 mmproj) are both served by a local
# `llama-server` process. Every path / port / context size lives here;
# business code must never hardcode absolute model paths.
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _find_llama_server() -> str | None:
    """Locate llama-server executable: env override → project tools/ → winget → PATH."""
    explicit = os.getenv("MMAI_LLAMA_SERVER")
    if explicit:
        return explicit
    candidates = [
        PROJECT_ROOT / "tools" / "llama.cpp" / "llama-server.exe",
        Path("D:/mm_ai_models/llama.cpp") / "llama-server.exe",
    ]
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            Path(local_app_data)
            / "Microsoft/WinGet/Packages"
            / "ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "llama-server.exe"
        )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    from shutil import which

    return which("llama-server")


LLAMA_CONFIG = {
    "server": _find_llama_server(),
    # 文本模型：Qwen3.5-4B IQ4_XS（bartowski GGUF，官方文件 sha256 628124c6…）
    "text": {
        "model": Path(
            os.getenv(
                "MMAI_TEXT_GGUF",
                str(PROJECT_ROOT / "models" / "qwen3.5-4b" / "Qwen_Qwen3.5-4B-IQ4_XS.gguf"),
            )
        ),
        "mmproj": None,
        "alias": os.getenv("MMAI_TEXT_ALIAS", "qwen3.5-4b-iq4_xs"),
        "port": int(os.getenv("MMAI_TEXT_PORT", "8080")),
        "n_ctx": int(os.getenv("MMAI_TEXT_CTX", "8192")),
        "n_gpu_layers": os.getenv("MMAI_TEXT_NGL", "auto"),
        "device": os.getenv("MMAI_TEXT_DEVICE", ""),
        "disable_thinking": _env_flag("MMAI_TEXT_DISABLE_THINKING", True),
        "keep_alive_seconds": int(os.getenv("MMAI_TEXT_KEEP_ALIVE", "600")),
    },
    # 视觉模型：Qwen3-VL-4B-Instruct Q4_K_M + Q8_0 mmproj（Qwen 官方 GGUF）
    "vision": {
        "model": Path(
            os.getenv(
                "MMAI_VL_GGUF",
                str(PROJECT_ROOT / "models" / "qwen3-vl-4b" / "Qwen3VL-4B-Instruct-Q4_K_M.gguf"),
            )
        ),
        "mmproj": Path(
            os.getenv(
                "MMAI_VL_MMPROJ",
                str(PROJECT_ROOT / "models" / "qwen3-vl-4b" / "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"),
            )
        ),
        "alias": os.getenv("MMAI_VL_ALIAS", "qwen3-vl-4b-instruct"),
        "port": int(os.getenv("MMAI_VL_PORT", "8081")),
        "n_ctx": int(os.getenv("MMAI_VL_CTX", "8192")),
        "n_gpu_layers": os.getenv("MMAI_VL_NGL", "auto"),
        "device": os.getenv("MMAI_VL_DEVICE", ""),
        "disable_thinking": _env_flag("MMAI_VL_DISABLE_THINKING", False),
        "keep_alive_seconds": int(os.getenv("MMAI_VL_KEEP_ALIVE", "300")),
    },
    # --fit 让 llama.cpp 根据当前设备显存自动决定 CPU/GPU offload 层数。
    "fit": _env_flag("MMAI_LLAMA_FIT", True),
    "fit_target_mb": int(os.getenv("MMAI_LLAMA_FIT_TARGET_MB", "1024")),
    # 独占模式：任何时刻最多只有一个大模型驻留显存（8 GB GPU 安全策略）。
    "exclusive": _env_flag("MMAI_LLAMA_EXCLUSIVE", True),
    "start_timeout": int(os.getenv("MMAI_LLAMA_START_TIMEOUT", "300")),
    "request_timeout": float(os.getenv("MMAI_LLAMA_TIMEOUT", "300")),
    "log_dir": PATHS["logs"],
}

MODEL_CONFIG = {
    "asr": os.getenv(
        "MMAI_ASR_MODEL",
        str(EXTERNAL_MODEL_ROOT / "SenseVoiceSmall")
        if (EXTERNAL_MODEL_ROOT / "SenseVoiceSmall").exists()
        else str(PROJECT_ROOT / "models" / "SenseVoiceSmall")
        if (PROJECT_ROOT / "models" / "SenseVoiceSmall").exists()
        else "iic/SenseVoiceSmall",
    ),
    "embedding": os.getenv(
        "MMAI_EMBEDDING_MODEL",
        str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")
        if (PROJECT_ROOT / "models" / "all-MiniLM-L6-v2").exists()
        else "sentence-transformers/all-MiniLM-L6-v2",
    ),
    "text": LLAMA_CONFIG["text"]["alias"],
    "vision": LLAMA_CONFIG["vision"]["alias"],
    "text_gguf": LLAMA_CONFIG["text"]["model"],
    "vl_gguf": LLAMA_CONFIG["vision"]["model"],
    "vl_mmproj": LLAMA_CONFIG["vision"]["mmproj"],
}

AUDIO_CONFIG = {
    "rate": 16_000,
    "chunk": 1_024,
    "channels": 1,
    "sample_width": 2,
    "input_device_index": (
        int(os.environ["MMAI_AUDIO_DEVICE"])
        if os.getenv("MMAI_AUDIO_DEVICE")
        else None
    ),
    "vad_aggressiveness": 2,
    "silence_duration_ms": 600,
    "min_speech_duration_ms": 500,
}

VIDEO_CONFIG = {
    "camera_id": int(os.getenv("MMAI_CAMERA_ID", "0")),
    "width": 640,
    "height": 480,
    "fps": 30,
    "frame_buffer_size": 60,
}

TTS_CONFIG = {
    "zh_voice": os.getenv("MMAI_ZH_VOICE", "zh-CN-XiaoxiaoNeural"),
    "en_voice": os.getenv("MMAI_EN_VOICE", "en-US-JennyNeural"),
    "enabled": os.getenv("MMAI_TTS_ENABLED", "1") not in {"0", "false", "False"},
}

LOG_CONFIG = {
    "level": os.getenv("MMAI_LOG_LEVEL", "INFO"),
    "file": PATHS["logs"] / "multimodal_assistant.log",
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
