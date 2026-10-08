"""多模态助手的全局配置中心（唯一配置来源）。

约定：所有路径、端口、超参都集中在本文件，且每一项都可以用 MMAI_* 环境变量覆盖——
因此仓库里不含「某台机器专有」的绝对路径，换机器 / 换模型只需改环境变量。

分区：
  - 设备与种子：DEVICE / SEED / set_seed()
  - PATHS：项目内目录（data / models / outputs / logs / knowledge）
  - LLAMA_CONFIG：llama-server 运行时 + 文本、视觉两个模型的全部参数
  - MODEL_CONFIG：ASR / 嵌入模型 + 两个 LLM 的别名与 GGUF 文件路径
  - AUDIO_CONFIG / VIDEO_CONFIG / TTS_CONFIG / LOG_CONFIG：采集与输出参数
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from src.inference_options import InferenceOptions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = int(os.getenv("MMAI_SEED", "42"))
EXTERNAL_MODEL_ROOT = Path(os.getenv("MMAI_EXTERNAL_MODEL_ROOT", "D:/mm_ai_models"))


def _detect_device() -> str:
    """决定推理设备：MMAI_DEVICE 可强制 cuda/cpu；默认 auto 探测 torch。"""
    requested = os.getenv("MMAI_DEVICE", "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# 全局推理设备（ASR/嵌入等走 PyTorch 的模块使用；llama.cpp 的 GPU 选择见 LLAMA_CONFIG）
DEVICE = _detect_device()

PATHS = {
    "data": PROJECT_ROOT / "data",
    "models": PROJECT_ROOT / "models",
    "outputs": PROJECT_ROOT / "outputs",
    "temp": PROJECT_ROOT / "outputs" / "temp",
    "logs": PROJECT_ROOT / "outputs" / "logs",
    "knowledge": PROJECT_ROOT / "data" / "knowledge",
}
# 首次运行时自动创建这些目录，业务代码无需再关心「目录不存在」
for _path in PATHS.values():
    _path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# llama.cpp（llama-server）运行时 —— 本项目的推理后端。
#
# 文本模型（Qwen3.5-4B IQ4_XS）与视觉语言模型（Qwen3-VL-4B Q4_K_M + Q8_0 mmproj）
# 都由本机 llama-server 进程提供服务。所有路径 / 端口 / 上下文大小的默认值都在
# 这里定义；业务代码不得硬编码模型绝对路径，一律从本配置读取。
# ---------------------------------------------------------------------------


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔型环境变量：0/false/no/off/空字符串 视为 False，其它视为 True。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _find_llama_server() -> str | None:
    """定位 llama-server 可执行文件：环境变量 → 项目 tools/ → winget 安装 → PATH。"""
    explicit = os.getenv("MMAI_LLAMA_SERVER")
    if explicit:
        return explicit
    candidates = [
        PROJECT_ROOT / "tools" / "llama.cpp" / "llama-server.exe",
        Path("D:/mm_ai_models/llama.cpp") / "llama-server.exe",
    ]
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        # winget 安装 ggml.llamacpp 后的默认包目录
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
        "mmproj": None,  # 文本模型没有视觉投影层
        "alias": os.getenv("MMAI_TEXT_ALIAS", "qwen3.5-4b-iq4_xs"),  # API 里的模型名
        "port": int(os.getenv("MMAI_TEXT_PORT", "8080")),
        "n_ctx": int(os.getenv("MMAI_TEXT_CTX", "8192")),  # 上下文长度
        "n_gpu_layers": os.getenv("MMAI_TEXT_NGL", "auto"),  # auto = 交给 --fit 决定
        "device": os.getenv("MMAI_TEXT_DEVICE", ""),  # 多设备时可指定（如 Vulkan1）
        "disable_thinking": _env_flag("MMAI_TEXT_DISABLE_THINKING", True),
        "keep_alive_seconds": int(os.getenv("MMAI_TEXT_KEEP_ALIVE", "600")),  # 空闲自动卸载
        "inference": InferenceOptions.from_environment("MMAI_TEXT"),  # 进阶参数（需显式开启）
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
        "inference": InferenceOptions.from_environment("MMAI_VL"),
    },
    # --fit 让 llama.cpp 根据当前设备显存自动决定 CPU/GPU offload 层数。
    "fit": _env_flag("MMAI_LLAMA_FIT", True),
    "fit_target_mb": int(os.getenv("MMAI_LLAMA_FIT_TARGET_MB", "1024")),
    # 独占模式：任何时刻最多只有一个大模型驻留显存（8 GB GPU 安全策略）。
    "exclusive": _env_flag("MMAI_LLAMA_EXCLUSIVE", True),
    "start_timeout": int(os.getenv("MMAI_LLAMA_START_TIMEOUT", "300")),  # 模型加载最长等待
    "request_timeout": float(os.getenv("MMAI_LLAMA_TIMEOUT", "300")),
    "log_dir": PATHS["logs"],  # llama-server 进程日志目录
}

MODEL_CONFIG = {
    # ASR 模型路径优先级：环境变量 > D:/mm_ai_models > 项目 models/ > ModelScope 名称
    "asr": os.getenv(
        "MMAI_ASR_MODEL",
        str(EXTERNAL_MODEL_ROOT / "SenseVoiceSmall")
        if (EXTERNAL_MODEL_ROOT / "SenseVoiceSmall").exists()
        else str(PROJECT_ROOT / "models" / "SenseVoiceSmall")
        if (PROJECT_ROOT / "models" / "SenseVoiceSmall").exists()
        else "iic/SenseVoiceSmall",
    ),
    # 嵌入模型同理（本地目录优先，缺省时退回 Sentence Transformers 官方名）
    "embedding": os.getenv(
        "MMAI_EMBEDDING_MODEL",
        str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")
        if (PROJECT_ROOT / "models" / "all-MiniLM-L6-v2").exists()
        else "sentence-transformers/all-MiniLM-L6-v2",
    ),
    # 两个 LLM 的别名与 GGUF 文件路径（从 LLAMA_CONFIG 派生，保持单一来源）
    "text": LLAMA_CONFIG["text"]["alias"],
    "vision": LLAMA_CONFIG["vision"]["alias"],
    "text_gguf": LLAMA_CONFIG["text"]["model"],
    "vl_gguf": LLAMA_CONFIG["vision"]["model"],
    "vl_mmproj": LLAMA_CONFIG["vision"]["mmproj"],
}

AUDIO_CONFIG = {
    "rate": 16_000,  # 采样率：VAD 与 ASR 都按 16kHz 工作
    "chunk": 1_024,  # 每次读取的采样点数
    "channels": 1,  # 单声道
    "sample_width": 2,  # 16-bit PCM
    "input_device_index": (
        int(os.environ["MMAI_AUDIO_DEVICE"])
        if os.getenv("MMAI_AUDIO_DEVICE")
        else None
    ),
    "vad_aggressiveness": 2,  # WebRTC VAD 灵敏度 0-3（2 = 平衡）
    "silence_duration_ms": int(os.getenv('MMAI_SILENCE_MS', '450')),
    "min_speech_duration_ms": 500,  # 短于 500ms 的语音段丢弃（防咳嗽/键盘声误触）
    "max_segment_duration_ms": int(os.getenv('MMAI_MAX_SPEECH_MS', '8000')),
}

VIDEO_CONFIG = {
    "camera_id": int(os.getenv("MMAI_CAMERA_ID", "0")),
    "width": 640,
    "height": 480,
    "fps": 30,
    "frame_buffer_size": 60,  # 帧缓冲容量（视觉问答从缓冲取「最近两帧」）
}

TTS_CONFIG = {
    "zh_voice": os.getenv("MMAI_ZH_VOICE", "zh-CN-XiaoxiaoNeural"),
    "en_voice": os.getenv("MMAI_EN_VOICE", "en-US-JennyNeural"),
    "enabled": os.getenv("MMAI_TTS_ENABLED", "1") not in {"0", "false", "False"},
    "timeout_seconds": float(os.getenv('MMAI_TTS_TIMEOUT', '10')),
    "backend": os.getenv('MMAI_TTS_BACKEND', 'auto'),
}

LOG_CONFIG = {
    "level": os.getenv("MMAI_LOG_LEVEL", "INFO"),
    "file": PATHS["logs"] / "multimodal_assistant.log",
}


def set_seed(seed: int = SEED) -> None:
    """固定 random / numpy / torch 随机种子，保证推理过程可复现（默认 42）。"""
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
