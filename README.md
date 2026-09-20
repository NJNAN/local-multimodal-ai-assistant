# Multimodal AI Assistant

一个运行在 Windows 本地的多模态 AI 助手，集成文字对话、语音识别、视觉理解、知识库问答、人体姿态与手势识别，以及语音合成。桌面界面基于 PyQt5，文本与视觉模型通过 `llama.cpp` 的 `llama-server` 按需加载。

## 主要功能

- 麦克风采集、WebRTC VAD 与 SenseVoice 语音识别
- Qwen3.5 文本对话与流式输出
- Qwen3-VL 图像理解和多轮视觉问答
- Sentence Transformers + FAISS 本地知识库检索
- MediaPipe 人体姿态和手势识别
- Edge TTS 中英文语音合成
- PyQt5 桌面界面及端到端测试脚本

## 环境要求

- Windows 10/11
- Python 3.10（建议使用，项目支持范围为 3.10–3.11）
- 麦克风和摄像头（使用相关功能时）
- `llama-server`，可通过 `winget install ggml.llamacpp` 安装
- NVIDIA GPU 为可选项；CUDA 依赖见 `requirements-cuda.txt`

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如需使用 NVIDIA GPU，再安装 CUDA 版 PyTorch：

```powershell
pip install -r requirements-cuda.txt
```

## 模型准备

模型权重体积较大，不包含在仓库中。可使用 Hugging Face CLI 下载：

```powershell
hf download bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-IQ4_XS.gguf --local-dir models/qwen3.5-4b
hf download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf --local-dir models/qwen3-vl-4b
```

SenseVoice 和嵌入模型会在首次使用时由依赖库下载，也可通过环境变量指定本地路径：

- `MMAI_ASR_MODEL`
- `MMAI_EMBEDDING_MODEL`
- `MMAI_TEXT_GGUF`
- `MMAI_VL_GGUF`
- `MMAI_VL_MMPROJ`
- `MMAI_LLAMA_SERVER`

## 运行

双击 `start_assistant.bat`，或在 PowerShell 中运行：

```powershell
python scripts/00_check_env.py
python main.py
```

无麦克风或摄像头时可使用：

```powershell
python main.py --no-audio --no-video
```

## 测试

```powershell
pytest
```

`scripts/` 还提供音频采集、视频采集、ASR、LLM、知识库、视觉理解、TTS、检测和端到端检查等独立测试入口。

## 项目结构

```text
config/       应用配置
scripts/      环境检查与功能测试脚本
src/          核心业务代码
tests/        自动化测试
tools/        模型校验与辅助工具
main.py       应用入口
```

模型权重、知识库原始文档、课程资料、运行日志、测试媒体与生成报告均未纳入版本控制。
