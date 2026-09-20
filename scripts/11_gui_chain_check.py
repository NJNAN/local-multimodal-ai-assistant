"""GUI → 模型 → TTS 基本调用链检查（无窗口版本）。

构建与 main.py 相同的核心组件（文本模型 + RAG + 视觉模型 + 模态路由），
走一遍 AssistantService.process()，再把回答送进 Edge TTS 合成，
最后确认 llama-server 退出无残留。

用法:

    D:\\mm_ai_env\\Scripts\\python.exe scripts\\11_gui_chain_check.py
"""

from __future__ import annotations

import subprocess
import time

from _bootstrap import PROJECT_ROOT

from config import MODEL_CONFIG, PATHS, TTS_CONFIG
from src.assistant_service import AssistantService
from src.knowledge_base import VectorStore
from src.llama_backend import create_text_client, create_vision_client, shutdown
from src.modal_router import ModalRouter
from src.rag import RAGPipeline
from src.text_llm import TextLLM
from src.tts import TTSEngine
from src.vl_model import VLModel


class _OfflineVideo:
    """无摄像头环境下的占位（本检查只走文本链路）。"""

    def recent_frames(self, count: int = 2):
        return []


def main() -> int:
    text_llm = TextLLM(MODEL_CONFIG["text"], client=create_text_client())
    store = VectorStore(MODEL_CONFIG["embedding"])
    rag = RAGPipeline(store, text_llm)
    vl_model = VLModel(MODEL_CONFIG["vision"], client=create_vision_client())
    service = AssistantService(text_llm, rag, vl_model, ModalRouter(), _OfflineVideo())

    chunks: list[str] = []
    started = time.perf_counter()
    answer = service.process("你好，请用一句话介绍你自己。", chunks.append)
    elapsed = time.perf_counter() - started
    print(f"模型回答（{elapsed:.2f}s，{len(chunks)} 个流式片段）: {answer[:120]}")
    assert answer.strip(), "模型没有返回内容"

    tts = TTSEngine(PATHS["temp"], TTS_CONFIG["zh_voice"], TTS_CONFIG["en_voice"])
    output = tts.synthesize_sync(answer, PATHS["outputs"] / "chain_tts_test.mp3")
    size = output.stat().st_size
    print(f"TTS 合成: {output}（{size} 字节）")
    assert size > 0, "TTS 输出为空"

    shutdown()
    time.sleep(1.5)
    result = subprocess.run(
        ["tasklist"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    residue = [line for line in result.stdout.splitlines() if "llama-server" in line.lower()]
    if residue:
        print("残留 llama-server 进程:")
        print("\n".join(residue))
        return 1
    print("退出清理: 无 llama-server 残留")
    print("GUI → 模型 → TTS 链路检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
