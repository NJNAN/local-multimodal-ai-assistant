from __future__ import annotations

import time

from _bootstrap import PROJECT_ROOT

from config import MODEL_CONFIG, PATHS
from src.text_llm import TextLLM


def main() -> int:
    llm = TextLLM(MODEL_CONFIG["text"])
    print(f"模型: {llm.model_name}（llama.cpp / llama-server 后端）")
    prompts = [
        "介绍一下你自己",
        "用两句话解释什么是推荐系统",
        "中文测试：请用一句话说明本地大模型的好处。",
    ]
    records = []
    for prompt in prompts:
        print(f"\n你: {prompt}\n助手: ", end="", flush=True)
        started = time.perf_counter()
        first_token = None
        chunks = []
        for chunk in llm.stream_generate(prompt, max_new_tokens=160):
            if first_token is None:
                first_token = time.perf_counter() - started
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        elapsed = time.perf_counter() - started
        answer = "".join(chunks)
        print(
            f"\n首 token {first_token or elapsed:.3f}s，总耗时 {elapsed:.3f}s，输出 {len(answer)} 字"
        )
        records.append(
            f"问题: {prompt}\n回答: {answer}\n首token: {first_token or elapsed:.3f}s\n耗时: {elapsed:.3f}s\n"
        )
    (PATHS["outputs"] / "llm_test_log.txt").write_text("\n".join(records), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
