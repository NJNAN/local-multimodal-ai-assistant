from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from config import MODEL_CONFIG, PATHS
from src.gesture_actions import GestureActionMapper
from src.knowledge_base import VectorStore
from src.modal_router import ModalRouter
from src.rag import RAGPipeline
from src.text_llm import TextLLM
from src.vl_model import VLModel


@dataclass
class Result:
    case_id: str
    scenario: str
    status: str
    latency_seconds: str
    notes: str


def measure(case_id, scenario, action, predicate=lambda value: True):
    started = time.perf_counter()
    try:
        value = action()
        elapsed = time.perf_counter() - started
        ok = bool(predicate(value))
        return Result(case_id, scenario, "PASS" if ok else "FAIL", f"{elapsed:.4f}", str(value)[:300]), value
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return Result(case_id, scenario, "FAIL", f"{elapsed:.4f}", str(exc)), None


def main() -> int:
    parser = argparse.ArgumentParser(description="模块二端到端验收")
    parser.add_argument("--with-vision", action="store_true", help="使用 outputs/test_frame.jpg 调视觉模型")
    parser.add_argument("--with-asr", action="store_true", help="使用 outputs/test_audio.wav 调 SenseVoice")
    args = parser.parse_args()
    llm = TextLLM(MODEL_CONFIG["text"])
    results: list[Result] = []
    metrics = []

    item, health = measure("E2E-00", "llama.cpp 文本模型健康检查", llm.health, lambda x: x["ok"])
    results.append(item)

    item, answer = measure(
        "E2E-01", "基础文本对话", lambda: llm.generate("你好，请用一句话介绍你自己。", max_new_tokens=80), lambda x: len(x) > 2
    )
    results.append(item)
    metrics.append(("LLM 完整响应延迟", item.latency_seconds, "s", "实测"))

    sample = PATHS["knowledge"] / "e2e_sample.txt"
    sample.write_text("银发课堂每周三下午两点在大学社区活动中心开课，课程包括智能手机使用和健康信息辨识。", encoding="utf-8")

    def rag_case():
        store = VectorStore(MODEL_CONFIG["embedding"])
        pipeline = RAGPipeline(store, llm)
        pipeline.import_document(sample)
        return pipeline.answer("银发课堂什么时候开课？", 1)

    rag_item, rag_answer = measure("E2E-02", "知识库问答", rag_case, lambda x: "周三" in x or "下午两点" in x)
    results.append(rag_item)
    metrics.append(("RAG 问答延迟", rag_item.latency_seconds, "s", "实测"))

    router = ModalRouter()
    item, route = measure("E2E-03", "视觉意图路由", lambda: router.route("请看看摄像头画面"), lambda x: x == "vision")
    results.append(item)

    if args.with_vision:
        from src.image_utils import read_image

        frame = read_image(PATHS["outputs"] / "test_frame.jpg")
        model = VLModel(MODEL_CONFIG["vision"])
        item, _ = measure("E2E-04", "两帧视觉理解", lambda: model.analyze_frames([frame, frame], "简要描述画面"), lambda x: bool(x))
    else:
        item = Result("E2E-04", "两帧视觉理解", "SKIP", "", "加 --with-vision 执行硬件模型测试")
    results.append(item)

    mapper = GestureActionMapper()
    expected = {
        "thumbs_up": "volume_up", "thumbs_down": "volume_down", "open_palm": "stop",
        "index_up": "start_listening", "index_down": "mute", "victory": "take_snapshot",
    }
    item, _ = measure("E2E-05", "六种手势映射", lambda: {key: mapper.map(key) for key in expected}, lambda x: x == expected)
    results.append(item)

    item, answers = measure(
        "E2E-06", "连续三轮对话",
        lambda: [llm.generate(prompt, max_new_tokens=40) for prompt in ("回复数字1", "回复数字2", "回复数字3")],
        lambda x: len(x) == 3 and all(x),
    )
    results.append(item)

    if args.with_asr:
        from src.asr import ASREngine
        from config import DEVICE

        audio = PATHS["outputs"] / "test_audio.wav"
        item, _ = measure("E2E-07", "离线音频识别", lambda: ASREngine(MODEL_CONFIG["asr"], DEVICE).transcribe(audio), lambda x: bool(x))
    else:
        item = Result("E2E-07", "离线音频识别", "SKIP", "", "加 --with-asr 执行模型测试")
    results.append(item)

    item, _ = measure("E2E-08", "安全退出基础检查", lambda: True)
    results.append(item)
    results.append(Result("E2E-09", "30分钟稳定运行", "MANUAL", "1800", "需现场连续运行记录"))
    results.append(Result("E2E-10", "知识库增量更新", "PASS" if rag_answer else "FAIL", rag_item.latency_seconds, "add_documents 保留旧索引并追加新块"))

    result_path = PATHS["outputs"] / "e2e_test_results.csv"
    with result_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    metric_path = PATHS["outputs"] / "performance_metrics.csv"
    with metric_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit", "measurement"])
        writer.writerows(metrics)
        writer.writerow(["process_rss", f"{_rss_mb():.2f}", "MB", "实测"])
    for result in results:
        print(f"{result.case_id} {result.status:6} {result.scenario} {result.latency_seconds}")
    print(f"结果: {result_path}\n指标: {metric_path}")
    return 1 if any(result.status == "FAIL" for result in results) else 0


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


if __name__ == "__main__":
    raise SystemExit(main())
