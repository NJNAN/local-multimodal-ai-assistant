"""性能采集脚本：模型大小 / 加载时间 / RAM / VRAM / 首 token / tokens/s / 图片延迟。

用法（模型下载完成后）:

    D:\\mm_ai_env\\Scripts\\python.exe scripts\\10_perf_report.py

结果写入 ``outputs/perf_report.json`` 与 ``outputs/perf_report.md``。
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT

from config import MODEL_CONFIG, PATHS
from src.image_utils import read_image
from src.llama_backend import shutdown, get_manager
from src.vl_model import VLModel


def _creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def gpu_mem_used_mb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_creationflags(),
        )
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return None


def tree_rss_mb() -> float:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        total = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                continue
        return total / 1024 / 1024
    except Exception:
        return 0.0


def fit_params_hint(server) -> str | None:
    """llama-fit-params：--fit 在当前设备状态下会选择什么参数。"""
    if not server.server_path or not Path(server.server_path).is_file():
        return None
    exe = Path(server.server_path).with_name("llama-fit-params.exe")
    if not exe.is_file():
        return None
    cmd = [str(exe), "-m", str(server.model_path), "-c", str(server.n_ctx)]
    if server.device:
        cmd += ["--device", server.device]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
            creationflags=_creationflags(),
        )
    except Exception:
        return None
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def gpu_process_names() -> list[str]:
    """当前占用 GPU 的进程名（用于标注测试环境是否有干扰）。"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
            creationflags=_creationflags(),
        )
        return sorted(set(re.findall(r"[\\/]([A-Za-z0-9_.\-]+\.exe)", result.stdout or "")))
    except Exception:
        return []


def parse_offload_layers(log_path: str | Path) -> dict | None:
    path = Path(log_path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"offloaded (\d+)/(\d+) layers", text)
    if match:
        return {"offloaded": int(match.group(1)), "total": int(match.group(2))}
    return None


def file_size_mb(path: str | Path) -> float:
    return round(Path(path).stat().st_size / 1024 / 1024, 1)


def main() -> int:
    report: dict = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    manager = get_manager()

    # --- 模型文件大小 -------------------------------------------------------
    report["files"] = {
        "text_gguf": {"name": Path(MODEL_CONFIG["text_gguf"]).name, "size_mb": file_size_mb(MODEL_CONFIG["text_gguf"])},
        "vl_gguf": {"name": Path(MODEL_CONFIG["vl_gguf"]).name, "size_mb": file_size_mb(MODEL_CONFIG["vl_gguf"])},
        "vl_mmproj": {"name": Path(MODEL_CONFIG["vl_mmproj"]).name, "size_mb": file_size_mb(MODEL_CONFIG["vl_mmproj"])},
    }

    vram_baseline = gpu_mem_used_mb()

    # --- 文本模型 -----------------------------------------------------------
    from src.llama_backend import create_text_client

    text_client = create_text_client()
    text_server = manager.ensure("text")
    report["text"] = {
        "load_seconds": round(text_server.load_seconds or 0, 2),
        "vram_used_mb_after_load": gpu_mem_used_mb(),
        "tree_rss_mb_after_load": round(tree_rss_mb(), 1),
    }

    # 非流式调用（拿 timings: tokens/s）
    started = time.perf_counter()
    response = text_client.chat(
        {
            "model": text_client.model_name,
            "messages": [{"role": "user", "content": "请用三句话介绍一下你自己。"}],
            "think": False,
            "options": {"temperature": 0.4, "num_predict": 128},
        }
    )
    nonstream_seconds = time.perf_counter() - started
    timings = response.get("timings") or {}
    report["text"].update(
        {
            "nonstream_seconds": round(nonstream_seconds, 2),
            "predicted_per_second": round(timings.get("predicted_per_second", 0), 2),
            "prompt_per_second": round(timings.get("prompt_per_second", 0), 2),
            "predicted_n": timings.get("predicted_n"),
            "answer_preview": str(response.get("message", {}).get("content", ""))[:120],
        }
    )

    # 流式调用（首 token 延迟）
    started = time.perf_counter()
    first_token = None
    streamed = []
    stream_timings = {}
    for event in text_client.stream_chat(
        {
            "model": text_client.model_name,
            "messages": [{"role": "user", "content": "用一句话介绍本地大模型的优点。"}],
            "think": False,
            "options": {"temperature": 0.4, "num_predict": 96},
        }
    ):
        content = event.get("message", {}).get("content")
        if content:
            if first_token is None:
                first_token = time.perf_counter() - started
            streamed.append(content)
        if event.get("timings"):
            stream_timings = event["timings"]
    report["text"].update(
        {
            "ttft_seconds": round(first_token or -1, 3),
            "stream_total_seconds": round(time.perf_counter() - started, 2),
            "stream_predicted_per_second": round(stream_timings.get("predicted_per_second", 0), 2),
        }
    )

    # --- 视觉模型（独占切换：先停文本） ---------------------------------------
    vl = VLModel(MODEL_CONFIG["vision"])
    vision_server = manager.ensure("vision")
    report["vision"] = {
        "load_seconds": round(vision_server.load_seconds or 0, 2),
        "vram_used_mb_after_load": gpu_mem_used_mb(),
        "tree_rss_mb_after_load": round(tree_rss_mb(), 1),
    }

    image_path = PATHS["outputs"] / "vl_test_1.jpg"
    if not image_path.exists():
        fallback = PATHS["outputs"] / "test_frame.jpg"
        if fallback.exists():
            image_path = fallback
        else:
            raise FileNotFoundError("缺少 outputs/vl_test_1.jpg 或 outputs/test_frame.jpg 测试图")
    frame = read_image(image_path)

    started = time.perf_counter()
    answer = vl.analyze_frames([frame], "请描述这张图片的内容。", max_new_tokens=128)
    single_seconds = time.perf_counter() - started
    report["vision"].update(
        {
            "single_image_seconds": round(single_seconds, 2),
            "single_image_answer_preview": answer[:120],
        }
    )

    # 图片细节问答（带 timings 的直接调用）
    started = time.perf_counter()
    messages = vl.create_messages([str(image_path)], "图中的文字是什么？如果没有文字请说明。")
    response = vl.client.chat(
        {
            "model": vl.model_name,
            "messages": messages,
            "options": {"temperature": 0.2, "num_predict": 128},
        }
    )
    detail_seconds = time.perf_counter() - started
    v_timings = response.get("timings") or {}
    report["vision"].update(
        {
            "detail_query_seconds": round(detail_seconds, 2),
            "detail_predicted_per_second": round(v_timings.get("predicted_per_second", 0), 2),
            "detail_answer_preview": str(response.get("message", {}).get("content", ""))[:120],
        }
    )

    # --- offload / OOM / 收尾 -------------------------------------------------
    report["offload"] = {
        "text_layers": parse_offload_layers(manager.servers["text"].log_path),
        "vision_layers": parse_offload_layers(manager.servers["vision"].log_path),
        "fit": "on（llama.cpp --fit 自动适配显存）",
        "device_hint": manager.servers["vision"].device or "auto",
        "text_fit_params_hint": fit_params_hint(manager.servers["text"]),
        "vision_fit_params_hint": fit_params_hint(manager.servers["vision"]),
        "gpu_processes_during_test": gpu_process_names(),
    }
    logs = ""
    for server in manager.servers.values():
        if server.log_path and Path(server.log_path).exists():
            logs += Path(server.log_path).read_text(encoding="utf-8", errors="replace")
    report["oom_detected"] = bool(re.search(r"out of memory|failed to allocate", logs, re.IGNORECASE))

    shutdown()
    time.sleep(1.5)
    report["vram_used_mb_final"] = gpu_mem_used_mb()
    report["vram_baseline_mb"] = vram_baseline

    json_path = PATHS["outputs"] / "perf_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# llama.cpp 性能采集", "", f"时间: {report['timestamp']}", ""]
    lines.append("## 模型文件")
    for name, info in report["files"].items():
        lines.append(f"- {info['name']}: {info['size_mb']} MB")
    lines.append("")
    lines.append("## 文本模型 (Qwen3.5-4B IQ4_XS)")
    for key, value in report["text"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 视觉模型 (Qwen3-VL-4B Q4_K_M + mmproj Q8_0)")
    for key, value in report["vision"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## 其他")
    for key, value in report["offload"].items():
        lines.append(f"- offload.{key}: {value}")
    lines.append(f"- OOM detected: {report['oom_detected']}")
    lines.append(f"- VRAM 最终: {report['vram_used_mb_final']} MB（基线 {report['vram_baseline_mb']} MB）")
    md_path = PATHS["outputs"] / "perf_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n结果: {json_path}\n摘要: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
