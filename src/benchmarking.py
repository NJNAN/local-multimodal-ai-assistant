"""Shared measurement, quality gates and deterministic reporting for experiments."""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

from .inference_metrics import counter_delta, response_measurements


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


class GPUSampler:
    """Whole-device sampled usage; never claimed as isolated process allocation."""
    def __init__(self, interval=0.25):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self):
        try:
            raw = command_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            self.samples.append({"seconds": time.perf_counter(), "used_mib": float(raw.splitlines()[0])})
        except (ValueError, IndexError):
            pass

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=16)
        self._sample()

    def summary(self):
        return {"gpu_baseline_mib": self.samples[0]["used_mib"] if self.samples else None,
                "gpu_peak_mib": max(p["used_mib"] for p in self.samples) if self.samples else None,
                "gpu_sample_count": len(self.samples), "sampling_interval_seconds": self.interval}


def check_answer(answer: str, case: dict) -> bool:
    if "expected_text" in case:
        return answer.strip() == case["expected_text"].strip()
    # Accept JSON-only markdown fences; reject commentary or partial substrings.
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(clean) == case["expected"]
    except (ValueError, TypeError):
        return False


def measure_request(client, case: dict, *, images=None, cache_prompt=False, max_tokens=256) -> dict:
    before = client.server.metrics()
    messages = [{"role": "system", "content": "严格遵循用户格式要求，不输出思考过程。"},
                {"role": "user", "content": case["prompt"]}]
    if images:
        messages[-1]["images"] = images
    payload = {"messages": messages, "options": {"temperature": 0, "num_predict": max_tokens,
               "seed": 42, "id_slot": 0, "cache_prompt": cache_prompt}}
    first, chunks = None, []
    with GPUSampler() as gpu:
        start = time.perf_counter()
        for event in client.stream_chat(payload):
            text = event.get("message", {}).get("content", "")
            if text:
                if first is None:
                    first = time.perf_counter() - start
                chunks.append(text)
        elapsed = time.perf_counter() - start
    answer = "".join(chunks)
    return {"case": case["id"], "category": case["category"], "answer": answer,
            "correct": check_answer(answer, case), "cache_prompt": cache_prompt,
            **response_measurements(client.last_timings, client.last_usage, elapsed, first),
            **gpu.summary(), "speculative": counter_delta(before, client.server.metrics())}


def percentile(values, percent):
    values = sorted(x for x in values if x is not None)
    if not values:
        return None
    position = (len(values) - 1) * percent
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def aggregate(profile: dict) -> dict:
    rows = profile.get("requests", [])
    result = {"count": len(rows), "correct": sum(r["correct"] for r in rows),
              "accuracy": sum(r["correct"] for r in rows) / len(rows) if rows else None}
    for field in ("ttft_seconds", "tpot_ms", "throughput_tokens_s"):
        result[field + "_p50"] = percentile([r[field] for r in rows], 0.5)
        result[field + "_p95"] = percentile([r[field] for r in rows], 0.95)
    peaks = [r["gpu_peak_mib"] for r in rows if r["gpu_peak_mib"] is not None]
    result["gpu_peak_mib"] = max(peaks) if peaks else None
    return result


def fmt(value, digits=3):
    return f"{value:.{digits}f}" if value is not None else "N/A"


def render_results(data: dict) -> str:
    lines = ["# 8GB 本地多模态推理实验结果", "", f"实验时间：{data['created_at']}。数据：`benchmarks/results/latest.json`。",
             "", "由脚本从原始请求生成，运行 `python scripts/16_check_benchmarks.py` 检查数字漂移。",
             "", "|配置|状态|样本数|严格正确率|TTFT p50 / p95 (s)|TPOT p50 (ms)|吞吐 p50 (tok/s)|设备显存采样峰值 (MiB)|",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for p in data["profiles"]:
        a = aggregate(p)
        lines.append(f"|{p['id']}|{p['status']}|{a['count']}|{a['correct']}/{a['count']}|{fmt(a['ttft_seconds_p50'])} / {fmt(a['ttft_seconds_p95'])}|{fmt(a['tpot_ms_p50'])}|{fmt(a['throughput_tokens_s_p50'])}|{fmt(a['gpu_peak_mib'], 0)}|")
    lines += ["", "## 分任务对照", "", "|配置|题目|严格匹配|吞吐 p50 (tok/s)|草稿 token 增量|接受率|", "|---|---|---:|---:|---:|---:|"]
    for p in data["profiles"]:
        for case in sorted(set(r["case"] for r in p.get("requests", []))):
            rows = [r for r in p["requests"] if r["case"] == case]
            draft_values = [r["speculative"]["draft_tokens"] for r in rows]
            accept_values = [r["speculative"]["accepted_tokens"] for r in rows]
            drafts = sum(draft_values) if all(v is not None for v in draft_values) else None
            accepted = sum(accept_values) if all(v is not None for v in accept_values) else None
            acceptance = accepted / drafts if drafts and accepted is not None else None
            rate = percentile([r["throughput_tokens_s"] for r in rows], .5)
            lines.append(f"|{p['id']}|{case}|{sum(r['correct'] for r in rows)}/{len(rows)}|{fmt(rate)}|{fmt(drafts, 0)}|{fmt(acceptance)}|")
    lines += ["", "## 测量条件", "", f"- GPU / 驱动：`{data['environment']['gpu']}`。",
              f"- 引擎：`{data['environment']['server_version']}`。",
              f"- 重复次数：每道题 {data['repeats']} 次；各配置单独启动，只保持一个模型驻留，模型加载耗时另存。",
              "- 温度 0、seed 42、单槽位；文本按固定上下文扫描，禁用自动 fit 调整，配置和实际命令均记录。",
              "- TTFT 从发出已就绪服务请求到首个非空内容片段；TPOT=(流结束时间-首片段时间)/(输出 token 数-1)，是 SSE 观测值。",
              "- tok/s 是含 prefill 的端到端吞吐；原始 server timings 同时保留。拒绝用字符数代替 token 数。",
              "- 普通矩阵请求关闭 prompt 复用；cache_study 单独测冷、热、保存重启恢复。加载时间和 GPU 采样开销不计入请求延迟。",
              "- 显存是 nvidia-smi 采样的整卡占用，包含桌面/其他进程，且可能漏掉瞬时峰值；不是引擎独占显存或 OOM 边界。",
              "- 没有额外预热请求；首次请求的 shader 编译/分配开销保留在原始数据中。配置按固定顺序运行，未随机交错，也未控制功率与热状态。",
              "- 小型合成题集采用严格答案匹配，不代表通用模型能力；低样本 p95 不具有生产容量规划意义。",
              "- 缺失指标保留 N/A；失败配置与无收益的结果都保留，不自动切换回基线掩盖失败。",
              "", "## 原生缓存实验", "", "```json", json.dumps(data.get("cache_study", {}), ensure_ascii=False, indent=2), "```",
              "", "## 可复核材料", "", "- 完整模型/引擎/题集 SHA256、源码哈希、实际参数、响应、usage、timings、指标差值见原始 JSON。",
              "- [实验图](benchmarks/results/overview.svg) / [交互结果页](benchmarks/results/dashboard.html)。",
              "- [负面结果](NEGATIVE-RESULTS.md) / [简历与面试材料](RESUME.md)。", ""]
    return "\n".join(lines)


def render_resume(data: dict) -> str:
    profiles = data["profiles"]
    requests = sum(len(p.get("requests", [])) for p in profiles)
    baseline = next((p for p in profiles if p["id"] == "text_f16_8192"), None)
    speculative = next((p for p in profiles if p["id"] == "text_ngram-cache_8192"), None)
    vision_base = next((p for p in profiles if p["id"] == "vision_default"), None)
    vision_budget = next((p for p in profiles if p["id"] == "vision_256"), None)
    lines = ["# 简历与面试材料", "", "由实测 JSON 自动生成，数字纳入 evidence checker。",
             "", "## 推荐项目名称", "", "**LocalInferLab：8GB 显存约束下的本地多模态推理系统与可复现实验平台**",
             "", "## 可直接使用的项目描述", "",
             "技术栈：Python / llama.cpp（本机 Vulkan 后端）/ PyQt5 / Prometheus / FAISS / MediaPipe。",
             "", "- 集成文本、视觉、语音和本地知识库问答，设计独占模型生命周期管理、空闲卸载、流式取消与手势稳定确认，在消费级 GPU 上保留完整桌面交互链路。",
             f"- 构建 KV 类型×上下文、N-gram 投机解码、视觉 token 预算实验矩阵，完成 {len(profiles)} 组配置、{requests} 次真实推理；记录引擎/模型 SHA256、逐题质量门禁与 TTFT/TPOT/显存采样。"]
    if baseline and speculative:
        b = [r for r in baseline.get("requests", []) if r["case"] == "repetitive_copy"]
        s = [r for r in speculative.get("requests", []) if r["case"] == "repetitive_copy"]
        br, sr = percentile([r["throughput_tokens_s"] for r in b], .5), percentile([r["throughput_tokens_s"] for r in s], .5)
        if br and sr:
            lines += [f"- 在固定重复性中文复制题的 {len(s)} 次重复中，N-gram cache 将端到端吞吐 p50 从 {br:.2f} 提升到 {sr:.2f} tok/s（{sr/br:.2f}×），严格匹配 {sum(r['correct'] for r in s)}/{len(s)}；记录普通事实问答无投机收益等边界。"]
    if vision_base and vision_budget:
        a, b = aggregate(vision_base), aggregate(vision_budget)
        old, new = a["ttft_seconds_p50"], b["ttft_seconds_p50"]
        if old and new:
            lines += [f"- 在固定合成视觉题中，将图像 token 上限设为 256，TTFT p50 从 {old:.3f}s 降到 {new:.3f}s（下降 {(1-new/old)*100:.1f}%），严格匹配 {b['correct']}/{b['count']}；生成原始数据、图表及自动校验报告。"]
    lines += ["", "若简历空间只够三条，保留第一条、实验矩阵条、与你投递方向最相关的一条实测收益。岗位更偏桌面应用时，可保留 RAG 与多模态交互；岗位偏推理工程时，优先保留指标、资源限制与收益边界。",
              "", "## 面试演示（约 5 分钟）", "",
              "1. 启动助手：文字流式问答、图片上传追问，演示原有语音和六手势入口。",
              "2. 打开“推理监控”：说明模型驻留、活动请求与缺失指标为何保留未知。",
              "3. 打开 `benchmarks/results/dashboard.html`：筛选重复性题与事实题，展示收益取决于草稿命中。",
              "4. 展示 KV/视觉预算矩阵与质量门禁：解释整卡显存采样、真实 token 数和首片段时间。",
              "5. 修改一份临时报告中的数字运行 checker：演示证据漂移会失败；再展示缓存恢复未命中的负面结果。",
              "", "## 需要能解释的问题", "",
              "- TTFT、TPOT、prefill、引擎解码 tok/s 和端到端 tok/s 有什么区别？为什么视觉完整回答的首段时间不能叫首 token？",
              "- 为什么 no-draft N-gram 对重复输出收益大，而事实题可能没收益？怎样用计数器差值看接受率？",
              "- 为什么同时记录正确率？本题集的二进制转换失败说明模型能力问题，不能说 KV 量化造成了所有错误。",
              "- 为什么 n_restored 成功而 cache_n=0 仍不能宣称缓存优化成功？下一步应怎样用引擎日志和相同前缀定位？",
              "- 为什么不同配置的整卡显存峰值不能直接当成 KV 分配量？首次 shader 开销、热状态和背景进程如何干扰比较？",
              "- 为什么应用采用独占模型切换？并发服务需要怎样协调 in-flight 请求、模型切换和内存预算？",
              "", "## 证据与边界", "",
              "所有数值来自 [RESULTS.md](RESULTS.md) 和 `benchmarks/results/latest.json`，运行 `python scripts/16_check_benchmarks.py` 可复核。",
              "这是单机、单次矩阵内的少量重复测量；不能写成通用场景加速、生产高并发能力、最大上下文扩容、已成功的落盘加速或 CUDA 算子开发。",
              "CI 配置已加入，GitHub 上的实际运行状态需要推送后确认。", ""]
    return "\n".join(lines)


def render_negative(data: dict) -> str:
    lines = ["# 负面结果与证据边界", "", "本文件自动生成；失败和质量损失也是结果。", ""]
    for p in data["profiles"]:
        if p["status"] != "ok":
            lines += [f"- `{p['id']}`：{p.get('error', 'not completed')}。"]
        a = aggregate(p)
        if a["count"] and a["correct"] < a["count"]:
            failed = sorted(set(r["case"] for r in p["requests"] if not r["correct"]))
            lines += [f"- `{p['id']}` 严格匹配仅 {a['correct']}/{a['count']}；失败题：{', '.join(failed)}。不能只报告速度。"]
        if p.get("options", {}).get("spec_type", "none") != "none":
            drafts = [r["speculative"]["draft_tokens"] for r in p.get("requests", [])]
            if not any(x for x in drafts if x is not None):
                lines += [f"- `{p['id']}` 未观测到非零草稿 token 增量；启用参数不等于已经获得投机加速。"]
            baseline = next((b for b in data["profiles"] if b.get("mode") == "text" and b.get("options", {}).get("spec_type", "none") == "none" and b.get("n_ctx") == p.get("n_ctx") and b.get("options", {}).get("cache_type_k", "f16") == "f16"), None)
            if baseline:
                b = aggregate(baseline)
                if a["throughput_tokens_s_p50"] is not None and b["throughput_tokens_s_p50"] is not None:
                    lines += [f"- `{p['id']}` 吞吐 p50={fmt(a['throughput_tokens_s_p50'])} tok/s；对应 f16 基线={fmt(b['throughput_tokens_s_p50'])} tok/s。样本量有限，不能推广为所有场景的收益。"]
    cache = data.get("cache_study", {})
    if cache.get("status") != "ok":
        lines += [f"- 原生缓存恢复：{cache.get('error', '未运行')}。未完成的实验不能写成已优化。"]
    elif not cache.get("reuse_verified"):
        lines += [f"- 缓存端点返回成功，n_restored={cache.get('restore', {}).get('n_restored')}，但恢复后 cache_n={cache.get('restored', {}).get('cached_tokens')}。未验证到前缀复用；不能把文件恢复成功写成推理加速。"]
    lines += ["- 本机引擎无法识别中文 slot-save-path 目录，实验改用 ASCII 系统临时目录。KV 文件不纳入公开材料。"]
    lines += ["", "尚未测量最大可用上下文、OOM 边界、长期热稳定性、大规模并发和官方 router 对照。",
              "没有实现 CUDA 算子、训练基础设施或新推理算法；实现的是引擎集成、测量和资源约束下的取舍。",
              "调查中的独占赛道、零额外显存、必然加速等说法不作为本项目结论。", ""]
    return "\n".join(lines)


def write_visuals(data: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    profiles = [p for p in data["profiles"] if p.get("requests")]
    names = [p["id"] for p in profiles]
    aggregates = [aggregate(p) for p in profiles]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, field, label in zip(axes, ("ttft_seconds_p50", "gpu_peak_mib", "accuracy"),
                                  ("Median TTFT (s)", "Sampled device VRAM (MiB)", "Exact-match accuracy")):
        axis.barh(names, [a[field] or 0 for a in aggregates], color="#15a695")
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=.2)
    figure.suptitle("8GB local inference: latency, memory, quality (small synthetic set)")
    figure.tight_layout()
    figure.savefig(output / "overview.svg")
    plt.close(figure)
    # Offline, self-contained result explorer. Escape HTML script closing sequences in raw responses.
    raw = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>8GB 推理实验台</title>
<style>body{font:16px system-ui;background:#0c1421;color:#dce7f5;margin:40px}h1{color:#2dd4bf}select{padding:8px}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #33445b;padding:9px;text-align:left}pre{white-space:pre-wrap}img{background:white;width:100%}</style>
<h1>8GB 本地多模态推理实验台</h1><p>小型合成题集 · SSE 计时 · 整卡显存采样 · 原始数据可复核</p>
<img src="overview.svg" alt="延迟、显存与正确率"><p>筛选配置 <select id="filter"></select></p><table><thead><tr><th>配置 / 题目</th><th>正确</th><th>TTFT(s)</th><th>TPOT(ms)</th><th>输出 tok</th><th>草稿接受率</th></tr></thead><tbody id="rows"></tbody></table><details><summary>环境与原始数据</summary><pre id="raw"></pre></details>
<script>const data=__DATA__;const filter=document.getElementById('filter');for(const name of ['全部',...data.profiles.map(p=>p.id)]){const o=document.createElement('option');o.textContent=name;filter.append(o)}
function render(){const rows=document.getElementById('rows');rows.replaceChildren();for(const p of data.profiles){if(filter.value!=='全部'&&filter.value!==p.id)continue;for(const r of p.requests||[]){const tr=document.createElement('tr');for(const v of [p.id+' / '+r.case,r.correct,r.ttft_seconds,r.tpot_ms,r.output_tokens,r.speculative.acceptance_rate]){const td=document.createElement('td');td.textContent=v==null?'N/A':typeof v==='number'?v.toFixed(3):String(v);tr.append(td)}rows.append(tr)}}}filter.onchange=render;render();document.getElementById('raw').textContent=JSON.stringify(data,null,2);</script></html>'''
    (output / "dashboard.html").write_text(html.replace("__DATA__", raw), encoding="utf-8")
