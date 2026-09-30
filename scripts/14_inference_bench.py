"""Run isolated, real llama.cpp experiments; failures are first-class results."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import platform
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from config import LLAMA_CONFIG
from src.benchmarking import GPUSampler, command_output, measure_request, sha256
from src.llama_backend import LlamaCppClient, LlamaServerManager, _build_servers


def plan(suite):
    profiles = []
    if suite in {"text", "all"}:
        for context in (4096, 8192):
            for kv in ("f16", "q8_0", "q4_0"):
                profiles.append({"id": f"text_{kv}_{context}", "mode": "text", "n_ctx": context,
                                 "options": {"cache_type_k": kv, "cache_type_v": kv, "flash_attn": "on", "parallel": 1}})
        for spec in ("ngram-simple", "ngram-cache"):
            profiles.append({"id": f"text_{spec}_8192", "mode": "text", "n_ctx": 8192,
                             "options": {"spec_type": spec, "flash_attn": "on", "parallel": 1}})
    if suite in {"vision", "all"}:
        for budget in (None, 256, 1024):
            options = {"parallel": 1, "flash_attn": "on"}
            if budget:
                options.update(image_min_tokens=64, image_max_tokens=budget, mtmd_batch_max_tokens=1024)
            profiles.append({"id": f"vision_{budget or 'default'}", "mode": "vision", "n_ctx": 8192, "options": options})
    return profiles


def cache_study(manager, client, case):
    """Same request/prefix cold vs warm vs save/restart/restore; no mock reuse."""
    mode = client.mode
    case = {**case, "id": "long_prefix", "category": "cache", "prompt":
            "以下是背景资料：\n" + ("本地系统需要测量首字延迟、显存和输出正确率。模型切换时释放上一模型。\n" * 48)
            + "\n只输出 JSON：{\"value\":7300}", "expected": {"value": 7300}}
    case.pop("expected_text", None)
    study = {"status": "error"}
    try:
        manager.slot_action(mode, "erase")
        study["cold"] = measure_request(client, case, cache_prompt=True, max_tokens=48)
        study["warm"] = measure_request(client, case, cache_prompt=True, max_tokens=48)
        study["save"] = manager.save_slot(mode, "benchmark-prefix.bin")
        client.server.stop()
        manager.ensure(mode)
        study["restore"] = manager.restore_slot(mode, "benchmark-prefix.bin")
        study["restored"] = measure_request(client, case, cache_prompt=True, max_tokens=48)
        restored_count = study["restore"].get("n_restored", 0)
        cached_count = study["restored"].get("cached_tokens")
        study["reuse_verified"] = bool(restored_count and cached_count and cached_count > 0)
        study["restore_plus_ttft_seconds"] = (study["restore"].get("timings", {}).get("restore_ms", 0) / 1000
                                               + (study["restored"]["ttft_seconds"] or 0))
        study["status"] = "ok"
    except Exception as exc:
        study["error"] = str(exc)
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("text", "vision", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks/results/latest.json")
    parser.add_argument("--profile", action="append", help="Run selected profile ids only")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    profiles = [p for p in plan(args.suite) if not args.profile or p["id"] in args.profile]
    if not profiles:
        parser.error("No profiles selected")
    if args.plan_only:
        print(json.dumps(profiles, indent=2))
        return 0
    case_path = PROJECT_ROOT / "benchmarks/cases.json"
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Private runtime cache stays in ignored outputs; no conversation caches in public data.
    run_dir = PROJECT_ROOT / "outputs/inference_bench" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    slot_temp = tempfile.TemporaryDirectory(prefix="mmai_bench_slots_")
    image_data = None
    if any(p["mode"] == "vision" for p in profiles):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (768, 768), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((100, 90, 260, 250), fill="red")
        draw.ellipse((470, 90, 630, 250), fill="blue")
        draw.rectangle((100, 480, 260, 640), fill="red")
        draw.polygon(((550, 470), (450, 650), (650, 650)), fill="green")
        path = run_dir / "synthetic.png"
        image.save(path)
        image_data = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    data = {"schema_version": 1, "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "repeats": args.repeats, "environment": {"python": platform.python_version(), "os": platform.platform(),
            "gpu": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
            "server_version": command_output([LLAMA_CONFIG["server"], "--version"]),
            "server_devices": command_output([LLAMA_CONFIG["server"], "--list-devices"]),
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "server_sha256": sha256(Path(LLAMA_CONFIG["server"])),
            "cases_sha256": sha256(case_path), "source_sha256": {}, "models": {}}, "profiles": []}
    for relative in ("src/llama_backend.py", "src/inference_options.py", "src/inference_metrics.py", "src/benchmarking.py", "scripts/14_inference_bench.py"):
        data["environment"]["source_sha256"][relative] = sha256(PROJECT_ROOT / relative)
    for mode in sorted(set(p["mode"] for p in profiles)):
        for key in ("model", "mmproj"):
            path = LLAMA_CONFIG[mode].get(key)
            if path:
                path = Path(path)
                data["environment"]["models"][f"{mode}_{key}"] = {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
    for profile in profiles:
        config = copy.deepcopy(LLAMA_CONFIG)
        # Dedicated ports; never reclaim or terminate the user's running assistant.
        config["text"]["port"], config["vision"]["port"] = 18080, 18081
        config["log_dir"], config["fit"], config["start_timeout"] = run_dir, False, 120
        mode = profile["mode"]
        config[mode].update(n_ctx=profile["n_ctx"], disable_thinking=True, keep_alive_seconds=0)
        options = {**profile["options"]}
        if profile["id"] == "text_f16_8192":
            options["slot_save_path"] = slot_temp.name
        config[mode]["inference"] = options
        # Avoid inheriting unrelated environment tuning into the comparison.
        config["vision" if mode == "text" else "text"]["inference"] = {}
        manager = LlamaServerManager(_build_servers(config), exclusive=True, start_timeout=120)
        record = {**profile, "status": "error", "requests": []}
        data["profiles"].append(record)
        print(f"START {profile['id']}", flush=True)
        try:
            server = manager.servers[mode]
            # Port conflicts are reported, rather than killing arbitrary model processes.
            if manager._port_in_use(server):
                raise RuntimeError(f"Benchmark port {server.port} is occupied; no process was terminated")
            with GPUSampler() as loading:
                manager.ensure(mode)
            record["load_seconds"], record["loading_gpu"] = server.load_seconds, loading.summary()
            record["command"] = ["ASCII_TEMP_SLOTS" if token == slot_temp.name else str(Path(token).name)
                                  if str(PROJECT_ROOT) in token or token == config["server"] else token for token in server.build_command()]
            record["props"] = {k: v for k, v in server.props().items() if k in {"build_info", "model_ftype", "total_slots", "model_meta"}}
            client = LlamaCppClient(manager, mode, config=config)
            for repeat in range(args.repeats):
                for case in cases[mode]:
                    request = measure_request(client, case, images=[image_data] if mode == "vision" else None)
                    request["repeat"] = repeat
                    record["requests"].append(request)
                    print(f"  {case['id']}: correct={request['correct']} ttft={request['ttft_seconds']} tokens={request['output_tokens']}", flush=True)
            record["status"] = "ok"
            if profile["id"] == "text_f16_8192":
                data["cache_study"] = cache_study(manager, client, cases["text"][0])
        except Exception as exc:
            record["error"] = str(exc).replace(str(PROJECT_ROOT), "$PROJECT_ROOT")
            print(f"FAILED {profile['id']}: {record['error']}", flush=True)
        finally:
            manager.begin_shutdown()
            manager.stop_all()
            path = run_dir / f"llama_server_{mode}.log"
            if path.exists():
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                record["allocation_log"] = [line.replace(str(PROJECT_ROOT), "$PROJECT_ROOT") for line in lines
                                             if any(term in line.lower() for term in ("kv", "cache size", "offloaded", "n_ctx", "not support", "error:"))][-50:]
            args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"DATA {args.output}", flush=True)
    slot_temp.cleanup()
    return 0 if any(p["status"] == "ok" for p in data["profiles"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
