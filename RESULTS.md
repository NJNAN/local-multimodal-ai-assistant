# 8GB 本地多模态推理实验结果

实验时间：2026-09-30T18:06:20.867078+08:00。数据：`benchmarks/results/latest.json`。

由脚本从原始请求生成，运行 `python scripts/16_check_benchmarks.py` 检查数字漂移。

|配置|状态|样本数|严格正确率|TTFT p50 / p95 (s)|TPOT p50 (ms)|吞吐 p50 (tok/s)|设备显存采样峰值 (MiB)|
|---|---|---:|---:|---:|---:|---:|---:|
|text_f16_4096|ok|9|6/9|0.238 / 0.309|15.923|45.404|4409|
|text_q8_0_4096|ok|9|6/9|0.222 / 0.289|15.582|47.013|4248|
|text_q4_0_4096|ok|9|6/9|0.236 / 0.262|15.364|48.841|4534|
|text_f16_8192|ok|9|6/9|0.209 / 0.245|15.129|49.575|4769|
|text_q8_0_8192|ok|9|6/9|0.212 / 0.245|15.596|48.340|4421|
|text_q4_0_8192|ok|9|6/9|0.210 / 0.245|15.298|50.789|4521|
|text_ngram-simple_8192|ok|9|6/9|0.209 / 0.246|15.417|49.215|4710|
|text_ngram-cache_8192|ok|9|6/9|0.201 / 0.236|15.549|47.166|4710|
|vision_default|ok|6|6/6|0.473 / 0.544|13.374|29.316|6183|
|vision_256|ok|6|6/6|0.244 / 0.259|13.170|43.457|6118|
|vision_1024|ok|6|6/6|0.458 / 0.490|13.422|31.552|6184|

## 分任务对照

|配置|题目|严格匹配|吞吐 p50 (tok/s)|草稿 token 增量|接受率|
|---|---|---:|---:|---:|---:|
|text_f16_4096|fact_qa|0/3|45.404|0|N/A|
|text_f16_4096|repetitive_copy|3/3|53.988|0|N/A|
|text_f16_4096|structured_extraction|3/3|34.730|0|N/A|
|text_q8_0_4096|fact_qa|0/3|47.013|0|N/A|
|text_q8_0_4096|repetitive_copy|3/3|55.577|0|N/A|
|text_q8_0_4096|structured_extraction|3/3|40.401|0|N/A|
|text_q4_0_4096|fact_qa|0/3|46.485|0|N/A|
|text_q4_0_4096|repetitive_copy|3/3|56.732|0|N/A|
|text_q4_0_4096|structured_extraction|3/3|47.317|0|N/A|
|text_f16_8192|fact_qa|0/3|49.575|0|N/A|
|text_f16_8192|repetitive_copy|3/3|58.367|0|N/A|
|text_f16_8192|structured_extraction|3/3|40.969|0|N/A|
|text_q8_0_8192|fact_qa|0/3|48.340|0|N/A|
|text_q8_0_8192|repetitive_copy|3/3|58.498|0|N/A|
|text_q8_0_8192|structured_extraction|3/3|41.281|0|N/A|
|text_q4_0_8192|fact_qa|0/3|50.789|0|N/A|
|text_q4_0_8192|repetitive_copy|3/3|58.149|0|N/A|
|text_q4_0_8192|structured_extraction|3/3|47.495|0|N/A|
|text_ngram-simple_8192|fact_qa|0/3|49.215|0|N/A|
|text_ngram-simple_8192|repetitive_copy|3/3|112.008|402|0.672|
|text_ngram-simple_8192|structured_extraction|3/3|41.575|0|N/A|
|text_ngram-cache_8192|fact_qa|0/3|47.166|0|N/A|
|text_ngram-cache_8192|repetitive_copy|3/3|118.278|312|0.942|
|text_ngram-cache_8192|structured_extraction|3/3|40.373|24|0.250|
|vision_default|color_count|3/3|34.276|0|N/A|
|vision_default|layout|3/3|26.609|0|N/A|
|vision_256|color_count|3/3|49.364|0|N/A|
|vision_256|layout|3/3|38.937|0|N/A|
|vision_1024|color_count|3/3|35.269|0|N/A|
|vision_1024|layout|3/3|27.814|0|N/A|

## 测量条件

- GPU / 驱动：`NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 616.92`。
- 引擎：`version: 0.3.0-dev (build 10689, commit 57291f264)
built with Clang 20.1.8 for Windows x86_64`。
- 重复次数：每道题 3 次；各配置单独启动，只保持一个模型驻留，模型加载耗时另存。
- 温度 0、seed 42、单槽位；文本按固定上下文扫描，禁用自动 fit 调整，配置和实际命令均记录。
- TTFT 从发出已就绪服务请求到首个非空内容片段；TPOT=(流结束时间-首片段时间)/(输出 token 数-1)，是 SSE 观测值。
- tok/s 是含 prefill 的端到端吞吐；原始 server timings 同时保留。拒绝用字符数代替 token 数。
- 普通矩阵请求关闭 prompt 复用；cache_study 单独测冷、热、保存重启恢复。加载时间和 GPU 采样开销不计入请求延迟。
- 显存是 nvidia-smi 采样的整卡占用，包含桌面/其他进程，且可能漏掉瞬时峰值；不是引擎独占显存或 OOM 边界。
- 没有额外预热请求；首次请求的 shader 编译/分配开销保留在原始数据中。配置按固定顺序运行，未随机交错，也未控制功率与热状态。
- 小型合成题集采用严格答案匹配，不代表通用模型能力；低样本 p95 不具有生产容量规划意义。
- 缺失指标保留 N/A；失败配置与无收益的结果都保留，不自动切换回基线掩盖失败。

## 原生缓存实验

```json
{
  "status": "ok",
  "cold": {
    "case": "long_prefix",
    "category": "cache",
    "answer": "{\"value\":7300}",
    "correct": true,
    "cache_prompt": true,
    "e2e_seconds": 0.6902220999982092,
    "ttft_seconds": 0.5659148999984609,
    "output_tokens": 9,
    "throughput_tokens_s": 13.03928112418213,
    "tpot_ms": 15.538399999968533,
    "server_decode_tokens_s": 64.45842834237094,
    "prefill_ms": 539.376,
    "cached_tokens": 0,
    "timings": {
      "cache_n": 0,
      "prompt_n": 1149,
      "prompt_ms": 539.376,
      "prompt_per_token_ms": 0.4694308093994778,
      "prompt_per_second": 2130.2393877369404,
      "predicted_n": 9,
      "predicted_ms": 124.111,
      "predicted_per_token_ms": 15.513875,
      "predicted_per_second": 64.45842834237094
    },
    "usage": {
      "completion_tokens": 9,
      "prompt_tokens": 1149,
      "total_tokens": 1158,
      "prompt_tokens_details": {
        "cached_tokens": 0
      }
    },
    "gpu_baseline_mib": 4769.0,
    "gpu_peak_mib": 4777.0,
    "gpu_sample_count": 4,
    "sampling_interval_seconds": 0.25,
    "speculative": {
      "draft_tokens": 0.0,
      "accepted_tokens": 0.0,
      "acceptance_rate": null,
      "accepted_per_position": [],
      "available": true
    }
  },
  "warm": {
    "case": "long_prefix",
    "category": "cache",
    "answer": "{\"value\":7300}",
    "correct": true,
    "cache_prompt": true,
    "e2e_seconds": 0.18868490000022575,
    "ttft_seconds": 0.0659148999984609,
    "output_tokens": 9,
    "throughput_tokens_s": 47.69857047378583,
    "tpot_ms": 15.346250000220607,
    "server_decode_tokens_s": 65.30825496342737,
    "prefill_ms": 62.289,
    "cached_tokens": 1145,
    "timings": {
      "cache_n": 1145,
      "prompt_n": 4,
      "prompt_ms": 62.289,
      "prompt_per_token_ms": 15.57225,
      "prompt_per_second": 64.21679590296841,
      "predicted_n": 9,
      "predicted_ms": 122.496,
      "predicted_per_token_ms": 15.312,
      "predicted_per_second": 65.30825496342737
    },
    "usage": {
      "completion_tokens": 9,
      "prompt_tokens": 1149,
      "total_tokens": 1158,
      "prompt_tokens_details": {
        "cached_tokens": 1145
      }
    },
    "gpu_baseline_mib": 4777.0,
    "gpu_peak_mib": 4777.0,
    "gpu_sample_count": 2,
    "sampling_interval_seconds": 0.25,
    "speculative": {
      "draft_tokens": 0.0,
      "accepted_tokens": 0.0,
      "acceptance_rate": null,
      "accepted_per_position": [],
      "available": true
    }
  },
  "save": {
    "id_slot": 0,
    "filename": "benchmark-prefix.bin",
    "n_saved": 1157,
    "n_written": 90636748,
    "timings": {
      "save_ms": 67.762
    }
  },
  "restore": {
    "id_slot": 0,
    "filename": "benchmark-prefix.bin",
    "n_restored": 1157,
    "n_read": 90636748,
    "timings": {
      "restore_ms": 29.874
    }
  },
  "restored": {
    "case": "long_prefix",
    "category": "cache",
    "answer": "{\"value\":7300}",
    "correct": true,
    "cache_prompt": true,
    "e2e_seconds": 0.6981097000025329,
    "ttft_seconds": 0.5724866000018665,
    "output_tokens": 9,
    "throughput_tokens_s": 12.891956665216577,
    "tpot_ms": 15.702887500083307,
    "server_decode_tokens_s": 63.577843121672096,
    "prefill_ms": 545.418,
    "cached_tokens": 0,
    "timings": {
      "cache_n": 0,
      "prompt_n": 1149,
      "prompt_ms": 545.418,
      "prompt_per_token_ms": 0.4746892950391645,
      "prompt_per_second": 2106.641144956712,
      "predicted_n": 9,
      "predicted_ms": 125.83,
      "predicted_per_token_ms": 15.72875,
      "predicted_per_second": 63.577843121672096
    },
    "usage": {
      "completion_tokens": 9,
      "prompt_tokens": 1149,
      "total_tokens": 1158,
      "prompt_tokens_details": {
        "cached_tokens": 0
      }
    },
    "gpu_baseline_mib": 4481.0,
    "gpu_peak_mib": 4504.0,
    "gpu_sample_count": 4,
    "sampling_interval_seconds": 0.25,
    "speculative": {
      "draft_tokens": 0.0,
      "accepted_tokens": 0.0,
      "acceptance_rate": null,
      "accepted_per_position": [],
      "available": true
    }
  },
  "reuse_verified": false,
  "restore_plus_ttft_seconds": 0.6023606000018664
}
```

## 可复核材料

- 完整模型/引擎/题集 SHA256、源码哈希、实际参数、响应、usage、timings、指标差值见原始 JSON。
- [实验图](benchmarks/results/overview.svg) / [交互结果页](benchmarks/results/dashboard.html)。
- [负面结果](NEGATIVE-RESULTS.md) / [简历与面试材料](RESUME.md)。
