# 8GB 推理实验台

目标是在消费级 GPU 上测量推理配置的收益边界，同时保留桌面助手的文字、语音、视觉、RAG 与六种手势控制。这里研究引擎集成与服务化，不宣称发明推理算法。

## 一条命令复现

准备 README 中的模型与 llama-server，安装 `requirements-bench.txt`，关闭其他推理工作负载后执行：

```powershell
.\scripts\run_all_bench.ps1 -Suite all -Repeats 3
```

默认 Python 为项目 `.venv`；可传 `-Python` 指定解释器。脚本兼容 Windows PowerShell 5.1，不需要 Linux。使用 18080/18081 实验端口；占用时报错，不回收用户进程。实验进程退出后释放模型。若用户正在运行另一套模型，整卡显存和性能会受干扰，应关闭该负载后重测。

分阶段运行或查看计划：

```powershell
python scripts/14_inference_bench.py --plan-only
python scripts/14_inference_bench.py --suite text --repeats 3
python scripts/14_inference_bench.py --suite all --profile text_f16_8192 --profile text_ngram-simple_8192 --repeats 5
python scripts/15_render_benchmarks.py
python scripts/16_check_benchmarks.py
```

单次运行覆盖 `latest.json`；历史实验需要传 `--output outputs/my_run.json` 保存。渲染和校验也支持 `--input`。原始逐题记录先落盘，报告和图从 JSON 生成。设备日志与 KV 二进制留在本地，公开材料不包含摄像头照片或私有知识库。

## 实验与判据

|方向|实现|成功判据|
|---|---|---|
|KV 量化|f16 / q8_0 / q4_0 × 4096 / 8192 上下文|延迟、整卡显存采样峰值与严格正确率同时报告|
|无 draft 模型的 N-gram|`ngram-simple` / `ngram-cache` 与相同上下文 f16 基线对照|草稿 token 增量、接受率与真实吞吐；未产生草稿不算加速|
|Prefix cache|同一长前缀的冷请求、热请求、保存、重启、恢复|`n_restored` 与下一请求 `cache_n` 都要检查；保存成功不等于复用成功|
|视觉预算|默认预算 / 256 / 1024 上限，固定合成图|颜色、数量和位置答案严格匹配，同时报告 SSE TTFT|
|可观测性|原生 `/metrics` 差值、usage、timings、UI 监控|未知保持 N/A，计数器重置不产生负接受率|
|证据校验|确定性报告再生成与题集 SHA256 校验|篡改表格数字或换题集时 checker 失败|

事实题、结构化提取、重复性中文复制分层记录。每题温度 0、seed 42、同一输入重复三次。错误答案保留，不删除失败配置；严格正确率同时衡量内容和格式，不是泛化模型能力。当前上下文矩阵只验证 4K/8K，不等于测出了最大上下文。

普通矩阵禁用 prompt 复用，另做缓存实验。模型加载时间独立记录；请求 TTFT 不包含加载。固定单槽和显式 `--fit off` 避免引擎自行改变上下文。记录实际命令和引擎资源日志，N-gram 可能改变 SSE 分块，观测 TPOT 需要连同原生 timings 解读。

显存每 0.25 秒通过 nvidia-smi 采样，包含桌面及其他进程；峰值可能漏采。无 GPU 指标时不记为 0。p50/p95 是有限样本的描述统计，不做生产容量保证。

## 助手中开启实验参数

默认仍用原来的 f16 KV、自动 Flash Attention、无投机。高级参数只对明确配置的模型生效；请求的开关不在本机 `--help` 中会报错，避免静默失效。

```powershell
$env:MMAI_TEXT_CTK = 'q8_0'
$env:MMAI_TEXT_CTV = 'q8_0'
$env:MMAI_TEXT_FLASH_ATTN = 'on'
$env:MMAI_TEXT_SPEC_TYPE = 'ngram-simple'
$env:MMAI_TEXT_PARALLEL = '1'
$env:MMAI_VL_IMAGE_MIN_TOKENS = '64'
$env:MMAI_VL_IMAGE_MAX_TOKENS = '256'
python main.py
```

其他可选项：`MMAI_TEXT_CACHE_REUSE`、`MMAI_TEXT_SLOT_SAVE_PATH`，以及相应的 `MMAI_VL_*`。缓存目录在 Windows 上应使用可写的 ASCII 路径。本机测试引擎不能识别中文 slot-save-path，因此基准自动使用系统临时目录；KV 文件退出时清理，避免私有上下文进入仓库。保存、恢复仅针对空闲的运行中服务；in-flight 时拒绝操作。

取消环境变量或重开终端即可回到默认配置。矩阵基准不继承这些实验参数，避免环境污染。桌面“推理监控”只读状态和指标，不会为了刷新而加载模型；对话 JSON 同时包含新增的真实 token 与引擎 timing 信息。

## 版本与来源

本机引擎版本、GPU/驱动、模型和可执行文件完整 SHA256、被测源码哈希、题集版本均在 `results/latest.json`。模型路径遵循项目配置，不硬编码到公共说明中。

`.gitattributes` 固定被测文件的换行方式，避免 Windows/Linux checkout 自动转换导致源码或题集哈希漂移。checker 同时核对当前被测源码；修改推理或测量实现后应重跑，而不是手改报告数值。

参数和端点以 [llama.cpp 官方 server 文档](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) 及本机 `--help` 为准。`--lookup-cache-dynamic` 需要文件名；本实现使用显式 `--spec-type`，没有将 `-lcd` 当布尔开关。没有额外 draft 模型不代表总内存或显存开销严格为零。

原有独占模型管理器继续承担应用的切换、空闲卸载和退出清理。官方 router 对照实验尚未运行，不能把现有管理器描述成已优于官方网关；连续批处理默认开启也不作为新成果。

CI 分为 Windows 无模型回归和无 GPU 证据校验。GPU 实验在本机运行，CI 不编造性能数字；当前配置文件尚需推送后才能在 GitHub 验证。
