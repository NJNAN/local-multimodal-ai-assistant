param([switch]$SmokeTest)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "D:\mm_ai_env\Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python"
}

# llama.cpp（llama-server）与两个 GGUF 模型由环境检查统一验证。
# 运行时按需加载模型，无需预先启动任何服务进程。
& $Python "scripts\00_check_env.py"
if ($LASTEXITCODE -ne 0) {
    throw "Environment check failed. Review the messages above."
}
if ($SmokeTest) {
    & $Python "main.py" "--smoke-test"
} else {
    & $Python "main.py"
}
