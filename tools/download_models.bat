@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

rem 查找 aria2c（优先 PATH，其次 winget 安装目录）
set "ARIA="
for /f "delims=" %%i in ('where aria2c 2^>nul') do if not defined ARIA set "ARIA=%%i"
if not defined ARIA set "ARIA=%LOCALAPPDATA%\Microsoft\WinGet\Packages\aria2.aria2_Microsoft.Winget.Source_8wekyb3d8bbwe\aria2-1.37.0-win-64bit-build1\aria2c.exe"
if not exist "%ARIA%" (
  echo [错误] 未找到 aria2c。请先执行: winget install aria2.aria2
  pause
  exit /b 1
)

echo ============================================================
echo  模型下载（支持断点续传，可反复运行；中途 Ctrl+C 可暂停）
echo  文本模型 Qwen3.5-4B IQ4_XS
echo ============================================================
"%ARIA%" -x8 -s8 -k 1M --file-allocation=none -c --retry-wait=10 --max-tries=0 --check-certificate=false --lowest-speed-limit=5000 --summary-interval=30 --console-log-level=warn --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -d models/qwen3.5-4b -o "Qwen_Qwen3.5-4B-IQ4_XS.gguf" "https://modelscope.cn/api/v1/models/bartowski/Qwen_Qwen3.5-4B-GGUF/repo?Revision=master&FilePath=Qwen_Qwen3.5-4B-IQ4_XS.gguf"

echo ============================================================
echo  视觉模型主文件 Qwen3VL-4B-Instruct-Q4_K_M
echo ============================================================
"%ARIA%" -x8 -s8 -k 1M --file-allocation=none -c --retry-wait=10 --max-tries=0 --check-certificate=false --lowest-speed-limit=5000 --summary-interval=30 --console-log-level=warn --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -d models/qwen3-vl-4b -o "Qwen3VL-4B-Instruct-Q4_K_M.gguf" "https://modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct-GGUF/repo?Revision=master&FilePath=Qwen3VL-4B-Instruct-Q4_K_M.gguf"

echo ============================================================
echo  视觉 mmproj Q8_0
echo ============================================================
"%ARIA%" -x8 -s8 -k 1M --file-allocation=none -c --retry-wait=10 --max-tries=0 --check-certificate=false --lowest-speed-limit=5000 --summary-interval=30 --console-log-level=warn --user-agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)" -d models/qwen3-vl-4b -o "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf" "https://modelscope.cn/api/v1/models/Qwen/Qwen3-VL-4B-Instruct-GGUF/repo?Revision=master&FilePath=mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"

echo.
echo ============================================================
echo  校验 SHA256
echo ============================================================
if exist "D:\mm_ai_env\Scripts\python.exe" (
  "D:\mm_ai_env\Scripts\python.exe" tools\verify_models.py
) else (
  python tools\verify_models.py
)
pause
endlocal
