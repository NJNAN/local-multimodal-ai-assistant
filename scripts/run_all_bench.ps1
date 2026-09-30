param(
    [ValidateSet('text', 'vision', 'all')][string]$Suite = 'all',
    [int]$Repeats = 3,
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe')
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { $Python = 'python' }
& $Python (Join-Path $PSScriptRoot '14_inference_bench.py') --suite $Suite --repeats $Repeats
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python (Join-Path $PSScriptRoot '15_render_benchmarks.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python (Join-Path $PSScriptRoot '16_check_benchmarks.py')
exit $LASTEXITCODE
