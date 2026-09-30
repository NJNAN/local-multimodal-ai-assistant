param(
    [switch]$SmokeTest,
    [switch]$NoAudio,
    [switch]$NoVideo,
    [switch]$CheckEnvironment
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = $env:MMAI_PYTHON

Set-Location -LiteralPath $ProjectRoot
if ($Python) {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "MMAI_PYTHON points to a missing file: $Python"
    }
} else {
    $Candidates = @(
        (Join-Path $ProjectRoot '.venv\Scripts\python.exe'),
        'D:\mm_ai_env\Scripts\python.exe'
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $Python = $Candidate
            break
        }
    }
    if (-not $Python) {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($PythonCommand) { $Python = $PythonCommand.Source }
    }
    if (-not $Python) {
        throw 'Python not found. Create .venv or set MMAI_PYTHON to your python.exe.'
    }
}

# Keep this launcher ASCII for Windows PowerShell 5.1 encoding compatibility.
# Optional dependency/hardware diagnostics; model servers load on demand.
if ($CheckEnvironment) {
    & $Python 'scripts\00_check_env.py'
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Some checks failed. The assistant will try to start with available features.'
    }
}
$AssistantArgs = @('main.py')
if ($SmokeTest) { $AssistantArgs += '--smoke-test' }
if ($NoAudio) { $AssistantArgs += '--no-audio' }
if ($NoVideo) { $AssistantArgs += '--no-video' }
& $Python @AssistantArgs
exit $LASTEXITCODE
