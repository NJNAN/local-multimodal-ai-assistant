param(
    [switch]$SmokeTest,
    [switch]$NoAudio,
    [switch]$NoVideo,
    [switch]$CheckEnvironment
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $ProjectRoot 'start_assistant.ps1') @PSBoundParameters
exit $LASTEXITCODE
