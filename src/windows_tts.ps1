# Persistent local SAPI worker. JSON is the only protocol; no console windows.
$ErrorActionPreference = 'Stop'
[Console]::InputEncoding = New-Object System.Text.UTF8Encoding($false)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$voice = New-Object -ComObject SAPI.SpVoice
$tokens = @{}
foreach ($root in @('HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices', 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices')) {
    try {
        $category = New-Object -ComObject SAPI.SpObjectTokenCategory
        $category.SetId($root, $false)
        foreach ($entry in @(@('zh','804'), @('en','409'))) {
            $list = $category.EnumerateTokens("Language=$($entry[1])", '')
            if ($list.Count -gt 0 -and -not $tokens.ContainsKey($entry[0])) {
                $tokens[$entry[0]] = $list.Item(0)
            }
        }
    } catch { }
}
@{ready=$true; zh=$tokens.ContainsKey('zh'); en=$tokens.ContainsKey('en')} | ConvertTo-Json -Compress
while ($null -ne ($line = [Console]::ReadLine())) {
    $stream = $null
    try {
        $request = $line | ConvertFrom-Json
        if (-not $tokens.ContainsKey($request.language)) { throw 'Requested voice is unavailable' }
        $voice.Voice = $tokens[$request.language]
        $stream = New-Object -ComObject SAPI.SpFileStream
        $stream.Open($request.path, 3, $false)
        $voice.AudioOutputStream = $stream
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $null = $voice.Speak($request.text, 0)
        $timer.Stop()
        $stream.Close()
        $stream = $null
        @{ok=$true; seconds=$timer.Elapsed.TotalSeconds; voice=$voice.Voice.GetDescription()} | ConvertTo-Json -Compress
    } catch {
        @{ok=$false; error=$_.Exception.Message} | ConvertTo-Json -Compress
    } finally {
        if ($null -ne $stream) { $stream.Close() }
    }
}
