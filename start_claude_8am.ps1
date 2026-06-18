$ErrorActionPreference = "Stop"

Set-Location "D:\Aegra"

$logDir = "D:\Aegra\runs\claude-start"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$ts = Get-Date -Format "yyyyMMdd-HHmmss"

$prompt = "Reply with one sentence only: Claude Code daily work window has started."

& claude -p $prompt *> "$logDir\$ts.txt"
