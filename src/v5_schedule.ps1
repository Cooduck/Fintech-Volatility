param(
    [ValidateSet('Install','Predict','Verify')]
    [string]$Action = 'Install'
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $root '.venv/Scripts/python.exe'
$watcher = Join-Path $PSScriptRoot 'v4_forward_watcher.py'
$logs = Join-Path $root '.log'
New-Item -ItemType Directory -Path $logs -Force | Out-Null

if ($Action -ne 'Install') {
    $flag = if ($Action -eq 'Predict') { '--run-once' } else { '--verify' }
    $logFile = Join-Path $logs ("v5_forward_" + $Action.ToLowerInvariant() + ".log")
    Add-Content -LiteralPath $logFile -Encoding utf8 -Value ([DateTimeOffset]::Now.ToString('o'))
    & $python $watcher --phase a --revision v5 $flag *>&1 | Out-File -LiteralPath $logFile -Encoding utf8 -Append
    exit $LASTEXITCODE
}

$calendar = @"
import json, sys
import exchange_calendars as xc
import pandas as pd
from pathlib import Path

sys.path.insert(0,str(Path.cwd()/"src"))
from v4_forward_watcher import load_freeze

manifest,_=load_freeze("a","v5")
instant=max(pd.Timestamp.now(tz="UTC"), pd.Timestamp(manifest["created_utc"]))
today=instant.tz_convert("America/New_York").tz_localize(None).normalize()
schedule=xc.get_calendar("XNYS",start=today,end=today+pd.Timedelta(days=55)).schedule
dates=[]
for i,row in enumerate(schedule.itertuples(index=False)):
    pred=row.open-pd.Timedelta(minutes=8)
    if pred<=instant:
        continue
    next_session=schedule.index[i+1].strftime("%Y-%m-%d")
    verify=pd.Timestamp(next_session+" 16:30",tz="America/New_York").tz_convert("UTC")
    dates.append(dict(predict=pred.isoformat(),verify=verify.isoformat()))
    if len(dates)==20:
        break
assert len(dates)==20
print(json.dumps(dates))
"@
Push-Location -LiteralPath $root
try {
    $rows = ($calendar | & $python -) | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $rows.Count -ne 20) { throw 'Could not resolve 20 exchange sessions.' }
} finally {
    Pop-Location
}

$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
$schedulerScript = Join-Path $PSScriptRoot 'v5_schedule.ps1'
$shell = Join-Path $env:WINDIR 'System32/WindowsPowerShell/v1.0/powershell.exe'
if (-not (Test-Path -LiteralPath $shell -PathType Leaf)) { throw 'Windows PowerShell executable missing.' }
foreach($kind in @('Predict','Verify')) {
    $name = "FintechVol_Forward_$kind"
    $prior = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
    if ($prior -and $prior.Actions.Arguments -notlike "*$schedulerScript*") {
        throw "Existing task $name has a different action."
    }
    $trigger = foreach ($row in $rows) {
        $at = [DateTimeOffset]::Parse($row.($kind.ToLowerInvariant())).LocalDateTime
        New-ScheduledTaskTrigger -Once -At $at
    }
    $actionArguments = "-NoLogo -NoProfile -File `"$schedulerScript`" -Action $kind"
    $taskAction = New-ScheduledTaskAction -Execute $shell -Argument $actionArguments -WorkingDirectory $root
    Register-ScheduledTask -TaskName $name -Action $taskAction -Trigger $trigger -Principal $principal -Settings $settings -Description 'Fintech Volatility V5 forward research, twenty exchange sessions' -Force | Out-Null
    Get-ScheduledTaskInfo -TaskName $name | Select-Object TaskName,NextRunTime
}

