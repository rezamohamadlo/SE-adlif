# One bounded user-authorized sequence using the existing official entrypoints.
# Wait for the existing20 stage; resume the candidate to40; freeze/test/report.
param([switch]$PreflightOnly)
$ErrorActionPreference = 'Stop'
$studyDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [IO.Path]::GetFullPath((Join-Path $studyDirectory '..\..\..\..'))
$pythonPath = Join-Path $repositoryRoot '.ven\Scripts\python.exe'
$evaluationRelative = 'results/experiments/dth_complement_v1/b001_rdrop_ssc_v1/evaluations/user_completed40_ssc_test_v1'
$statusPath = Join-Path $studyDirectory 'STAGE040_SEQUENCE_STATUS.json'
function Save-SequenceState($phase, $extra) {
    $record = @{phase=$phase;sequence_pid=$PID;updated_utc=[DateTime]::UtcNow.ToString('o');details=$extra}
    $record | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
}
function Invoke-ExistingEntrypoint($arguments, $prefix) {
    $process = Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repositoryRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $studyDirectory ($prefix+'_stdout.log')) -RedirectStandardError (Join-Path $studyDirectory ($prefix+'_stderr.log')) -PassThru
    Save-SequenceState $prefix @{python_launcher_pid=$process.Id}
    while (-not $process.WaitForExit(1000)) { }
    if ($process.ExitCode -ne 0) { throw "$prefix failed (exit $($process.ExitCode)); preserve all evidence and inspect logs. No automatic retry." }
}
if ($PreflightOnly) {
    if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Missing project virtual environment' }
    foreach ($name in @('FREEZE_SOURCE.py','EVALUATION_SOURCE.txt','REPORT_SOURCE.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot ($evaluationRelative+'/'+$name)))) { throw "Missing $name" }
    }
    Write-Output 'Paths verified; preflight performs no waiting, training, testing or writes.'
    exit 0
}
$lock = $null
try {
    $lock = [IO.File]::Open((Join-Path $studyDirectory '.user_stage040_sequence.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    Save-SequenceState 'waiting_for_existing20' @{target_completed_epochs=40;automatic_new_ideas=$false}
    # Retain handles to exact current study processes. Never stop another job.
    $existing = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'dth_complement_B001_ssc_v1|SSC_DTH_RDROP_SE_adLIF' })
    $handles = @()
    foreach ($item in $existing) {
        if ($item.CommandLine -match '(?:^|\s)(?:[^\s"]*[\\/])?runner\.py"?(?:\s|$)' -and $item.CommandLine -notmatch '--stage-ceiling 20') { throw 'Unexpected study launcher already active; refusing duplicate work' }
        try { $handles += Get-Process -Id $item.ProcessId -ErrorAction Stop } catch { }
    }
    foreach ($process in $handles) { while (-not $process.HasExited) { Start-Sleep -Seconds 5; $process.Refresh() } }
    $receiptPath=Join-Path $studyDirectory 'runs/DTH_RDROP_SE_adLIF/SSC/seed_42/stages/020/RESULT.json'
    if (-not (Test-Path -LiteralPath $receiptPath)) { throw 'No completed20 stage receipt; do not restart an interrupted job automatically' }
    $receipt=Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.status -ne 'stage_complete' -or $receipt.completed_epochs -ne 20) { throw 'Existing20 stage incomplete' }
    $remaining=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'dth_complement_B001_ssc_v1|SSC_DTH_RDROP_SE_adLIF' })
    if ($remaining.Count -gt 0) { throw 'Study process still active; refusing duplicate launch' }
    $metrics=@(Get-ChildItem -LiteralPath (Join-Path $studyDirectory 'runs/DTH_RDROP_SE_adLIF/SSC/seed_42') -Filter 'metrics.csv' -Recurse -File)
    if ($metrics.Count -ne 1) { throw 'Ambiguous metric history before40 resume' }
    $historyPath=Join-Path $studyDirectory 'METRICS_BEFORE_STAGE040.csv'
    if (Test-Path -LiteralPath $historyPath) { throw 'Existing40 preparation history; inspect before repeating sequence' }
    Copy-Item -LiteralPath $metrics[0].FullName -Destination $historyPath
    Invoke-ExistingEntrypoint '-u runner.py --models DTH_RDROP_SE_adLIF --datasets SSC --seeds 42 --study-file config/studies/dth_complement_B001_ssc_v1.yaml --stage-ceiling 40' 'stage040'
    foreach ($step in @(@('FREEZE_SOURCE.py','stage040_freeze'),@('EVALUATION_SOURCE.txt','stage040_test'),@('REPORT_SOURCE.py','stage040_report'))) {
        $code="import runpy; runpy.run_path('$evaluationRelative/$($step[0])',run_name='__main__')"
        Invoke-ExistingEntrypoint ('-u -c "'+$code+'"') $step[1]
    }
    Save-SequenceState 'complete_training_and_test_report' @{report=($evaluationRelative+'/REPORT_FA.md');no_more_work_scheduled=$true}
} catch {
    Save-SequenceState 'stopped_on_error_no_retry' @{error=$_.Exception.Message}
    throw
} finally {
    if ($null -ne $lock) { $lock.Dispose() }
}
