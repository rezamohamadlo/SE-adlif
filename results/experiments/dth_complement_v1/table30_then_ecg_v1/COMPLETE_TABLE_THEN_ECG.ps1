# Explicit user request: finish the four30epoch-window test cells, then ECG.
# Main training always uses runner.py. No searches or automatic retries.
param([switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$taskDirectory=Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot=[IO.Path]::GetFullPath((Join-Path $taskDirectory '..\..\..\..'))
$pythonPath=Join-Path $repositoryRoot '.ven\Scripts\python.exe'
$actionsRelative='results/experiments/dth_complement_v1/table30_then_ecg_v1/ACTIONS.py'
$statusPath=Join-Path $taskDirectory 'MASTER_STATUS.json'
function Save-State($phase,$details) {
    @{phase=$phase;sequence_pid=$PID;updated_utc=[DateTime]::UtcNow.ToString('o');details=$details} | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
}
if ($PreflightOnly) {
    if (-not (Test-Path -LiteralPath $pythonPath)) {throw 'Missing project venv'}
    foreach ($name in @('ACTIONS.py','ECG_DTH_RDROP_SE_adLIF_2layer.template.yaml','dth_complement_B001_ecg30_v1.template.yaml','USER_SCOPE.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $taskDirectory $name))) {throw "Missing $name"}
    }
    foreach ($name in @('data/QTDB/QTDB_train.mat','data/QTDB/QTDB_test.mat')) {
        if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot $name))) {throw "Missing exact $name"}
    }
    Write-Output 'Portable paths and prerequisite files verified; no training, testing, changes or waiting.'
    exit 0
}
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class TableStudyExit { [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GetExitCodeProcess(IntPtr handle,out uint exitCode); }'
function Invoke-Existing($arguments,$prefix) {
    $process=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repositoryRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskDirectory ($prefix+'_stdout.log')) -RedirectStandardError (Join-Path $taskDirectory ($prefix+'_stderr.log')) -PassThru
    $processHandle=$process.Handle
    Save-State $prefix @{python_launcher_pid=$process.Id}
    while (-not $process.WaitForExit(1000)) { }
    [uint32]$exitCode=0
    if (-not [TableStudyExit]::GetExitCodeProcess($processHandle,[ref]$exitCode)) {throw "Cannot verify $prefix exit"}
    if ($exitCode -ne 0) {throw "$prefix failed with exit $exitCode; preserve evidence; no automatic retry"}
}
function Invoke-Action($action) {
    $code="import runpy,sys; sys.argv=['ACTIONS.py','$action']; runpy.run_path('$actionsRelative',run_name='__main__')"
    Invoke-Existing ('-u -c "'+$code+'"') $action
}
$lock=$null
try {
    $lock=[IO.File]::Open((Join-Path $taskDirectory '.table_then_ecg.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    Save-State 'waiting_existing_SHD30_tests' @{existing_sequence_pid=22392;target='four complete official-test cells then ECG30'}
    $shdStatus=Join-Path $repositoryRoot 'results/experiments/dth_complement_v1/b001_rdrop_shd_v1/SEQUENCE_STATUS.json'
    $existing=Get-CimInstance Win32_Process -Filter 'ProcessId=22392'
    if ($null -ne $existing) {
        if ($existing.CommandLine -notmatch 'USER_SHD030_SEQUENCE.ps1') {throw 'Recorded SHD PID belongs to a different job; do not touch it'}
        $process=Get-Process -Id 22392
        while (-not $process.HasExited) {Start-Sleep -Seconds 5;$process.Refresh()}
    }
    $finished=Get-Content -LiteralPath $shdStatus -Raw | ConvertFrom-Json
    if ($finished.phase -ne 'completed_training_tests_report') {throw 'Existing SHD sequence did not finish successfully; inspect before continuing'}
    $related=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'dth_complement_B001_(ssc|shd|ecg30)_v1|experiment=(SSC|SHD|ECG)_DTH_RDROP_SE_adLIF' })
    if ($related.Count -gt 0) {throw 'Related job remains active; refusing concurrent training'}
    Invoke-Action 'prepare'
    Invoke-Existing '-u runner.py --models DTH_SE_adLIF --datasets SSC --seeds 42 --study-file config/studies/dth_complement_B001_ssc_v1.yaml --stage-ceiling 30 --dry-run' 'ssc30_preflight'
    Invoke-Existing '-u runner.py --models DTH_SE_adLIF --datasets SSC --seeds 42 --study-file config/studies/dth_complement_B001_ssc_v1.yaml --stage-ceiling 30' 'ssc_control30'
    Invoke-Action 'freeze_ssc'
    $code="import runpy; runpy.run_path('results/experiments/dth_complement_v1/table30_then_ecg_v1/ssc_control30_test/EVALUATION_SOURCE.txt',run_name='__main__')"
    Invoke-Existing ('-u -c "'+$code+'"') 'ssc_control30_test'
    Invoke-Action 'table'
    Save-State 'four_test_cells_complete_before_ECG' @{report='TEST_TABLE_30_FA.md';results='TABLE_RESULTS.json'}
    Invoke-Action 'calibrate_ecg'
    Invoke-Existing '-u runner.py --models DTH_RDROP_SE_adLIF --datasets ECG --seeds 42 --study-file config/studies/dth_complement_B001_ecg30_v1.yaml --stage-ceiling 30 --dry-run' 'ecg30_preflight'
    Invoke-Existing '-u runner.py --models DTH_RDROP_SE_adLIF --datasets ECG --seeds 42 --study-file config/studies/dth_complement_B001_ecg30_v1.yaml --stage-ceiling 30' 'ecg30_training'
    Invoke-Action 'freeze_ecg'
    Invoke-Action 'evaluate_ecg'
    Invoke-Action 'final_report'
    Save-State 'complete_all_requested_tests_and_ECG' @{report='FINAL_REPORT_FA.md';no_further_work_scheduled=$true}
} catch {
    Save-State 'stopped_on_error_no_retry' @{error=$_.Exception.Message}
    throw
} finally {
    if ($null -ne $lock) {$lock.Dispose()}
}
