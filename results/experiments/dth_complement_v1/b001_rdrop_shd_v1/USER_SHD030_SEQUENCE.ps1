# Bounded user-authorized transfer study using the existing training runner.
param([switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$studyDirectory=Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot=[IO.Path]::GetFullPath((Join-Path $studyDirectory '..\..\..\..'))
$pythonPath=Join-Path $repositoryRoot '.ven\Scripts\python.exe'
$evaluationRelative='results/experiments/dth_complement_v1/b001_rdrop_shd_v1/evaluations/user_shd030_test_v1'
$statusPath=Join-Path $studyDirectory 'SEQUENCE_STATUS.json'
function Save-SequenceState($phase,$details) {
    @{phase=$phase;sequence_pid=$PID;updated_utc=[DateTime]::UtcNow.ToString('o');details=$details} | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
}
if ($PreflightOnly) {
    if (-not (Test-Path -LiteralPath $pythonPath)) {throw 'Missing project venv'}
    foreach ($name in @('FREEZE_SOURCE.py','EVALUATION_SOURCE.txt','REPORT_SOURCE.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot ($evaluationRelative+'/'+$name)))) {throw "Missing $name"}
    }
    foreach ($name in @('shd_train.h5','shd_test.h5')) {
        if (-not (Test-Path -LiteralPath (Join-Path $repositoryRoot ('data/SHDWrapper/'+$name)))) {throw "Missing exact local $name"}
    }
    Write-Output 'Portable paths and inputs verified; no training, testing or output writes.'
    exit 0
}
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class StudyProcessExit { [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GetExitCodeProcess(IntPtr handle,out uint exitCode); }'
function Invoke-ExistingEntrypoint($arguments,$prefix) {
    $process=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repositoryRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $studyDirectory ($prefix+'_stdout.log')) -RedirectStandardError (Join-Path $studyDirectory ($prefix+'_stderr.log')) -PassThru
    $processHandle=$process.Handle
    Save-SequenceState $prefix @{python_launcher_pid=$process.Id}
    while (-not $process.WaitForExit(1000)) { }
    [uint32]$exitCode=0
    if (-not [StudyProcessExit]::GetExitCodeProcess($processHandle,[ref]$exitCode)) {throw "Cannot verify $prefix exit"}
    if ($exitCode -ne 0) {throw "$prefix exited $exitCode; preserve evidence; no automatic restart"}
}
$lock=$null
try {
    $lock=[IO.File]::Open((Join-Path $studyDirectory '.user_shd030_sequence.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    $existing=@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'dth_complement_B001_(ssc|shd)_v1|experiment=(SSC|SHD)_DTH_RDROP_SE_adLIF' })
    if ($existing.Count -gt 0) {throw 'Related training still active; refusing duplicate GPU work'}
    Invoke-ExistingEntrypoint '-u runner.py --models DTH_SE_adLIF DTH_RDROP_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/dth_complement_B001_shd_v1.yaml --stage-ceiling 30' 'stage030'
    foreach ($step in @(@('FREEZE_SOURCE.py','freeze030'),@('EVALUATION_SOURCE.txt','test030'),@('REPORT_SOURCE.py','report030'))) {
        $code="import runpy; runpy.run_path('$evaluationRelative/$($step[0])',run_name='__main__')"
        Invoke-ExistingEntrypoint ('-u -c "'+$code+'"') $step[1]
    }
    Save-SequenceState 'completed_training_tests_report' @{report='CROSS_DATASET_TEST_REPORT_FA.md';no_more_work_scheduled=$true}
} catch {
    Save-SequenceState 'stopped_on_error_no_retry' @{error=$_.Exception.Message}
    throw
} finally {
    if ($null -ne $lock) {$lock.Dispose()}
}
