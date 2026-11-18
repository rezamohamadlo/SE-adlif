# Orchestration only: runner.py owns every training and resume operation.
param([switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$studyRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot=[IO.Path]::GetFullPath((Join-Path $studyRoot '..\..\..\..'))
$pythonPath=$null
foreach ($name in @('.ven/Scripts/python.exe','.venv/Scripts/python.exe')) {
    $candidate=Join-Path $repositoryRoot $name
    if (Test-Path -LiteralPath $candidate) {$pythonPath=$candidate;break}
}
if ($null -eq $pythonPath) {throw 'Missing project virtual environment'}
$helper='results/experiments/dth_complement_v1/direction_aug_shd100_v1/AUGMENTATION_ACTIONS.py'
$training='-u runner.py --models DIRECTION_AUG_DTH_RDROP_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/dth_direction_aug_shd100_v1.yaml --stage-ceiling 100'
if ($PreflightOnly) {
    Push-Location $repositoryRoot
    try { & $pythonPath $helper verify; if ($LASTEXITCODE -ne 0) {throw 'Source verification failed'} }
    finally {Pop-Location}
    Write-Output 'Portable paths and frozen evidence verified; no training.'
    exit 0
}
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class DirectionStudyExit { [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GetExitCodeProcess(IntPtr handle,out uint exitCode); }'
function Save-State($phase,$details) {
    @{phase=$phase;sequence_pid=$PID;updated_utc=[DateTime]::UtcNow.ToString('o');details=$details} | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $studyRoot 'SEQUENCE_STATUS.json')
}
function Invoke-Existing($arguments,$prefix) {
    $process=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repositoryRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $studyRoot ($prefix+'_stdout.log')) -RedirectStandardError (Join-Path $studyRoot ($prefix+'_stderr.log')) -PassThru
    $processHandle=$process.Handle
    Save-State $prefix @{python_launcher_pid=$process.Id}
    while (-not $process.WaitForExit(1000)) { }
    [uint32]$exitCode=0
    if (-not [DirectionStudyExit]::GetExitCodeProcess($processHandle,[ref]$exitCode)) {throw "Cannot verify $prefix exit"}
    if ($exitCode -ne 0) {throw "$prefix failed with exit $exitCode; preserve evidence; no automatic retry"}
}
$lock=$null
try {
    $lock=[IO.File]::Open((Join-Path $studyRoot '.sequence.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    if (Test-Path -LiteralPath (Join-Path $studyRoot 'evaluation/ALL_RESULTS.json')) {throw 'Final test already exists; do not rerun'}
    $existing=@(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -match '(?:^|\s)(?:[^\s"]*[\\/])?(?:run|runner|tune_shd|search_shd|pilot)\.py"?(?:\s|$)'
    })
    if ($existing.Count -gt 0) {throw 'Other training/queue exists; do not stop or duplicate it'}
    Invoke-Existing ('-u '+$helper+' verify') 'verify'
    Invoke-Existing ($training+' --dry-run') 'preflight'
    Invoke-Existing $training 'training'
    Invoke-Existing ('-u '+$helper+' freeze') 'freeze'
    Invoke-Existing ('-u '+$helper+' evaluate') 'evaluation'
    Invoke-Existing ('-u '+$helper+' report') 'report'
    Save-State 'complete_training_test_report' @{report='COMPARISON_FA.md';no_further_work_scheduled=$true}
} catch {
    Save-State 'stopped_on_error_no_retry' @{error=$_.Exception.Message}
    throw
} finally {if ($null -ne $lock) {$lock.Dispose()}}
