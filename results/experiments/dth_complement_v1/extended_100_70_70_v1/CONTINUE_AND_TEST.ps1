# Existing official runner; audited full-state continuations, then frozen tests.
param([switch]$PreflightOnly)
$ErrorActionPreference='Stop'
$baseDirectory=Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot=[IO.Path]::GetFullPath((Join-Path $baseDirectory '..\..\..\..'))
$pythonPath=$null
foreach ($name in @('.ven/Scripts/python.exe','.venv/Scripts/python.exe')) {
    $candidate=Join-Path $repositoryRoot $name
    if (Test-Path -LiteralPath $candidate) {$pythonPath=$candidate;break}
}
$actionRelative='results/experiments/dth_complement_v1/extended_100_70_70_v1/ACTIONS.py'
$studyRelative='config/studies/dth_complement_extended_100_70_70_v1.yaml'
$statusPath=Join-Path $baseDirectory 'SEQUENCE_STATUS.json'
function Save-State($phase,$details) {
    @{phase=$phase;sequence_pid=$PID;updated_utc=[DateTime]::UtcNow.ToString('o');details=$details} | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
}
if ($PreflightOnly) {
    if ($null -eq $pythonPath) {throw 'Project virtual environment missing'}
    foreach ($dataset in @('SHD','ECG','SSC')) {
        $checkpoint=Join-Path $baseDirectory ('runs/DTH_RDROP_SE_adLIF/'+$dataset+'/seed_42/ckpt/last.ckpt')
        if (-not (Test-Path -LiteralPath $checkpoint)) {throw "Missing prepared $dataset checkpoint"}
    }
    foreach ($name in @('CONTINUATION_PLAN.json','SOURCE_SNAPSHOT.json','DATA_IDENTITIES.json','ACTIONS.py')) {
        if (-not (Test-Path -LiteralPath (Join-Path $baseDirectory $name))) {throw "Missing $name"}
    }
    Write-Output 'Portable paths, venv and prepared checkpoints verified; no training, tests or writes.'
    exit 0
}
Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public class ExtendedStudyExit { [DllImport("kernel32.dll",SetLastError=true)] public static extern bool GetExitCodeProcess(IntPtr handle,out uint exitCode); }'
function Invoke-Existing($arguments,$prefix) {
    $process=Start-Process -FilePath $pythonPath -ArgumentList $arguments -WorkingDirectory $repositoryRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $baseDirectory ($prefix+'_stdout.log')) -RedirectStandardError (Join-Path $baseDirectory ($prefix+'_stderr.log')) -PassThru
    $processHandle=$process.Handle
    Save-State $prefix @{python_launcher_pid=$process.Id}
    while (-not $process.WaitForExit(1000)) { }
    [uint32]$exitCode=0
    if (-not [ExtendedStudyExit]::GetExitCodeProcess($processHandle,[ref]$exitCode)) {throw "Cannot verify $prefix exit"}
    if ($exitCode -ne 0) {throw "$prefix failed with exit $exitCode; preserve evidence; no automatic retry"}
}
function Invoke-Action($action,$dataset) {
    $arguments="['ACTIONS.py','$action'"
    if ($dataset) {$arguments+=",'$dataset'"}
    $arguments+=']'
    $code="import runpy,sys; sys.argv=$arguments; runpy.run_path('$actionRelative',run_name='__main__')"
    Invoke-Existing ('-u -c "'+$code+'"') ($action+'_'+$dataset)
}
$lock=$null
try {
    $lock=[IO.File]::Open((Join-Path $baseDirectory '.continuation_sequence.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
    foreach ($job in @(@('SHD',100),@('ECG',70),@('SSC',70))) {
        $dataset=$job[0];$target=$job[1]
        $existing=@(Get-CimInstance Win32_Process | Where-Object {
            $_.Name -eq 'python.exe' -and $_.CommandLine -match '(?:^|\s)(?:[^\s"]*[\\/])?(?:run|runner|tune_shd|search_shd|pilot)\.py"?(?:\s|$)' -and $_.CommandLine -notmatch 'device=cpu'
        })
        if ($existing.Count -gt 0) {throw 'A training/queue process is already active; do not stop unrelated work or duplicate GPU training'}
        Invoke-Action 'memory' $dataset
        $trainingArguments="-u runner.py --models DTH_RDROP_SE_adLIF --datasets $dataset --seeds 42 --study-file $studyRelative --stage-ceiling $target"
        Invoke-Existing ($trainingArguments+' --dry-run') ('preflight_'+$dataset)
        Invoke-Existing $trainingArguments ('training_'+$dataset)
        Invoke-Action 'freeze' $dataset
        Invoke-Action 'evaluate' $dataset
        Invoke-Action 'report' $null
    }
    Save-State 'complete_all_three_continuations_tests_report' @{report='UPDATED_TEST_TABLE_FA.md';no_further_work_scheduled=$true}
} catch {
    Save-State 'stopped_on_error_no_retry' @{error=$_.Exception.Message}
    throw
} finally {
    if ($null -ne $lock) {$lock.Dispose()}
}
