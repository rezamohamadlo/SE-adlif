<#
Windows/NVIDIA bootstrap. No training, data downloads, result merges or system installs.
Run from any directory; paths are resolved against this script's repository.
#>
[CmdletBinding()]
param(
    [ValidateSet('.ven', '.venv')][string]$VenvName = '.venv',
    [string]$PythonExe = '',
    [string]$ExpectedCommit = '',
    [switch]$PlanOnly,
    [switch]$VerifyOnly,
    [switch]$RunTests
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$venvRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot $VenvName))
if (-not $venvRoot.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Virtual environment must stay inside this checkout.'
}
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'
$requirements = Join-Path $repoRoot 'requirements.txt'
if (-not (Test-Path -LiteralPath $requirements)) { throw 'requirements.txt is missing.' }
if ($PlanOnly -and $VerifyOnly) { throw 'Choose PlanOnly or VerifyOnly.' }
if ($PlanOnly) {
    Write-Host "Repository: $repoRoot"
    Write-Host "Environment: $venvRoot (Windows x64, Python 3.11)"
    Write-Host 'Install pinned pip 26.2.1, then requirements.txt with verbose output/progress bars and a pip log.'
    Write-Host 'Check exact package pins, pip dependencies, CUDA execution and all three baseline dataset sections.'
    Write-Host 'Resolve runner plans without training. Data/checkpoints must be copied separately.'
    Write-Host 'Plan only: no installation, downloads, training or filesystem changes.'
    return
}
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code ${LASTEXITCODE}: $Executable $Arguments" }
}
Push-Location -LiteralPath $repoRoot
try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Install Git for Windows first; see SETUP_THIRD_PC.md.' }
    $actualCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot read the repository commit.' }
    if ($ExpectedCommit -and $actualCommit -ne $ExpectedCommit) {
        throw "Checkout differs from expected commit: $actualCommit versus $ExpectedCommit. No installation performed."
    }
    $sourceChanges = & git diff --name-only HEAD -- requirements.txt run.py runner.py config models datasets functional module evaluation.yaml evaluation_protocol.py project_paths.py
    if ($LASTEXITCODE -ne 0) { throw 'Cannot check source/config changes.' }
    if ($sourceChanges) { throw "Source/config edits differ from the recorded commit. Preserve them and resolve before setup: $sourceChanges" }
    Write-Host "Git commit: $actualCommit"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        if ($VerifyOnly) { throw "Missing environment: $venvPython. Run without VerifyOnly to install." }
        if (Test-Path -LiteralPath $venvRoot) { throw 'An incomplete environment exists. Inspect it; this script never deletes it.' }
        if ($PythonExe) {
            $basePython = (Get-Command $PythonExe -ErrorAction Stop).Source
            $baseArgs = @()
        } else {
            if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install 64-bit Python 3.11 with its Windows launcher first.' }
            $basePython = (Get-Command py).Source
            $baseArgs = @('-3.11')
        }
        Invoke-Checked -Executable $basePython -Arguments ($baseArgs + @('-c', "import sys,struct; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, 'Requires 64-bit Python 3.11'"))
        Invoke-Checked -Executable $basePython -Arguments ($baseArgs + @('-m', 'venv', $venvRoot))
    }
    Invoke-Checked -Executable $venvPython -Arguments @('-c', "import sys,struct; assert sys.prefix!=sys.base_prefix and sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, 'Requires a local 64-bit Python 3.11 venv'")
    if (-not $VerifyOnly) {
        # Keep downloads and status visible. Built-in pip logging preserves console progress.
        $logRoot = Join-Path $repoRoot 'results\setup'
        New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $pipLog = Join-Path $logRoot "pip-$stamp.log"
        Write-Host "Installing packages; full pip log: $pipLog"
        Invoke-Checked -Executable $venvPython -Arguments @('-m', 'pip', '--log', $pipLog, 'install', '--verbose', '--progress-bar', 'on', 'pip==26.2.1')
        Invoke-Checked -Executable $venvPython -Arguments @('-m', 'pip', '--log', $pipLog, 'install', '--verbose', '--progress-bar', 'on', '-r', $requirements)
    }
    Invoke-Checked -Executable $venvPython -Arguments @('-m', 'pip', 'check')
    $verifyArgs = @((Join-Path $PSScriptRoot 'verify_setup.py'), '--expected-prefix', $venvRoot, '--output', 'results/setup/SETUP_REPORT.json')
    if ($ExpectedCommit) { $verifyArgs += @('--expected-commit', $ExpectedCommit) }
    Invoke-Checked -Executable $venvPython -Arguments $verifyArgs
    Invoke-Checked -Executable $venvPython -Arguments @('runner.py', '--models', 'SE_adLIF', '--datasets', 'SHD', 'SSC', 'ECG', '--logdir', 'results/setup/preflight', '--dry-run')
    Invoke-Checked -Executable $venvPython -Arguments @('runner.py', '--models', 'MT_DTH_SE_adLIF', '--datasets', 'SHD', '--logdir', 'results/setup/preflight', '--dry-run')
    if ($RunTests) {
        # Existing unit tests include tiny synthetic fit/resume smoke tests, never full jobs.
        Invoke-Checked -Executable $venvPython -Arguments @('-m', 'unittest', 'discover', '-s', 'tests')
    }
    Write-Host 'Environment checks passed. Review results/setup/SETUP_REPORT.json for missing data/checkpoints.'
    Write-Host 'No full training job started. Read SETUP_THIRD_PC.md before scheduling jobs.'
} finally {
    Pop-Location
}
