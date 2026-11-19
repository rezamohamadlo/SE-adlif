param(
    [switch] $CheckOnly,
    [switch] $VerifyOnly,
    [string] $Device = 'cuda:0',
    [string] $OutputRoot = 'results/experiments/adaptation_mechanism_full_validation_v1'
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Missing project .venv Python.'
}
if ($CheckOnly -and $VerifyOnly) { throw 'Choose CheckOnly or VerifyOnly.' }
$arguments = @((Join-Path $projectRoot 'scripts\adaptation_analysis\evaluate.py'),
               '--device', $Device, '--output-root', $OutputRoot)
if ($CheckOnly) { $arguments += '--check-only' }
if ($VerifyOnly) { $arguments += '--verify-only' }
Push-Location -LiteralPath $projectRoot
try {
    & $python @arguments
    $analysisExitCode = $LASTEXITCODE
}
finally { Pop-Location }
exit $analysisExitCode
