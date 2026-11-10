param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('pc1', 'pc2', 'pc3')]
    [string] $PC,

    [switch] $CheckOnly,
    [switch] $SelfCheck,

    [ValidateRange(1, 2147483647)]
    [int] $Samples = 256,

    [ValidateNotNullOrEmpty()]
    [int[]] $Seeds = @(42, 123, 456),

    [ValidateNotNullOrEmpty()]
    [string] $Device = 'cuda:0',

    [ValidateNotNullOrEmpty()]
    [string] $OutputRoot = 'results/experiments/adaptation_mechanism_analysis_v1'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$python = @('.venv\Scripts\python.exe', '.ven\Scripts\python.exe') |
    ForEach-Object { Join-Path $projectRoot $_ } |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1
if (-not $python) {
    throw 'Missing project Python. Create the project .venv or .ven environment first.'
}

$pilot = Join-Path $projectRoot 'scripts\adaptation_analysis\pilot.py'
if (-not (Test-Path -LiteralPath $pilot -PathType Leaf)) {
    throw 'Missing analysis pilot. Pull the complete implementation before launching.'
}

$arguments = @(
    $pilot, '--pc', $PC.ToLowerInvariant(),
    '--samples', [string] $Samples,
    '--device', $Device,
    '--output-root', $OutputRoot,
    '--seeds'
)
$arguments += @($Seeds | ForEach-Object { [string] $_ })
if ($CheckOnly) { $arguments += '--check-only' }
if ($SelfCheck) { $arguments += '--self-check' }

Push-Location -LiteralPath $projectRoot
try {
    & $python @arguments
    $pilotExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $pilotExitCode
