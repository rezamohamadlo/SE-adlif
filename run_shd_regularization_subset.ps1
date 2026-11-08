param(
    [Parameter(Mandatory = $true)]
    [string[]] $Job,
    [switch] $DryRun
)

$ErrorActionPreference = 'Stop'
$python = Join-Path (Get-Location) '.ven\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing project Python: $python"
}

$temporary = Join-Path $env:TEMP ("shd_regularization_subset_{0}.py" -f $PID)
$code = @'
import sys
import os
sys.path.insert(0, os.getcwd())
import runner
import shd_speaker_holdout as speaker
import shd_speaker_regularization as regularization

jobs = []
for value in sys.argv[1:]:
    if value == '--dry-run':
        continue
    profile, model, seed = value.split(':', 2)
    if profile not in regularization.PROFILES:
        raise ValueError(f'Unknown profile: {profile}')
    if model not in speaker.MODELS:
        raise ValueError(f'Unknown model: {model}')
    jobs.append((profile, model, int(seed)))
if not jobs:
    raise ValueError('At least one job is required')

valid = set(regularization.ordered_jobs('all'))
if any(job not in valid for job in jobs):
    raise ValueError('Job is not part of the frozen regularization queue')
if len(set(jobs)) != len(jobs):
    raise ValueError('Duplicate jobs are not allowed')

full_jobs = list(regularization.ordered_jobs('all'))
regularization.ordered_jobs = lambda phase: list(jobs)
original_frozen_json = regularization.frozen_json
def frozen_json(path, value):
    if path.name == 'QUEUE.json':
        value = dict(value)
        value['jobs'] = [dict(profile=p, model=m, seed=s) for p, m, s in full_jobs]
    return original_frozen_json(path, value)
regularization.frozen_json = frozen_json
arguments = [
    '--shd-speaker-phase', 'all',
    '--shd-speaker-profile', 'regularization_v1',
]
if '--dry-run' in sys.argv:
    arguments.append('--dry-run')
raise SystemExit(speaker.main(arguments, runner))
'@

Set-Content -LiteralPath $temporary -Value $code -Encoding UTF8
try {
    $arguments = @($Job)
    if ($DryRun) { $arguments += '--dry-run' }
    & $python $temporary @arguments
    exit $LASTEXITCODE
}
finally {
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
}
