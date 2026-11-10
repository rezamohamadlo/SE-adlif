# User-run cleanup only. Automatic approval review blocked the agent's deletion.
# All original result/checkpoint files have verified copies under artifacts/.
$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path.TrimEnd('\')
$experimentRoot = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot 'results\experiments')).TrimEnd('\')
$combinedRoot = Join-Path $experimentRoot 'shd_speaker_combined_v1'
$receiptPath = Join-Path $workspaceRoot 'results\provenance\shd-speaker-consolidation-2026-11-10.json'
$receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
$names = @(
    'shd_speaker_augmented_v1', 'shd_speaker_regularization_v1',
    'shd_regularization_test_seed42_v1', 'shd_speaker_regularization_tests_v1',
    'shd_speaker_regularization_test_v1', 'shd_speaker_augmented_tta_v1'
)
$active = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match [regex]::Escape($workspaceRoot) -and
    $_.CommandLine -match 'runner.py|run.py|EVALUATION.py|regularization_subset'
})
if ($active.Count -gt 0) { throw 'An active project process was found. Close it before cleanup.' }
$known = @{}
foreach ($file in $receipt.files) {
    $source = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $file.source))
    $target = [System.IO.Path]::GetFullPath((Join-Path $workspaceRoot $file.destination))
    if (-not $source.StartsWith($experimentRoot + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $target.StartsWith($combinedRoot + '\artifacts\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw 'A receipt path escaped the intended experiment directories.'
    }
    if (-not (Test-Path -LiteralPath $target -PathType Leaf) -or
        (Get-Item -LiteralPath $target).Length -ne $file.bytes -or
        (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
        throw "Consolidated copy failed verification: $($file.destination)"
    }
    if (Test-Path -LiteralPath $source) {
        if ((Get-Item -LiteralPath $source).Length -ne $file.bytes -or
            (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $file.sha256) {
            throw "Source changed since consolidation: $($file.source)"
        }
    }
    $known[$file.source] = $true
}
foreach ($file in $receipt.excluded_disposable_files) { $known[$file] = $true }
$targets = @()
foreach ($name in $names) {
    $expected = [System.IO.Path]::GetFullPath((Join-Path $experimentRoot $name)).TrimEnd('\')
    if (-not (Test-Path -LiteralPath $expected)) { continue }
    $target = (Resolve-Path -LiteralPath $expected).Path.TrimEnd('\')
    if ($target -ne $expected -or (Split-Path -Parent $target) -ne $experimentRoot -or $target -eq $combinedRoot) {
        throw "Unsafe source directory: $target"
    }
    $items = @(Get-Item -LiteralPath $target) + @(Get-ChildItem -LiteralPath $target -Recurse -Force)
    if (@($items | Where-Object { ($_.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0 }).Count -gt 0) {
        throw 'A reparse point was found; cleanup stopped.'
    }
    foreach ($item in $items | Where-Object { -not $_.PSIsContainer }) {
        $relative = $item.FullName.Substring($workspaceRoot.Length + 1).Replace('\', '/')
        if (-not $known.ContainsKey($relative)) { throw "Uninventoried file retained: $relative" }
    }
    $targets += $target
}
foreach ($target in $targets) { Remove-Item -LiteralPath $target -Recurse -Force }
foreach ($target in $targets) {
    if (Test-Path -LiteralPath $target) { throw "Removal failed: $target" }
}
$receipt.original_folders_removed = $true
$receipt | Add-Member -NotePropertyName cleanup_status -NotePropertyValue 'completed_by_user_script' -Force
$receipt | Add-Member -NotePropertyName cleanup_verified_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('o')) -Force
[System.IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 50) + [char]10,
    [System.Text.UTF8Encoding]::new($false))
Write-Host "Removed $($targets.Count) separate folders. Combined artifacts and hashes remain intact."
