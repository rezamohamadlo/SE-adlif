param(
    # [int[]]$Seeds = @(42, 123, 456),
    [int[]]$Seeds = @(42),
    # [ValidateSet("SHD", "SSC", "ECG")]
    [ValidateSet("SHD")]
    [Alias("Dataset")]
    [string[]]$Datasets = @("SHD"),
    # Experiment-config suffix. For example, SE_adLIF resolves to
    # config/experiment/SHD_SE_adLIF.yaml.
    [ValidatePattern("^[A-Za-z][A-Za-z0-9_-]*$")]
    [string]$ModelVariant = "SE_adLIF",
    [ValidateRange(0, 10000)]
    [int]$Epochs = 0,
    [ValidateRange(0, 100000)]
    [int]$BatchSize = 0,
    [ValidateSet(0, 1)]
    [int]$EarlyStopping = 0,
    [int]$EarlyStoppingPatience = 50,
    [double]$EarlyStoppingMinDelta = 0.001,
    [ValidateRange(0, 10000)]
    [int]$LrSchedulerPatience = 15,
    [ValidateRange(0.000001, 0.999999)]
    [double]$LrSchedulerFactor = 0.9,
    [ValidateSet("reference", "ours")]
    [string]$Mode = "reference",
    [ValidateSet(0, 1)]
    [int]$Resume = 0,
    [string]$LogDir = "results/phase1_models"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "run.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual-environment Python was not found at: $Python"
}

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Experiment runner was not found at: $Runner"
}

$ModelName = "$ModelVariant ($Mode)"

$DatasetDefaults = @{
    SHD = @{ Epochs = 300; BatchSize = 512 }
    SSC = @{ Epochs = 300; BatchSize = 512 }
    ECG = @{ Epochs = 300; BatchSize = 512 }
}

Write-Host "Starting a minimal model-analysis run"
Write-Host "  Mode:    $Mode"
Write-Host "  Model variant: $ModelVariant"
Write-Host "  Datasets: $($Datasets -join ', ')"
Write-Host "  Seeds:   $($Seeds -join ', ')"
Write-Host "  Early stopping: $EarlyStopping"
if ($EarlyStopping -eq 1) {
    Write-Host "    Patience:  $EarlyStoppingPatience"
    Write-Host "    Min delta: $EarlyStoppingMinDelta"
}
Write-Host "  LR scheduler patience: $LrSchedulerPatience"
Write-Host "  LR scheduler factor:   $LrSchedulerFactor"
Write-Host "  Resume:  $Resume"

Push-Location $ProjectRoot
try {
    foreach ($Dataset in $Datasets) {
        $DatasetName = $Dataset.ToUpperInvariant()
        $ExperimentCandidates = if ($DatasetName -eq "ECG") {
            @(
                "${DatasetName}_${ModelVariant}_2layer"
                "${DatasetName}_${ModelVariant}"
            )
        } else {
            @("${DatasetName}_${ModelVariant}")
        }

        $Experiment = $null
        $ExperimentConfig = $null
        foreach ($Candidate in $ExperimentCandidates) {
            $CandidateConfig = Join-Path $ProjectRoot "config\experiment\$Candidate.yaml"
            if (Test-Path -LiteralPath $CandidateConfig) {
                $Experiment = $Candidate
                $ExperimentConfig = $CandidateConfig
                break
            }
        }

        if ($null -eq $ExperimentConfig) {
            $ExpectedConfigs = $ExperimentCandidates |
                ForEach-Object { "config/experiment/$_.yaml" }
            throw (
                "Model variant '$ModelVariant' is not implemented for $DatasetName. " +
                "Create one of: $($ExpectedConfigs -join ', ')"
            )
        }

        $DatasetEpochs = if ($Epochs -eq 0) {
            $DatasetDefaults[$DatasetName].Epochs
        } else {
            $Epochs
        }
        $DatasetBatchSize = if ($BatchSize -eq 0) {
            $DatasetDefaults[$DatasetName].BatchSize
        } else {
            $BatchSize
        }
        # Keep variants isolated so development runs cannot overwrite the
        # reference baseline or another proposed-model variant.
        $ModeLogDir = "$LogDir/$Mode/$ModelVariant/$DatasetName"

        Write-Host ""
        Write-Host "Starting dataset $DatasetName"
        Write-Host "  Experiment: $Experiment"
        Write-Host "  Epochs:     $DatasetEpochs"
        Write-Host "  Batch:      $DatasetBatchSize"
        Write-Host "  Log dir:    $ModeLogDir"

    foreach ($Seed in $Seeds) {
        $SeedLogDir = "$ModeLogDir/seed_$Seed"
        $CompletionMarker = Join-Path $SeedLogDir "completed.txt"

        if (($Resume -eq 1) -and (Test-Path -LiteralPath $CompletionMarker)) {
            Write-Host ""
            Write-Host "Skipping completed seed $Seed."
            continue
        }

        $ResumeCheckpoint = $null
        $SeedRunDir = $SeedLogDir
        if (($Resume -eq 1) -and (Test-Path -LiteralPath $SeedLogDir)) {
            $ResumeCheckpoint = Get-ChildItem `
                -LiteralPath $SeedLogDir `
                -Filter "last*.ckpt" `
                -File `
                -Recurse `
                -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTimeUtc -Descending |
                Select-Object -First 1
        }

        if ($null -ne $ResumeCheckpoint) {
            # Reuse the checkpoint's original Hydra working directory. This
            # also supports runs created by the older date/time layout.
            $CheckpointDir = Split-Path -Parent $ResumeCheckpoint.FullName
            $SeedRunDir = Split-Path -Parent $CheckpointDir
        }
        $HydraRunDir = $SeedRunDir -replace "\\", "/"

        Write-Host ""
        Write-Host "Running seed $Seed"
        Write-Host "  Output: $SeedRunDir"

        $Overrides = @(
            "experiment=$Experiment"
            "random_seed=$Seed"
            "n_epochs=$DatasetEpochs"
            "batch_size=$DatasetBatchSize"
            "dataset.batch_size=$DatasetBatchSize"
            "logdir=$SeedLogDir"
            "dataset.num_workers=0"
            "early_stopping=$($EarlyStopping -eq 1)"
            "early_stopping_patience=$EarlyStoppingPatience"
            "early_stopping_min_delta=$EarlyStoppingMinDelta"
            "patience=$LrSchedulerPatience"
            "factor=$LrSchedulerFactor"
            "hydra.run.dir=$HydraRunDir"
        )

        if ($null -ne $ResumeCheckpoint) {
            Write-Host "  Resuming: $($ResumeCheckpoint.FullName)"
            $ResumeCheckpointPath = $ResumeCheckpoint.FullName -replace "\\", "/"
            $Overrides += "ckpt_path=$ResumeCheckpointPath"
        }
        elseif ($Resume -eq 1) {
            Write-Host "  No last checkpoint found; starting this seed from the beginning."
        }

        & $Python $Runner @Overrides

        $TrainingExitCode = $LASTEXITCODE
        if ($TrainingExitCode -ne 0) {
            throw "$ModelName training for seed $Seed failed with exit code $TrainingExitCode."
        }

        New-Item -ItemType Directory -Path $SeedLogDir -Force | Out-Null
        Set-Content -LiteralPath $CompletionMarker -Value "Completed $(Get-Date -Format o)"
        Write-Host "Seed $Seed completed successfully."
    }
        Write-Host "Dataset $DatasetName completed successfully."
    }
}
finally {
    Pop-Location
}

Write-Host "All requested $ModelName runs completed successfully."
