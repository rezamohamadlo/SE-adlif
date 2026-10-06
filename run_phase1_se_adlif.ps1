param(
    [int[]]$Seeds = @(42, 123, 456),
    [ValidateSet("SHD", "SSC", "ECG")]
    [Alias("Dataset")]
    [string[]]$Datasets = @("SSC", "ECG"),
    [ValidateRange(0, 10000)]
    [int]$Epochs = 0,
    [ValidateRange(0, 100000)]
    [int]$BatchSize = 0,
    [ValidateSet(0, 1)]
    [int]$EarlyStopping = 1,
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

$ModelName = if ($Mode -eq "reference") {
    "SE-adLIF reference"
} else {
    "DA-LIF proposed model"
}

$DatasetDefaults = @{
    SHD = @{ Epochs = 300; BatchSize = 256 }
    SSC = @{ Epochs = 40; BatchSize = 256 }
    ECG = @{ Epochs = 400; BatchSize = 64 }
}

Write-Host "Starting the Phase 1 model runs"
Write-Host "  Mode:    $Mode ($ModelName)"
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
        switch ("$Mode/$Dataset") {
            "reference/SHD" { $Experiment = "SHD_SE_adLIF" }
            "reference/SSC" { $Experiment = "SSC_SE_adLIF" }
            "reference/ECG" { $Experiment = "ECG_SE_adLIF_2layer" }
            "ours/SHD" { $Experiment = "SHD_DA_LIF" }
            "ours/SSC" { $Experiment = "SSC_DA_LIF" }
            "ours/ECG" { $Experiment = "ECG_DA_LIF_2layer" }
        }

        $ExperimentConfig = Join-Path $ProjectRoot "config\experiment\$Experiment.yaml"
        if (-not (Test-Path -LiteralPath $ExperimentConfig)) {
            throw "The '$Mode' model is not implemented yet. Expected configuration: $ExperimentConfig"
        }

        $DatasetName = $Dataset.ToUpperInvariant()
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
        $ModeLogDir = "$LogDir/$Mode/$DatasetName"

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
