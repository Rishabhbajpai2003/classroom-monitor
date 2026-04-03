param(
    [string]$ManifestRoot = "data\ava_active_speaker",
    [string]$OutputDir = "data\ava_active_speaker\videos",
    [switch]$Monitor,
    [switch]$Resume
)

$jobName = "AVA-ActiveSpeaker-TrainVal"

if ($Monitor) {
    Get-BitsTransfer | Where-Object { $_.DisplayName -eq $jobName } |
        Select-Object DisplayName, JobState, BytesTransferred, BytesTotal, FilesTransferred, FilesTotal
    exit 0
}

$trainManifest = Join-Path $ManifestRoot "train_video_files.txt"
$valManifest = Join-Path $ManifestRoot "val_video_files.txt"

if (-not (Test-Path $trainManifest)) {
    throw "Missing manifest: $trainManifest"
}
if (-not (Test-Path $valManifest)) {
    throw "Missing manifest: $valManifest"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$files = @(
    Get-Content $trainManifest
    Get-Content $valManifest
) | Where-Object { $_ -and $_.Trim() -ne "" } | Sort-Object -Unique

$sources = New-Object System.Collections.Generic.List[string]
$destinations = New-Object System.Collections.Generic.List[string]

foreach ($file in $files) {
    $destination = Join-Path $OutputDir $file
    if (Test-Path $destination) {
        continue
    }
    $sources.Add(("https://s3.amazonaws.com/ava-dataset/trainval/{0}" -f $file))
    $destinations.Add($destination)
}

if ($sources.Count -eq 0) {
    Write-Output "All AVA ActiveSpeaker train/val videos already exist in $OutputDir"
    exit 0
}

if ($Resume) {
    $existing = Get-BitsTransfer | Where-Object { $_.DisplayName -eq $jobName }
    if ($existing) {
        Resume-BitsTransfer -BitsJob $existing
        Write-Output "Resumed existing BITS job: $jobName"
        exit 0
    }
}

Start-BitsTransfer -DisplayName $jobName -Description "AVA ActiveSpeaker train/val video download" -Source $sources -Destination $destinations -TransferType Download -Priority Normal -Asynchronous | Out-Null
Write-Output ("Queued BITS download job '{0}' with {1} files into {2}" -f $jobName, $sources.Count, $OutputDir)
