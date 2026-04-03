# launch_cloudflare.ps1
# ------------------------------------------------------------------
# Downloads cloudflared.exe (if not present) and starts a Cloudflare
# Quick Tunnel that exposes the local Gradio app at a public HTTPS URL.
#
# Inference runs on THIS machine's GPU - Cloudflare only forwards HTTP.
#
# Usage (run AFTER starting gradio_app.py in another terminal):
#   powershell -ExecutionPolicy Bypass -File launch_cloudflare.ps1
#   powershell -ExecutionPolicy Bypass -File launch_cloudflare.ps1 -Port 7861
# ------------------------------------------------------------------

param(
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
$CloudflaredExe = Join-Path $PSScriptRoot "cloudflared.exe"
$DownloadUrl    = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   Classroom Monitor - Cloudflare Quick Tunnel          " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Download cloudflared if not already present
if (-not (Test-Path $CloudflaredExe)) {
    Write-Host "[1/2] cloudflared.exe not found. Downloading..." -ForegroundColor Yellow
    try {
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $CloudflaredExe -UseBasicParsing
        Write-Host "      [OK] Saved to: $CloudflaredExe" -ForegroundColor Green
    } catch {
        Write-Host "" 
        Write-Host "      [ERROR] Download failed: $_" -ForegroundColor Red
        Write-Host "      Please download cloudflared manually from:" -ForegroundColor Yellow
        Write-Host "      https://github.com/cloudflare/cloudflared/releases" -ForegroundColor Yellow
        Write-Host "      Place cloudflared.exe in: $PSScriptRoot" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[1/2] cloudflared.exe already present - skipping download." -ForegroundColor Green
}

# Step 2: Verify that the Gradio app is reachable
Write-Host ""
Write-Host "[2/2] Starting Cloudflare Quick Tunnel -> http://localhost:$Port" -ForegroundColor Yellow
Write-Host ""

try {
    $null = Invoke-WebRequest -Uri "http://localhost:$Port" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    Write-Host "      [OK] Gradio app detected on port $Port." -ForegroundColor Green
} catch {
    Write-Host "      [WARNING] Gradio app not detected on port $Port yet." -ForegroundColor Yellow
    Write-Host "               Make sure 'python gradio_app.py' is running." -ForegroundColor Yellow
    Write-Host "               Starting tunnel anyway..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "      [URL] Your public URL will appear in the output below." -ForegroundColor Cyan
Write-Host "            Look for a line containing 'trycloudflare.com'." -ForegroundColor Cyan
Write-Host "            Press Ctrl+C to stop the tunnel." -ForegroundColor Gray
Write-Host ""
Write-Host "--------------------------------------------------------" -ForegroundColor DarkGray

& $CloudflaredExe tunnel --url "http://localhost:$Port"
