$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "frontend")

$flutterBin = Join-Path $env:USERPROFILE "tools\flutter\bin"
if (Test-Path (Join-Path $flutterBin "flutter.bat")) {
    $env:Path = "$flutterBin;$env:Path"
}

flutter run -d chrome
