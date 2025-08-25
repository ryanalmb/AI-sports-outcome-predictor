# Setup and run Sports Prediction Bot (Windows PowerShell)
# - Creates a Python 3.12 venv
# - Installs required packages
# - Copies .env.example to .env (if missing)
# - Loads .env into process environment
# - Runs the bot

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[1/7] Setting Execution Policy for this process..." -ForegroundColor Cyan
Try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force } Catch {}

# Helper to run a command safely
Function Invoke-CommandSafe {
  Param([Parameter(Mandatory=$true)][string]$Command)
  Write-Host "[cmd] $Command" -ForegroundColor DarkGray
  Invoke-Expression $Command
}

Write-Host "[2/7] Ensuring virtual environment (.venv) exists..." -ForegroundColor Cyan
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  Try {
    Invoke-CommandSafe 'py -3.12 -m venv .venv'
  } Catch {
    Write-Warning "Python launcher 'py -3.12' failed. Trying 'python -m venv'..."
    Invoke-CommandSafe 'python -m venv .venv'
  }
}

Write-Host "[3/7] Activating virtual environment..." -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1

Write-Host "[4/7] Installing Python dependencies..." -ForegroundColor Cyan
python -m pip install -U pip
# Remove conflicting package if present
python -m pip uninstall -y telegram | Out-Null
# Install required deps
python -m pip install "python-telegram-bot==20.7" aiohttp requests numpy pandas scikit-learn google-generativeai

Write-Host "[5/7] Preparing .env file..." -ForegroundColor Cyan
if (-not (Test-Path ".env")) {
  if (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env" -Force
    Write-Host "Created .env from .env.example" -ForegroundColor Green
  } else {
    Write-Warning ".env.example not found. Creating empty .env"
    New-Item -ItemType File -Path .env -Force | Out-Null
  }
} else {
  Write-Host ".env already exists; leaving as-is" -ForegroundColor Yellow
}

Write-Host "[6/7] Loading environment variables from .env..." -ForegroundColor Cyan
(Get-Content .env) |
  ForEach-Object { $_.Trim() } |
  Where-Object { $_ -and -not $_.StartsWith('#') } |
  ForEach-Object {
    $kv = $_ -split '=', 2
    if ($kv.Length -eq 2) {
      $name = $kv[0].Trim()
      $value = $kv[1].Trim()
      Set-Item -Path Env:$name -Value $value
      Write-Host "  set $name" -ForegroundColor DarkGray
    }
  }

Write-Host "[7/7] Running bot..." -ForegroundColor Cyan
python -V
python -c "import sys; print('Python exe:', sys.executable)"
python -m pip show python-telegram-bot | Select-String -Pattern "Version" -Quiet | Out-Null
if (-not $?) { Write-Warning "python-telegram-bot not detected; attempting to continue" }

python fixed_bot.py
