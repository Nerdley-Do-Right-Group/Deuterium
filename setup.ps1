<#
.SYNOPSIS
    Sets up the Deuterium Python virtual environment (.venv) on Windows.
.DESCRIPTION
    Creates a .venv folder, upgrades pip, and installs Deuterium + its dev
    dependencies in editable mode so you can run `deuterium` from the CLI.
#>

$ErrorActionPreference = "Stop"

# Always run relative to the repository root, regardless of where the user invoked the script
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Host "=== Deuterium Setup (Windows) ===" -ForegroundColor Cyan

# Find a working Python 3.10+ executable (prefer `python`, fall back to `py -3`)
$PythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonCmd = @("python")
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonCmd = @("py", "-3")
} else {
    Write-Error "Python was not found on PATH. Install Python 3.10+ and try again."
    exit 1
}

function Invoke-Python {
    param([string[]]$Args)
    if ($PythonCmd.Count -eq 2) {
        & $PythonCmd[0] $PythonCmd[1] @Args
    } else {
        & $PythonCmd[0] @Args
    }
}

# Verify Python is actually >= 3.10 (and not a dusty 2.7/3.8 relic)
$VersionCheck = Invoke-Python -Args @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}'); sys.exit(0 if sys.version_info >= (3, 10) else 1)")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python 3.10+ is required (found $VersionCheck)."
    exit 1
}
Write-Host "Using Python $VersionCheck" -ForegroundColor Green

# Create the virtual environment if it doesn't already exist
$VenvDir = Join-Path $RepoRoot ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment at .venv ..."
    Invoke-Python -Args @("-m", "venv", ".venv")
} else {
    Write-Host "Virtual environment .venv already exists. Reusing it."
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing Deuterium and dependencies in editable mode..."
& $VenvPython -m pip install -e ".[dev]"

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the venv:  .\.venv\Scripts\Activate.ps1"
Write-Host "  2. Train the model:    deuterium train"
Write-Host '  3. Predict a message:  deuterium predict "Congratulations! You won a free prize!"'
