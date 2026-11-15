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

# Find a working native Windows Python 3.10+ executable (prefer `python`, then `py -3`, then `python3`).
# We check sysconfig.get_platform() so we don't accidentally pick an MSYS2/mingw Python
# (like C:\msys64\clang64\bin\python3.exe), which lacks PyPI binary wheels and tries
# to compile numpy/scikit-learn from source.
$PythonCmd = $null
$Candidates = @(
    @("python"),
    @("py", "-3"),
    @("python3")
)

foreach ($Candidate in $Candidates) {
    if (Get-Command $Candidate[0] -ErrorAction SilentlyContinue) {
        try {
            $PlatCheck = if ($Candidate.Count -eq 2) {
                & $Candidate[0] $Candidate[1] -c "import sys, sysconfig; p=sysconfig.get_platform(); print(p); sys.exit(0 if sys.version_info >= (3, 10) and 'mingw' not in p else 1)" 2>$null
            } else {
                & $Candidate[0] -c "import sys, sysconfig; p=sysconfig.get_platform(); print(p); sys.exit(0 if sys.version_info >= (3, 10) and 'mingw' not in p else 1)" 2>$null
            }
            if ($LASTEXITCODE -eq 0) {
                $PythonCmd = $Candidate
                break
            }
        } catch {
            continue
        }
    }
}

if (-not $PythonCmd) {
    Write-Error "Native Windows Python 3.10+ was not found on PATH (MSYS2/mingw Python is skipped because PyPI wheels aren't available for it). Install Python 3.10+ from python.org and try again."
    exit 1
}

# Note: never name a PowerShell parameter `$Args` — `$args` is a reserved automatic variable
# and splatting `@Args` silently drops your arguments and opens an interactive Python REPL.
function Invoke-Python {
    param([string[]]$PythonArgs)
    if ($PythonCmd.Count -eq 2) {
        & $PythonCmd[0] $PythonCmd[1] @PythonArgs
    } else {
        & $PythonCmd[0] @PythonArgs
    }
}

# Verify Python is actually >= 3.10 (and not a dusty 2.7/3.8 relic)
$VersionCheck = Invoke-Python -PythonArgs @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}'); sys.exit(0 if sys.version_info >= (3, 10) else 1)")
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python 3.10+ is required (found $VersionCheck)."
    exit 1
}
Write-Host "Using Python $VersionCheck ($($PythonCmd -join ' '))" -ForegroundColor Green

# Create the virtual environment if it doesn't already exist (or recreate if a broken/MSYS2 .venv is left over)
$VenvDir = Join-Path $RepoRoot ".venv"
$ExpectedVenvPython = Join-Path $VenvDir "Scripts\python.exe"
if ((Test-Path $VenvDir) -and -not (Test-Path $ExpectedVenvPython)) {
    Write-Host "Existing .venv is not a native Windows venv; recreating it..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment at .venv ..."
    Invoke-Python -PythonArgs @("-m", "venv", ".venv")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment at .venv."
        exit 1
    }
} else {
    Write-Host "Virtual environment .venv already exists. Reusing it."
}

# Standard Windows Python uses .venv\Scripts\python.exe, while MSYS2/Cygwin/Git-Bash
# Python on Windows sometimes creates .venv\bin\python3 instead. Support both!
$VenvPythonCandidates = @(
    (Join-Path $VenvDir "Scripts\python.exe"),
    (Join-Path $VenvDir "Scripts\python3.exe"),
    (Join-Path $VenvDir "bin\python3.exe"),
    (Join-Path $VenvDir "bin\python.exe"),
    (Join-Path $VenvDir "bin\python3"),
    (Join-Path $VenvDir "bin\python")
)
$VenvPython = $VenvPythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $VenvPython) {
    Write-Error "Could not find a Python executable inside '$VenvDir'. Delete .venv and try again."
    exit 1
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip."
    exit 1
}

Write-Host "Installing Deuterium and dependencies in editable mode..."
& $VenvPython -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Deuterium."
    exit 1
}

$ActivateHint = if (Test-Path (Join-Path $VenvDir "Scripts\Activate.ps1")) {
    ".\.venv\Scripts\Activate.ps1"
} else {
    ".\.venv\bin\Activate.ps1"
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the venv:  $ActivateHint"
Write-Host "  2. Train the model:    deuterium train"
Write-Host '  3. Predict a message:  deuterium predict "Congratulations! You won a free prize!"'
