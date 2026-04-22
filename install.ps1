# install.ps1 — One-command installer for Local IDE RL Agent (Windows PowerShell)
#
# Usage (after cloning):
#   .\install.ps1
#
# Or, run directly without cloning:
#   irm https://raw.githubusercontent.com/your-org/local-ide-agent/main/install.ps1 | iex

$ErrorActionPreference = "Stop"

function Write-Ok   { Write-Host "[local-ide-agent] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[warn] $args" -ForegroundColor Yellow }
function Write-Err  { Write-Host "[error] $args" -ForegroundColor Red; exit 1 }

# ── Check Python ──────────────────────────────────────────────────────────────

$python = $null
foreach ($py in @("python3.12", "python3.11", "python3", "python")) {
    try {
        $ver = & $py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($ver -match "^(\d+)\.(\d+)$") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 11) { $python = $py; break }
        }
    } catch { }
}

if (-not $python) {
    Write-Err "Python 3.11 or higher is required. Download from https://python.org"
}
Write-Ok "Using $python ($ver)"

# ── Install ───────────────────────────────────────────────────────────────────

if (Test-Path "pyproject.toml") {
    Write-Ok "Installing from local source (editable)..."
    & $python -m pip install -e ".[dev]" --quiet
} else {
    Write-Ok "Installing from PyPI..."
    & $python -m pip install local-ide-agent --quiet
}

# ── Verify ────────────────────────────────────────────────────────────────────

$cmd = Get-Command local-ide-agent -ErrorAction SilentlyContinue
if (-not $cmd) {
    Write-Warn "CLI not found in PATH. Try: python -m local_ide_agent.main"
} else {
    Write-Ok "Installation successful!"
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "    local-ide-agent train --episodes 10    # Train for 10 episodes"
    Write-Host "    local-ide-agent dashboard              # Open the live dashboard"
    Write-Host "    local-ide-agent eval                   # Run the evaluation harness"
    Write-Host ""
    Write-Host "  Copy and edit the example config:"
    Write-Host "    copy settings.example.yaml settings.yaml"
    Write-Host ""
}
