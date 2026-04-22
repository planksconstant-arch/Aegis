#!/usr/bin/env bash
# install.sh — One-command installer for Local IDE RL Agent (Linux / macOS)
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/your-org/local-ide-agent/main/install.sh | bash
#
# Or, after cloning:
#   bash install.sh

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

info()    { echo -e "${GREEN}[local-ide-agent]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC} $*"; }
error()   { echo -e "${RED}[error]${NC} $*"; exit 1; }

# ── Check Python ──────────────────────────────────────────────────────────────

PYTHON=""
for py in python3.12 python3.11 python3 python; do
    if command -v "$py" &>/dev/null; then
        ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=${ver%%.*}; minor=${ver##*.}
        if [[ $major -ge 3 && $minor -ge 11 ]]; then
            PYTHON="$py"
            break
        fi
    fi
done

[[ -z "$PYTHON" ]] && error "Python 3.11 or higher is required. Install from https://python.org"
info "Using $PYTHON ($ver)"

# ── Install ───────────────────────────────────────────────────────────────────

if [[ -f "pyproject.toml" ]]; then
    info "Installing from local source (editable)..."
    "$PYTHON" -m pip install -e ".[dev]" --quiet
else
    info "Installing from PyPI..."
    "$PYTHON" -m pip install local-ide-agent --quiet
fi

# ── Verify ────────────────────────────────────────────────────────────────────

if ! command -v local-ide-agent &>/dev/null; then
    warn "CLI not found in PATH. Try: python -m local_ide_agent.main"
else
    info "Installation successful!"
    echo ""
    echo "  Next steps:"
    echo ""
    echo "    local-ide-agent train --episodes 10    # Train for 10 episodes"
    echo "    local-ide-agent dashboard              # Open the live dashboard"
    echo "    local-ide-agent eval                   # Run the evaluation harness"
    echo ""
    echo "  Copy and edit the example config:"
    echo "    cp settings.example.yaml settings.yaml"
    echo ""
fi
