#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
PYTHON_SCRIPT="$SCRIPT_DIR/main.py"
ENV_FILE="$SCRIPT_DIR/.env"
ENV_EXAMPLE="$SCRIPT_DIR/.env.example"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}ok${NC}   $*"; }
warn() { echo -e "${YELLOW}warn${NC} $*"; }
die()  { echo "$*" >&2; exit 1; }

APP_MODE="interactive"
if [[ $# -gt 0 ]]; then
    case "$1" in
        interactive|toggle)
            APP_MODE="$1"
            shift
            ;;
    esac
fi
APP_EXTRA_ARGS=("$@")

find_python() {
    if command -v python3 &>/dev/null; then
        echo "python3"
        return
    fi

    if command -v python &>/dev/null; then
        echo "python"
        return
    fi

    die "Python is not installed."
}

SYSTEM_PYTHON="$(find_python)"

activate_venv() {
    set +u
    source "$VENV_DIR/bin/activate"
    set -u
    ok "Virtual environment activated: $VENV_DIR"
}

ensure_env_file() {
    if [[ ! -f "$ENV_FILE" ]]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        warn ".env file created. Add your Groq API key to it:"
        warn "  $ENV_FILE"
        warn "Get a key from: https://console.groq.com/keys"
    elif grep -q "your_groq_api_key_here" "$ENV_FILE"; then
        warn ".env exists but GROQ_API_KEY is still a placeholder."
        warn "Edit $ENV_FILE and replace 'your_groq_api_key_here' with your actual Groq API key."
    else
        ok ".env file found"
    fi
}

load_env_file() {
    if [[ -f "$ENV_FILE" ]] && ! grep -q "your_groq_api_key_here" "$ENV_FILE"; then
        unset GROQ_API_KEY
        set -a
        # shellcheck disable=SC1090
        source "$ENV_FILE"
        set +a
        ok "Loaded GROQ_API_KEY from $ENV_FILE and cleared any stale shell override"
    fi
}

full_setup() {
    if [[ ! -d "$VENV_DIR" ]]; then
        "$SYSTEM_PYTHON" -m venv "$VENV_DIR"
        ok "Virtual environment created: $VENV_DIR"
    else
        ok "Virtual environment found: $VENV_DIR"
    fi

    activate_venv

    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y libportaudio2 wl-clipboard wtype libnotify-bin 2>/dev/null \
            && ok "Installed system packages for audio, clipboard, typing, and notifications" \
            || warn "Could not install one or more system packages automatically; install libportaudio2 wl-clipboard wtype libnotify-bin manually if needed"
    else
        warn "Install libportaudio2, wl-clipboard, wtype, and libnotify-bin for the full Wayland workflow"
    fi

    echo "Installing Python dependencies..."
    python -m pip install --quiet --upgrade pip 2>&1 | tail -3
    python -m pip install --quiet -r "$REQUIREMENTS_FILE" 2>&1 | tail -3
    ok "Python dependencies installed from $REQUIREMENTS_FILE"

    ensure_env_file
    load_env_file
}

fast_launch_ready() {
    [[ -d "$VENV_DIR" ]] || die "Missing $VENV_DIR. Run '$SCRIPT_DIR/setup.sh' once in a terminal first."
    [[ -f "$ENV_FILE" ]] || die "Missing $ENV_FILE. Run '$SCRIPT_DIR/setup.sh' once in a terminal first."
    if grep -q "your_groq_api_key_here" "$ENV_FILE"; then
        die "GROQ_API_KEY is still a placeholder in $ENV_FILE."
    fi
}

launch_app() {
    chmod +x "$PYTHON_SCRIPT"
    python "$PYTHON_SCRIPT" "$APP_MODE" "${APP_EXTRA_ARGS[@]}"
}

if [[ "$APP_MODE" == "toggle" ]]; then
    fast_launch_ready
    activate_venv
    load_env_file
    launch_app
    exit 0
fi

full_setup
launch_app

echo
echo "Run the app with:"
echo "  source \"$VENV_DIR/bin/activate\" && unset GROQ_API_KEY && set -a && source \"$ENV_FILE\" && set +a && python \"$PYTHON_SCRIPT\""
echo
echo "Ubuntu keyboard shortcut command:"
echo "  bash -lc 'bash \"$SCRIPT_DIR/setup.sh\" toggle'"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    warn "The virtual environment was activated for this setup run only."
    warn "Run 'source \"$VENV_DIR/bin/activate\"' to keep it active in your shell."
else
    ok "The virtual environment remains active in this shell."
fi
