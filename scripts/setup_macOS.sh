#!/usr/bin/env bash
set -euo pipefail

if [[ "${OSTYPE:-}" != darwin* ]]; then
  echo "This script only supports macOS hosts." >&2
  exit 1
fi

if ! xcode-select -p >/dev/null 2>&1; then
  echo "Xcode Command Line Tools are required. Run 'xcode-select --install' and re-run this script." >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Homebrew is required but not found.
Install it from https://brew.sh/ and then re-run this script.
EOF
  exit 1
fi

SHELL_CONFIGS=("$HOME/.zprofile" "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.profile")
ensure_shell_line() {
  local line="$1"
  for rc in "${SHELL_CONFIGS[@]}"; do
    if [[ -e "$rc" ]]; then
      if ! grep -Fqx "$line" "$rc"; then
        printf '\n%s\n' "$line" >>"$rc"
      fi
    else
      printf '%s\n' "$line" >>"$rc"
    fi
  done
}

brew update
BREW_PACKAGES=(cmake git python@3.12 protobuf pkg-config)
for pkg in "${BREW_PACKAGES[@]}"; do
  if ! brew list --versions "$pkg" >/dev/null 2>&1; then
    brew install "$pkg"
  fi
done

if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
if [[ -f "$HOME/.cargo/env" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.cargo/env"
  ensure_shell_line '. "$HOME/.cargo/env"'
fi

mkdir -p "$HOME/.local"
cd "$HOME/.local"

if [[ -d "emsdk" ]]; then
  git -C emsdk pull --ff-only
else
  git clone https://github.com/emscripten-core/emsdk.git
fi

cd emsdk
./emsdk install latest
./emsdk activate latest
cd ..

ensure_shell_line 'source "$HOME/.local/emsdk/emsdk_env.sh"'

BINARYEN_VERSION="version_120_b"
BINARYEN_DIR="$HOME/.local/binaryen-${BINARYEN_VERSION}"
if [[ ! -d "$BINARYEN_DIR" ]]; then
  ARCH="$(uname -m)"
  case "$ARCH" in
    arm64) ARCHIVE_SUFFIX="arm64-macos" ;;
    x86_64) ARCHIVE_SUFFIX="x86_64-macos" ;;
    *)
      echo "Unsupported macOS architecture: $ARCH" >&2
      exit 1
      ;;
  esac
  ARCHIVE="binaryen-${BINARYEN_VERSION}-${ARCHIVE_SUFFIX}.tar.gz"
  curl -L -o "$ARCHIVE" "https://github.com/WebAssembly/binaryen/releases/download/${BINARYEN_VERSION}/${ARCHIVE}"
  tar -xzf "$ARCHIVE"
  rm -f "$ARCHIVE"
fi
ensure_shell_line 'export PATH="$PATH:$HOME/.local/binaryen-version_120_b/bin"'