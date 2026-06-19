#!/bin/zsh
set -euo pipefail

cd -- "${0:A:h}"
source .venv/bin/activate
export PYINSTALLER_CONFIG_DIR="$PWD/.pyinstaller-cache"

python -m PyInstaller --noconfirm --clean StreamCensor.spec
codesign --force --deep --sign - "dist/Stream Censor.app"
ditto -c -k --sequesterRsrc --keepParent \
  "dist/Stream Censor.app" "dist/Stream-Censor-macOS-arm64.zip"

echo "Built: dist/Stream Censor.app"
echo "Archive: dist/Stream-Censor-macOS-arm64.zip"
