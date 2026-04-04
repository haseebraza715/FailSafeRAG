#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/x/Downloads/Thesis-Paper/Code"
"$ROOT/.venv/bin/python" -m pip install -e "$ROOT"
"$ROOT/.venv/bin/faar-demo" run-example --example-id "$1"
