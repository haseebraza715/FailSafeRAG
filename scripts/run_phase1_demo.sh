#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if [ -x "$ROOT/.venv/bin/faar-demo" ]; then
    FAAR_DEMO="$ROOT/.venv/bin/faar-demo"
else
    FAAR_DEMO=faar-demo
    if ! command -v "$FAAR_DEMO" >/dev/null 2>&1; then
        printf '%s\n' "faar-demo not found under $ROOT/.venv/bin or on PATH" >&2
        printf '%s\n' "Create the Python environment first; this script never installs dependencies." >&2
        exit 1
    fi
fi

if [ "$#" -lt 1 ]; then
    printf 'usage: %s <example-id> [faar-demo args...]\n' "$0" >&2
    exit 2
fi

example_id=$1
shift

exec "$FAAR_DEMO" run-example --project-root "$ROOT" --example-id "$example_id" "$@"
