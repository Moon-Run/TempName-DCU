#!/usr/bin/env bash
set -euo pipefail
dcu_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
dcu_python=${DCU_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
cd -- "$dcu_repo"
exec "$dcu_python" tools/dcu/phase6/init.py "$@"
