#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ $# -gt 0 ]]; then
  exec python3 "$SCRIPT_DIR/evaluator.py" "$@"
fi

for SOLUTION in "${FRONTIER_LSM_SOLUTION_PATH:-}" \
  "/app/solution.py" \
  "/work/execution_env/solution_env/solution.py"; do
  if [[ -n "$SOLUTION" && -f "$SOLUTION" ]]; then
    exec python3 "$SCRIPT_DIR/evaluator.py" "$SOLUTION"
  fi
done

echo "Error: missing solution.py; tried FRONTIER_LSM_SOLUTION_PATH, /app/solution.py, and /work/execution_env/solution_env/solution.py" >&2
exit 1
