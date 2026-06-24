#!/usr/bin/env bash
# reset.sh — wipe evolved agents, logs, and results for a clean run
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Cleaning evolved agents..."
find agents -name "*.py" \
  ! -name "__init__.py" \
  ! -name "no_memory.py" \
  ! -name "fewshot_memory.py" \
  ! -name "fewshot_all.py" \
  -delete
echo "Cleaning logs..."
rm -rf logs/
echo "Cleaning results..."
rm -rf results/
echo "Done. Ready for a fresh run:"
echo "  uv run python meta_harness.py --iterations 1"