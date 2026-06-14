#!/usr/bin/env bash
set -euo pipefail

# Run the full pipeline in the documented order from the repository root.
# This script stops immediately if any stage fails.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "[1/3] Training surrogate models and building artifacts..."
python3 pipeline/surrogate_model.py

echo "[2/3] Running risk-to-cash frontier simulations..."
python3 pipeline/RiskToCashFrontier.py

echo "[3/3] Rendering risk-to-cash plots..."
python3 pipeline/plot_risk_to_cash_results.py

echo "Pipeline complete."
