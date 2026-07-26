#!/usr/bin/env bash
set -euo pipefail

# Autoresearch benchmark for SGCC F1 optimization.
# Runs the ImprovedMetaLearner v2 and emits METRIC lines.

export KMP_DUPLICATE_LIB_OK=TRUE

OUTPUT=$(python run_meta_v2.py --dataset sgcc --label-source original 2>&1)

echo "$OUTPUT"

# Parse the F1 printed under "FINAL: ImprovedMetaLearner v2"
F1=$(echo "$OUTPUT" | awk '/FINAL: ImprovedMetaLearner v2/{flag=1; next} flag && /F1=/{print $2; exit}')
AUC=$(echo "$OUTPUT" | awk '/FINAL: ImprovedMetaLearner v2/{flag=1; next} flag && /AUC=/{print $2; exit}')
REC=$(echo "$OUTPUT" | awk '/FINAL: ImprovedMetaLearner v2/{flag=1; next} flag && /Rec=/{print $2; exit}')
PREC=$(echo "$OUTPUT" | awk '/FINAL: ImprovedMetaLearner v2/{flag=1; next} flag && /Prec=/{print $2; exit}')

# Fallbacks if parsing failed
F1=${F1:-0}
AUC=${AUC:-0}
REC=${REC:-0}
PREC=${PREC:-0}

echo "METRIC f1=$F1"
echo "METRIC auc=$AUC"
echo "METRIC recall=$REC"
echo "METRIC precision=$PREC"
