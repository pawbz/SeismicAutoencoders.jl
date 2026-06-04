#!/bin/bash
#
# train_vqvae_parallel.sh — train station pairs in parallel across multiple GPUs.
#
# Each GPU gets a disjoint subset of pairs (round-robin split). One Julia process
# is launched per GPU; it compiles the XLA graph once and trains all its assigned
# pairs x seeds sequentially, reusing the compiled graph.
#
# Usage: train_vqvae_parallel [pairs] [--gpus LIST] [options]
#
# Parallel Options:
#   --gpus LIST    Comma-separated GPU indices (default: "0,1")
#                  e.g. --gpus 0,1,2,3 for four GPUs
#
# All other options are forwarded verbatim to train_vqvae.jl on every GPU.
# See train_vqvae.jl --help for the full option list.
#
# Examples:
#   ./train_vqvae_parallel.sh --data-dir /path/to/data --nepoch 100
#   ./train_vqvae_parallel.sh AP-BK,AP-CL,AP-MN --data-dir /path/to/data
#   ./train_vqvae_parallel.sh --gpus 0,1,2,3 --data-dir /path/to/data --nepoch 100
#   ./train_vqvae_parallel.sh --gpus 1 --data-dir /path/to/data

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

# Defaults
GPUS="0,1"
DATA_DIR="$(pwd)"
PAIRS_ARG=""
PASSTHROUGH=()

usage_parallel() {
    cat <<'EOF'
Usage: train_vqvae_parallel [pairs] [--gpus LIST] [options]

Arguments:
  pairs                  Comma-separated station pairs e.g. "AP-BK,AP-CL"
                         Use "all" or omit to train all discovered pairs (default: all)
                         Pairs are split round-robin across GPUs.

Parallel Options:
  --gpus LIST            Comma-separated GPU indices to use (default: "0,1")
                         e.g. --gpus 0,1,2,3 for four GPUs
                         One Julia process is launched per GPU; each compiles XLA once
                         and trains its assigned pairs x seeds sequentially.

Info Options (forwarded, run in foreground):
  --help, -h             Show this message
  --list-pairs, -l       Print discovered pairs and exit
  --sample-pair          Sample random pair from the given pairs subset and show plots

All other options are forwarded verbatim to train_vqvae.jl on every GPU:
  --data-dir, --save-dir, --seeds, --nepoch, --batchsize, --lr, --verbose,
  --period-min, --period-max, --dt, --nwindows, --whitening-kernel-length,
  --K, --d, --n-filters, --ratios, --n-residual-layers, --entropy-weight,
  --autodiff-backend

Examples:
  # Train all pairs across 2 GPUs (default):
  ./train_vqvae_parallel.sh --data-dir /path/to/data --nepoch 100

  # Train specific pairs across 2 GPUs:
  ./train_vqvae_parallel.sh AP-BK,AP-CL,AP-MN --data-dir /path/to/data

  # Use 4 GPUs:
  ./train_vqvae_parallel.sh --gpus 0,1,2,3 --data-dir /path/to/data --nepoch 100

  # Single GPU (explicit):
  ./train_vqvae_parallel.sh --gpus 1 --data-dir /path/to/data
EOF
}

# Parse arguments: consume --gpus and --data-dir; forward everything else
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        usage_parallel
        exit 0
    elif [ "$arg" = "--list-pairs" ] || [ "$arg" = "-l" ]; then
        # Forward trivial op to Julia in foreground
        julia "${SCRIPT_DIR}/train_vqvae.jl" "$@"
        exit $?
    elif [ "$arg" = "--sample-pair" ]; then
        # Forward trivial op to Julia in foreground (passes pairs arg too)
        julia "${SCRIPT_DIR}/train_vqvae.jl" "$@"
        exit $?
    elif [ "$arg" = "--gpus" ]; then
        i=$((i+1))
        GPUS="${!i}"
    elif [ "$arg" = "--data-dir" ]; then
        i=$((i+1))
        DATA_DIR="${!i}"
        PASSTHROUGH+=("--data-dir" "${!i}")
    elif [[ "$arg" != --* && "$arg" != -* ]]; then
        # Positional arg = pairs
        PAIRS_ARG="$arg"
    else
        PASSTHROUGH+=("$arg")
    fi
    i=$((i+1))
done

# Discover all pairs from the data directory
ALL_PAIRS_RAW=$(julia "${SCRIPT_DIR}/train_vqvae.jl" --list-pairs --data-dir "$DATA_DIR" 2>/dev/null \
    | grep -E '^\s+[A-Za-z0-9]+-[A-Za-z0-9]+' \
    | sed 's/^[[:space:]]*//')

if [ -z "$ALL_PAIRS_RAW" ]; then
    echo "Error: no station pairs found in ${DATA_DIR}" >&2
    exit 1
fi

# Build the candidate pair list
if [ -z "$PAIRS_ARG" ] || [ "$PAIRS_ARG" = "all" ]; then
    # Use all discovered pairs
    mapfile -t CANDIDATE_PAIRS <<< "$ALL_PAIRS_RAW"
else
    # Use the user-supplied pairs (comma-separated)
    IFS=',' read -ra CANDIDATE_PAIRS <<< "$PAIRS_ARG"
fi

NPAIRS=${#CANDIDATE_PAIRS[@]}
if [ "$NPAIRS" -eq 0 ]; then
    echo "Error: no pairs to train." >&2
    exit 1
fi

# Parse GPU list
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NGPUS=${#GPU_IDS[@]}

# Round-robin split: GPU_PAIRS[i] gets pairs at positions i, i+NGPUS, i+2*NGPUS, ...
declare -a GPU_PAIRS
for ((i=0; i<NGPUS; i++)); do
    GPU_PAIRS[$i]=""
done

for ((p=0; p<NPAIRS; p++)); do
    slot=$((p % NGPUS))
    pair="${CANDIDATE_PAIRS[$p]}"
    if [ -z "${GPU_PAIRS[$slot]}" ]; then
        GPU_PAIRS[$slot]="$pair"
    else
        GPU_PAIRS[$slot]="${GPU_PAIRS[$slot]},${pair}"
    fi
done

echo "Splitting ${NPAIRS} pair(s) across ${NGPUS} GPU(s): ${GPUS}"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILES=()

for ((i=0; i<NGPUS; i++)); do
    gpu="${GPU_IDS[$i]}"
    pairs_for_gpu="${GPU_PAIRS[$i]}"

    if [ -z "$pairs_for_gpu" ]; then
        echo "GPU ${gpu}: no pairs assigned, skipping."
        continue
    fi

    LOG="${RUN_DIR}/train_vqvae_gpu${gpu}_${TIMESTAMP}.out"
    LOG_FILES+=("$LOG")

    CUDA_VISIBLE_DEVICES=$gpu julia "${SCRIPT_DIR}/train_vqvae.jl" \
        "$pairs_for_gpu" "${PASSTHROUGH[@]}" > "$LOG" 2>&1 &

    PID=$!
    echo "GPU ${gpu}: PID=${PID}  pairs=${pairs_for_gpu}"
    echo "         log=${LOG}"
done

echo ""
if [ ${#LOG_FILES[@]} -gt 1 ]; then
    echo "Monitor all:  tail -f ${LOG_FILES[*]}"
elif [ ${#LOG_FILES[@]} -eq 1 ]; then
    echo "Monitor:      tail -f ${LOG_FILES[0]}"
fi
