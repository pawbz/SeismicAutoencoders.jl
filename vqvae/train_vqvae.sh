#!/bin/bash
#
# train_vqvae.sh — launch VQ-VAE training on one or more GPUs.
#
# Station pairs are split round-robin across GPUs. One Julia process is
# launched per GPU; it compiles the XLA graph once and trains all its
# assigned pairs x seeds sequentially, reusing the compiled graph.
#
# Usage: train_vqvae [pairs] [options]
#
# GPU Options:
#   --gpus LIST        Comma-separated GPU indices (default: "0")
#                      e.g. --gpus 0,1   or   --gpus 0,1,2,3
#   --foreground, -f   Stream all GPU logs to the terminal in real time.
#                      Logs are still written to files. Blocks until done.
#
# All other options are forwarded verbatim to train_vqvae.jl on every GPU.
# See train_vqvae.jl --help for the full option list.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

GPUS="0"
DATA_DIR="$(pwd)"
PAIRS_ARG=""
FOREGROUND=false
PASSTHROUGH=()

usage() {
    cat <<'EOF'
Usage: train_vqvae [pairs] [options]

Arguments:
  pairs                  Comma-separated station pairs e.g. "AP-BK,AP-CL"
                         Use "all" or omit to train all discovered pairs (default: all)
                         Pairs are split round-robin across GPUs.

GPU Options:
  --gpus LIST            Comma-separated GPU indices to use (default: "0")
                         e.g. --gpus 0,1,2,3 for four GPUs
                         One Julia process is launched per GPU; each compiles XLA once
                         and trains its assigned pairs x seeds sequentially.
  --foreground, -f       Stream all GPU log files to the terminal in real time.
                         Log files are still written; output is live-tailed.
                         The script blocks until all GPU jobs finish.

Info Options (forwarded, run in foreground):
  --help, -h             Show this message
  --list-pairs, -l       Print discovered pairs and exit
  --sample-pair          Sample a random pair and show waveform/PSD plots

All other options are forwarded verbatim to train_vqvae.jl on every GPU:
  --data-dir, --save-dir, --seeds, --nepoch, --batchsize, --lr, --verbose,
  --period-min, --period-max, --dt, --nwindows, --whitening-kernel-length,
  --K, --d, --n-filters, --ratios, --n-residual-layers, --entropy-weight,
  --autodiff-backend

Examples:
  # Train all pairs on GPU 0 (default):
  ./train_vqvae.sh --data-dir /path/to/data --nepoch 100

  # Train all pairs on GPU 0, print to terminal:
  ./train_vqvae.sh --foreground --data-dir /path/to/data --nepoch 100

  # Train across two GPUs in background:
  ./train_vqvae.sh --gpus 0,1 --data-dir /path/to/data --nepoch 100

  # Train across two GPUs, stream to terminal:
  ./train_vqvae.sh --gpus 0,1 --foreground --data-dir /path/to/data --nepoch 100

  # Train specific pairs on GPU 1:
  ./train_vqvae.sh AP-BK,AP-CL --gpus 1 --data-dir /path/to/data

  # List available pairs:
  ./train_vqvae.sh --list-pairs --data-dir /path/to/data

  # Inspect a random pair:
  ./train_vqvae.sh --sample-pair --data-dir /path/to/data
EOF
}

# Parse arguments
i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        usage
        exit 0
    elif [ "$arg" = "--list-pairs" ] || [ "$arg" = "-l" ]; then
        julia "${SCRIPT_DIR}/train_vqvae.jl" "$@"
        exit $?
    elif [ "$arg" = "--sample-pair" ]; then
        julia "${SCRIPT_DIR}/train_vqvae.jl" "$@"
        exit $?
    elif [ "$arg" = "--foreground" ] || [ "$arg" = "-f" ]; then
        FOREGROUND=true
    elif [ "$arg" = "--gpus" ]; then
        i=$((i+1))
        GPUS="${!i}"
    elif [ "$arg" = "--data-dir" ]; then
        i=$((i+1))
        DATA_DIR="${!i}"
        PASSTHROUGH+=("--data-dir" "${!i}")
    elif [[ "$arg" != --* && "$arg" != -* ]]; then
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
    mapfile -t CANDIDATE_PAIRS <<< "$ALL_PAIRS_RAW"
else
    IFS=',' read -ra CANDIDATE_PAIRS <<< "$PAIRS_ARG"
fi

NPAIRS=${#CANDIDATE_PAIRS[@]}
if [ "$NPAIRS" -eq 0 ]; then
    echo "Error: no pairs to train." >&2
    exit 1
fi

# Parse GPU list and split pairs round-robin
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NGPUS=${#GPU_IDS[@]}

declare -a GPU_PAIRS
for ((i=0; i<NGPUS; i++)); do GPU_PAIRS[$i]=""; done

for ((p=0; p<NPAIRS; p++)); do
    slot=$((p % NGPUS))
    pair="${CANDIDATE_PAIRS[$p]}"
    if [ -z "${GPU_PAIRS[$slot]}" ]; then
        GPU_PAIRS[$slot]="$pair"
    else
        GPU_PAIRS[$slot]="${GPU_PAIRS[$slot]},${pair}"
    fi
done

if [ "$NGPUS" -gt 1 ]; then
    echo "Splitting ${NPAIRS} pair(s) across ${NGPUS} GPU(s): ${GPUS}"
else
    echo "Training ${NPAIRS} pair(s) on GPU ${GPU_IDS[0]}"
fi
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILES=()
PIDS=()

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
    PIDS+=("$PID")
    if [ "$NGPUS" -gt 1 ]; then
        echo "GPU ${gpu}: PID=${PID}  pairs=${pairs_for_gpu}"
        echo "         log=${LOG}"
    else
        echo "PID=${PID}  log=${LOG}"
    fi
done

echo ""

if $FOREGROUND; then
    echo "Streaming output (Ctrl-C to detach; jobs continue in background)..."
    echo "---"
    sleep 1
    tail -f "${LOG_FILES[@]}" &
    TAIL_PID=$!
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    kill "$TAIL_PID" 2>/dev/null
    wait "$TAIL_PID" 2>/dev/null
    echo ""
    echo "All jobs finished."
else
    if [ ${#LOG_FILES[@]} -gt 1 ]; then
        echo "Monitor all:  tail -f ${LOG_FILES[*]}"
    elif [ ${#LOG_FILES[@]} -eq 1 ]; then
        echo "Monitor:      tail -f ${LOG_FILES[0]}"
    fi
    echo "To stop all:  kill ${PIDS[*]}"
fi
