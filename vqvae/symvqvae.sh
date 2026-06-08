#!/bin/bash
#
# symvqvae.sh — launch SymVQVAE training on one or more GPUs.
#
# For the train subcommand, station pairs are split round-robin across GPUs.
# One Julia process is launched per GPU; it compiles the XLA graph once and
# trains all its assigned pairs × seeds sequentially, reusing the compiled graph.
#
# Usage:
#   symvqvae train [pairs] [--gpus LIST] [options]
#   symvqvae inspect PAIR [options]
#   symvqvae --help
#
# GPU options (train only):
#   --gpus LIST        Comma-separated GPU indices (default: "0")
#                      e.g. --gpus 0,1  or  --gpus 0,1,2,3
#   --foreground, -f   Stream all GPU log files to the terminal in real time.
#                      Logs are still written to files. Blocks until all done.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

JULIA="${HOME}/.juliaup/bin/julialauncher"
if [ ! -x "$JULIA" ]; then
    JULIA="julia"
fi

# ── Fast path: help, inspect, no args ────────────────────────────────────────
if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ "$1" = "inspect" ]; then
    "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" "$@"
    exit $?
fi

# ── list-pairs: optionally show GPU distribution if --gpus passed ─────────────
if [ "$1" = "list-pairs" ]; then
    shift  # consume "list-pairs"
    LP_DATA_DIR="$(pwd)"
    LP_GPUS=""
    LP_PASSTHROUGH=()
    i=1
    while [ $i -le $# ]; do
        arg="${!i}"
        if [ "$arg" = "--data-dir" ]; then
            i=$((i+1)); LP_DATA_DIR="${!i}"
            LP_PASSTHROUGH+=("--data-dir" "${!i}")
        elif [ "$arg" = "--gpus" ]; then
            i=$((i+1)); LP_GPUS="${!i}"
        else
            LP_PASSTHROUGH+=("$arg")
        fi
        i=$((i+1))
    done

    if [ -z "$LP_GPUS" ]; then
        # No --gpus: plain listing from Julia
        "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" \
            list-pairs "${LP_PASSTHROUGH[@]}"
        exit $?
    fi

    # With --gpus: discover pairs then show round-robin distribution
    ALL_PAIRS_LP=$(
        "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" \
            list-pairs --data-dir "$LP_DATA_DIR" 2>/dev/null \
            | grep -E '^\s+[A-Za-z0-9]+-[A-Za-z0-9]+' \
            | sed 's/^[[:space:]]*//'
    )
    if [ -z "$ALL_PAIRS_LP" ]; then
        echo "No station pairs found in ${LP_DATA_DIR}"
        exit 0
    fi
    mapfile -t ALL_PAIRS_ARR <<< "$ALL_PAIRS_LP"
    NPAIRS_LP=${#ALL_PAIRS_ARR[@]}
    IFS=',' read -ra LP_GPU_IDS <<< "$LP_GPUS"
    NGPUS_LP=${#LP_GPU_IDS[@]}

    declare -a LP_GPU_PAIRS
    for ((i=0; i<NGPUS_LP; i++)); do LP_GPU_PAIRS[$i]=""; done
    for ((p=0; p<NPAIRS_LP; p++)); do
        slot=$((p % NGPUS_LP))
        pair="${ALL_PAIRS_ARR[$p]}"
        if [ -z "${LP_GPU_PAIRS[$slot]}" ]; then
            LP_GPU_PAIRS[$slot]="$pair"
        else
            LP_GPU_PAIRS[$slot]="${LP_GPU_PAIRS[$slot]},${pair}"
        fi
    done

    echo "${NPAIRS_LP} pair(s) in ${LP_DATA_DIR} — distribution across ${NGPUS_LP} GPU(s) [${LP_GPUS}]:"
    echo ""
    for ((i=0; i<NGPUS_LP; i++)); do
        gpu="${LP_GPU_IDS[$i]}"
        pairs_str="${LP_GPU_PAIRS[$i]}"
        IFS=',' read -ra pair_list <<< "$pairs_str"
        echo "  GPU ${gpu} (${#pair_list[@]} pair(s)):"
        for p in "${pair_list[@]}"; do echo "    $p"; done
    done
    exit 0
fi

# ── Only the train subcommand gets multi-GPU logic ───────────────────────────
if [ "$1" != "train" ]; then
    # Unknown subcommand — forward as-is
    "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" "$@"
    exit $?
fi
shift  # consume "train"

# ── Parse train arguments ─────────────────────────────────────────────────────
GPUS="0"
DATA_DIR="$(pwd)"
PAIRS_ARG=""
FOREGROUND=false
PASSTHROUGH=()

i=1
while [ $i -le $# ]; do
    arg="${!i}"
    if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
        "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" train --help
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
    elif [ "$arg" = "--save-dir" ] || \
         [ "$arg" = "--nepoch" ] || \
         [ "$arg" = "--batchsize" ] || \
         [ "$arg" = "--Nmax" ] || \
         [ "$arg" = "--lr" ] || \
         [ "$arg" = "--seeds" ] || \
         [ "$arg" = "--periods" ] || \
         [ "$arg" = "--dt" ] || \
         [ "$arg" = "--K" ] || \
         [ "$arg" = "--d" ] || \
         [ "$arg" = "--n-filters" ] || \
         [ "$arg" = "--ratios" ] || \
         [ "$arg" = "--n-residual-layers" ] || \
         [ "$arg" = "--entropy-weight" ] || \
         [ "$arg" = "--whitening-kernel-length" ] || \
         [ "$arg" = "--autodiff-backend" ]; then
        i=$((i+1))
        if [ $i -gt $# ]; then
            echo "Error: missing value for ${arg}" >&2
            exit 1
        fi
        PASSTHROUGH+=("$arg" "${!i}")
    elif [[ "$arg" != --* && "$arg" != -* && -z "$PAIRS_ARG" ]]; then
        # First positional arg is the pairs specification
        PAIRS_ARG="$arg"
    else
        PASSTHROUGH+=("$arg")
    fi
    i=$((i+1))
done

# ── Single-GPU shortcut (no pair splitting needed) ───────────────────────────
IFS=',' read -ra GPU_IDS <<< "$GPUS"
NGPUS=${#GPU_IDS[@]}

if [ "$NGPUS" -eq 1 ]; then
    PAIRS_PASSTHROUGH=()
    [ -n "$PAIRS_ARG" ] && PAIRS_PASSTHROUGH=("$PAIRS_ARG")

    if [ "${SYMVQVAE_DRY_RUN:-}" = "1" ]; then
        printf 'CUDA_VISIBLE_DEVICES=%s\n' "${GPU_IDS[0]}"
        printf 'julia_args:'
        printf ' %q' train "${PAIRS_PASSTHROUGH[@]}" "${PASSTHROUGH[@]}"
        printf '\n'
        exit 0
    fi

    if $FOREGROUND; then
        CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
            "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" \
            train "${PAIRS_PASSTHROUGH[@]}" "${PASSTHROUGH[@]}"
        exit $?
    fi

    LOG_FILE="${RUN_DIR}/symvqvae_$(date +%Y%m%d_%H%M%S).out"
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" \
        "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" \
        train "${PAIRS_PASSTHROUGH[@]}" "${PASSTHROUGH[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "Training started in background (PID: $PID)"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "To monitor: tail -f $LOG_FILE"
    echo "To stop:    kill $PID"
    exit 0
fi

# ── Multi-GPU path: discover pairs and split round-robin ─────────────────────

# Discover available pairs from the data directory (lightweight — no Lux/Reactant)
ALL_PAIRS_RAW=$(
    "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" \
        list-pairs --data-dir "$DATA_DIR" 2>/dev/null \
        | grep -E '^\s+[A-Za-z0-9]+-[A-Za-z0-9]+' \
        | sed 's/^[[:space:]]*//'
)

if [ -z "$ALL_PAIRS_RAW" ]; then
    echo "Error: no station pairs found in ${DATA_DIR}" >&2
    exit 1
fi

# Build candidate pair list
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

# Split pairs round-robin across GPUs
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

echo "Splitting ${NPAIRS} pair(s) across ${NGPUS} GPUs: ${GPUS}"
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

    LOG="${RUN_DIR}/symvqvae_gpu${gpu}_${TIMESTAMP}.out"
    LOG_FILES+=("$LOG")

    CUDA_VISIBLE_DEVICES=$gpu \
        "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" \
        train "$pairs_for_gpu" "${PASSTHROUGH[@]}" > "$LOG" 2>&1 &

    PID=$!
    PIDS+=("$PID")
    echo "GPU ${gpu}: PID=${PID}  pairs=${pairs_for_gpu}"
    echo "         log=${LOG}"
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
    echo "Monitor all:  tail -f ${LOG_FILES[*]}"
    echo "To stop all:  kill ${PIDS[*]}"
fi
