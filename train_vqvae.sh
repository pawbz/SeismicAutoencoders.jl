#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get the directory where the script is being run from
RUN_DIR="$(pwd)"

# Check if this is a trivial operation (--help, --list-pairs, --sample-pair)
for arg in "$@"; do
    if [[ "$arg" == "--help" ]] || [[ "$arg" == "-h" ]] || [[ "$arg" == "--list-pairs" ]] || [[ "$arg" == "-l" ]] || [[ "$arg" == "--sample-pair" ]]; then
        # Run trivial operations in foreground
        julia "${SCRIPT_DIR}/train_vqvae.jl" "$@"
        exit $?
    fi
done

# For training, run in background with logging
LOG_FILE="${RUN_DIR}/train_vqvae_$(date +%Y%m%d_%H%M%S).out"

# Run the Julia script in the background and redirect output to log file
julia "${SCRIPT_DIR}/train_vqvae.jl" "$@" > "$LOG_FILE" 2>&1 &

# Get the PID of the background process
PID=$!

echo "Training started in background (PID: $PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor: tail -f $LOG_FILE"
echo "To check status: ps aux | grep $PID"
echo "To stop: kill $PID"

exit 0
