#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$(pwd)"

# Use julialauncher from juliaup to get Julia 1.12.6 (respects `juliaup default`)
JULIA="${HOME}/.juliaup/bin/julialauncher"
if [ ! -x "$JULIA" ]; then
    JULIA="julia"  # fallback to system julia if julialauncher not found
fi

# Check if this is a trivial operation (--help, inspect)
for arg in "$@"; do
    if [[ "$arg" == "--help" ]] || [[ "$arg" == "-h" ]] || [[ "$arg" == "inspect" ]]; then
        # Run info/inspect operations in foreground
        "$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" "$@"
        exit $?
    fi
done

# For training, run in background with logging
LOG_FILE="${RUN_DIR}/symvqvae_$(date +%Y%m%d_%H%M%S).out"

"$JULIA" --project="${SCRIPT_DIR}/.." "${SCRIPT_DIR}/symvqvae.jl" "$@" > "$LOG_FILE" 2>&1 &

PID=$!

echo "Training started in background (PID: $PID)"
echo "Log file: $LOG_FILE"
echo ""
echo "To monitor: tail -f $LOG_FILE"
echo "To check status: ps aux | grep $PID"
echo "To stop: kill $PID"

exit 0
