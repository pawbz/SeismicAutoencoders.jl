#!/usr/bin/env julia
# Batch driver: run N independent full-resolution simulations, each with its own
# random seed (=> different scatterer medium), sharing ONE worker pool. Each run
# produces one impulse-response gather (n_sources=1) written to data/ as
# event_s<seed>_k1.{jld2,png}.
#
# Usage:  julia -p <N> run_batch.jl [n_runs]     (default n_runs = 10)
#
# Sequential runs share the worker pool so the per-frequency pmap always has full
# parallelism; launching all runs at once would oversubscribe the cores.

using Distributed
if nworkers() == 1
    addprocs()
end
@everywhere import Pkg
@everywhere Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "generate_mscatter_data.jl"))

n_runs = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10

@info "Batch start" n_runs nworkers = nworkers()
all_files = String[]
for i in 1:n_runs
    @info "════════ Run $i / $n_runs ════════"
    cfg = MScatterConfig(n_sources=1)     # fresh random seed => new medium
    @info "run config" seed = cfg.seed Nt = cfg.Nt res = cfg.res
    files = generate_dataset(cfg)
    append!(all_files, files)
    flush(stderr)
end
@info "BATCH COMPLETE" n_files = length(all_files)
println("BATCHDONE $(length(all_files)) files")
