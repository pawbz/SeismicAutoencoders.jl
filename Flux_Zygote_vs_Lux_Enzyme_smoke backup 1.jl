### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook
# outside Pluto, this mock gives bound variables a default value.
macro bind(def, element)
    return quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 00000001-0000-0000-0000-000000000001
begin
    using ADTypes
    using BenchmarkTools
    using CUDA
    using Enzyme
    using Flux
    using Lux
    using NNlib
    using Optimisers
    using PlutoUI
    using Random
    using Statistics
    using Zygote
end

# ╔═╡ 00000002-0000-0000-0000-000000000001
md"""
# Flux/Zygote vs Lux/Enzyme Smoke Test

Minimal encoder-only benchmark on dummy 1D waveform batches.

The goal is not model quality. The goal is to check whether a Lux migration is
worth deeper testing for your VQ-VAE-style encoder path.
"""

# ╔═╡ 00000003-0000-0000-0000-000000000001
begin
    rng = MersenneTwister(20260502)
    CUDA.allowscalar(false)
    use_cuda = CUDA.functional()
    dev = use_cuda ? gpu : identity
    device_label = use_cuda ? "CUDA GPU" : "CPU"
end

# ╔═╡ 00000004-0000-0000-0000-000000000001
md"Device: **$(device_label)**"

# ╔═╡ 00000005-0000-0000-0000-000000000001
begin
    nt = 2048
    batchsize = 128
    channels = 1
    encoder_widths = (8, 16, 24)
    kernels = (17, 9, 5)
end

# ╔═╡ 00000006-0000-0000-0000-000000000001
function dummy_waveforms(rng, nt::Int, batchsize::Int)
    t = reshape(Float32.(range(0, 1; length=nt)), nt, 1, 1)
    freqs = reshape(rand(rng, Float32, 1, 1, batchsize) .* 8f0 .+ 2f0, 1, 1, batchsize)
    phases = reshape(rand(rng, Float32, 1, 1, batchsize) .* 2f0 .* Float32(pi), 1, 1, batchsize)
    x = sin.(2f0 .* Float32(pi) .* freqs .* t .+ phases)
    x .+= 0.1f0 .* randn(rng, Float32, nt, 1, batchsize)
    return x
end

# ╔═╡ 00000007-0000-0000-0000-000000000001
x_cpu = dummy_waveforms(rng, nt, batchsize)

# ╔═╡ 00000008-0000-0000-0000-000000000001
x = dev(x_cpu)

# ╔═╡ 00000009-0000-0000-0000-000000000001
activation(x) = NNlib.leakyrelu(x, 0.1f0)

# ╔═╡ 0000000a-0000-0000-0000-000000000001
function build_flux_encoder()
    Flux.Chain(
        Flux.Conv((kernels[1],), channels => encoder_widths[1], activation; pad=Flux.SamePad()),
        Flux.Conv((kernels[2],), encoder_widths[1] => encoder_widths[2], activation; pad=Flux.SamePad()),
        Flux.Conv((kernels[3],), encoder_widths[2] => encoder_widths[3], activation; pad=Flux.SamePad()),
        Flux.Conv((1,), encoder_widths[3] => encoder_widths[3]; pad=Flux.SamePad()),
    )
end

# ╔═╡ 0000000b-0000-0000-0000-000000000001
function build_lux_encoder()
    Lux.Chain(
        Lux.Conv((kernels[1],), channels => encoder_widths[1], activation; pad=Lux.SamePad()),
        Lux.Conv((kernels[2],), encoder_widths[1] => encoder_widths[2], activation; pad=Lux.SamePad()),
        Lux.Conv((kernels[3],), encoder_widths[2] => encoder_widths[3], activation; pad=Lux.SamePad()),
        Lux.Conv((1,), encoder_widths[3] => encoder_widths[3]; pad=Lux.SamePad()),
    )
end

# ╔═╡ 0000000c-0000-0000-0000-000000000001
begin
    flux_model = build_flux_encoder() |> dev
    lux_model = build_lux_encoder()
    lux_ps, lux_st = Lux.setup(rng, lux_model)
    lux_ps = dev(lux_ps)
    lux_st = dev(lux_st)
end

# ╔═╡ 0000000d-0000-0000-0000-000000000001
flux_loss(m, x) = mean(abs2, m(x))

# ╔═╡ 0000000e-0000-0000-0000-000000000001
function lux_loss(model, ps, st, x)
    y, st = Lux.apply(model, x, ps, st)
    return mean(abs2, y), st, (;)
end

# ╔═╡ 0000000f-0000-0000-0000-000000000001
function flux_forward(m, x)
    y = m(x)
    use_cuda && CUDA.synchronize()
    return y
end

# ╔═╡ 00000010-0000-0000-0000-000000000001
function flux_zygote_gradient(m, x)
    grad = Flux.gradient(m) do mm
        flux_loss(mm, x)
    end
    use_cuda && CUDA.synchronize()
    return grad[1]
end

# ╔═╡ 00000011-0000-0000-0000-000000000001
function lux_forward(model, ps, st, x)
    y, st = Lux.apply(model, x, ps, st)
    use_cuda && CUDA.synchronize()
    return y, st
end

# ╔═╡ 00000012-0000-0000-0000-000000000001
function lux_enzyme_gradient(model, ps, st, x)
    tstate = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(1f-3))
    gs, loss, stats, tstate = Lux.Training.compute_gradients(
        ADTypes.AutoEnzyme(), lux_loss, x, tstate
    )
    use_cuda && CUDA.synchronize()
    return gs, loss, stats
end

# ╔═╡ 00000013-0000-0000-0000-000000000001
function lux_enzyme_step(model, ps, st, x)
    tstate = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(1f-3))
    gs, loss, stats, tstate = Lux.Training.single_train_step!(
        ADTypes.AutoEnzyme(), lux_loss, x, tstate
    )
    use_cuda && CUDA.synchronize()
    return loss, tstate
end

# ╔═╡ 00000014-0000-0000-0000-000000000001
function try_result(label, f)
    try
        value = f()
        return (; label, ok=true, value, error="")
    catch err
        return (; label, ok=false, value=nothing, error=sprint(showerror, err))
    end
end

# ╔═╡ 00000015-0000-0000-0000-000000000001
warmup_results = [
    try_result("Flux forward", () -> flux_forward(flux_model, x)),
    try_result("Flux/Zygote gradient", () -> flux_zygote_gradient(flux_model, x)),
    try_result("Lux forward", () -> lux_forward(lux_model, lux_ps, lux_st, x)),
    try_result("Lux/Enzyme gradient", () -> lux_enzyme_gradient(lux_model, lux_ps, lux_st, x)),
    try_result("Lux/Enzyme train step", () -> lux_enzyme_step(lux_model, lux_ps, lux_st, x)),
]

# ╔═╡ 00000016-0000-0000-0000-000000000001
md"""
## Smoke Status

$(join(["- $(r.label): $(r.ok ? "ok" : "failed: $(r.error)")" for r in warmup_results], "\n"))
"""

# ╔═╡ 00000017-0000-0000-0000-000000000001
run_benchmarks_button = @bind run_benchmarks CounterButton("Run benchmarks")

# ╔═╡ 00000018-0000-0000-0000-000000000001
function benchmark_if_ok(label, ok, f)
    ok || return (; label, ok=false, benchmark=nothing, error="warmup failed")
    try
        b = @benchmark $f()
        return (; label, ok=true, benchmark=b, error="")
    catch err
        return (; label, ok=false, benchmark=nothing, error=sprint(showerror, err))
    end
end

# ╔═╡ 00000019-0000-0000-0000-000000000001
benchmarks = let
    run_benchmarks
    flux_fwd_ok = warmup_results[1].ok
    flux_grad_ok = warmup_results[2].ok
    lux_fwd_ok = warmup_results[3].ok
    lux_grad_ok = warmup_results[4].ok
    lux_step_ok = warmup_results[5].ok
    [
        benchmark_if_ok("Flux forward", flux_fwd_ok, () -> flux_forward(flux_model, x)),
        benchmark_if_ok("Flux/Zygote gradient", flux_grad_ok, () -> flux_zygote_gradient(flux_model, x)),
        benchmark_if_ok("Lux forward", lux_fwd_ok, () -> lux_forward(lux_model, lux_ps, lux_st, x)),
        benchmark_if_ok("Lux/Enzyme gradient", lux_grad_ok, () -> lux_enzyme_gradient(lux_model, lux_ps, lux_st, x)),
        benchmark_if_ok("Lux/Enzyme train step", lux_step_ok, () -> lux_enzyme_step(lux_model, lux_ps, lux_st, x)),
    ]
end

# ╔═╡ 0000001a-0000-0000-0000-000000000001
function bench_summary(row)
    if !row.ok
        return (label=row.label, status="failed", median_ms=NaN, memory_kib=NaN, error=row.error)
    end
    b = row.benchmark
    return (
        label=row.label,
        status="ok",
        median_ms=BenchmarkTools.median(b).time / 1e6,
        memory_kib=BenchmarkTools.median(b).memory / 1024,
        error="",
    )
end

# ╔═╡ 0000001b-0000-0000-0000-000000000001
benchmark_summary = bench_summary.(benchmarks)

# ╔═╡ 0000001c-0000-0000-0000-000000000001
md"""
## Benchmark Summary

$(join(["- $(r.label): $(r.status), median=$(round(r.median_ms; digits=3)) ms, memory=$(round(r.memory_kib; digits=1)) KiB$(isempty(r.error) ? "" : ", error=$(r.error)")" for r in benchmark_summary], "\n"))
"""

# ╔═╡ 0000001d-0000-0000-0000-000000000001
md"""
## Notes

- This intentionally benchmarks only a small 1D convolutional encoder.
- Flux path: `Flux.Chain` + `Flux.gradient` / Zygote.
- Lux path: `Lux.Chain` + `Lux.Training.compute_gradients(ADTypes.AutoEnzyme(), ...)`.
- If Lux/Enzyme fails but Lux forward works, the failure is an AD/backend compatibility signal, not a model-shape problem.
- If GPU timings look suspiciously small, rerun the benchmark cell; the functions call `CUDA.synchronize()` when CUDA is active.
"""

# ╔═╡ Cell order:
# ╠═00000001-0000-0000-0000-000000000001
# ╟─00000002-0000-0000-0000-000000000001
# ╠═00000003-0000-0000-0000-000000000001
# ╟─00000004-0000-0000-0000-000000000001
# ╠═00000005-0000-0000-0000-000000000001
# ╠═00000006-0000-0000-0000-000000000001
# ╠═00000007-0000-0000-0000-000000000001
# ╠═00000008-0000-0000-0000-000000000001
# ╠═00000009-0000-0000-0000-000000000001
# ╠═0000000a-0000-0000-0000-000000000001
# ╠═0000000b-0000-0000-0000-000000000001
# ╠═0000000c-0000-0000-0000-000000000001
# ╠═0000000d-0000-0000-0000-000000000001
# ╠═0000000e-0000-0000-0000-000000000001
# ╠═0000000f-0000-0000-0000-000000000001
# ╠═00000010-0000-0000-0000-000000000001
# ╠═00000011-0000-0000-0000-000000000001
# ╠═00000012-0000-0000-0000-000000000001
# ╠═00000013-0000-0000-0000-000000000001
# ╠═00000014-0000-0000-0000-000000000001
# ╠═00000015-0000-0000-0000-000000000001
# ╟─00000016-0000-0000-0000-000000000001
# ╠═00000017-0000-0000-0000-000000000001
# ╠═00000018-0000-0000-0000-000000000001
# ╠═00000019-0000-0000-0000-000000000001
# ╠═0000001a-0000-0000-0000-000000000001
# ╠═0000001b-0000-0000-0000-000000000001
# ╟─0000001c-0000-0000-0000-000000000001
# ╟─0000001d-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ADTypes = "47edcb42-4c32-461d-b0d4-f288d3bc2ce3"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5b0c-9f4c-02a8a2a7b438"
Statistics = "10745b16-90e8-5b5f-9d7d-2c0c31e46c0f"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
ADTypes = "~1.18"
BenchmarkTools = "~1.7"
CUDA = "~5.11"
Enzyme = "~0.13"
Flux = "~0.16"
Lux = "~1.20"
NNlib = "~0.9"
Optimisers = "~0.4"
PlutoUI = "~0.7"
Zygote = "~0.7"
"""

PLUTO_MANIFEST_TOML_CONTENTS = """
# This notebook intentionally does not pin a manifest.
# Pluto will resolve the compact project above in the active Julia depot.
"""
