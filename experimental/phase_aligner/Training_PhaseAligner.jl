### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 4b3a86f2-3b14-11f1-a498-c9fc074b7df6
begin
    using CUDA, cuDNN, Flux, Zygote, FFTW
    using Optimisers, Random, Statistics, LinearAlgebra
    using JLD2, DSP, ProgressLogging
    using PlutoUI, PlutoHooks, PlutoLinks, PlutoPlotly
    CUDA.device!(0)
end

# ╔═╡ 4b3a88c8-3b14-11f1-903e-7d32036b9cd0
using PlutoLinks: @ingredients

# ╔═╡ 4b3a8918-3b14-11f1-8a03-c593d58ce280
TableOfContents(include_definitions=true)

# ╔═╡ 4b3a8954-3b14-11f1-adfa-b11a44aa35cf
xpu = gpu

# ╔═╡ 4b3a8990-3b14-11f1-a14d-8fcdc511b3b0
md"""# Phase Aligner — Training Notebook

Train a **Design C siamese phase aligner** to self-supervisedly align a set of
waveforms to a common frame, then compute the coherent stack.

## Workflow
1. Load / generate waveforms
2. Build model from `PhaseAligner_architecture.jl`
3. **Phase 1** — equivariance pre-training (no baseline needed)
4. **Phase 2** — EMA baseline alignment
5. Plot diagnostics: φ distribution, shift scatter, aligned stack
"""

# ╔═╡ 4b3a8b2a-3b14-11f1-90be-218eee387b52
md"## Load Architecture"

# ╔═╡ 4b3a8b72-3b14-11f1-8f89-2b2a746100d7
pa = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/PhaseAligner_architecture.jl")

# ╔═╡ 4b3a8bb6-3b14-11f1-9c74-2f9dd23ccb58
md"---
## Synthetic Validation

Generate Ricker waveforms with known shifts to verify the aligner works before
running on real data.
"

# ╔═╡ 4b3a8c60-3b14-11f1-ae5d-b7ed043ed9b9
begin
    # ── Synthetic dataset parameters ─────────────────────────────────────
    nt_syn        = 300
    f0_syn        = 0.05f0          # dominant frequency (cycles/sample)
    noise_std_syn = 0.3f0           # noise level relative to peak
    N_syn         = 800
    max_shift_syn = 40              # ±40 samples true shift range
    rng_syn       = Random.MersenneTwister(42)
end

# ╔═╡ 4b3a8e10-3b14-11f1-b90b-9ddb16ca3290
"""
Generate N Ricker waveforms with known true shifts.
Returns `(X_noisy, tau_true_samples)` — both on CPU, Float32.
"""
function make_ricker_dataset_syn(; nt=nt_syn, f0=f0_syn,
                                   noise_std=noise_std_syn,
                                   N=N_syn, max_shift=max_shift_syn,
                                   rng=rng_syn)
    t  = collect(Float32, range(-(nt/2), nt/2; length=nt))
    w0 = @. (1f0 - 2f0*Float32(π)^2*f0^2*t^2) * exp(-Float32(π)^2*f0^2*t^2)
    w0 ./= max(maximum(abs.(w0)), 1f-10)

    tau = Float32.(rand(rng, Float64, N) .* 2 .- 1) .* Float32(max_shift)

    # Fourier shift on CPU
    sg  = -im .* Float32.(fftfreq(nt) .* 2π)
    tau_mat = reshape(tau, 1, N)
    W   = repeat(w0, 1, N)
    W_s = real(ifft(fft(W, 1) .* exp.(sg .* tau_mat), 1))

    noise = noise_std .* randn(rng, Float32, nt, N)
    W_n   = W_s .+ noise

    # Per-trace zero-mean unit-std normalization
    m = mean(W_n; dims=1); s = std(W_n; dims=1)
    W_norm = (W_n .- m) ./ max.(s, 1f-8)
    return Float32.(W_norm), tau
end

# ╔═╡ 4b3a91b0-3b14-11f1-9552-0925253b8f86
X_syn, tau_true_syn = make_ricker_dataset_syn()

# ╔═╡ 4b3a91f6-3b14-11f1-b940-2984fb5fb327
begin
    ntrain_syn = round(Int, 0.85 * N_syn)
    idx_syn    = randperm(rng_syn, N_syn)
    X_train_syn = xpu(X_syn[:, idx_syn[1:ntrain_syn]])
    X_test_syn  = xpu(X_syn[:, idx_syn[ntrain_syn+1:end]])
    tau_true_train_syn = tau_true_syn[idx_syn[1:ntrain_syn]]
    tau_true_test_syn  = tau_true_syn[idx_syn[ntrain_syn+1:end]]
end;

# ╔═╡ f70f832a-20f0-4ce5-a476-2f0078407de7
plot(cpu(X_train_syn[:, 1:10]))

# ╔═╡ 4b3a9340-3b14-11f1-9734-95d754f356e5
md"## Model Parameters"

# ╔═╡ 4b3a9386-3b14-11f1-a082-796d75cf2b46
pa_params_syn = pa.PhaseAligner_Para(
    nt              = nt_syn,
    max_shift_samples = max_shift_syn + 10,   # a bit wider than true range
    enc_kernels     = [64, 32, 16],
    enc_filters     = [8, 16, 32],
    gamma           = 0.0001f0,
    ema_decay       = 0.99f0,
    seed            = 42,
)

# ╔═╡ 4b3a949e-3b14-11f1-ad90-f37908baa471
reload_syn_button = @bind reload_syn_pa CounterButton("Reload Model")

# ╔═╡ 4b3a94e6-3b14-11f1-b11e-a33f4ef14e4c
model_syn_pa = @use_memo([reload_syn_pa, pa_params_syn]) do
    reload_syn_pa
    pa.get_phase_aligner(pa_params_syn)
end

# ╔═╡ 4b3a9566-3b14-11f1-af48-23a204f629b2
md"## Training"

# ╔═╡ 4b3a95a2-3b14-11f1-8332-39ec2b9e7016
pa_training_para_syn = pa.PhaseAligner_Training_Para(
    batchsize         = 256,
    nepoch_phase1     = 150,
    nepoch_phase2     = 400,
    initial_lr_phase1 = 0.01,
    initial_lr_phase2 = 0.01,
    restart_period    = 50,
    nprint            = 10,
)

# ╔═╡ 4b3a9688-3b14-11f1-9840-233591c7a890
trained_syn_pa = @use_memo([]) do
    pa.train_phase_aligner(
        model_syn_pa,
        X_train_syn, X_test_syn,
        pa_params_syn, pa_training_para_syn,
    )
end

# ╔═╡ 4b3a973c-3b14-11f1-a221-352efa534949
baseline_syn, loss_history_syn_pa = trained_syn_pa

# ╔═╡ 4b3a9778-3b14-11f1-aa28-9511b4c8a5b8
md"## Diagnostics"

# ╔═╡ 4b3a97b6-3b14-11f1-ac19-314a898ba2eb
md"### Loss curves"

# ╔═╡ 4b3a97f0-3b14-11f1-a539-85bead0164cc
let
    h = loss_history_syn_pa
    n1 = pa_training_para_syn.nepoch_phase1
    n2 = pa_training_para_syn.nepoch_phase2
    e1 = 1:n1
    e2 = n1+1:n1+n2
    font = attr(family="Computer Modern, serif")
    traces = [
        PlutoPlotly.scatter(x=collect(e1), y=h.train_total[e1], mode="lines",
            name="Train Phase 1", line=attr(color="#1f77b4", width=1.5)),
        PlutoPlotly.scatter(x=collect(e1), y=h.test_total[e1], mode="lines",
            name="Test Phase 1", line=attr(color="#1f77b4", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=collect(e2), y=h.train_total[e2], mode="lines",
            name="Train Phase 2", line=attr(color="#d62728", width=1.5)),
        PlutoPlotly.scatter(x=collect(e2), y=h.test_total[e2], mode="lines",
            name="Test Phase 2", line=attr(color="#d62728", width=1.5, dash="dash")),
    ]
    layout = Layout(
        title=attr(text="Training loss (Phase 1 + Phase 2)", font=merge(font, attr(size=16))),
        xaxis=attr(title="Epoch", type="linear"),
        yaxis=attr(title="Loss", type="log"),
        height=350, width=900,
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=[attr(type="line", x0=n1, x1=n1, y0=0, y1=1,
                     yref="paper", line=attr(color="grey", dash="dot", width=1))],
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 4b3a9c5a-3b14-11f1-80cc-8bb641826f56
md"### φ distribution"

# ╔═╡ 4b3a9c96-3b14-11f1-b7e0-35fd57855f6f
let
    phi_train = vec(cpu(model_syn_pa(X_train_syn)))
    bins = range(minimum(phi_train) - 1, maximum(phi_train) + 1; length=40)
    counts = [sum(bins[i] .<= phi_train .< bins[i+1]) for i in 1:length(bins)-1]
    centers = [(bins[i] + bins[i+1]) / 2 for i in 1:length(bins)-1]
    traces = [
        PlutoPlotly.bar(x=centers, y=counts, name="φ(x_i)",
            marker=attr(color="#1f77b4", opacity=0.7)),
        PlutoPlotly.scatter(x=[baseline_syn.phi_baseline, baseline_syn.phi_baseline],
            y=[0, maximum(counts)], mode="lines",
            name="φ_B (baseline)", line=attr(color="red", dash="dash", width=2)),
    ]
    layout = Layout(
        title=attr(text="Phase coordinate distribution (train set)"),
        xaxis=attr(title="φ (samples)"), yaxis=attr(title="Count"),
        height=300, width=800, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 4b3a9fb6-3b14-11f1-a24c-2fd152d6ce55
md"### True vs predicted shift scatter"

# ╔═╡ 4b3a9ff4-3b14-11f1-abb8-472e9a4900cc
let
    tau_pred_train, _ = pa.predict_shifts(model_syn_pa, X_train_syn, baseline_syn)
    tau_pred = vec(cpu(tau_pred_train))
    tau_true = tau_true_train_syn

    # Scatter: predicted vs true
    lim = max(maximum(abs.(tau_true)), maximum(abs.(tau_pred))) * 1.1
    ref = PlutoPlotly.scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
        name="ideal", line=attr(color="grey", dash="dot", width=1))
    sc  = PlutoPlotly.scatter(x=tau_true, y=tau_pred, mode="markers",
        name="train", marker=attr(size=4, color="#1f77b4", opacity=0.6))
    layout = Layout(
        title=attr(text="True shift vs predicted shift (samples)"),
        xaxis=attr(title="τ_true (samples)", range=[-lim, lim]),
        yaxis=attr(title="τ_pred (samples)", range=[-lim, lim]),
        height=400, width=450, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot([ref, sc], layout)
end

# ╔═╡ e8e91d6b-72d0-4493-aa4a-bb0e36deda9e
md"### Coherent stack"

# ╔═╡ 5142d77f-3232-4b70-b3e9-86b668972529
function taper(x)
    nt = size(x, 1)
    w = cat(DSP.tukey(nt, 0.1), dims=ndims(x))
    return x .* w
end

# ╔═╡ 4b3aa33a-3b14-11f1-856e-7f736cf0e1cd
let
    tau_all, X_aligned_all = pa.apply_aligner(model_syn_pa, X_syn, baseline_syn)
    stack_aligned = pa.coherent_stack(X_aligned_all)

    # Raw mean (unaligned) for comparison
    stack_raw = vec(mean(X_syn; dims=2))

    ts = 1:nt_syn
    traces = [
        PlutoPlotly.scatter(x=ts, y=stack_raw, mode="lines",
            name="Raw mean (unaligned)", line=attr(color="grey", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=ts, y=stack_aligned, mode="lines",
            name="Aligned stack (PhaseAligner)", line=attr(color="#1f77b4", width=2)),
    ]
    layout = Layout(
        title=attr(text="Coherent stack: aligned vs raw mean"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 4b3aa61e-3b14-11f1-b62c-919b45f517b5
md"---
## Real Data Section

Replace the synthetic data above with your real station waveforms.
The workflow is identical — just substitute `X_train_syn` / `X_test_syn`
with your preprocessed waveform matrix `(nt, N)`.
"

# ╔═╡ 9f87709f-e8ba-4dcc-9c73-c0a866fe801e
 fldir = "/mnt/NAS/EQData/RFData"

# ╔═╡ 4b3aa704-3b14-11f1-abfd-6514b40c32d7
# ╠═╡ disabled = true
#=╠═╡
begin
   
    dfile = "$(fldir)/Syn410Ps_snr1.5.jld2"
    EqR      = load(dfile, "Syn")["Data"]
    TrueRF   = load(dfile, "Syn")["TrueRF"][1][451:750, :]
    StaName  = load(dfile, "Syn")["Sta"][1]
    EvtLoc   = load(dfile, "Syn")["EventLoc"]
	    s_sel = "1"
    stik  = findall(x -> x == s_sel, StaName)[1]
    raw_data = EqR[stik][451:750, :]   # (nt_real, N_real) trim for 410Ps window
	StaN = basename(dfile)
end
  ╠═╡ =#

# ╔═╡ f941a320-d11e-49d8-8277-5673417a283c
plot(TrueRF[:, 1])

# ╔═╡ 4b3aa812-3b14-11f1-bce1-f36ed58c8131
begin




snr_tres=0.0
dn="GSN_150ZTR_Bandpass_0.05_0.8_29nov_rf_iter_f1.jld2"
snrf="GSN_150ZTR_Bandpass_0.05_0.8_29nov_snr.jld2"

# Directory on NAS1 10.130.10.240: /share/EQData/RFData
# fldir="/mnt/NAS/RFData" # mounted NAS1 on Chuong system 
dfile="$(fldir)/$dn"
         
StaName=load(dfile)["Sta"][1]
StaLoc = load(dfile)["Sta"][2]
EvtLoc=load(dfile)["EventLoc"]
EvtMg=load(dfile)["EventMag"]
EvtDep=load(dfile)["EventDep"]
ses_snr=load("$(fldir)/$snrf","SNR")


# ### one station data ######    
# StaN="PALK"      #select stations
	StaN="POHA"
	# StaN = "FURI"
ix=findall(x-> x == StaN,StaName)[1]
    
R = load(dfile)["Data"][ix]
sel_snr =findall(x-> x > snr_tres ,ses_snr[ix]) # sesimogram snr filter
R= R[:,sel_snr]

EvtLoc=EvtLoc[ix][sel_snr]
EvtMg=EvtMg[ix][sel_snr]
EvtDep=EvtDep[ix][sel_snr]
	StaLoc = StaLoc[ix]
raw_data=R[501:800,:]  # trim RF at 410Ps window
end


# ╔═╡ 9b150acb-7d71-4b0b-a452-6782c49de61a
EvtLoc

# ╔═╡ e5fab98e-620f-4a99-80d3-0bd27891cfee
StaLoc

# ╔═╡ b9d9618e-5e00-4729-94bb-20063b49ba72
begin
	  nt_real  = size(raw_data, 1)
	    N_real   = size(raw_data, 2)
end

# ╔═╡ cf24fb96-36d0-4158-887d-5d09c93c2f6b
function taper_sin(x)

    nt = size(x, 1)

    t = range(0, π, length=nt)

    w = sin.(t).^1   # exactly zero at t=0 and t=π

    w = reshape(w, nt, ntuple(_ -> 1, ndims(x)-1)...)

    return x .* w

end

# ╔═╡ 4b3aa902-3b14-11f1-83d6-750ac8faf647
begin
    # Per-trace normalization
    mr = mean(raw_data; dims=1); sr = std(raw_data; dims=1)
    X_real = Float32.(taper_sin(((raw_data .- mr) ./ max.(sr, 1f-8))))
	# X_real = randn(Float32, size(X_real)...)
end

# ╔═╡ b9316cca-dfc8-47c3-84cd-af227994f727
plot(taper_sin(ones(128)))

# ╔═╡ 28c9aee3-ecf1-4851-a75a-fc592533fe07
std(X_real, dims=1)

# ╔═╡ 4b3aa9c0-3b14-11f1-bfe7-33c9135341b1
begin
    rng_real = Random.MersenneTwister(7)
    ntrain_real = round(Int, 0.85 * N_real)
    idx_real    = randperm(rng_real, N_real)
    X_train_real = xpu(X_real[:, idx_real[1:ntrain_real]])
    X_test_real  = xpu(X_real[:, idx_real[ntrain_real+1:end]])
end;

# ╔═╡ 4b3aaac4-3b14-11f1-9f27-dfbc0c9f7b57
pa_params_real = pa.PhaseAligner_Para(
    nt                = nt_real,
    max_shift_samples = 30,          # adjust to your expected lag range
    enc_kernels       = [128, 64, 32],
    enc_filters       = [16, 32, 64],
    gamma             = 0.0001f0,
    ema_decay         = 0.99f0,
    seed              = 42,
)

# ╔═╡ 4b3aabfa-3b14-11f1-88df-17f63d97bee9
reload_real_button = @bind reload_real_pa CounterButton("Reload Real Model")

# ╔═╡ 4b3aac40-3b14-11f1-87d8-fba529fb8df1
model_real_pa = @use_memo([reload_real_pa, pa_params_real]) do
    reload_real_pa
    pa.get_phase_aligner(pa_params_real)
end

# ╔═╡ 4b3aacca-3b14-11f1-8651-951f9425536b
pa_training_para_real = pa.PhaseAligner_Training_Para(
    batchsize         = 512,
    nepoch_phase1     = 200,
    nepoch_phase2     = 1000,
    initial_lr_phase1 = 0.01,
    initial_lr_phase2 = 0.01,
    restart_period    = 50,
    nprint            = 50,
)

# ╔═╡ 4b3aadb2-3b14-11f1-bc51-9f79b497d1d3
trained_real_pa = @use_memo([]) do
    pa.train_phase_aligner(
        model_real_pa,
        X_train_real, X_test_real,
        pa_params_real, pa_training_para_real,
    )
end

# ╔═╡ 4b3aae5c-3b14-11f1-b885-cf4473314b06
baseline_real, loss_history_real_pa = trained_real_pa

# ╔═╡ 0ae8f88f-f974-4449-b065-d00c7a7252d0
let
    h = loss_history_real_pa
    n1 = pa_training_para_real.nepoch_phase1
    n2 = pa_training_para_real.nepoch_phase2
    e1 = 1:n1
    e2 = n1+1:n1+n2
    font = attr(family="Computer Modern, serif")
    traces = [
        PlutoPlotly.scatter(x=collect(e1), y=h.train_total[e1], mode="lines",
            name="Train Phase 1", line=attr(color="#1f77b4", width=1.5)),
        PlutoPlotly.scatter(x=collect(e1), y=h.test_total[e1], mode="lines",
            name="Test Phase 1", line=attr(color="#1f77b4", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=collect(e2), y=h.train_total[e2], mode="lines",
            name="Train Phase 2", line=attr(color="#d62728", width=1.5)),
        PlutoPlotly.scatter(x=collect(e2), y=h.test_total[e2], mode="lines",
            name="Test Phase 2", line=attr(color="#d62728", width=1.5, dash="dash")),
    ]
    layout = Layout(
        title=attr(text="Training loss (Phase 1 + Phase 2)", font=merge(font, attr(size=16))),
        xaxis=attr(title="Epoch", type="linear"),
        yaxis=attr(title="Loss", type="log"),
        height=350, width=900,
        plot_bgcolor="white", paper_bgcolor="white",
        shapes=[attr(type="line", x0=n1, x1=n1, y0=0, y1=1,
                     yref="paper", line=attr(color="grey", dash="dot", width=1))],
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 4b3aaea2-3b14-11f1-aad2-e7e3e09e008c
md"### Real Data: Aligned Stack"

# ╔═╡ 4b3aaed6-3b14-11f1-80e4-179f8d163898
let
    tau_all, X_aligned_all = pa.apply_aligner(model_real_pa, X_real, baseline_real)
    stack_aligned = pa.coherent_stack(X_aligned_all)
    stack_raw     = vec(mean(X_real; dims=2))

    ts = 1:nt_real
    traces = [
        PlutoPlotly.scatter(x=ts, y=stack_raw, mode="lines",
            name="Raw mean", line=attr(color="grey", width=1.5, dash="dash")),
        PlutoPlotly.scatter(x=ts, y=stack_aligned, mode="lines",
            name="Aligned mean", line=attr(color="#d62728", width=2)),
    ]
    layout = Layout(
        title=attr(text="Real data $StaN"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white",
    )
    WideCell(PlutoPlotly.plot(traces, layout))
end

# ╔═╡ 4b3ab17c-3b14-11f1-bb14-418aed6192ee
md"### Real Data: Shift Distribution"

# ╔═╡ 4b3ab1e0-3b14-11f1-a866-ef43937275d1
let
    tau_all, _ = pa.apply_aligner(model_real_pa, X_real, baseline_real)
    bins = range(minimum(tau_all) - 0.5, maximum(tau_all) + 0.5; length=41)
    counts  = [sum(bins[i] .<= tau_all .< bins[i+1]) for i in 1:length(bins)-1]
    centers = [(bins[i] + bins[i+1]) / 2 for i in 1:length(bins)-1]
    traces = [PlutoPlotly.bar(x=centers, y=counts,
        name="shifts", marker=attr(color="#d62728", opacity=0.7))]
    layout = Layout(
        title=attr(text="Predicted shift distribution (samples)"),
        xaxis=attr(title="τ (samples)"), yaxis=attr(title="Count"),
        height=300, width=700, plot_bgcolor="white", paper_bgcolor="white",
    )
    PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 2f3d2f72-7d4c-46ab-aef5-7db6f23b8d0a
md"## Multi-Station Per-Station Training"

# ╔═╡ 1b0d27a4-42af-4b2f-bf34-74d6fca5e65c
begin
    station_loop_cfg = (
        snr_threshold = 0.0,
        trim_range = (501, 800),
        min_traces = 80,
        train_frac = 0.85,
        base_seed = 17,
        max_stations = typemax(Int),
        K_eval = 8,
        eval_batchsize = 256,
    )

    station_model_template = (
        max_shift_samples = 30,
        enc_kernels = [128, 64, 32],
        enc_filters = [16, 32, 64],
        gamma = 0.0001f0,
        ema_decay = 0.99f0,
    )

    station_training_para = pa.PhaseAligner_Training_Para(
        batchsize = 512,
        nepoch_phase1 = 200,
        nepoch_phase2 = 1000,
        initial_lr_phase1 = 0.01,
        initial_lr_phase2 = 0.01,
        restart_period = 50,
        nprint = 50,
    )
end

# ╔═╡ 6ba46c66-4e7d-43bc-a8ab-f61b8a7e453c


# ╔═╡ fbf112da-4bda-4d65-ab05-26f7482d9751
md"## Appendix"

# ╔═╡ 32e695ba-1443-475b-a25e-1d6c7b2fbf01
"""
Per-waveform equivariance loss:
mean over K random shifts of (phi(shift(x,delta)) - phi(x) - delta)^2 + gamma*phi(x)^2
Returns CPU Vector{Float32} length N.
"""
function equiv_loss_per_waveform(model, X, pa_params; K=8, batchsize=256)
    N = size(X, 2)
    out = zeros(Float32, N)
    loader = Flux.DataLoader(X; batchsize=min(batchsize, N), shuffle=false)
    offset = 0

    for x in loader
        xb = xpu(x)                       # (nt, B)
        B  = size(xb, 2)
        phi_x = vec(cpu(model(xb)))       # (B,)
        acc = zeros(Float32, B)

        for _ in 1:K
            delta = (2f0 .* CUDA.rand(Float32, 1, B) .- 1f0) .* Float32(pa_params.max_shift_samples)
            xs = Zygote.@ignore pa.shift_traces_Fourier(xb, delta, model.grid)
            phi_xs = vec(cpu(model(xs)))
            d = vec(cpu(delta))
            acc .+= (phi_xs .- phi_x .- d).^2
        end

        equiv_i = acc ./ Float32(K)
        reg_i   = pa_params.gamma .* (phi_x .^ 2)
        out[offset+1:offset+B] .= equiv_i .+ reg_i
        offset += B
    end
    return out
end

# ╔═╡ 9200f3bb-dcdb-4c25-a0f1-bcaa00800c19
loss_syn_per = equiv_loss_per_waveform(model_syn_pa, X_syn, pa_params_syn; K=8, batchsize=256)

# ╔═╡ ff79d140-c5a9-4207-9c9f-42518a3edbda
let
    idx = collect(1:length(loss_syn_per))
    tr = PlutoPlotly.scatter(
        x=idx, y=loss_syn_per, mode="markers",
        marker=attr(size=4, color="#1f77b4", opacity=0.65),
        name="Synthetic per-waveform equiv loss"
    )
    layout = Layout(
        title=attr(text="Synthetic: equivariance loss per waveform"),
        xaxis=attr(title="Waveform index"),
        yaxis=attr(title="Loss", type="log"),
        height=350, width=900,
        plot_bgcolor="white", paper_bgcolor="white"
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ 98bb3517-14b9-4b4b-aa9f-466c416fbe3a
begin
	trained_real_pa
	loss_real_per = equiv_loss_per_waveform(model_real_pa, X_real, pa_params_real; K=8, batchsize=256)
end

# ╔═╡ efbe1b87-1839-43bc-a745-0ca3c577423f
let
    idx = collect(1:length(loss_real_per))
    tr = PlutoPlotly.scatter(
        x=idx, y=loss_real_per, mode="markers",
        marker=attr(size=4, color="#d62728", opacity=0.65),
        name="Real per-waveform equiv loss"
    )
    layout = Layout(
        title=attr(text="Real: equivariance loss per waveform"),
        xaxis=attr(title="Waveform index"),
        yaxis=attr(title="Loss", type="log"),
        height=350, width=900,
        plot_bgcolor="white", paper_bgcolor="white"
    )
    WideCell(PlutoPlotly.plot([tr], layout))
end

# ╔═╡ da9ccbd5-1120-4174-bc98-32432c73b593
let
q_keep = 0.7f0
thr_real = quantile(loss_real_per, q_keep)
keep_real = findall(loss_real_per .<= thr_real)

X_real_keep = X_real[:, keep_real]
tau_keep, X_aligned_keep = pa.apply_aligner(model_real_pa, X_real_keep, baseline_real)
	@show size(X_aligned_keep)
stack_keep = pa.coherent_stack(X_aligned_keep)

tau_all, X_aligned_all = pa.apply_aligner(model_real_pa, X_real, baseline_real)
stack_all = pa.coherent_stack(X_aligned_all)


    ts = 1:size(X_real,1)
    tr1 = PlutoPlotly.scatter(x=ts, y=stack_all, mode="lines",
        name="All traces", line=attr(color="grey", width=1.5, dash="dash"))
    tr2 = PlutoPlotly.scatter(x=ts, y=stack_keep, mode="lines",
        name="Low-loss only", line=attr(color="#d62728", width=2))
    layout = Layout(
        title=attr(text="Aligned stack: all vs low-equiv-loss subset"),
        xaxis=attr(title="Sample"), yaxis=attr(title="Amplitude"),
        height=350, width=900, plot_bgcolor="white", paper_bgcolor="white"
    )
    WideCell(PlutoPlotly.plot([tr1, tr2], layout))
end

# ╔═╡ 5b26ae9a-4f5e-4fd8-ad3c-0042b91fec8f

let
	trained_real_pa
q_keep = 0.1f0
thr_real = quantile(loss_real_per, q_keep)
keep_real = findall(loss_real_per .<= thr_real)

X_real_keep = X_real[:, keep_real]
tau_keep, X_aligned_keep = pa.apply_aligner(model_real_pa, X_real_keep, baseline_real)

	plot(heatmap(z=X_aligned_keep))
end

# ╔═╡ b2f7aa59-1e67-4a35-b5b6-e9f3dc88f935
begin
    function station_latlon(loc)
        vals = Float64[]
        if loc isa AbstractVector
            for v in loc
                v isa Number && push!(vals, Float64(v))
            end
        elseif loc isa Tuple
            for v in loc
                v isa Number && push!(vals, Float64(v))
            end
        elseif loc isa NamedTuple
            for v in values(loc)
                v isa Number && push!(vals, Float64(v))
            end
        end

        if length(vals) < 2
            return (NaN, NaN)
        end

        a, b = vals[1], vals[2]
        if abs(a) <= 90 && abs(b) <= 180
            return (a, b)  # (lat, lon)
        elseif abs(b) <= 90 && abs(a) <= 180
            return (b, a)  # (lat, lon)
        else
            return (a, b)
        end
    end

    function train_phase_aligner_per_station(pa, dfile, snr_file;
            snr_threshold=0.0,
            trim_range=(501, 800),
            min_traces=80,
            train_frac=0.85,
            base_seed=17,
            max_stations=typemax(Int),
            K_eval=8,
            eval_batchsize=256,
            model_template,
            training_para)

        sta_names = load(dfile)["Sta"][1]
        sta_locs = load(dfile)["Sta"][2]
        data_all = load(dfile)["Data"]
        ses_snr_all = load(snr_file, "SNR")

        stations = unique(sta_names)
        results = NamedTuple[]
        ntrained = 0

        for sta in stations
            ntrained >= max_stations && break

            ix = findfirst(x -> x == sta, sta_names)
            ix === nothing && continue

            R = data_all[ix]
            snr_vec = ses_snr_all[ix]
            sel = findall(x -> x > snr_threshold, snr_vec)
            n_after_snr = length(sel)

            lat, lon = station_latlon(sta_locs[ix])

            if n_after_snr < min_traces
                push!(results, (
                    station = String(sta),
                    status = :skipped,
                    n_traces = n_after_snr,
                    lat = lat,
                    lon = lon,
                    loss_median = NaN,
                    loss_q25 = NaN,
                    loss_q75 = NaN,
                    loss_iqr = NaN,
                    raw_average = nothing,
                    aligned_average = nothing,
                ))
                continue
            end

            raw = Float32.(R[trim_range[1]:trim_range[2], sel])
            m = mean(raw; dims=1)
            s = std(raw; dims=1)
            X = Float32.(taper((raw .- m) ./ max.(s, 1f-8)))

            N = size(X, 2)
            ntrain = round(Int, train_frac * N)
            ntrain = clamp(ntrain, 1, N - 1)

            sta_seed = base_seed + abs(hash(String(sta))) % 100000
            rng = Random.MersenneTwister(sta_seed)
            idx = randperm(rng, N)

            X_train = xpu(X[:, idx[1:ntrain]])
            X_test = xpu(X[:, idx[ntrain+1:end]])

            para = pa.PhaseAligner_Para(
                nt = size(X, 1),
                max_shift_samples = model_template.max_shift_samples,
                enc_kernels = model_template.enc_kernels,
                enc_filters = model_template.enc_filters,
                gamma = model_template.gamma,
                ema_decay = model_template.ema_decay,
                seed = sta_seed,
            )

            model = pa.get_phase_aligner(para)
            baseline, _ = pa.train_phase_aligner(model, X_train, X_test, para, training_para)

            raw_average = Float32.(vec(mean(X; dims=2)))
            _, X_aligned = pa.apply_aligner(model, xpu(X), baseline)
            aligned_average = Float32.(vec(cpu(pa.coherent_stack(X_aligned))))

            lvec = equiv_loss_per_waveform(model, X, para; K=K_eval, batchsize=eval_batchsize)
            q25 = quantile(lvec, 0.25)
            q50 = quantile(lvec, 0.50)
            q75 = quantile(lvec, 0.75)

            push!(results, (
                station = String(sta),
                status = :ok,
                n_traces = N,
                lat = lat,
                lon = lon,
                loss_median = Float32(q50),
                loss_q25 = Float32(q25),
                loss_q75 = Float32(q75),
                loss_iqr = Float32(q75 - q25),
                phi_baseline = Float32(baseline.phi_baseline),
                raw_average = raw_average,
                aligned_average = aligned_average,
            ))

            ntrained += 1
        end

        return results
    end
end

# ╔═╡ 4d68f98f-c426-4f5e-b9db-8d00e946c042
station_results = train_phase_aligner_per_station(
    pa,
    dfile,
    "$(fldir)/$snrf";
    snr_threshold = station_loop_cfg.snr_threshold,
    trim_range = station_loop_cfg.trim_range,
    min_traces = station_loop_cfg.min_traces,
    train_frac = station_loop_cfg.train_frac,
    base_seed = station_loop_cfg.base_seed,
    max_stations = station_loop_cfg.max_stations,
    K_eval = station_loop_cfg.K_eval,
    eval_batchsize = station_loop_cfg.eval_batchsize,
    model_template = station_model_template,
    training_para = station_training_para,
)

# ╔═╡ 9f66f8be-1297-4f59-a157-aa214ec24fa0
let
    ok = filter(r -> r.status == :ok, station_results)
    sort!(ok; by = r -> r.loss_median, rev = true)
    ok
end

# ╔═╡ e0c03af8-7f79-4eb5-a561-a4a746723f34
let
    ok_geo = filter(r -> r.status == :ok && isfinite(r.lat) && isfinite(r.lon), station_results)

    if isempty(ok_geo)
        md"No valid station coordinates found for geoscatter."
    else
        lats = Float64[r.lat for r in ok_geo]
        lons = Float64[r.lon for r in ok_geo]
        med  = Float64[r.loss_median for r in ok_geo]
        ntr  = Float64[r.n_traces for r in ok_geo]
        med_log = log10.(max.(med, 1f-12))

        nmax = max(maximum(ntr), 1.0)
        sizes = 5 .+ 10 .* sqrt.(ntr ./ nmax)

        labels = [
            "$(r.station)<br>N=$(r.n_traces)<br>med=$(round(r.loss_median, sigdigits=3))<br>IQR=$(round(r.loss_iqr, sigdigits=3))"
            for r in ok_geo
        ]

        tr = PlutoPlotly.scattergeo(
            lat=lats,
            lon=lons,
            mode="markers",
            text=labels,
            hoverinfo="text",
			textposition="top center",
            marker=attr(
                size=sizes,
                color=med/100,
                colorscale="Inferno",
                colorbar=attr(title="log10(median equiv loss)"),
                line=attr(width=0.5, color="black"),
                opacity=0.9,
            ),
            name="Stations",
        )

        layout = Layout(
            title=attr(text="Per-station convergence equivariance loss (High loss means more <b>diverse</b> RFs)"),
            geo=attr(
                projection=attr(type="natural earth"),
                showland=true,
                landcolor="rgb(240,240,240)",
                showcountries=true,
                countrycolor="rgb(150,150,150)",
                fitbounds="locations",
            ),
            height=600,
            width=950,
            margin=attr(l=0, r=0, t=50, b=0),
        )

        WideCell(PlutoPlotly.plot([tr], layout))
    end
end

# ╔═╡ 8bb89846-6d1d-4b49-bf88-72b6482af47c

	  ok_geo = filter(r -> r.status == :ok && isfinite(r.lat) && isfinite(r.lon), station_results)

# ╔═╡ 1eaebecb-cc7d-4910-b704-8aad8b3f6d7d
station_results[1].loss_median

# ╔═╡ 0c8ec297-2fd7-4fc9-9fd4-d57a2c1bb4d3
begin
    station_results_file = joinpath(fldir, "station_results_phase_aligner.jld2")
    jldsave(
        station_results_file;
        station_results=station_results,
        station_loop_cfg=station_loop_cfg,
        station_model_template=station_model_template,
        station_training_para=station_training_para,
    )
    station_results_file
end

# ╔═╡ da191791-1c4d-4f03-9a7e-b7850d86ecb2
let
	   ok_geo = filter(r -> r.status == :ok && isfinite(r.lat) && isfinite(r.lon), station_results)
	raw = hcat([vec(s.raw_average) for s in ok_geo]...)
	plot(heatmap(z=raw))
end

# ╔═╡ a3dfdf81-925e-410d-914d-83929718d7eb
let
	   ok_geo = filter(r -> r.status == :ok && isfinite(r.lat) && isfinite(r.lon), station_results)
	raw = hcat([vec(s.aligned_average) for s in ok_geo]...)
	plot(heatmap(z=raw))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[compat]
CUDA = "~5.11.0"
DSP = "~0.8.4"
FFTW = "~1.10.0"
Flux = "~0.16.9"
JLD2 = "~0.6.4"
Optimisers = "~0.4.7"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
ProgressLogging = "~0.1.6"
Zygote = "~0.7.10"
cuDNN = "~1.4.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "8b0e1dc5f9d439b019ff659e951287422ae407b7"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "2eeb2c9bef11013efc6f8f97f32ee59b146b09fb"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.44"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0761717147821d696c9470a7a86364b2fbd22fd8"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.2"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b8651b2eb5796a386b0398a20b519a6a6150f75c"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.3"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "e386db8b4753b42caac75ac81d0a4fe161a68a97"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.1"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "ea6a2ab8307059b6c9ea186ff7dfcd032a13b731"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.11.0"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8c19e97de5b7574672e4a7a3abd55714ad66d59a"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.2+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "061f39cc84e99928830aa1005d79f7e99097ba28"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.2.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "af17d37b5b8b4d7525f8902eba1ef6141a9a7d3b"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.21.0+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "70dea6a7133d2100a143b515a00d6d887e208500"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "9.20.0+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3c190c570fb3108c09f838607386d10c71701789"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.73.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "1a3ad7e16a321667698a19e77362b35a1e94c544"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.1"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnzymeCore]]
git-tree-sha1 = "24bbb6fc8fb87eb71c1f8d00184a60fc22c63903"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.19"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Flux]]
deps = ["ADTypes", "Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "ea6715b3d7a95a07a62109df1c9ede2641a50706"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.9"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxFiniteDifferencesExt = "FiniteDifferences"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]
    FluxMooncakeExt = "Mooncake"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cddeab6487248a39dae1a960fff0ac17b2a28888"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.3"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "6487601563e4a1d1dab796e88b4548bf5544209e"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.4.1"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "fedfe5e7db7035271c3f58359007f971da1dde87"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.9.1"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "a589b6c1a0eff953571f5d8b0474f5020831114d"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.1.1"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "941f87a0ae1b14d1ac2fa57245425b23a9d7a516"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.4"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "67c6f1f085cb2671c93fe34244c9cccde30f7a26"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "f2e76d3ced51a2a9e185abc0b97494c7273f649f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.41"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "69e4739502b7ab5176117e97e1664ed181c35036"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.6"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "5d4278f755440f70648d80cc6225f51e78e94094"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.1"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "39a69ca451c3e78b9a6a2e42ef894fdf7505e629"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.5"

    [deps.MLDataDevices.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    ChainRulesExt = "ChainRules"
    ComponentArraysExt = "ComponentArrays"
    FillArraysExt = "FillArrays"
    GPUArraysSparseArraysExt = ["GPUArrays", "SparseArrays"]
    MLUtilsExt = "MLUtils"
    MetalExt = ["GPUArrays", "Metal"]
    OneHotArraysExt = "OneHotArrays"
    OpenCLExt = ["GPUArrays", "OpenCL"]
    ReactantExt = "Reactant"
    RecursiveArrayToolsExt = "RecursiveArrayTools"
    ReverseDiffExt = "ReverseDiff"
    SparseArraysExt = "SparseArrays"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"
    cuDNNExt = ["CUDA", "cuDNN"]
    oneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "78cd28dbd5f03f99ccaba45c987107adcb61c115"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.34"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NVTX]]
deps = ["JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "a9083c3e469e63cca454d1fc3b19472d9d92c14a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.3"
weakdeps = ["Colors"]

    [deps.NVTX.extensions]
    NVTXColorsExt = "Colors"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "36b5d2b9dd06290cd65fcf5bdbc3a551ed133af5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.7"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "aea4eede5ab3ee188906d0cf3bbfa36eb543dccc"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.8"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "2d99b4c8a7845ab1342921733fa29366dae28b24"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.1"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"
    PolynomialsRecipesBaseExt = "RecipesBase"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "5f4f629c085b87e71125eec6773f5f872c74a47a"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "ac4b837d89a58c848e85e698e2a2514e9d59d8f6"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "fa95b3b097bcef5845c142ea2e085f1b2591e92c"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.7.1"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "0f30765c32d66d58e41f4cb5624d4fc8a82ec13b"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.1"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a29cbf3968d36022198bcc6f23fdfd70f7caf737"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.10"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "5494b0ae3ddc5ca0f64159d5ed3a396f36e0fcfe"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.4.7"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═4b3a86f2-3b14-11f1-a498-c9fc074b7df6
# ╠═4b3a88c8-3b14-11f1-903e-7d32036b9cd0
# ╠═4b3a8918-3b14-11f1-8a03-c593d58ce280
# ╠═4b3a8954-3b14-11f1-adfa-b11a44aa35cf
# ╠═4b3a8990-3b14-11f1-a14d-8fcdc511b3b0
# ╠═4b3a8b2a-3b14-11f1-90be-218eee387b52
# ╠═4b3a8b72-3b14-11f1-8f89-2b2a746100d7
# ╠═4b3a8bb6-3b14-11f1-9c74-2f9dd23ccb58
# ╠═4b3a8c60-3b14-11f1-ae5d-b7ed043ed9b9
# ╠═4b3a8e10-3b14-11f1-b90b-9ddb16ca3290
# ╠═4b3a91b0-3b14-11f1-9552-0925253b8f86
# ╠═4b3a91f6-3b14-11f1-b940-2984fb5fb327
# ╠═f70f832a-20f0-4ce5-a476-2f0078407de7
# ╠═4b3a9340-3b14-11f1-9734-95d754f356e5
# ╠═4b3a9386-3b14-11f1-a082-796d75cf2b46
# ╠═4b3a949e-3b14-11f1-ad90-f37908baa471
# ╠═4b3a94e6-3b14-11f1-b11e-a33f4ef14e4c
# ╠═4b3a9566-3b14-11f1-af48-23a204f629b2
# ╠═4b3a95a2-3b14-11f1-8332-39ec2b9e7016
# ╠═4b3a9688-3b14-11f1-9840-233591c7a890
# ╠═4b3a973c-3b14-11f1-a221-352efa534949
# ╠═4b3a9778-3b14-11f1-aa28-9511b4c8a5b8
# ╠═4b3a97b6-3b14-11f1-ac19-314a898ba2eb
# ╠═4b3a97f0-3b14-11f1-a539-85bead0164cc
# ╠═4b3a9c5a-3b14-11f1-80cc-8bb641826f56
# ╠═4b3a9c96-3b14-11f1-b7e0-35fd57855f6f
# ╠═4b3a9fb6-3b14-11f1-a24c-2fd152d6ce55
# ╠═4b3a9ff4-3b14-11f1-abb8-472e9a4900cc
# ╠═e8e91d6b-72d0-4493-aa4a-bb0e36deda9e
# ╠═5142d77f-3232-4b70-b3e9-86b668972529
# ╠═4b3aa33a-3b14-11f1-856e-7f736cf0e1cd
# ╠═9200f3bb-dcdb-4c25-a0f1-bcaa00800c19
# ╠═ff79d140-c5a9-4207-9c9f-42518a3edbda
# ╠═4b3aa61e-3b14-11f1-b62c-919b45f517b5
# ╠═9f87709f-e8ba-4dcc-9c73-c0a866fe801e
# ╠═4b3aa704-3b14-11f1-abfd-6514b40c32d7
# ╠═f941a320-d11e-49d8-8277-5673417a283c
# ╠═4b3aa812-3b14-11f1-bce1-f36ed58c8131
# ╠═9b150acb-7d71-4b0b-a452-6782c49de61a
# ╠═e5fab98e-620f-4a99-80d3-0bd27891cfee
# ╠═b9d9618e-5e00-4729-94bb-20063b49ba72
# ╠═4b3aa902-3b14-11f1-83d6-750ac8faf647
# ╠═cf24fb96-36d0-4158-887d-5d09c93c2f6b
# ╠═b9316cca-dfc8-47c3-84cd-af227994f727
# ╠═28c9aee3-ecf1-4851-a75a-fc592533fe07
# ╠═4b3aa9c0-3b14-11f1-bfe7-33c9135341b1
# ╠═4b3aaac4-3b14-11f1-9f27-dfbc0c9f7b57
# ╠═4b3aabfa-3b14-11f1-88df-17f63d97bee9
# ╠═4b3aac40-3b14-11f1-87d8-fba529fb8df1
# ╠═4b3aacca-3b14-11f1-8651-951f9425536b
# ╠═4b3aadb2-3b14-11f1-bc51-9f79b497d1d3
# ╠═4b3aae5c-3b14-11f1-b885-cf4473314b06
# ╠═0ae8f88f-f974-4449-b065-d00c7a7252d0
# ╠═4b3aaea2-3b14-11f1-aad2-e7e3e09e008c
# ╠═4b3aaed6-3b14-11f1-80e4-179f8d163898
# ╟─4b3ab17c-3b14-11f1-bb14-418aed6192ee
# ╠═4b3ab1e0-3b14-11f1-a866-ef43937275d1
# ╠═98bb3517-14b9-4b4b-aa9f-466c416fbe3a
# ╠═efbe1b87-1839-43bc-a745-0ca3c577423f
# ╠═da9ccbd5-1120-4174-bc98-32432c73b593
# ╠═5b26ae9a-4f5e-4fd8-ad3c-0042b91fec8f
# ╟─2f3d2f72-7d4c-46ab-aef5-7db6f23b8d0a
# ╠═1b0d27a4-42af-4b2f-bf34-74d6fca5e65c
# ╠═b2f7aa59-1e67-4a35-b5b6-e9f3dc88f935
# ╠═4d68f98f-c426-4f5e-b9db-8d00e946c042
# ╠═9f66f8be-1297-4f59-a157-aa214ec24fa0
# ╠═e0c03af8-7f79-4eb5-a561-a4a746723f34
# ╠═6ba46c66-4e7d-43bc-a8ab-f61b8a7e453c
# ╠═8bb89846-6d1d-4b49-bf88-72b6482af47c
# ╠═1eaebecb-cc7d-4910-b704-8aad8b3f6d7d
# ╠═fbf112da-4bda-4d65-ab05-26f7482d9751
# ╠═32e695ba-1443-475b-a25e-1d6c7b2fbf01
# ╠═0c8ec297-2fd7-4fc9-9fd4-d57a2c1bb4d3
# ╠═da191791-1c4d-4f03-9a7e-b7850d86ecb2
# ╠═a3dfdf81-925e-410d-914d-83929718d7eb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
