### A Pluto.jl notebook ###
# v0.20.27

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

# ╔═╡ 10000001-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate("/mnt/NAS/EQData/SeismicAutoencoders")

    using JLD2,
        PlutoLinks,
        PlutoPlotly,
        PlutoUI,
        Random,
        Statistics

end

# ╔═╡ e0c79630-dbe0-4110-b880-a9ee9e4e1186
using Distances

# ╔═╡ f3d2428d-fcd6-4218-85d1-ef016cd8b999
using FFTW

# ╔═╡ c672d254-b73d-4191-86a5-ae11be0df3cb
using ProgressLogging

# ╔═╡ bdeb2eb2-6b5d-4677-837c-7072e3588430
begin
    using CUDA, ConcreteStructs,
        Dates,
        DSP,
        Enzyme,
        EnzymeCore,
        LinearAlgebra,
        Lux,
        MLUtils,
        NNlib,
        Optimisers,
        Reactant,
        StatsBase
end

# ╔═╡ 2a5aed3b-1319-4870-b086-6ecdc6996fc6
using ProgressMeter

# ╔═╡ f3a39849-5f08-46ae-bbae-562f3fe73cf5
using UnicodePlots

# ╔═╡ 96817feb-6aa9-4f35-9277-9a4560e9a2a7
# using FFTW, Peaks, ColorSchemes, Colors, InlineStrings

# ╔═╡ 206a2c26-b3cc-4e74-83e1-fa92aa0bdd10
# using PlutoHooks

# ╔═╡ 10000002-0000-0000-0000-000000000001
md"""
# VQ-VAE v10 Training — Split-Decoder Interferometric VQ

Short orchestration notebook.  The v10 architecture notebook owns the model,
training loop, split VQ, kNN, saving/loading, source-state averaging, and plotting
helpers.  This notebook selects station pairs, starts training, and connects the
trained source states to MFT.
"""

# ╔═╡ 10000003-0000-0000-0000-000000000001

    vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/vqvae/VQVAE_architecture_v9.jl")


# ╔═╡ 787a0d87-e6ef-4b44-83db-480489795df6


# ╔═╡ 10000004-0000-0000-0000-000000000001
md"## Data and Pair Selection"

# ╔═╡ 10000005-0000-0000-0000-000000000001
begin
    # data_filepath = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/"
	# data_filepath = "/mnt/NAS2/Sanket_data/California_XJ_13032026/"
	data_filepath = "/mnt/NAS2/Sanket_data/California_XJ_13032026/"

	# data_filepath =  "/mnt/NAS2/Sanket_data/Minneapolis_pairs_SS_29052026/"
		# data_filepath =  "/mnt/sanket1/Minneapolis_pairs_SM_29052026_new/"
	# data_filepath = "/mnt/NAS2/Pushkar_Data/uttaranchal_data/jldfiles/30mins_dt_1p0_band_0p01_0p5_250maxlag_selected/Z/"

    dt = 1.0
    period_min = 3
    period_max = 10
    mft_nT = 20        # number of periods log-spaced between period_min and period_max
    velocity_range = (1.0, 8.0)
    mft_nperiods = 100
    mft_max_modes = 6
    bandwidth_factor = 0.15
    zero_pad_factor = 4
    use_latest_run_per_seed = true
    training_n_max = 10_000  # maximum pooled waveforms per pair; use nothing for no cap
end

# ╔═╡ 1000000c-0000-0000-0000-000000000004
begin
	whitening_kernel_length = 256   # FIR tap count for spectral whitening (longer = sharper freq resolution)
	per_waveform_whitening_kernel_length = 128  # per-waveform FIR taps; longer = closer to phase-only
end

# ╔═╡ c0000002-0000-0000-0000-000000000002
bp_filter = let
    responsetype = Bandpass(inv(period_max), inv(period_min))
    digitalfilter(responsetype, Butterworth(2); fs=inv(dt))
end

# ╔═╡ 10000009-0000-0000-0000-000000000001
md"## Hyperparameters"

# ╔═╡ 1000000a-0000-0000-0000-000000000001
vqvae_parameters = (;
    d=40,
    beta_commit=0.25f0,
    K=[5,3],
    ema_decay=0.99f0,
    n_filters=32,
    ratios=[2, 5],
    n_residual_layers=3,
    dilation_base=2,
    residual_kernel_size=3,
    enc_kernel_size=7,
    dec_kernel_size=7,
    use_bn=false,
    dead_threshold=50,
    entropy_weight=0.1f0,
    codebook_exclusivity_weight=0.0f0,
    reconstruction_loss=:l1,
)

# ╔═╡ 1000000b-0000-0000-0000-000000000001
training_para = vqvae.VQVAE_Training_Para(
    batchsize=4096,
    nepoch=100,
    initial_learning_rate=0.001,
    weight_decay=0.0,
    Mnn_schedule=[(1, 128), (5, 256), (26, 256)],
    warmup_epochs=0,
	verbose = true,
	autodiff_backend = :auto,
	knn_search_chunk_size_fraction = 0.25,
    index_refresh_every=2,
)

# ╔═╡ 1000000c-0000-0000-0000-000000000003
run_seeds = [1236, 1237, 1238]

# ╔═╡ 1000000c-0000-0000-0000-000000000002
training_device = vqvae.default_xdev(; force=true)

# ╔═╡ 1000000c-0000-0000-0000-000000000001
md"## Train"

# ╔═╡ 1000000d-0000-0000-0000-000000000002
@bind compile_button CounterButton("Compile selected pairs")

# ╔═╡ 1000000d-0000-0000-0000-000000000001
@bind train_button CounterButton("Train compiled pairs")

# ╔═╡ 10000008-0000-0000-0000-000000000002
begin
    if !@isdefined(compiled_model_cache_v9)
        const compiled_model_cache_v9 = Ref{Any}(nothing)
    end
    if !@isdefined(compiled_model_key_cache_v9)
        const compiled_model_key_cache_v9 = Ref{Any}(nothing)
    end
end

# ╔═╡ 70b49b49-8d0e-4b1b-a361-be7f1014607a
gpu_device(force=true)

# ╔═╡ 1000000f-0000-0000-0000-000000000001
md"## Inspect One Result"

# ╔═╡ 7f4e1dc2-f3dd-45b2-8818-6eff1ed4f7b5
@bind save_recon_examples_button CounterButton("Save 100 raw recon examples")

# ╔═╡ b00b61c7-81c6-467a-b8bd-9f9e2ebd4d0c
result_title_context(result) = begin
    pair = result.pair
    distance_label = isnothing(result.data_bundle.distance) ?
        "distance unavailable" : "$(round(Int, result.data_bundle.distance))km"
    seed_label = hasproperty(result, :seed) ? " seed=$(result.seed)" : ""
    "$(pair[1])-$(pair[2])$(seed_label) $(distance_label) $(period_min)-$(period_max)s"
end

# ╔═╡ 9c18f181-8d76-46a4-836d-c4fe2cad19c8
function list_station_pairs(filepath::String)
    files = readdir(filepath)
    pairs = Set{Tuple{String,String}}()
    for f in files
        m = match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)", basename(f))
        m === nothing && continue
        push!(pairs, (m.captures[1], m.captures[2]))
    end
    return sort!(collect(pairs), by=x -> (x[1], x[2]))
end

# ╔═╡ 10000006-0000-0000-0000-000000000001
available_pairs = list_station_pairs(data_filepath)

# ╔═╡ 10000007-0000-0000-0000-000000000001
begin
    isempty(available_pairs) && error("No station pairs found in $(data_filepath)")
    pair_options = ["$(p[1])-$(p[2])" for p in available_pairs]
    default_pair_names = pair_options[1:min(end, 1)]
    
end

# ╔═╡ 0bf6c416-bf64-4b2a-b55e-623a4f9daae2
@bind selected_pair_names confirm(MultiCheckBox(pair_options; default=default_pair_names, select_all=true))

# ╔═╡ 272932ae-4c54-41a2-8c9a-e1cd4834150a
selected_pair_names

# ╔═╡ 10000008-0000-0000-0000-000000000001
selected_pairs = begin
    isempty(selected_pair_names) && error("Select at least one station pair.")
    [begin
        parts = split(name, "-", limit=2)
        length(parts) == 2 || error("Invalid pair name $(name).")
        (String(parts[1]), String(parts[2]))
    end for name in selected_pair_names]
end

# ╔═╡ 10000008-0000-0000-0000-000000000005
# Load and whiten only the first pair for preview/compilation geometry — discarded after use
pairs_data_preview = let
    pd_raw = only(vqvae.load_pairs_data(selected_pairs[1:1];
        filepath=data_filepath, seed=1234,
        dt=dt, period_min=period_min, period_max=period_max,
        n_max=training_n_max))
    [vqvae.whiten_pair_entry(pd_raw;
        bp_filter, per_waveform_whitening_kernel_length)]
end

# ╔═╡ d5000002-0000-0000-0000-000000000002
# Alias for downstream preview cells that reference pairs_data_whitened
pairs_data_whitened = pairs_data_preview

# ╔═╡ 66bc6645-a459-40fb-a871-2dd674e5bf9d
pairs_data_whitened[1].data

# ╔═╡ c0000003-0000-0000-0000-000000000003
let
    colors_raw = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f"]
    colors_whi = ["#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                  "#c49c94", "#f7b6d2", "#c7c7c7"]
    traces = [scatter()]
    for (pi, (pd_raw, pd_whi)) in enumerate(zip(pairs_data_preview, pairs_data_whitened))
        pair_label = "$(pd_raw.pair[1])-$(pd_raw.pair[2])"
        X_raw = Float64.(pd_raw.data.D_train)
        X_whi = Float64.(pd_whi.data.D_train)
        nt = size(X_raw, 1)
        fft_freqs = FFTW.fftfreq(nt, inv(Float64(dt)))
        pos = fft_freqs .> 0
        order = sortperm(1.0 ./ fft_freqs[pos])
        periods_plot = (1.0 ./ fft_freqs[pos])[order]
        P_raw = vec(mean(abs2.(FFTW.fft(X_raw, 1)); dims=2))
        P_whi = vec(mean(abs2.(FFTW.fft(X_whi, 1)); dims=2))
        norm = maximum(P_raw[pos])
        ci = mod1(pi, length(colors_raw))
        push!(traces, PlutoPlotly.scatter(
            x=periods_plot, y=P_raw[pos][order]./norm, mode="lines",
            line=PlutoPlotly.attr(color=colors_raw[ci], width=2),
            name="$(pair_label) raw", legendgroup=pair_label))
        push!(traces, PlutoPlotly.scatter(
            x=periods_plot, y=P_whi[pos][order]./norm, mode="lines",
            line=PlutoPlotly.attr(color=colors_whi[ci], width=1.5, dash="dash"),
            name="$(pair_label) whitened+filtered", legendgroup=pair_label))
    end
    WideCell(PlutoPlotly.plot(traces, PlutoPlotly.Layout(
        title="Average PSD: raw (solid) → whitened + bandpass (dashed)",
        xaxis_title="Period (s)", yaxis_title="Power (avg across waveforms, normalized to raw max)",
        yaxis_type="log",
		xaxis=attr(range=(0, 100)),
        height=420, margin=PlutoPlotly.attr(l=60, r=20, t=60, b=50),
    )))
end

# ╔═╡ 10000008-0000-0000-0000-000000000003
selected_pairs_key = (
    path=:direct_vq_notebook,
    selected_pairs=Tuple(selected_pairs),
    vqvae_parameters=vqvae_parameters,
    training_batchsize=training_para.batchsize,
    training_nepoch=training_para.nepoch,
    warmup_epochs=training_para.warmup_epochs,
    Mnn_schedule=Tuple(training_para.Mnn_schedule),
    index_refresh_every=training_para.index_refresh_every,
    training_n_max=training_n_max,
)

# ╔═╡ d1eec798-b7aa-4387-9a1e-71064e2660fd
analysis_settings = (;
    dt,
    period_min,
    period_max,
    mft_nperiods,
    mft_max_modes,
    velocity_range,
    bandwidth_factor,
    zero_pad_factor,
    use_latest_run_per_seed,
)

# ╔═╡ d5000002-0000-0000-0000-000000000003
let
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]
    n_sample = 10  # number of individual waveforms to overlay
    traces = [scatter()]
    for (pi, (pd_raw, pd_whi)) in enumerate(zip(pairs_data, pairs_data_whitened))
        pair_label = "$(pd_raw.pair[1])-$(pd_raw.pair[2])"
        X_raw = Float64.(pd_raw.data.D_train)
        X_whi = Float64.(pd_whi.data.D_train)
        nt, nw = size(X_raw)
        fft_freqs = FFTW.fftfreq(nt, inv(Float64(dt)))
        pos = fft_freqs .> 0
        order = sortperm(1.0 ./ fft_freqs[pos])
        periods_plot = (1.0 ./ fft_freqs[pos])[order]
        ci = mod1(pi, length(colors))
        # std of log-PSD across all waveforms
        logP_raw = log10.(abs2.(FFTW.fft(X_raw, 1))[pos, :] .+ 1f-30)
        logP_whi = log10.(abs2.(FFTW.fft(X_whi, 1))[pos, :] .+ 1f-30)
        std_raw = vec(std(logP_raw[order, :]; dims=2))
        std_whi = vec(std(logP_whi[order, :]; dims=2))
        push!(traces, PlutoPlotly.scatter(
            x=periods_plot, y=std_raw, mode="lines",
            line=PlutoPlotly.attr(color=colors[ci], width=2),
            name="$(pair_label) raw σ", legendgroup="$(pair_label)_std"))
        push!(traces, PlutoPlotly.scatter(
            x=periods_plot, y=std_whi, mode="lines",
            line=PlutoPlotly.attr(color=colors[ci], width=2, dash="dash"),
            name="$(pair_label) whitened σ", legendgroup="$(pair_label)_std"))
        # overlay a few individual waveform spectra (whitened only)
        idx_sample = round.(Int, range(1, nw; length=min(n_sample, nw)))
        for (si, i) in enumerate(idx_sample)
            push!(traces, PlutoPlotly.scatter(
                x=periods_plot, y=logP_whi[order, i], mode="lines",
                line=PlutoPlotly.attr(color=colors[ci], width=0.5),
                opacity=0.3, showlegend=(si == 1),
                name="$(pair_label) individual",
                legendgroup="$(pair_label)_ind"))
        end
    end
    WideCell(PlutoPlotly.plot(traces, PlutoPlotly.Layout(
        title="Per-waveform whitening check: σ of log-PSD across waveforms (solid=raw, dashed=whitened) + individual whitened spectra",
        xaxis_title="Period (s)",
        yaxis_title="σ of log₁₀(PSD)  /  log₁₀(PSD) per waveform",
        xaxis=attr(range=(0, 100)),
        height=480, margin=PlutoPlotly.attr(l=60, r=20, t=60, b=50),
    )))
end

# ╔═╡ d5000002-0000-0000-0000-000000000004
let
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]
    traces = [scatter()]
    for (pi, (pd_raw, pd_whi)) in enumerate(zip(pairs_data, pairs_data_whitened))
        pair_label = "$(pd_raw.pair[1])-$(pd_raw.pair[2])"
        ci = mod1(pi, length(colors))
        nt = size(pd_raw.data.D_ac_all, 1)
        t = range(0; step=Float64(dt), length=nt)
        avg_raw = vec(mean(Float64.(pd_raw.data.D_ac_all); dims=2))
        avg_whi = vec(mean(Float64.(pd_whi.data.D_ac_all); dims=2))
        norm_raw = maximum(abs.(avg_raw))
        norm_whi = maximum(abs.(avg_whi))
        push!(traces, PlutoPlotly.scatter(
            x=collect(t), y=avg_raw ./ norm_raw, mode="lines",
            line=PlutoPlotly.attr(color=colors[ci], width=2),
            name="$(pair_label) raw (ac)", legendgroup=pair_label))
        push!(traces, PlutoPlotly.scatter(
            x=collect(t), y=avg_whi ./ norm_whi, mode="lines",
            line=PlutoPlotly.attr(color=colors[ci], width=2, dash="dash"),
            name="$(pair_label) whitened (ac)", legendgroup=pair_label))
    end
    WideCell(PlutoPlotly.plot(traces, PlutoPlotly.Layout(
        title="Global average waveform: raw (solid) vs per-waveform whitened + bandpass (dashed), acausal branch",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (normalized to peak)",
        height=420, margin=PlutoPlotly.attr(l=60, r=20, t=60, b=50),
    )))
end


# ╔═╡ a6f36936-6199-11f1-b72f-69a0d065f9cd
begin
    if !@isdefined(DirectVQRealLoss)
        struct DirectVQRealLoss
            para
        end
        function (l::DirectVQRealLoss)(model, ps, st, batch)
            return vqvae.vqvae_loss(model, ps, st, batch.x, batch.target, l.para; training=true)
        end
    end

    function direct_real_allfinite(x)
        if x isa Number
            return isfinite(x)
        elseif x isa AbstractArray
            return all(isfinite, Array(x))
        elseif x isa NamedTuple || x isa Tuple
            return all(direct_real_allfinite, values(x))
        else
            return true
        end
    end

    direct_real_cpu_float(x; cdev=vqvae.default_cdev()) = Float32(cdev(x))
    direct_real_effective_mnn(requested::Integer, n::Integer) = min(Int(requested), max(1, n - 1))

    function direct_real_train_metrics(total_sum, recon_sum, commit_sum, entropy_sum,
            codebook_excl_sum, perp_sum, nbatches)
        denom = Float32(max(nbatches, 1))
        return (;
            total=total_sum / denom,
            recon_loss=recon_sum / denom,
            commit_loss=commit_sum / denom,
            entropy_loss=entropy_sum / denom,
            codebook_exclusivity_loss=codebook_excl_sum / denom,
            perplexity=perp_sum / denom,
        )
    end

    function compile_direct_real_train_step(model, ps, st, train_x_cpu, para, training_para;
            device=identity)
        start = time()
        opt = Optimisers.AdamW(; eta=Float64(training_para.initial_learning_rate),
            lambda=Float64(training_para.weight_decay))
        dummy_batch_cpu = Float32.(train_x_cpu[:, 1:training_para.batchsize])
        dummy_target_cpu = training_para.normalize_target ?
            Float32.(MLUtils.normalise(dummy_batch_cpu; dims=1)) : dummy_batch_cpu
        dummy_bdev = (; x=device(dummy_batch_cpu), target=device(dummy_target_cpu))
        ts = Lux.Training.TrainState(model, ps, Lux.trainmode(st), opt)
        ad_backend = vqvae.training_backend(training_para, device)
        loss_fn = DirectVQRealLoss(para)
        _, _, _, ts_warmed = Lux.Training.single_train_step!(
            ad_backend, loss_fn, dummy_bdev, ts; return_gradients=Val(false))
        return (; cache=ts_warmed.cache, compile_time_s=time() - start)
    end

    function compile_direct_real_model(pair_entry; vqvae_parameters, training_para,
            seed::Integer=1234, device=training_device)
        train_x_cpu = Float32.(pair_entry.data.D_train)
        nt = size(train_x_cpu, 1)
        n_train = size(train_x_cpu, 2)
        n_train >= training_para.batchsize || error("Direct VQ compile needs train N=$(n_train) >= batchsize=$(training_para.batchsize).")
        para = vqvae.VQVAE_Para(; merge(vqvae_parameters, (; nt, seed))...)
        model, ps, st, _ = vqvae.get_vqvae(para; rng=Random.Xoshiro(seed), device)

        enc_start = time()
        sample_full = device(train_x_cpu)
        encode_z_e = @compile vqvae.encode_z_e_inference(model, ps, Lux.testmode(st), sample_full)
        encoder_compile_time_s = time() - enc_start

        train_step = compile_direct_real_train_step(model, ps, st, train_x_cpu,
            para, training_para; device)

        return (;
            model,
            para,
            compiled=(; encode_z_e, encode_z_e_train=nothing),
            train_step_cache=train_step.cache,
            compile_seed=seed,
            n_train,
            direct_compile_timings=(;
                encoder_compile_time_s,
                train_step_compile_time_s=train_step.compile_time_s,
            ),
        )
    end

    function run_direct_real_update(model, ps, st, loss_history, train_data, test_data,
            para, training_para; device=training_device, cdev=vqvae.default_cdev(),
            compiled, train_step_cache, n_compiled::Integer)
        train_x_cpu = Float32.(cdev(vqvae.flatten_batch(train_data)))
        test_x_cpu = Float32.(cdev(vqvae.flatten_batch(test_data)))
        n_train = size(train_x_cpu, 2)
        n_train >= training_para.batchsize || error("Training set N=$(n_train) is smaller than batchsize=$(training_para.batchsize).")
        opt = Optimisers.AdamW(; eta=Float64(training_para.initial_learning_rate),
            lambda=Float64(training_para.weight_decay))
        train_state = Lux.Training.TrainState(model, ps, Lux.trainmode(st), opt)
        train_state = vqvae.replace_train_state_cache(train_state, train_step_cache)
        ad_backend = vqvae.training_backend(training_para, device)
        loss_fn = DirectVQRealLoss(para)
        idx = vqvae.LatentIndex(min(vqvae.max_Mnn(training_para), max(1, n_train - 1)))
        last_index_Mnn = 0
        ensemble_targets_cpu = nothing
        test_eval_x_cpu = test_x_cpu[:, 1:min(512, size(test_x_cpu, 2))]

        latent_index_times = Float64[]
        target_cache_times = Float64[]
        step_times = Float64[]
        effective_Mnns = Int[]

        for epoch in 1:training_para.nepoch
            epoch_start = time()
            phase = vqvae.ensemble_phase(epoch, training_para)
            if phase.post_epoch > 0
                effective_Mnn = direct_real_effective_mnn(phase.Mnn, n_train)
                if effective_Mnn != last_index_Mnn || mod(phase.post_epoch - 1, training_para.index_refresh_every) == 0
                    index_start = time()
                    effective_Mnn != phase.Mnn && @warn "Clamping Mnn for available samples" epoch requested_Mnn=phase.Mnn effective_Mnn n_train
                    vqvae.rebuild_latent_index!(idx, train_state.model, train_state.parameters,
                        train_state.states, train_x_cpu;
                        Mnn=effective_Mnn, device, cdev,
                        encode_compiled=compiled.encode_z_e,
                        knn_search_chunk_size_fraction=training_para.knn_search_chunk_size_fraction,
                        n_compiled)
                    push!(latent_index_times, time() - index_start)
                    target_start = time()
                    ensemble_targets_cpu = vqvae.build_ensemble_targets(train_x_cpu, idx, 1:n_train; Mnn=effective_Mnn)
                    push!(target_cache_times, time() - target_start)
                    last_index_Mnn = effective_Mnn
                else
                    push!(latent_index_times, 0.0)
                    push!(target_cache_times, 0.0)
                end
                push!(effective_Mnns, effective_Mnn)
            else
                push!(latent_index_times, 0.0)
                push!(target_cache_times, 0.0)
                push!(effective_Mnns, 0)
            end

            batches = vqvae.make_batches(train_x_cpu, training_para.batchsize)
            total_sum = 0f0
            recon_sum = 0f0
            commit_sum = 0f0
            entropy_sum = 0f0
            codebook_excl_sum = 0f0
            perp_sum = 0f0
            step_sum = 0.0
            for batch in batches
                target_cpu = if phase.post_epoch == 0 || isnothing(ensemble_targets_cpu)
                    batch.x
                else
                    ensemble_targets_cpu[:, batch.indices]
                end
                if training_para.normalize_target
                    target_cpu = Float32.(MLUtils.normalise(target_cpu; dims=1))
                end
                bdev = (; x=device(batch.x), target=device(Float32.(target_cpu)))
                step_start = time()
                _, loss, stats, train_state = Lux.Training.single_train_step!(
                    ad_backend, loss_fn, bdev, train_state; return_gradients=Val(false))
                step_sum += time() - step_start
                total_sum += direct_real_cpu_float(loss; cdev)
                recon_sum += direct_real_cpu_float(stats.recon_loss; cdev)
                commit_sum += direct_real_cpu_float(stats.commit_loss; cdev)
                entropy_sum += direct_real_cpu_float(stats.entropy_loss; cdev)
                codebook_excl_sum += direct_real_cpu_float(stats.codebook_exclusivity_loss; cdev)
                perp_sum += direct_real_cpu_float(stats.perplexity; cdev)
            end
            push!(step_times, step_sum)
            train_m = direct_real_train_metrics(total_sum, recon_sum, commit_sum,
                entropy_sum, codebook_excl_sum, perp_sum, length(batches))
            test_recon_mse = direct_real_cpu_float(vqvae.recon_mse_inference(
                train_state.model, cdev(train_state.parameters), cdev(train_state.states),
                test_eval_x_cpu; normalize_target=training_para.normalize_target); cdev)
            throughput = n_train / max(time() - epoch_start, eps())
            vqvae.record_train_metrics!(loss_history, train_m, test_recon_mse,
                time() - epoch_start, throughput)
            training_para.verbose && @info "Direct VQ epoch" epoch objective=train_m.total train_target_mse=train_m.recon_loss test_recon_mse perplexity=train_m.perplexity effective_Mnn=effective_Mnns[end]
        end

        direct_stats = (;
            latent_index_time_s=latent_index_times,
            target_cache_time_s=target_cache_times,
            step_time_s=step_times,
            effective_Mnn=effective_Mnns,
            finite_metrics=all(isfinite, loss_history.train_objective) &&
                all(isfinite, loss_history.train_target_mse) &&
                all(isfinite, loss_history.test_recon_mse),
            ps_finite=direct_real_allfinite(cdev(train_state.parameters)),
            rvq_finite=direct_real_allfinite(cdev(train_state.states.rvq)),
        )
        return train_state.parameters, train_state.states, loss_history, direct_stats
    end

    function train_selected_pairs_direct_lazy(selected_pairs, compiled_model;
            seeds, training_para, save_root::String, filepath::String,
            dt::Real, period_min::Real, period_max::Real, n_max,
            bp_filter, per_waveform_whitening_kernel_length::Int,
            device=training_device, analysis_settings=(;))
        results = Any[]
        rng = Random.Xoshiro(1234)
        for pair_raw in selected_pairs
            pair = (String(pair_raw[1]), String(pair_raw[2]))
            bundle = vqvae.build_training_bundle(pair; filepath, dt, period_min, period_max)
            bundle = vqvae.trim_training_bundle(bundle; n_max, rng)
            data = vqvae.make_pooled_split(bundle.D1fac, bundle.D1fc; rng)
            pd_raw = (; pair, data, data_bundle=bundle)
            pd = vqvae.whiten_pair_entry(pd_raw; bp_filter, per_waveform_whitening_kernel_length)
            for (run_index, seed) in enumerate(seeds)
                ps, st = vqvae.reset_vqvae(compiled_model.model; seed, device)
                loss_history = vqvae.fresh_loss_history()
                ps, st, loss_history, direct_stats = run_direct_real_update(
                    compiled_model.model, ps, st, loss_history,
                    pd.data.D_train, pd.data.D_test,
                    compiled_model.para, training_para;
                    device, cdev=vqvae.default_cdev(),
                    compiled=compiled_model.compiled,
                    train_step_cache=compiled_model.train_step_cache,
                    n_compiled=compiled_model.n_train)
                run_dir = vqvae.run_dir_for_seed(save_root, pair, seed)
                vqvae.save_vqvae_run(run_dir; model=compiled_model.model, ps, st,
                    para=compiled_model.para, training_para, loss_history, pair,
                    data_bundle=pd.data_bundle, analysis_settings)
                push!(results, (; pair, run_index, seed, run_dir,
                    model=compiled_model.model, ps, st, para=compiled_model.para,
                    training_para, loss_history, data=pd.data,
                    data_bundle=pd.data_bundle, direct_stats))
            end
        end
        return results
    end
end

# ╔═╡ 10000008-0000-0000-0000-000000000004
compiled_model = let
    if compile_button === missing || compile_button == 0
        compiled_model_cache_v9[]
    elseif compiled_model_key_cache_v9[] == selected_pairs_key && !isnothing(compiled_model_cache_v9[])
        compiled_model_cache_v9[]
    else
        compiled_model_cache_v9[] = compile_direct_real_model(pairs_data_whitened[1];
            vqvae_parameters=vqvae_parameters,
            training_para=training_para,
            seed=1234,
            device=training_device,
        )
        compiled_model_key_cache_v9[] = selected_pairs_key
        compiled_model_cache_v9[]
    end
end

# ╔═╡ 77e454ce-619f-11f1-ba90-7fee4b9c9de0
compiled_model_for_training = begin
    if isnothing(compiled_model)
        nothing
    else
        ps0, st0 = vqvae.reset_vqvae(compiled_model.model;
            seed=compiled_model.compile_seed,
            device=training_device)
        train_step_refresh = compile_direct_real_train_step(
            compiled_model.model, ps0, st0,
            Float32.(pairs_data_whitened[1].data.D_train),
            compiled_model.para, training_para;
            device=training_device)
        refreshed = merge(compiled_model, (;
            train_step_cache=train_step_refresh.cache,
            direct_compile_timings=merge(compiled_model.direct_compile_timings, (;
                train_step_recompile_time_s=train_step_refresh.compile_time_s,
            )),
        ))
        compiled_model_cache_v9[] = refreshed
        refreshed
    end
end

# ╔═╡ 1000000e-0000-0000-0000-000000000001
train_results = begin
    cm = compiled_model_for_training
    if isnothing(cm)
        error("Compile model before training.")
    else
        train_selected_pairs_direct_lazy(
            selected_pairs, cm;
            seeds=run_seeds,
            training_para=training_para,
            save_root=joinpath(data_filepath, "SavedModels", "vqvae_direct_K=$(string(cm.para.K))"),
            filepath=data_filepath,
            dt,
            period_min,
            period_max,
            n_max=training_n_max,
            bp_filter,
            per_waveform_whitening_kernel_length,
            device=training_device,
            analysis_settings,
        )
    end
end

# ╔═╡ 9757ced0-619a-11f1-b98d-815aa847ff5b
direct_vq_convergence_summary = let results = train_results
    if isnothing(results)
        "direct_vq_real_training\nstatus=not_started"
    else
        io = IOBuffer()
        println(io, "direct_vq_real_training")
        for result in results
            lh = result.loss_history
            finite_metrics = all(isfinite, lh.train_objective) &&
                all(isfinite, lh.train_target_mse) &&
                all(isfinite, lh.test_recon_mse)
            objective_decreased = !isempty(lh.train_objective) && last(lh.train_objective) < first(lh.train_objective)
            recon_decreased = !isempty(lh.train_target_mse) && last(lh.train_target_mse) < first(lh.train_target_mse)
            println(io, "pair=$(result.pair)")
            println(io, "seed=$(result.seed)")
            println(io, "train_size=$(size(result.data.D_train))")
            println(io, "batchsize=$(result.training_para.batchsize)")
            println(io, "nepoch=$(result.training_para.nepoch)")
            if @isdefined(compiled_model_for_training) && !isnothing(compiled_model_for_training)
                println(io, "encoder_compile_time_s=$(round(compiled_model_for_training.direct_compile_timings.encoder_compile_time_s; digits=3))")
                println(io, "train_step_compile_time_s=$(round(get(compiled_model_for_training.direct_compile_timings, :train_step_compile_time_s, NaN); digits=3))")
                println(io, "train_step_recompile_time_s=$(round(get(compiled_model_for_training.direct_compile_timings, :train_step_recompile_time_s, NaN); digits=3))")
            end
            println(io, "final_train_objective=$(isempty(lh.train_objective) ? missing : last(lh.train_objective))")
            println(io, "final_train_target_mse=$(isempty(lh.train_target_mse) ? missing : last(lh.train_target_mse))")
            println(io, "final_test_recon_mse=$(isempty(lh.test_recon_mse) ? missing : last(lh.test_recon_mse))")
            println(io, "finite_metrics=$(finite_metrics)")
            println(io, "objective_decreased=$(objective_decreased)")
            println(io, "recon_decreased=$(recon_decreased)")
            println(io)
        end
        String(take!(io))
    end
end

# ╔═╡ 10000010-0000-0000-0000-000000000001
result_pair_options = if isnothing(train_results)
    String[]
else
    unique(["$(result.pair[1])-$(result.pair[2])" for result in train_results])
end

# ╔═╡ 3c23d4cb-4bf9-4b4b-a57f-fefdc8f29cf1
selected_result_pair_ui = @bind selected_result_pair_label Select(
    isempty(result_pair_options) ? ["No training results"] : result_pair_options
)

# ╔═╡ 928d4238-216b-4f62-a4d7-62075275252b
selected_result_pair_ui

# ╔═╡ 3ef20afc-4c62-4aab-b96f-4e170f8d96dc
result_seed_options = if isnothing(train_results) || isempty(result_pair_options)
    String[]
else
    sort(unique([
        string(result.seed)
        for result in train_results
        if "$(result.pair[1])-$(result.pair[2])" == selected_result_pair_label
    ]))
end

# ╔═╡ 3ef20afc-4c62-4aab-b96f-4e170f8d96dd
selected_result_seed_ui = @bind selected_result_seed_label Select(
    isempty(result_seed_options) ? ["No trained seeds"] : result_seed_options
)

# ╔═╡ 3ef20afc-4c62-4aab-b96f-4e170f8d96de
selected_result_seed_ui

# ╔═╡ ea4761ee-461d-4429-a13a-e872ae52cc66
selected_result_seed_ui

# ╔═╡ f41c2e62-efbf-47c8-8b7e-adf828ffde4f
selected_result = begin
    if isnothing(train_results) || isempty(result_seed_options)
        nothing
    else
        idx = findfirst(result -> "$(result.pair[1])-$(result.pair[2])" == selected_result_pair_label &&
            string(result.seed) == selected_result_seed_label, train_results)
        isnothing(idx) ? nothing : train_results[idx]
    end
end

# ╔═╡ 10000011-0000-0000-0000-000000000001
if isnothing(selected_result)
    md"Compile selected pairs, then press **Train compiled pairs** to create v9 runs."
else
    WideCell(vqvae.plot_training_dashboard(selected_result.loss_history;
        title="VQ-VAE v9 Training Dashboard ($(result_title_context(selected_result)))"))
end

# ╔═╡ 10000013-0000-0000-0000-000000000001
if isnothing(selected_result)
    md""
else
    WideCell(vqvae.plot_codebook_heatmap(selected_result.st; stage=1,
        title="RVQ Stage 1 Codebook ($(result_title_context(selected_result)))"))
end

# ╔═╡ 10000014-0000-0000-0000-000000000001
if isnothing(selected_result)
    md""
else
    WideCell(vqvae.plot_reconstruction_examples(
        selected_result.model,
        selected_result.ps,
        selected_result.st,
        selected_result.data.D_ac_all;
        nsamples=8,
        dt=dt,
        device=training_device,
        title="Acausal Reconstruction Examples ($(result_title_context(selected_result)))",
    ))
end

# ╔═╡ 13c7f289-d698-453f-a9b1-65bf2c87f465
if isnothing(selected_result)
    md""
else
    WideCell(vqvae.plot_reconstruction_examples(
        selected_result.model,
        selected_result.ps,
        selected_result.st,
        selected_result.data.D_c_all;
        nsamples=8,
        dt=dt,
        device=training_device,
        title="Causal Reconstruction Examples ($(result_title_context(selected_result)))",
    ))
end

# ╔═╡ 9dbe9444-75f3-4d59-8b47-3f70d6bcf6b3
raw_recon_examples_save = let
    if isnothing(selected_result) || save_recon_examples_button === missing || save_recon_examples_button == 0
        md""
    else
        n_save = 100
        rng = MersenneTwister(20260511 + save_recon_examples_button)
        cdev = vqvae.default_cdev()

        raw_ac = hasproperty(selected_result.data_bundle, :D1fac_raw) ?
            selected_result.data_bundle.D1fac_raw : selected_result.data_bundle.D1fac
        raw_c = hasproperty(selected_result.data_bundle, :D1fc_raw) ?
            selected_result.data_bundle.D1fc_raw : selected_result.data_bundle.D1fc

        X_ac = Float32.(selected_result.data.D_ac_all)
        X_c = Float32.(selected_result.data.D_c_all)
        X_model = Float32.(hcat(X_ac, X_c))
        match_cols(X, n) = X[:, mod1.(1:n, size(X, 2))]
        X_raw = Float32.(hcat(match_cols(raw_ac, size(X_ac, 2)), match_cols(raw_c, size(X_c, 2))))
        branches = vcat(fill("acausal", size(X_ac, 2)), fill("causal", size(X_c, 2)))
        branch_indices = vcat(
            mod1.(1:size(X_ac, 2), size(raw_ac, 2)),
            mod1.(1:size(X_c, 2), size(raw_c, 2)),
        )

        cache = vqvae.encoded_cache(
            selected_result.model,
            selected_result.ps,
            selected_result.st,
            selected_result.data;
            device=training_device,
        )
        stage_indices = Int.(hcat(cache.stage_ac, cache.stage_c))
        if length(selected_result.para.K) >= 2
            K1, K2 = selected_result.para.K[1], selected_result.para.K[2]
            source_state = Int.((stage_indices[2, :] .- 1) .* K1 .+ stage_indices[1, :])
            state_labels = ["($k1,$k2)" for k2 in 1:K2 for k1 in 1:K1]
            nstates = K1 * K2
        else
            source_state = Int.(vec(hcat(cache.coarse_ac, cache.coarse_c)))
            state_labels = string.(1:selected_result.para.K[1])
            nstates = selected_result.para.K[1]
        end

        shuffled_by_state = [
            shuffle(rng, findall(==(state), source_state))
            for state in 1:nstates
        ]
        cursors = ones(Int, nstates)
        selected_ids = Int[]
        while length(selected_ids) < min(n_save, length(source_state))
            added_this_round = false
            for state in 1:nstates
                cursors[state] <= length(shuffled_by_state[state]) || continue
                push!(selected_ids, shuffled_by_state[state][cursors[state]])
                cursors[state] += 1
                added_this_round = true
                length(selected_ids) == min(n_save, length(source_state)) && break
            end
            added_this_round || break
        end
        sort!(selected_ids)

        result, _ = selected_result.model(
            X_model[:, selected_ids],
            cdev(selected_result.ps),
            Lux.testmode(cdev(selected_result.st));
            training=false,
        )
        recon = Float32.(cdev(result.xhat))
        selected_state = source_state[selected_ids]
        state_counts = [count(==(state), selected_state) for state in 1:nstates]
        save_path = joinpath(selected_result.run_dir, "raw_reconstruction_examples_100.jld2")

        jldsave(save_path;
            pair=selected_result.pair,
            seed=selected_result.seed,
            dt=dt,
            period_min=period_min,
            period_max=period_max,
            distance=selected_result.data_bundle.distance,
            selected_ids=selected_ids,
            branch=branches[selected_ids],
            branch_index=branch_indices[selected_ids],
            raw_waveforms=X_raw[:, selected_ids],
            model_input_waveforms=X_model[:, selected_ids],
            reconstruction_waveforms=recon,
            source_state=selected_state,
            state_labels=state_labels,
            stage_indices=stage_indices[:, selected_ids],
            state_counts=state_counts,
        #     note="Reconstructions are in the model-input preprocessing space; raw_waveforms are the original selected pair traces before whitening/filtering.")
			   )
			@show  save_path

        # md"Saved $(length(selected_ids)) examples to `$(save_path)`. State counts: $(state_counts)"
    end
end

# ╔═╡ Cell order:
# ╠═10000001-0000-0000-0000-000000000001
# ╠═96817feb-6aa9-4f35-9277-9a4560e9a2a7
# ╠═e0c79630-dbe0-4110-b880-a9ee9e4e1186
# ╠═f3d2428d-fcd6-4218-85d1-ef016cd8b999
# ╠═206a2c26-b3cc-4e74-83e1-fa92aa0bdd10
# ╠═c672d254-b73d-4191-86a5-ae11be0df3cb
# ╠═bdeb2eb2-6b5d-4677-837c-7072e3588430
# ╠═2a5aed3b-1319-4870-b086-6ecdc6996fc6
# ╠═f3a39849-5f08-46ae-bbae-562f3fe73cf5
# ╟─10000002-0000-0000-0000-000000000001
# ╠═10000003-0000-0000-0000-000000000001
# ╠═787a0d87-e6ef-4b44-83db-480489795df6
# ╟─10000004-0000-0000-0000-000000000001
# ╠═10000005-0000-0000-0000-000000000001
# ╠═10000006-0000-0000-0000-000000000001
# ╠═10000007-0000-0000-0000-000000000001
# ╠═0bf6c416-bf64-4b2a-b55e-623a4f9daae2
# ╠═272932ae-4c54-41a2-8c9a-e1cd4834150a
# ╠═10000008-0000-0000-0000-000000000001
# ╠═10000008-0000-0000-0000-000000000005
# ╠═1000000c-0000-0000-0000-000000000004
# ╠═c0000002-0000-0000-0000-000000000002
# ╠═d5000002-0000-0000-0000-000000000002
# ╠═c0000003-0000-0000-0000-000000000003
# ╟─10000009-0000-0000-0000-000000000001
# ╠═1000000a-0000-0000-0000-000000000001
# ╠═1000000b-0000-0000-0000-000000000001
# ╠═1000000c-0000-0000-0000-000000000003
# ╠═1000000c-0000-0000-0000-000000000002
# ╟─1000000c-0000-0000-0000-000000000001
# ╠═1000000d-0000-0000-0000-000000000002
# ╠═1000000d-0000-0000-0000-000000000001
# ╠═10000008-0000-0000-0000-000000000002
# ╠═10000008-0000-0000-0000-000000000003
# ╠═10000008-0000-0000-0000-000000000004
# ╠═77e454ce-619f-11f1-ba90-7fee4b9c9de0
# ╠═1000000e-0000-0000-0000-000000000001
# ╠═9757ced0-619a-11f1-b98d-815aa847ff5b
# ╠═70b49b49-8d0e-4b1b-a361-be7f1014607a
# ╠═66bc6645-a459-40fb-a871-2dd674e5bf9d
# ╟─1000000f-0000-0000-0000-000000000001
# ╟─10000010-0000-0000-0000-000000000001
# ╠═3c23d4cb-4bf9-4b4b-a57f-fefdc8f29cf1
# ╠═928d4238-216b-4f62-a4d7-62075275252b
# ╠═3ef20afc-4c62-4aab-b96f-4e170f8d96dc
# ╠═3ef20afc-4c62-4aab-b96f-4e170f8d96dd
# ╠═3ef20afc-4c62-4aab-b96f-4e170f8d96de
# ╠═f41c2e62-efbf-47c8-8b7e-adf828ffde4f
# ╠═10000011-0000-0000-0000-000000000001
# ╠═10000013-0000-0000-0000-000000000001
# ╠═10000014-0000-0000-0000-000000000001
# ╠═13c7f289-d698-453f-a9b1-65bf2c87f465
# ╠═7f4e1dc2-f3dd-45b2-8818-6eff1ed4f7b5
# ╠═9dbe9444-75f3-4d59-8b47-3f70d6bcf6b3
# ╠═b00b61c7-81c6-467a-b8bd-9f9e2ebd4d0c
# ╠═9c18f181-8d76-46a4-836d-c4fe2cad19c8
# ╠═ea4761ee-461d-4429-a13a-e872ae52cc66
# ╠═d1eec798-b7aa-4387-9a1e-71064e2660fd
# ╠═d5000002-0000-0000-0000-000000000003
# ╠═d5000002-0000-0000-0000-000000000004
# ╠═a6f36936-6199-11f1-b72f-69a0d065f9cd
