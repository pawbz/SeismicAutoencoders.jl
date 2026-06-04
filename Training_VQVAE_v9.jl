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

# ╔═╡ 10000000-0000-0000-0000-000000000000
begin
    import Pkg
    Pkg.activate("/home/pawan/.julia/vqvae_cli")
end

# ╔═╡ 10000001-0000-0000-0000-000000000001
begin
    using JLD2,
        PlutoLinks,
        PlutoPlotly,
        PlutoUI,
        Random,
        Statistics
	
end

# ╔═╡ 96817feb-6aa9-4f35-9277-9a4560e9a2a7
using FFTW, Peaks, ColorSchemes, Colors, InlineStrings

# ╔═╡ b30b5f82-cbeb-41aa-9a4f-212d6aafa760
using Zygote

# ╔═╡ e0c79630-dbe0-4110-b880-a9ee9e4e1186
using Distances

# ╔═╡ 206a2c26-b3cc-4e74-83e1-fa92aa0bdd10
using PlutoHooks

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

# ╔═╡ 10000002-0000-0000-0000-000000000001
md"""
# VQ-VAE v10 Training — Split-Decoder Interferometric VQ

Short orchestration notebook.  The v10 architecture notebook owns the model,
training loop, split VQ, kNN, saving/loading, source-state averaging, and plotting
helpers.  This notebook selects station pairs, starts training, and connects the
trained source states to MFT.
"""

# ╔═╡ 10000003-0000-0000-0000-000000000001

    vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_architecture_v9.jl")


# ╔═╡ 787a0d87-e6ef-4b44-83db-480489795df6


# ╔═╡ 10000004-0000-0000-0000-000000000001
md"## Data and Pair Selection"

# ╔═╡ 10000005-0000-0000-0000-000000000001
begin
    # data_filepath = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/"
	# data_filepath = "/mnt/NAS2/Sanket_data/California_XJ_13032026/"
	data_filepath = "/mnt/NAS2/Sanket_data/California_XJ_13032026/"

	data_filepath =  "/mnt/NAS2/Sanket_data/Minneapolis_pairs_SS_29052026/"
		data_filepath =  "/mnt/sanket1/Minneapolis_pairs_SM_29052026_new/"
	# data_filepath = "/mnt/NAS2/Pushkar_Data/uttaranchal_data/jldfiles/30mins_dt_1p0_band_0p01_0p5_250maxlag_selected/Z/"

    dt = 1.0
    period_min = 10
    period_max = 75
    mft_nT = 20        # number of periods log-spaced between period_min and period_max
    velocity_range = (1.0, 8.0)
    mft_nperiods = 100
    mft_max_modes = 6
    bandwidth_factor = 0.15
    zero_pad_factor = 4
    use_latest_run_per_seed = true
    training_n_max = 10  # maximum pooled waveforms per pair; use nothing for no cap
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
    batchsize=2,
    nepoch=100,
    initial_learning_rate=0.001,
    weight_decay=0.0,
    Mnn_schedule=[(1, 128), (5, 256), (26, 256)],
    warmup_epochs=0,
	verbose = false,
	autodiff_backend = :zygote,
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
    # force recompile when notebook is re-evaluated
    compiled_model_key_cache_v9[] = nothing
end

# ╔═╡ 10000008-0000-0000-0000-000000000003
selected_pairs_key = (
    vqvae_parameters=vqvae_parameters,
    training_batchsize=training_para.batchsize,
    warmup_epochs=training_para.warmup_epochs,
    index_refresh_every=training_para.index_refresh_every,
    training_n_max=training_n_max,
)

# ╔═╡ dc766e83-6a3f-4b1e-ac6c-c000024cec07
autodiff_backend = :zygote

# ╔═╡ 18e13701-fff1-41ea-9ee7-b81dde615eab
# @compile encode_z_e_inference(model, ps, Lux.testmode(st), sample_x)  # redundant: compile_model already does this

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

# ╔═╡ 10000008-0000-0000-0000-000000000004
compiled_model = let
    if compile_button === missing || compile_button == 0
        compiled_model_cache_v9[]
    else
        nt = size(pairs_data_whitened[1].data.D_train, 1)
        n_train = size(pairs_data_whitened[1].data.D_train, 2)
        compiled_model_cache_v9[] = vqvae.compile_model(nt, n_train;
            vqvae_parameters=vqvae_parameters,
            training_para=training_para,
            seed=1234,
            device=training_device,
        )
        compiled_model_key_cache_v9[] = selected_pairs_key
        compiled_model_cache_v9[]
    end
end

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

# ╔═╡ 1000000e-0000-0000-0000-000000000001
train_results = @use_memo([]) do
    if train_button === missing || train_button == 0
        nothing
    elseif isnothing(compiled_model)
        error("Compile model before training.")
    else
        vqvae.train_selected_pairs_lazy(
            selected_pairs, compiled_model;
            seeds=run_seeds,
            training_para=training_para,
            save_root=joinpath(data_filepath, "SavedModels", "vqvae_v10_K=$(string(compiled_model.para.K))"),
            filepath=data_filepath,
            dt, period_min, period_max,
            n_max=training_n_max,
            bp_filter,
            per_waveform_whitening_kernel_length,
            device=training_device,
            analysis_settings,
        )
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


# ╔═╡ Cell order:
# ╠═10000000-0000-0000-0000-000000000000
# ╠═10000001-0000-0000-0000-000000000001
# ╠═96817feb-6aa9-4f35-9277-9a4560e9a2a7
# ╠═b30b5f82-cbeb-41aa-9a4f-212d6aafa760
# ╠═e0c79630-dbe0-4110-b880-a9ee9e4e1186
# ╠═206a2c26-b3cc-4e74-83e1-fa92aa0bdd10
# ╠═c672d254-b73d-4191-86a5-ae11be0df3cb
# ╠═bdeb2eb2-6b5d-4677-837c-7072e3588430
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
# ╠═dc766e83-6a3f-4b1e-ac6c-c000024cec07
# ╠═18e13701-fff1-41ea-9ee7-b81dde615eab
# ╠═1000000e-0000-0000-0000-000000000001
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
