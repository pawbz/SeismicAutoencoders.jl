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
    using CUDA,
        Dates,
        DSP,
        Enzyme,
        EnzymeCore,
        Flux,
        LinearAlgebra,
        MLUtils,
        NNlib,
        Optimisers,
        StatsBase
end

# ╔═╡ 10000002-0000-0000-0000-000000000001
md"""
# VQ-VAE v9 Training (Flux) — Split-Decoder Interferometric VQ

Flux version of the v9 training notebook. The architecture notebook owns the model,
training loop, split VQ, kNN, saving/loading, source-state averaging, and plotting
helpers. This notebook selects station pairs, starts training, and connects the
trained source states to MFT.

No Reactant/XLA compilation — just plain Flux + Zygote on GPU.
"""

# ╔═╡ 10000003-0000-0000-0000-000000000001
vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_architecture_v9_flux.jl")

# ╔═╡ 10000004-0000-0000-0000-000000000001
md"## Data and Pair Selection"

# ╔═╡ 10000005-0000-0000-0000-000000000001
begin
    # data_filepath = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/"
	# data_filepath = "/mnt/NAS2/Sanket_data/California_XJ_13032026/"
	data_filepath = "/mnt/NAS2/Pushkar_Data/uttaranchal_data/jldfiles/30mins_dt_1p0_band_0p01_0p5_250maxlag_selected/Z/"

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
    training_n_max = 12000  # maximum pooled waveforms per pair; use nothing for no cap
end

# ╔═╡ 1000000c-0000-0000-0000-000000000004
begin
	whitening_kernel_length = 256   # FIR tap count for spectral whitening
	per_waveform_whitening_kernel_length = 128  # per-waveform FIR taps
end

# ╔═╡ c0000002-0000-0000-0000-000000000002
bp_filter = let
    responsetype = DSP.Bandpass(inv(period_max), inv(period_min))
    DSP.digitalfilter(responsetype, DSP.Butterworth(2); fs=inv(dt))
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
	verbose = false,
	knn_search_chunk_size_fraction = 0.25,
    index_refresh_every=2,
)

# ╔═╡ 1000000c-0000-0000-0000-000000000003
run_seeds = [1234, 1235]

# ╔═╡ 1000000c-0000-0000-0000-000000000002
training_device = vqvae.default_xdev()

# ╔═╡ 1000000c-0000-0000-0000-000000000001
md"## Train"

# ╔═╡ 1000000d-0000-0000-0000-000000000001
@bind train_button CounterButton("Train selected pairs")

# ╔═╡ 1000000f-0000-0000-0000-000000000001
md"## Pair Selection"

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
    else
        vqvae.train_selected_pairs_lazy(
            selected_pairs;
            vqvae_parameters=vqvae_parameters,
            seeds=run_seeds,
            training_para=training_para,
            save_root=joinpath(data_filepath, "SavedModels", "vqvae_v9_flux_K=$(string(vqvae_parameters.K))"),
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

# ╔═╡ b00b61c7-81c6-467a-b8bd-9f9e2ebd4d0c
result_title_context(result) = begin
    pair = result.pair
    distance_label = isnothing(result.data_bundle.distance) ?
        "distance unavailable" : "$(round(Int, result.data_bundle.distance))km"
    seed_label = hasproperty(result, :seed) ? " seed=$(result.seed)" : ""
    "$(pair[1])-$(pair[2])$(seed_label) $(distance_label) $(period_min)-$(period_max)s"
end

# ╔═╡ 10000011-0000-0000-0000-000000000001
if isnothing(selected_result)
    md"Select pairs, then press **Train selected pairs** to run v9_flux training."
else
    WideCell(vqvae.plot_training_dashboard(selected_result.loss_history;
        title="VQ-VAE v9 Flux Training Dashboard ($(result_title_context(selected_result)))"))
end

# ╔═╡ 10000013-0000-0000-0000-000000000001
if isnothing(selected_result)
    md""
else
    WideCell(vqvae.plot_codebook_heatmap(selected_result.rvq_st; stage=1,
        title="RVQ Stage 1 Codebook ($(result_title_context(selected_result)))"))
end

# ╔═╡ 10000014-0000-0000-0000-000000000001
if isnothing(selected_result)
    md""
else
    WideCell(vqvae.plot_reconstruction_examples(
        selected_result.model,
        selected_result.rvq_st,
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
        selected_result.rvq_st,
        selected_result.data.D_c_all;
        nsamples=8,
        dt=dt,
        device=training_device,
        title="Causal Reconstruction Examples ($(result_title_context(selected_result)))",
    ))
end

# ╔═╡ 7f4e1dc2-f3dd-45b2-8818-6eff1ed4f7b5
@bind save_recon_examples_button CounterButton("Save 100 raw recon examples")

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
            selected_result.rvq_st,
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
        n_per_state = max(1, n_save ÷ nstates)
        selected_indices = vcat([ids[1:min(n_per_state, length(ids))] for ids in shuffled_by_state]...)
        selected_indices = selected_indices[1:min(n_save, length(selected_indices))]

        model_cpu = cdev(selected_result.model)
        rvq_cpu = cdev(selected_result.rvq_st)
        X_sel = X_model[:, selected_indices]
        res, _ = model_cpu(X_sel, rvq_cpu; training=false)
        recon_sel = Float32.(res.xhat)

        save_dir = joinpath(selected_result.run_dir, "recon_examples_$(Dates.format(Dates.now(), "yyyymmdd_HHMMSS"))")
        mkpath(save_dir)
        JLD2.jldsave(joinpath(save_dir, "recon_examples.jld2");
            X_input=X_sel,
            X_recon=recon_sel,
            X_raw=X_raw[:, selected_indices],
            branches=branches[selected_indices],
            branch_indices=branch_indices[selected_indices],
            source_state=source_state[selected_indices],
            state_labels=state_labels,
            pair=selected_result.pair,
            seed=selected_result.seed,
            dt=dt,
        )
        md"Saved $(length(selected_indices)) reconstruction examples to `$(save_dir)`"
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
InlineStrings = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
CUDA = "~5"
DSP = "~0.8"
Distances = "~0.10"
Enzyme = "~0.13"
EnzymeCore = "~0.8"
FFTW = "~1"
Flux = "~0.14"
InlineStrings = "~1.4"
JLD2 = "~0.4"
MLUtils = "~0.4"
NNlib = "~0.9"
Optimisers = "~0.3"
PlutoPlotly = "~0.6"
StatsBase = "~0.34"
Zygote = "~0.6"
"""
