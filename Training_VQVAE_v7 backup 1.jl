### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook
# outside Pluto, this mock binding provides a default value.
macro bind(def, element)
    return quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
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

# ╔═╡ 10000002-0000-0000-0000-000000000001
md"""
# VQ-VAE v7 Training

Short orchestration notebook.  The v7 architecture notebook owns the model,
training loop, RVQ, kNN, saving/loading, source-state averaging, and plotting
helpers.  This notebook selects station pairs, starts training, and connects the
trained source states to MFT.
"""

# ╔═╡ 10000003-0000-0000-0000-000000000001
begin
    vqvae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/VQVAE_architecture_v7.jl")
    mft = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/MFT.jl")
end

# ╔═╡ 10000004-0000-0000-0000-000000000001
md"## Data and Pair Selection"

# ╔═╡ 10000005-0000-0000-0000-000000000001
begin
    data_filepath = "/mnt/NAS2/Sanket_data/California_TO_with_latlong/"
    dt = 1.0
    period_min = 10
    period_max = 50
end

# ╔═╡ 10000006-0000-0000-0000-000000000001
available_pairs = vqvae.list_station_pairs(data_filepath)

# ╔═╡ 10000007-0000-0000-0000-000000000001
begin
    isempty(available_pairs) && error("No station pairs found in $(data_filepath)")
    pair_options = ["$(p[1])-$(p[2])" for p in available_pairs]
    default_pair_names = pair_options[1:min(end, 1)]
    @bind selected_pair_names confirm(MultiCheckBox(pair_options; default=default_pair_names))
end

# ╔═╡ 10000008-0000-0000-0000-000000000001
selected_pairs = begin
    isempty(selected_pair_names) && error("Select at least one station pair.")
    [begin
        parts = split(name, "-", limit=2)
        length(parts) == 2 || error("Invalid pair name $(name).")
        (parts[1], parts[2])
    end for name in selected_pair_names]
end

# ╔═╡ 10000009-0000-0000-0000-000000000001
md"## Hyperparameters"

# ╔═╡ 1000000a-0000-0000-0000-000000000001
vqvae_parameters = (;
    d=24,
    beta_commit=0.25f0,
    K=[5],
    ema_decay=0.99f0,
    dead_threshold=50,
    entropy_weight=0.1f0,
    reconstruction_loss=:l2,
    velocity_range=(2.0, 6.0),
    envelope_floor=0.1f0,
)

# ╔═╡ 1000000b-0000-0000-0000-000000000001
training_para = vqvae.VQVAE_Training_Para(
    batchsize=512,
    nepoch=20,
    initial_learning_rate=0.001,
    weight_decay=0.0,
    Mnn_schedule=[(1, 10, :median), (6, 10, :mean), (26, 10, :mean)],
    warmup_epochs=5,
    index_refresh_every=1,
    latent_index_batch_size=256,
    latent_index_space=:z_metric_flat,
    compile_reactant=true,
)

# ╔═╡ 1000000c-0000-0000-0000-000000000001
md"## Train"

# ╔═╡ 1000000d-0000-0000-0000-000000000001
@bind train_button CounterButton("Train selected pairs")

# ╔═╡ 1000000e-0000-0000-0000-000000000001
train_results = let
    if train_button === missing || train_button == 0
        nothing
    else
        vqvae.train_selected_pairs(
            selected_pairs;
            filepath=data_filepath,
            vqvae_parameters=vqvae_parameters,
            training_para=training_para,
            save_root=joinpath(data_filepath, "SavedModels", "vqvae_v7"),
            seed=1234,
            dt=dt,
            period_min=period_min,
            period_max=period_max,
        )
    end
end

# ╔═╡ 1000000f-0000-0000-0000-000000000001
md"## Inspect One Result"

# ╔═╡ 10000010-0000-0000-0000-000000000001
selected_result = isnothing(train_results) ? nothing : first(train_results)

# ╔═╡ 10000011-0000-0000-0000-000000000001
if isnothing(selected_result)
    md"Press **Train selected pairs** to create v7 runs."
else
    vqvae.plot_training_dashboard(selected_result.loss_history;
        title="VQ-VAE v7 $(selected_result.pair[1])-$(selected_result.pair[2])")
end

# ╔═╡ 10000012-0000-0000-0000-000000000001
if isnothing(selected_result)
    md""
else
    vqvae.plot_envelope(selected_result.para)
end

# ╔═╡ 10000013-0000-0000-0000-000000000001
if isnothing(selected_result)
    md""
else
    vqvae.plot_codebook_heatmap(selected_result.st; stage=1)
end

# ╔═╡ 10000014-0000-0000-0000-000000000001
if isnothing(selected_result)
    md""
else
    vqvae.plot_reconstruction_examples(
        selected_result.model,
        selected_result.ps,
        selected_result.st,
        selected_result.data.D_ac_all;
        nsamples=8,
        dt=dt,
        device=vqvae.default_xdev(; force=true),
        title="Acausal reconstructions $(selected_result.pair[1])-$(selected_result.pair[2])",
    )
end

# ╔═╡ 10000015-0000-0000-0000-000000000001
md"## Source-State Averages and MFT"

# ╔═╡ 10000016-0000-0000-0000-000000000001
state_averages = if isnothing(selected_result)
    nothing
else
    vqvae.source_state_averages(
        selected_result.model,
        selected_result.ps,
        selected_result.st,
        selected_result.data;
        device=vqvae.default_xdev(; force=true),
    )
end

# ╔═╡ 10000017-0000-0000-0000-000000000001
if isnothing(state_averages)
    md""
else
    vqvae.plot_state_average_matrix(state_averages.acausal;
        title="Acausal source-state averages", dt=dt, reverse_time=false)
end

# ╔═╡ 10000018-0000-0000-0000-000000000001
if isnothing(state_averages)
    md""
else
    vqvae.plot_state_average_matrix(state_averages.causal;
        title="Causal source-state averages", dt=dt, reverse_time=false)
end

# ╔═╡ 10000019-0000-0000-0000-000000000001
mft_analysis = if isnothing(state_averages)
    nothing
else
    nstates = size(state_averages.acausal, 2)
    global_avg_ac = vec(mean(selected_result.data.D_ac_all; dims=2))
    global_avg_c = vec(mean(selected_result.data.D_c_all; dims=2))
    ac_traces = [
        mft.SeismicTrace(data=vec(state_averages.acausal[:, i]), dt=dt,
            distance=selected_result.data_bundle.distance)
        for i in 1:nstates
    ]
    push!(ac_traces, mft.SeismicTrace(data=global_avg_ac, dt=dt,
        distance=selected_result.data_bundle.distance))
    c_traces = [
        mft.SeismicTrace(data=vec(state_averages.causal[:, i]), dt=dt,
            distance=selected_result.data_bundle.distance)
        for i in 1:nstates
    ]
    push!(c_traces, mft.SeismicTrace(data=global_avg_c, dt=dt,
        distance=selected_result.data_bundle.distance))
    labels = vcat(string.(1:nstates), "Full")
    mft.analyze_causal_acausal_branches(
        ac_traces,
        c_traces;
        state_labels=labels,
        period_max=80.0,
        velocity_range=(1.0, 8.0),
        bandwidth_factor=0.15,
        zero_pad_factor=4,
    )
end

# ╔═╡ 1000001a-0000-0000-0000-000000000001
if isnothing(mft_analysis)
    md""
else
    @bind ui_period Slider(mft_analysis.periods; default=10, show_value=true)
end

# ╔═╡ 1000001b-0000-0000-0000-000000000001
if isnothing(mft_analysis)
    md""
else
    mft.plot_filtered_traces_by_period(
        mft_analysis;
        period=ui_period,
        correlation_threshold=nothing,
        normalize_each=true,
        scale=0.7,
        spacing=2.2,
        title="$(selected_result.pair[1])-$(selected_result.pair[2]) filtered source-state traces",
    )
end

# ╔═╡ 1000001c-0000-0000-0000-000000000001
if isnothing(mft_analysis)
    md""
else
    mft.plot_branch_correlation(
        mft_analysis;
        title="$(selected_result.pair[1])-$(selected_result.pair[2]) branch correlation",
    )
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
JLD2 = "~0.6.4"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.71"
"""

# ╔═╡ Cell order:
# ╠═10000001-0000-0000-0000-000000000001
# ╟─10000002-0000-0000-0000-000000000001
# ╠═10000003-0000-0000-0000-000000000001
# ╟─10000004-0000-0000-0000-000000000001
# ╠═10000005-0000-0000-0000-000000000001
# ╠═10000006-0000-0000-0000-000000000001
# ╠═10000007-0000-0000-0000-000000000001
# ╠═10000008-0000-0000-0000-000000000001
# ╟─10000009-0000-0000-0000-000000000001
# ╠═1000000a-0000-0000-0000-000000000001
# ╠═1000000b-0000-0000-0000-000000000001
# ╟─1000000c-0000-0000-0000-000000000001
# ╠═1000000d-0000-0000-0000-000000000001
# ╠═1000000e-0000-0000-0000-000000000001
# ╟─1000000f-0000-0000-0000-000000000001
# ╠═10000010-0000-0000-0000-000000000001
# ╠═10000011-0000-0000-0000-000000000001
# ╠═10000012-0000-0000-0000-000000000001
# ╠═10000013-0000-0000-0000-000000000001
# ╠═10000014-0000-0000-0000-000000000001
# ╟─10000015-0000-0000-0000-000000000001
# ╠═10000016-0000-0000-0000-000000000001
# ╠═10000017-0000-0000-0000-000000000001
# ╠═10000018-0000-0000-0000-000000000001
# ╠═10000019-0000-0000-0000-000000000001
# ╠═1000001a-0000-0000-0000-000000000001
# ╠═1000001b-0000-0000-0000-000000000001
# ╠═1000001c-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000001
