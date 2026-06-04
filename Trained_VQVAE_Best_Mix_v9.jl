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

# ╔═╡ b1000001-0000-0000-0000-000000000001
begin
    using Base.Threads
    using JLD2
    using LinearAlgebra
    using PlutoLinks
    using PlutoPlotly
    using PlutoUI
    using Printf
    using ProgressLogging
    using Random
    using Statistics
end

# ╔═╡ abd77dbc-e3e4-41cf-9b09-e10d50a3b465
using FFTW

# ╔═╡ 4d6e4b3b-239e-4009-9267-2c4ce447477f
using DSP

# ╔═╡ 527b90c6-e613-4ce4-ade7-80cc6b633c24
using Peaks

# ╔═╡ 16568c2d-2e04-438c-814d-8cb97943598c
using StatsBase

# ╔═╡ ff0c5258-7575-4603-8826-d7a18a798b74
using ColorSchemes, Colors

# ╔═╡ b1000003-0000-0000-0000-000000000001
md"""
# Trained VQ-VAE Best Mix v9

This producer notebook keeps MFT on the compute side of the tomography workflow.
It scores best waveform mixes against source-state causal/acausal symmetric pick
clouds, saves the winning best-mix picks, and saves global-mean branch picks for
the MFT-free `Prepare_Tomography_v9.jl` notebook.
"""

# ╔═╡ b100001a-0000-0000-0000-000000000001
begin
    @bind _artifact_write_controls PlutoUI.combine() do Child
        md"""
        | Artifact output | Value |
        |:---|:---|
        | Pick dataset path | $(Child("path", TextField(default=joinpath(@__DIR__, "DSurfTomo_runs", "v9_best_mix_pick_dataset.jld2")))) |
        | Write | $(Child("write", CounterButton("Write lightweight pick artifact"))) |
        """
    end
    
end

# ╔═╡ b21bc1cd-bf79-466e-9c84-5799957cdb72
@bind reload_saved_artifacts CounterButton("Reload saved source-state artifacts")

# ╔═╡ b100000c-0000-0000-0000-000000000001

    @bind _mft_compute_controls PlutoUI.combine() do Child
        md"""
        | MFT compute control | Value |
        |:---|:---|
        | Wavelength reference velocity (km/s) | $(Child("wavelength_ref_velocity", NumberField(0.5:0.5:5.0; default=2.0))) |
        | Wavelength fraction of distance | $(Child("wavelength_fraction", NumberField(0.05:0.01:1.0; default=0.33))) |
        | Upsample factor | $(Child("upsample_factor", confirm(NumberField(1:1:10; default=2)))) |
        | Numeric precision | $(Child("precision", Select(["Float32", "Float64"]; default="Float32"))) |
        """
    end


# ╔═╡ b1000008-0000-0000-0000-000000000001
begin
    @bind _best_mix_controls PlutoUI.combine() do Child
        md"""
        | Best-mix control | Value |
        |:---|:---|
        | Reference pick family | $(Child("reference_mode", Select(["Joint states (K1xK2)", "Marginal stages (K1+K2)"]; default="Joint states (K1xK2)"))) |
        | Candidate family | $(Child("best_mix_mode", Select(["stage1 + stage2", "marginal causal/acausal/mean atoms", "joint (K1xK2)", "stage2", "stage1"]; default="stage1 + stage2"))) |
        | Number of mixture trials | $(Child("mix_ntrials", confirm(NumberField(10:10:2000; default=256)))) |
        """
    end

end

# ╔═╡ b1000002-0000-0000-0000-000000000001
mft = @ingredients(joinpath(@__DIR__, "MFT_v2.jl"))

# ╔═╡ 7335cd68-e796-4f8e-9e89-221f7a2ead53
begin
	best_mix_pick_artifact_path = _artifact_write_controls.path
	    write_best_mix_pick_artifact = _artifact_write_controls.write
end

# ╔═╡ b1000004-0000-0000-0000-000000000001
begin
    saved_root = "/mnt/NAS2/Sanket_data/California_XJ_13032026/SavedModels/vqvae_v10_K=[5, 3]"

	saved_root ="/mnt/NAS2/Pushkar_Data/uttaranchal_data/jldfiles/30mins_dt_1p0_band_0p01_0p5_250maxlag_selected/Z/SavedModels/vqvae_v10_K=[5, 3]"
    
end

# ╔═╡ b1000005-0000-0000-0000-000000000001
begin
    function _parse_seed_timestamp(run_dir)
        m = match(r"^seed([0-9]+)_(.+)$", basename(run_dir))
        m === nothing && return (; seed=missing, timestamp=basename(run_dir))
        return (; seed=parse(Int, m.captures[1]), timestamp=m.captures[2])
    end
    _artifact_path(run) = joinpath(run.run_dir, "source_state_averages.jld2")
    function _read_settings(run_dir)
        d = load(joinpath(run_dir, "source_state_averages.jld2"))
        haskey(d, "analysis_settings") || error("analysis_settings missing in $(run_dir)")
        return d["analysis_settings"]
    end
    function discover_runs(root)
        isdir(root) || return NamedTuple[]
        out = NamedTuple[]
        for pair_dir in sort(filter(isdir, readdir(root, join=true)))
            stations = split(basename(pair_dir), "_")
            length(stations) == 2 || continue
            for run_dir in sort(filter(isdir, readdir(pair_dir, join=true)))
                isfile(joinpath(run_dir, "source_state_averages.jld2")) || continue
                parsed = _parse_seed_timestamp(run_dir)
                push!(out, (; pair=(String(stations[1]), String(stations[2])),
                    pair_label="$(stations[1])-$(stations[2])", run_dir,
                    parsed.seed, parsed.timestamp, analysis_settings=_read_settings(run_dir)))
            end
        end
        out
    end
    function load_source_artifact(run)
        d = load(_artifact_path(run))
        return (; pair=run.pair, pair_label=run.pair_label, run_dir=run.run_dir,
            seed=run.seed, timestamp=run.timestamp,
            causal=Float64.(d["causal"]), acausal=Float64.(d["acausal"]),
            combo_labels=String.(d["combo_labels"]),
            marginal_stage1_c=haskey(d, "marginal_stage1_c") ? Float64.(d["marginal_stage1_c"]) : zeros(0, 0),
            marginal_stage1_ac=haskey(d, "marginal_stage1_ac") ? Float64.(d["marginal_stage1_ac"]) : zeros(0, 0),
            marginal_stage2_c=haskey(d, "marginal_stage2_c") ? Float64.(d["marginal_stage2_c"]) : zeros(0, 0),
            marginal_stage2_ac=haskey(d, "marginal_stage2_ac") ? Float64.(d["marginal_stage2_ac"]) : zeros(0, 0),
            marginal_stage1_labels=haskey(d, "marginal_stage1_labels") ? String.(d["marginal_stage1_labels"]) : String[],
            marginal_stage2_labels=haskey(d, "marginal_stage2_labels") ? String.(d["marginal_stage2_labels"]) : String[],
            codebook_joint_waves=haskey(d, "codebook_joint_waves") ? Float64.(d["codebook_joint_waves"]) : zeros(0, 0),
            codebook_joint_labels=haskey(d, "codebook_joint_labels") ? String.(d["codebook_joint_labels"]) : String[],
            codebook_stage1_waves=haskey(d, "codebook_stage1_waves") ? Float64.(d["codebook_stage1_waves"]) : zeros(0, 0),
            codebook_stage1_labels=haskey(d, "codebook_stage1_labels") ? String.(d["codebook_stage1_labels"]) : String[],
            codebook_stage2_waves=haskey(d, "codebook_stage2_waves") ? Float64.(d["codebook_stage2_waves"]) : zeros(0, 0),
            codebook_stage2_labels=haskey(d, "codebook_stage2_labels") ? String.(d["codebook_stage2_labels"]) : String[],
            global_avg_c=Float64.(vec(d["global_avg_c"])),
            global_avg_ac=Float64.(vec(d["global_avg_ac"])),
            distance=Float64(d["distance"]), latitudes=d["latitudes"],
            longitudes=d["longitudes"], analysis_settings=d["analysis_settings"])
    end
end

# ╔═╡ b1000006-0000-0000-0000-000000000001
all_saved_runs = begin
    reload_saved_artifacts
    discover_runs(saved_root)
end

# ╔═╡ b1000007-0000-0000-0000-000000000001
begin
    pair_options = sort(unique(r.pair_label for r in all_saved_runs))
    @bind selected_pair_labels confirm(MultiCheckBox(pair_options; default=pair_options[1:min(end, 8)], select_all=true))
end

# ╔═╡ d1e98a61-0c2a-440e-a2b4-5894098aafa7
begin
	    ui_reference_mode = _best_mix_controls.reference_mode
	    ui_best_mix_mode = _best_mix_controls.best_mix_mode
	    ui_mix_ntrials = _best_mix_controls.mix_ntrials
end

# ╔═╡ b1000009-0000-0000-0000-000000000001
selected_runs = let
    raw = [r for r in all_saved_runs if r.pair_label in selected_pair_labels]
    use_latest = !isempty(raw) && hasproperty(first(raw).analysis_settings, :use_latest_run_per_seed) ?
        Bool(first(raw).analysis_settings.use_latest_run_per_seed) : false
    if use_latest
        latest = Dict{Tuple{String,Any},Any}()
        for run in raw
            key = (run.pair_label, run.seed)
            if !haskey(latest, key) || string(run.timestamp) > string(latest[key].timestamp)
                latest[key] = run
            end
        end
        sort(collect(values(latest)), by=r -> (r.pair_label, string(r.seed), r.timestamp))
    else
        sort(raw, by=r -> (r.pair_label, string(r.seed), r.timestamp))
    end
end

# ╔═╡ b100000a-0000-0000-0000-000000000001
source_artifacts = let
    out = Vector{Any}(undef, length(selected_runs))
    @progress name="Load source-state artifacts" for i in eachindex(selected_runs)
        out[i] = load_source_artifact(selected_runs[i])
    end
    out
end

# ╔═╡ b100000b-0000-0000-0000-000000000001
begin
    isempty(all_saved_runs) && error("No v9 source-state artifacts are available.")
    analysis_settings = first(all_saved_runs).analysis_settings
    _setting(s, name) = hasproperty(s, name) ? getproperty(s, name) : error("Missing setting $(name)")
    dt = Float64(_setting(analysis_settings, :dt))
    period_min = Float64(_setting(analysis_settings, :period_min))
    period_max = Float64(_setting(analysis_settings, :period_max))
    mft_nperiods = Int(_setting(analysis_settings, :mft_nperiods))
    mft_max_modes = hasproperty(analysis_settings, :mft_max_modes) ? Int(analysis_settings.mft_max_modes) : 6
    bandwidth_factor = Float64(_setting(analysis_settings, :bandwidth_factor))
    zero_pad_factor = Int(_setting(analysis_settings, :zero_pad_factor))
    saved_velocity_range = _setting(analysis_settings, :velocity_range)
    velocity_range = (Float64(saved_velocity_range[1]), Float64(saved_velocity_range[2]))
	velocity_range = (0.5, 8.0)
    mft_periods = exp10.(range(log10(period_min), log10(period_max); length=mft_nperiods))
end

# ╔═╡ 032ac7e1-7c31-4485-b44e-7f6a09b9dde0
begin
	    ui_wavelength_ref_velocity = _mft_compute_controls.wavelength_ref_velocity
	    ui_wavelength_fraction = _mft_compute_controls.wavelength_fraction
	    ui_mft_upsample_factor = _mft_compute_controls.upsample_factor
	    ui_mft_precision = _mft_compute_controls.precision
end

# ╔═╡ b100000d-0000-0000-0000-000000000001
begin
    mft_banks = Dict{Any,Any}()
    _precision_type() = String(ui_mft_precision) == "Float64" ? Float64 : Float32
    valid_periods(distance) = Float64[p for p in mft_periods if
        mft.wavelength_valid_period(p, distance;
            wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
            wavelength_fraction=Float64(ui_wavelength_fraction))]
    function bank_for(periods, npts, ncols)
        isempty(periods) && return nothing
        key = (Tuple(periods), npts, ncols, _precision_type())
        get!(mft_banks, key) do
            mft.MFTFilterBank(dt, npts, periods; bandwidth_factor, zero_pad_factor,
                upsample_factor=Float64(ui_mft_upsample_factor), velocity_range,
                precision=_precision_type(), storage_mode=:picks_only, N_initial=ncols)
        end
    end
end

# ╔═╡ b100000e-0000-0000-0000-000000000001
begin
    function _reference_specs(items)
        specs = NamedTuple[]
        for item in items
            if String(ui_reference_mode) == "Marginal stages (K1+K2)"
                choices = ((item.marginal_stage1_c, item.marginal_stage1_ac, item.marginal_stage1_labels, "S1"),
                    (item.marginal_stage2_c, item.marginal_stage2_ac, item.marginal_stage2_labels, "S2"))
                for (stage, (Wc, Wa, labels, prefix)) in enumerate(choices)
                    for k in axes(Wc, 2)
                        size(Wa, 2) >= k || continue
                        label = k <= length(labels) ? labels[k] : string(k)
                        family = stage == 1 ? "stage1" : "stage2"
                        push!(specs, (; pair_label=item.pair_label, distance=item.distance,
                            causal=vec(Wc[:, k]), acausal=vec(Wa[:, k]),
                            state_label="$(item.pair_label) seed $(item.seed) $(prefix) $(label)",
                            reference_family=family,
                            causal_reference_state_id=k,
                            acausal_reference_state_id=k,
                            reference_state_label="$(prefix) $(label)"))
                    end
                end
            else
                for k in axes(item.causal, 2)
                    label = k <= length(item.combo_labels) ? item.combo_labels[k] : string(k)
                    push!(specs, (; pair_label=item.pair_label, distance=item.distance,
                        causal=vec(item.causal[:, k]), acausal=vec(item.acausal[:, k]),
                        state_label="$(item.pair_label) seed $(item.seed) $(label)",
                        reference_family="joint",
                        causal_reference_state_id=k,
                        acausal_reference_state_id=k,
                        reference_state_label=label))
                end
            end
        end
        specs
    end
    function analyze_reference_specs(specs)
        by_pair = Dict{String,Any}()
        for pair_label in sort(unique(s.pair_label for s in specs))
            pair_specs = [s for s in specs if s.pair_label == pair_label]
            isempty(pair_specs) && continue
            n = minimum(min(length(s.causal), length(s.acausal)) for s in pair_specs)
            periods = valid_periods(first(pair_specs).distance)
            bank = bank_for(periods, n, 2 * length(pair_specs))
            isnothing(bank) && continue
            Wc = reduce(hcat, [s.causal[1:n] for s in pair_specs])
            Wa = reduce(hcat, [s.acausal[1:n] for s in pair_specs])
            by_pair[pair_label] = mft.analyze_causal_acausal_branches(Wc, Wa,
                [s.distance for s in pair_specs], bank;
                state_labels=[s.state_label for s in pair_specs], max_modes=mft_max_modes)
        end
        by_pair
    end
end

# ╔═╡ b100000f-0000-0000-0000-000000000001
begin
    reference_specs = _reference_specs(source_artifacts)
    reference_spec_same_by_label = Dict(s.state_label => (hasproperty(s, :same_codebook_state) ? s.same_codebook_state : false)
        for s in reference_specs)
    reference_batches = analyze_reference_specs(reference_specs)
end

# ╔═╡ b1000010-0000-0000-0000-000000000001
function matched_reference_rows(pair_label, batch; velocity_tolerance_fraction=0.10)
    rows = NamedTuple[]
    for (istate, st) in enumerate(batch.state_results), ip in eachindex(st.periods)
        corr = st.branch_correlation[ip]
        isfinite(corr) || continue
        causal = [(t=t, amplitude=a, velocity=st.distance / t, peak_rank=rank)
            for (rank, (t, a)) in enumerate(st.causal_result.all_peaks[ip]) if isfinite(t) && t > 0]
        acausal = [(t=t, amplitude=a, velocity=st.distance / t, peak_rank=rank)
            for (rank, (t, a)) in enumerate(st.acausal_result.all_peaks[ip]) if isfinite(t) && t > 0]
        used = falses(length(acausal))
        for c in causal
            best, best_rel = 0, Inf
            for (j, a) in enumerate(acausal)
                used[j] && continue
                denom = max((abs(c.velocity) + abs(a.velocity)) / 2, eps(Float64))
                rel = abs(c.velocity - a.velocity) / denom
                if rel <= velocity_tolerance_fraction && rel < best_rel
                    best, best_rel = j, rel
                end
            end
            best == 0 && continue
            used[best] = true
            a = acausal[best]
            push!(rows, (; pair_label=String(pair_label),
                state_index=istate, state_label=batch.state_labels[istate],
                same_codebook_state=get(reference_spec_same_by_label, batch.state_labels[istate], false),
                period=Float64(st.periods[ip]), group_velocity=0.5 * (c.velocity + a.velocity),
                peak_rank=min(c.peak_rank, a.peak_rank),
                causal_peak_rank=c.peak_rank, acausal_peak_rank=a.peak_rank,
                causal_peak_time=c.t, causal_amplitude=c.amplitude, causal_velocity=c.velocity,
                acausal_peak_time=a.t, acausal_amplitude=a.amplitude, acausal_velocity=a.velocity,
                velocity_relative_difference=best_rel, branch_correlation=corr, quality=corr))
        end
    end
    sort(rows, by=r -> (r.pair_label, r.period, r.state_index, r.group_velocity))
end

# ╔═╡ b1000011-0000-0000-0000-000000000001
reference_pick_rows = let
    chunks = [matched_reference_rows(pair, batch) for (pair, batch) in reference_batches]
    isempty(chunks) ? NamedTuple[] : vcat(chunks...)
end

# ╔═╡ b1000012-0000-0000-0000-000000000001
begin
    function global_branch_rows(items)
        rows = NamedTuple[]
        global_results = Dict{String,Any}()
        for pair_label in sort(unique(item.pair_label for item in items))
            pair_items = [item for item in items if item.pair_label == pair_label]
            n = minimum(min(length(item.global_avg_c), length(item.global_avg_ac)) for item in pair_items)
            n == 0 && continue
            c = vec(mean(reduce(hcat, [item.global_avg_c[1:n] for item in pair_items]); dims=2))
            ac = vec(mean(reduce(hcat, [item.global_avg_ac[1:n] for item in pair_items]); dims=2))
            dist = first(pair_items).distance
            periods = valid_periods(dist)
            bank = bank_for(periods, n, 2)
            isnothing(bank) && continue
            res = mft.analyze_causal_acausal_branches(c, ac, dist, bank; max_modes=mft_max_modes)
            global_results[pair_label] = res
            for (branch, mres) in (("causal", res.causal_result), ("acausal", res.acausal_result))
                for ip in eachindex(mres.periods), (peak_rank, (t, amp)) in enumerate(mres.all_peaks[ip])
                    isfinite(t) && t > 0 || continue
                    push!(rows, (; pair_label=String(pair_label), branch,
                        period=Float64(mres.periods[ip]), peak_time=t,
                        group_velocity=dist / t, peak_amplitude=amp,
                        peak_rank,
                        quality=mres.quality_factors[ip], branch_correlation=res.branch_correlation[ip]))
                end
            end
        end
        return sort(rows, by=r -> (r.pair_label, r.period, r.branch, r.group_velocity)), global_results
    end
end

# ╔═╡ b1000013-0000-0000-0000-000000000001
global_average_pick_rows, global_branch_results = global_branch_rows(source_artifacts)

# ╔═╡ b1000014-0000-0000-0000-000000000001
begin
    reference_cloud(pair_label) = [(r.period, r.group_velocity)
        for r in reference_pick_rows if r.pair_label == pair_label]
    function score_result(res, distance, reference; vtol=0.10)
        isempty(reference) && return 0.0
        candidate = Tuple{Float64,Float64}[]
        for ip in eachindex(res.periods), (t, _) in res.all_peaks[ip]
            isfinite(t) && t > 0 || continue
            push!(candidate, (res.periods[ip], distance / t))
        end
        isempty(candidate) && return 0.0
        return sum(any(abs(p - rp) / max(rp, 1e-8) < 0.02 &&
                abs(v - rv) / max((abs(v) + abs(rv)) / 2, 1e-8) < vtol
                for (p, v) in candidate) for (rp, rv) in reference) / length(reference)
    end
    function selected_codebooks(pair_items)
        columns = Vector{Float64}[]
        labels = String[]
        function append_family!(W, local_labels, family_label, seed)
            isempty(W) && return nothing
            for k in axes(W, 2)
                label = k <= length(local_labels) ? local_labels[k] : string(k)
                push!(columns, vec(W[:, k]))
                push!(labels, "seed$(seed)-$(family_label)-$(label)")
            end
            nothing
        end
        function append_marginal_branch_atoms!(Wc, Wa, local_labels, stage_label, seed)
            isempty(Wc) && return nothing
            isempty(Wa) && return nothing
            size(Wc, 2) == size(Wa, 2) || return nothing
            n = min(size(Wc, 1), size(Wa, 1))
            n > 0 || return nothing
            for k in axes(Wc, 2)
                label = k <= length(local_labels) ? local_labels[k] : string(k)
                c = vec(Wc[1:n, k])
                ac = vec(Wa[1:n, k])
                mean_wave = 0.5 .* (c .+ ac)
                push!(columns, c)
                push!(labels, "seed$(seed)-$(stage_label)C-$(label)")
                push!(columns, ac)
                push!(labels, "seed$(seed)-$(stage_label)AC-$(label)")
                push!(columns, mean_wave)
                push!(labels, "seed$(seed)-$(stage_label)MEAN-$(label)")
            end
            nothing
        end
        for item in pair_items
            mode = String(ui_best_mix_mode)
            if mode == "stage1 + stage2"
                append_family!(item.codebook_stage1_waves, item.codebook_stage1_labels, "S1", item.seed)
                append_family!(item.codebook_stage2_waves, item.codebook_stage2_labels, "S2", item.seed)
            elseif mode == "marginal causal/acausal/mean atoms"
                append_marginal_branch_atoms!(item.marginal_stage1_c, item.marginal_stage1_ac,
                    item.marginal_stage1_labels, "S1", item.seed)
                append_marginal_branch_atoms!(item.marginal_stage2_c, item.marginal_stage2_ac,
                    item.marginal_stage2_labels, "S2", item.seed)
            elseif mode == "joint (K1xK2)" && !isempty(item.codebook_joint_waves)
                append_family!(item.codebook_joint_waves, item.codebook_joint_labels, "J", item.seed)
            elseif mode == "stage2" && !isempty(item.codebook_stage2_waves)
                append_family!(item.codebook_stage2_waves, item.codebook_stage2_labels, "S2", item.seed)
            else
                append_family!(item.codebook_stage1_waves, item.codebook_stage1_labels, "S1", item.seed)
            end
        end
        return columns, labels
    end
    function l2_normalize_atom(w)
        wf = Float64.(w)
        all(isfinite, wf) || return nothing
        nrm = norm(wf)
        isfinite(nrm) && nrm > 0 || return nothing
        wf ./ nrm
    end
    function uniform_normalized_average(columns)
        atoms = Vector{Float64}[]
        for col in columns
            atom = l2_normalize_atom(col)
            isnothing(atom) && continue
            push!(atoms, atom)
        end
        isempty(atoms) && return nothing, 0
        n = minimum(length.(atoms))
        n > 0 || return nothing, 0
        avg = zeros(Float64, n)
        for atom in atoms
            avg .+= atom[1:n]
        end
        avg ./= length(atoms)
        return avg, length(atoms)
    end
end

# ╔═╡ b1000015-0000-0000-0000-000000000001
best_mix_results = let
    Random.seed!(20260522)
    results = NamedTuple[]
    @progress name="Best mix per pair" for pair_label in sort(unique(item.pair_label for item in source_artifacts))
        items = [item for item in source_artifacts if item.pair_label == pair_label]
        ref = reference_cloud(pair_label)
        (isempty(items) || isempty(ref)) && continue
        columns, labels = selected_codebooks(items)
        isempty(columns) && continue
        W = reduce(hcat, columns)
        n = size(W, 1)
        dist = first(items).distance
        periods = valid_periods(dist)
        bank = bank_for(periods, n, Int(ui_mix_ntrials))
        isnothing(bank) && continue
        K = size(W, 2)
        alphas = [begin a=zeros(K); a[k]=1; a end for k in 1:min(K, Int(ui_mix_ntrials))]
        while length(alphas) < Int(ui_mix_ntrials)
            x = -log.(rand(K) .+ 1e-12)
            push!(alphas, x ./ sum(x))
        end
        Wmix = reduce(hcat, [W * alpha for alpha in alphas])
        mres = mft.perform_mft_analysis_batch!(bank, Wmix, dist; compute_phase=false)
        scores = [score_result(mres[j], dist, ref) for j in eachindex(mres)]
        winner = argmax(scores)
        push!(results, (; pair_label=String(pair_label), pair=first(items).pair,
            distance=dist, result=mres[winner], score=scores[winner],
            reference_pick_count=length(ref), alpha=alphas[winner],
            codebook_labels=labels, winning_trial=winner, trial_scores=scores))
    end
    results
end

# ╔═╡ b100001c-0000-0000-0000-000000000001
begin
    result_labels = sort(unique(r.pair_label for r in best_mix_results))
    @bind _best_mix_plot_controls PlutoUI.combine() do Child
        md"""
        | Pick-cloud plot control | Value |
        |:---|:---|
        | Pair | $(Child("pair", Select(result_labels; default=isempty(result_labels) ? missing : first(result_labels)))) |
        | Reference picks | $(Child("same_only", CheckBox(default=false))) c == ac == k only |
        """
    end
end

# ╔═╡ b100001f-0000-0000-0000-000000000001
begin
    selected_best_mix_pair = _best_mix_plot_controls.pair
    ui_show_same_codebook_reference_only = _best_mix_plot_controls.same_only
end

# ╔═╡ b1000016-0000-0000-0000-000000000001
top_best_mix_pick_rows = let
    rows = NamedTuple[]
    for best in best_mix_results, ip in eachindex(best.result.periods), (peak_rank, (t, amp)) in enumerate(best.result.all_peaks[ip])
        isfinite(t) && t > 0 || continue
        push!(rows, (; pair_label=best.pair_label, period=Float64(best.result.periods[ip]),
            peak_time=t, group_velocity=best.distance / t, peak_amplitude=amp,
            peak_rank,
            quality=best.result.quality_factors[ip], winner_score=best.score,
            winner_trial=best.winning_trial, candidate_rank=1, state_label="top_best_mix"))
    end
    sort(rows, by=r -> (r.pair_label, r.period, r.group_velocity))
end

# ╔═╡ b1000017-0000-0000-0000-000000000001
pair_geometry_rows = let
    rows = NamedTuple[]
    for pair_label in sort(unique(item.pair_label for item in source_artifacts))
        item = first([x for x in source_artifacts if x.pair_label == pair_label])
        lat = isnothing(item.latitudes) ? [NaN, NaN] : Float64.(item.latitudes)
        lon = isnothing(item.longitudes) ? [NaN, NaN] : Float64.(item.longitudes)
        push!(rows, (; pair_label=String(pair_label), station1=String(item.pair[1]),
            station2=String(item.pair[2]), distance=Float64(item.distance),
            lat1=length(lat) >= 1 ? lat[1] : NaN, lat2=length(lat) >= 2 ? lat[2] : NaN,
            lon1=length(lon) >= 1 ? lon[1] : NaN, lon2=length(lon) >= 2 ? lon[2] : NaN))
    end
    rows
end

# ╔═╡ b1000015-0000-0000-0000-000000000002
uniform_average_results = let
    results = NamedTuple[]
    @progress name="Uniform normalized atom average per pair" for pair_label in sort(unique(item.pair_label for item in source_artifacts))
        items = [item for item in source_artifacts if item.pair_label == pair_label]
        isempty(items) && continue
        columns, labels = selected_codebooks(items)
        avg, atom_count = uniform_normalized_average(columns)
        isnothing(avg) && continue
        dist = first(items).distance
        periods = valid_periods(dist)
        bank = bank_for(periods, length(avg), 1)
        isnothing(bank) && continue
        Wavg = reshape(avg, :, 1)
        mres = mft.perform_mft_analysis_batch!(bank, Wavg, dist; compute_phase=false)
        push!(results, (; pair_label=String(pair_label), pair=first(items).pair,
            distance=dist, result=only(mres), atom_count,
            candidate_family=String(ui_best_mix_mode), codebook_labels=labels))
    end
    results
end

# ╔═╡ b1000016-0000-0000-0000-000000000002
uniform_average_pick_rows = let
    rows = NamedTuple[]
    for avg in uniform_average_results, ip in eachindex(avg.result.periods), (peak_rank, (t, amp)) in enumerate(avg.result.all_peaks[ip])
        isfinite(t) && t > 0 || continue
        push!(rows, (; pair_label=avg.pair_label, period=Float64(avg.result.periods[ip]),
            peak_time=t, group_velocity=avg.distance / t, peak_amplitude=amp,
            peak_rank,
            quality=avg.result.quality_factors[ip], atom_count=avg.atom_count,
            candidate_family=avg.candidate_family,
            state_label="uniform_normalized_atom_average"))
    end
    sort(rows, by=r -> (r.pair_label, r.period, r.group_velocity))
end

# ╔═╡ b1000018-0000-0000-0000-000000000001
begin
    best_mix_metadata = [(; pair_label=b.pair_label, winning_trial=b.winning_trial,
        score=b.score, reference_pick_count=b.reference_pick_count,
        codebook_labels=b.codebook_labels, alpha=b.alpha,
        nonzero_weights=[(; label=b.codebook_labels[k], weight=b.alpha[k])
            for k in eachindex(b.alpha) if b.alpha[k] > 0.05])
        for b in best_mix_results]
    run_provenance = [(; pair_label=item.pair_label, pair=item.pair,
        run_dir=item.run_dir, seed=item.seed, timestamp=item.timestamp) for item in source_artifacts]
    artifact_preview = (; pairs=length(pair_geometry_rows),
        reference_symmetric_picks=length(reference_pick_rows),
        global_branch_picks=length(global_average_pick_rows),
        uniform_normalized_average_picks=length(uniform_average_pick_rows),
        top_best_mix_picks=length(top_best_mix_pick_rows),
        best_mix_winners=length(best_mix_metadata))
end

# ╔═╡ b1000019-0000-0000-0000-000000000001
artifact_preview

# ╔═╡ b100001b-0000-0000-0000-000000000001
let
    write_best_mix_pick_artifact == 0 && return md"Artifact rows are ready; press the write button to save them."
    mkpath(dirname(best_mix_pick_artifact_path))
    artifact_settings = (; dt, period_min, period_max, mft_nperiods, mft_periods,
        velocity_range, bandwidth_factor, zero_pad_factor, mft_max_modes,
        upsample_factor=Float64(ui_mft_upsample_factor), precision=String(ui_mft_precision),
        storage_mode="picks_only", wavelength_ref_velocity=Float64(ui_wavelength_ref_velocity),
        wavelength_fraction=Float64(ui_wavelength_fraction),
        reference_mode=String(ui_reference_mode), best_mix_mode=String(ui_best_mix_mode),
        uniform_average_atom_normalization="unit_l2",
        uniform_average_candidate_family=String(ui_best_mix_mode),
        mix_trials=Int(ui_mix_ntrials), scoring_reference="source_state_symmetric",
        velocity_tolerance_fraction=0.10)
    jldsave(best_mix_pick_artifact_path;
        schema_version="v9_best_mix_pick_dataset_1",
        saved_root, artifact_settings, pair_geometry_rows, run_provenance,
        reference_pick_rows, global_average_pick_rows, top_best_mix_pick_rows,
        best_mix_metadata)
    md"Wrote lightweight pick artifact to `$(best_mix_pick_artifact_path)`."
end

# ╔═╡ b100001e-0000-0000-0000-000000000001
begin
    function _mft_peak_marker_size(rank)
        r = ismissing(rank) ? typemax(Int) : Int(rank)
        r <= 1 && return 20
        r == 2 && return 13
        r == 3 && return 8
        return 4
    end

    _mft_peak_marker_sizes(rows) = [_mft_peak_marker_size(r.peak_rank) for r in rows]

    # Draw low-priority peaks first so the strongest rank-1 picks stay visible.
    _mft_rank_plot_order(rows) = sort(rows, by=r -> (Int(r.peak_rank), r.period, r.group_velocity), rev=true)
end

# ╔═╡ b100001d-0000-0000-0000-000000000001
let
    reference_rows_all = [r for r in reference_pick_rows if r.pair_label == selected_best_mix_pair]
    reference_rows_filtered = ui_show_same_codebook_reference_only ?
        [r for r in reference_rows_all if r.same_codebook_state] :
        reference_rows_all
    reference_rows = _mft_rank_plot_order(reference_rows_filtered)
    best_rows = _mft_rank_plot_order([r for r in top_best_mix_pick_rows if r.pair_label == selected_best_mix_pair])
    uniform_rows = _mft_rank_plot_order([r for r in uniform_average_pick_rows if r.pair_label == selected_best_mix_pair])
    global_rows = _mft_rank_plot_order([r for r in global_average_pick_rows if r.pair_label == selected_best_mix_pair])
    isempty(reference_rows) && isempty(best_rows) && isempty(uniform_rows) && isempty(global_rows) &&
        return md"No saved dispersion pick clouds for the selected pair."

    traces = AbstractTrace[]
    !isempty(reference_rows) && push!(traces, PlutoPlotly.scatter(
        x=[r.period for r in reference_rows],
        y=[r.group_velocity for r in reference_rows],
        mode="markers",
        name=ui_show_same_codebook_reference_only ? "Reference c==ac==k" : "Source-state symmetric reference",
        text=["$(r.state_label)<br>c==ac==k: $(r.same_codebook_state)<br>MFT rank=$(r.peak_rank) (C=$(r.causal_peak_rank), AC=$(r.acausal_peak_rank))" for r in reference_rows],
        marker=PlutoPlotly.attr(symbol="circle", size=_mft_peak_marker_sizes(reference_rows), opacity=0.55,
            color=[r.branch_correlation for r in reference_rows],
            colorscale="Viridis", colorbar=PlutoPlotly.attr(title="Reference NCC")),
    ))
    !isempty(best_rows) && push!(traces, PlutoPlotly.scatter(
        x=[r.period for r in best_rows],
        y=[r.group_velocity for r in best_rows],
        mode="markers",
        name="Winning best-mix picks",
        text=["MFT rank=$(r.peak_rank)<br>winner trial=$(r.winner_trial)<br>score=$(round(r.winner_score; digits=3))" for r in best_rows],
        marker=PlutoPlotly.attr(symbol="triangle-up", size=_mft_peak_marker_sizes(best_rows), color="royalblue",
            opacity=0.75, line=PlutoPlotly.attr(width=1, color="navy")),
    ))
    !isempty(uniform_rows) && push!(traces, PlutoPlotly.scatter(
        x=[r.period for r in uniform_rows],
        y=[r.group_velocity for r in uniform_rows],
        mode="markers",
        name="Uniform normalized atom average",
        text=["MFT rank=$(r.peak_rank)<br>atoms=$(r.atom_count), family=$(r.candidate_family)" for r in uniform_rows],
        marker=PlutoPlotly.attr(symbol="star", size=_mft_peak_marker_sizes(uniform_rows), color="orange",
            opacity=0.82, line=PlutoPlotly.attr(width=1, color="darkorange")),
    ))
    for branch in ("causal", "acausal")
        rows = [r for r in global_rows if r.branch == branch]
        isempty(rows) && continue
        push!(traces, PlutoPlotly.scatter(
            x=[r.period for r in rows],
            y=[r.group_velocity for r in rows],
            mode="markers",
            name="Global mean $(branch)",
            text=["MFT rank=$(r.peak_rank)<br>$(branch) peak" for r in rows],
            marker=PlutoPlotly.attr(symbol="diamond", size=_mft_peak_marker_sizes(rows),
                color=branch == "causal" ? "gray" : "black",
                opacity=0.72, line=PlutoPlotly.attr(width=1, color="black")),
        ))
    end
    WideCell(PlutoPlotly.plot(traces, PlutoPlotly.Layout(
        title="Best-Mix Pick Clouds: $(selected_best_mix_pair)",
        xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
        yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"),
        width=1100, height=560,
        legend=PlutoPlotly.attr(orientation="h", x=0.5, xanchor="center",
            y=-0.18, yanchor="top"),
        margin=PlutoPlotly.attr(l=70, r=90, t=60, b=110),
        plot_bgcolor="white", paper_bgcolor="white")))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
ColorSchemes = "~3.31.0"
Colors = "~0.13.1"
DSP = "~0.8.5"
FFTW = "~1.10.0"
JLD2 = "~0.6.4"
Peaks = "~0.6.2"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
ProgressLogging = "~0.1.6"
StatsBase = "~0.34.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "d20b2f4cf6af3344d8e9830cc3bb0de279d9f21b"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

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

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

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

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d335b2929e1b6067951a1250df247cc5fab7d40e"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.5"

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

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

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
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "f76f7560267b840e492180f9899b472f30b88450"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

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

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

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

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

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
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

[[deps.Peaks]]
deps = ["SIMD"]
git-tree-sha1 = "a9b6680fb7fb097fb6eb1210c35549218d73da84"
uuid = "18e31ff7-3703-566c-8e60-38913d67486b"
version = "0.6.2"

    [deps.Peaks.extensions]
    MakieExt = "Makie"
    PlotsExt = "RecipesBase"

    [deps.Peaks.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

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
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

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

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

    [deps.Revise.weakdeps]
    Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

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

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

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

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

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

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

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
# ╟─b1000003-0000-0000-0000-000000000001
# ╟─b100001c-0000-0000-0000-000000000001
# ╟─b100001d-0000-0000-0000-000000000001
# ╠═b1000019-0000-0000-0000-000000000001
# ╟─b100001a-0000-0000-0000-000000000001
# ╟─b21bc1cd-bf79-466e-9c84-5799957cdb72
# ╠═b1000007-0000-0000-0000-000000000001
# ╟─b100000c-0000-0000-0000-000000000001
# ╟─b1000008-0000-0000-0000-000000000001
# ╠═b1000001-0000-0000-0000-000000000001
# ╠═abd77dbc-e3e4-41cf-9b09-e10d50a3b465
# ╠═4d6e4b3b-239e-4009-9267-2c4ce447477f
# ╠═527b90c6-e613-4ce4-ade7-80cc6b633c24
# ╠═16568c2d-2e04-438c-814d-8cb97943598c
# ╠═ff0c5258-7575-4603-8826-d7a18a798b74
# ╠═b1000002-0000-0000-0000-000000000001
# ╠═7335cd68-e796-4f8e-9e89-221f7a2ead53
# ╠═b1000004-0000-0000-0000-000000000001
# ╠═b1000005-0000-0000-0000-000000000001
# ╠═b1000006-0000-0000-0000-000000000001
# ╠═d1e98a61-0c2a-440e-a2b4-5894098aafa7
# ╠═b1000009-0000-0000-0000-000000000001
# ╠═b100000a-0000-0000-0000-000000000001
# ╠═b100000b-0000-0000-0000-000000000001
# ╠═032ac7e1-7c31-4485-b44e-7f6a09b9dde0
# ╠═b100000d-0000-0000-0000-000000000001
# ╠═b100000e-0000-0000-0000-000000000001
# ╠═b100000f-0000-0000-0000-000000000001
# ╠═b1000010-0000-0000-0000-000000000001
# ╠═b1000011-0000-0000-0000-000000000001
# ╠═b1000012-0000-0000-0000-000000000001
# ╠═b1000013-0000-0000-0000-000000000001
# ╠═b1000014-0000-0000-0000-000000000001
# ╠═b1000015-0000-0000-0000-000000000001
# ╠═b100001f-0000-0000-0000-000000000001
# ╠═b1000016-0000-0000-0000-000000000001
# ╠═b1000017-0000-0000-0000-000000000001
# ╠═b1000018-0000-0000-0000-000000000001
# ╠═b100001b-0000-0000-0000-000000000001
# ╠═b1000015-0000-0000-0000-000000000002
# ╠═b1000016-0000-0000-0000-000000000002
# ╠═b100001e-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
