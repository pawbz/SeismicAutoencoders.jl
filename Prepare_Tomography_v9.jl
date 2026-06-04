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

# ╔═╡ c1000001-0000-0000-0000-000000000001
begin
    using JLD2
    using PlutoPlotly
    using PlutoUI
    using Printf
    using Statistics
end

# ╔═╡ c1000002-0000-0000-0000-000000000001
begin
    include(joinpath(@__DIR__, "TomographySelection_v9.jl"))
    using .TomographySelection_v9
    tomo = TomographySelection_v9
end

# ╔═╡ c1000003-0000-0000-0000-000000000001
md"""
# Prepare Tomography v9

This notebook reads lightweight pick clouds produced by
`Trained_VQVAE_Best_Mix_v9.jl`. It has no MFT filter bank, filtered waveform, or
envelope dependency. It prepares global-average and best-mix pick-cloud
candidates, selects one candidate per receiver pair, and writes DSurfTomo tables.
"""

# ╔═╡ c1000004-0000-0000-0000-000000000001
begin
    @bind _pick_dataset_controls PlutoUI.combine() do Child
        md"""
        | Input dataset | Value |
        |:---|:---|
        | Best-mix pick artifact | $(Child("path", TextField(default=joinpath(@__DIR__, "DSurfTomo_runs", "v9_best_mix_pick_dataset.jld2")))) |
        | Station CSV paths | $(Child("station_csvs", TextField(default="/mnt/NAS2/Sanket_data/California_09032026/data/stationlists/Stations_California_XJ.csv;/mnt/NAS2/Sanket_data/California_09032026/data/stationlists/Stations_California_XJ_new.csv"))) |
        """
    end
end

# ╔═╡ c1000009-0000-0000-0000-000000000001
begin
    @bind _candidate_controls PlutoUI.combine() do Child
        md"""
        | Candidate selection control | Value |
        |:---|:---|
        | Velocity tolerance fraction | $(Child("tolerance", NumberField(0.02:0.01:0.30; default=0.10))) |
        | Minimum periods per candidate | $(Child("min_periods", NumberField(1:1:20; default=3))) |
        | Maximum missing period bins inside candidate | $(Child("max_gap_periods", NumberField(0:1:5; default=0))) |
        | Minimum candidate mean group velocity (km/s) | $(Child("mean_velocity_min", NumberField(0.0:0.05:10.0; default=1.5))) |
        | Maximum candidate mean group velocity (km/s) | $(Child("mean_velocity_max", NumberField(0.1:0.05:12.0; default=6.0))) |
        | Outlier sigma threshold | $(Child("outlier_nsigma", NumberField(0.5:0.1:5.0; default=0.8))) |
        """
    end
   
end

# ╔═╡ c100001f-0000-0000-0000-000000000001
begin
    @bind _dsurftomo_output_controls PlutoUI.combine() do Child
        md"""
        | DSurfTomo output | Value |
        |:---|:---|
        | Global-average dispersion path | $(Child("global_path", TextField(default=joinpath(@__DIR__, "DSurfTomo_runs", "prepare_v9_global_average_dispersion.txt")))) |
        | Best-mix dispersion path | $(Child("best_path", TextField(default=joinpath(@__DIR__, "DSurfTomo_runs", "prepare_v9_best_mix_dispersion.txt")))) |
        | Include count header | $(Child("count_header", CheckBox(default=false))) |
        """
    end
    
end

# ╔═╡ c1000021-0000-0000-0000-000000000001
begin
    @bind _dsurftomo_write_controls PlutoUI.combine() do Child
        md"""
        | DSurfTomo write action | Button |
        |:---|:---|
        | Global-average rows | $(Child("write_global", CounterButton("Write global-average DSurfTomo file"))) |
        | Best-mix rows | $(Child("write_best_mix", CounterButton("Write best-mix DSurfTomo file"))) |
        """
    end
 
end

# ╔═╡ 0e9a7ae1-09a6-4fa9-a5e7-a2c869c0c592
pick_artifact_path = _pick_dataset_controls.path

# ╔═╡ c1000005-0000-0000-0000-000000000001
pick_artifact = isfile(pick_artifact_path) ? load(pick_artifact_path) : nothing

# ╔═╡ c1000006-0000-0000-0000-000000000001
if isnothing(pick_artifact)
    md"Set `pick_artifact_path` to a JLD2 pick dataset written by `Trained_VQVAE_Best_Mix_v9.jl`."
else
    md"""
    Loaded schema **$(pick_artifact["schema_version"])**.

    Pairs: **$(length(pick_artifact["pair_geometry_rows"]))**.
    Reference symmetric picks: **$(length(pick_artifact["reference_pick_rows"]))**.
    Global branch picks: **$(length(pick_artifact["global_average_pick_rows"]))**.
    Top best-mix picks: **$(length(pick_artifact["top_best_mix_pick_rows"]))**.
    """
end

# ╔═╡ c1000007-0000-0000-0000-000000000001
begin
    artifact_settings = isnothing(pick_artifact) ? nothing : pick_artifact["artifact_settings"]
    pair_geometry_rows = isnothing(pick_artifact) ? NamedTuple[] : pick_artifact["pair_geometry_rows"]
    reference_pick_rows = isnothing(pick_artifact) ? NamedTuple[] : pick_artifact["reference_pick_rows"]
    global_branch_pick_rows = isnothing(pick_artifact) ? NamedTuple[] : pick_artifact["global_average_pick_rows"]
    top_best_mix_pick_rows = isnothing(pick_artifact) ? NamedTuple[] : pick_artifact["top_best_mix_pick_rows"]
end

# ╔═╡ c1000010-0000-0000-0000-000000000001
begin
    labels = sort(unique(vcat([r.pair_label for r in reference_pick_rows],
        [r.pair_label for r in global_branch_pick_rows], [r.pair_label for r in top_best_mix_pick_rows])))
    @bind selected_prepare_pair Select(labels; default=isempty(labels) ? missing : first(labels))
end

# ╔═╡ c1000011-0000-0000-0000-000000000001
let
    rows = [r for r in reference_pick_rows if r.pair_label == selected_prepare_pair]
    isempty(rows) && return md"No saved source-state symmetric reference picks for this pair."
    WideCell(PlutoPlotly.plot(PlutoPlotly.scatter(x=[r.period for r in rows],
        y=[r.group_velocity for r in rows], text=[r.state_label for r in rows],
        mode="markers", name="Reference",
        marker=PlutoPlotly.attr(size=7, color=[r.branch_correlation for r in rows],
            colorscale="Viridis", colorbar=PlutoPlotly.attr(title="NCC"))),
        PlutoPlotly.Layout(title="Saved best-mix scoring reference: $(selected_prepare_pair)",
            xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
            yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"), width=1050, height=500)))
end

# ╔═╡ c1000008-0000-0000-0000-000000000001
artifact_settings

# ╔═╡ 7839327c-bdb3-4ee0-a34f-c3530d4e9cf2
begin
 ui_candidate_tolerance = _candidate_controls.tolerance
	    ui_candidate_min_periods = _candidate_controls.min_periods
	    ui_candidate_max_gap_periods = _candidate_controls.max_gap_periods
	    ui_candidate_mean_velocity_min = _candidate_controls.mean_velocity_min
	    ui_candidate_mean_velocity_max = _candidate_controls.mean_velocity_max
	    ui_outlier_nsigma = _candidate_controls.outlier_nsigma
end

# ╔═╡ b3c368be-ac3d-4327-a68d-f11a3abb3aa2
# Distance can be used for diagnostics before station coordinates are available
# for a DSurfTomo export.
function _pair_distance_km(pair_label)
    idx = findfirst(g -> g.pair_label == pair_label, pair_geometry_rows)
    idx === nothing && return NaN
    d = Float64(pair_geometry_rows[idx].distance)
    return isfinite(d) && d > 0 ? d : NaN
end

# ╔═╡ 53921daa-6935-45a3-8e4a-eaa9ccb9c7f1
function _distance_color(distance, dmin, dmax; alpha=0.50)
    isfinite(distance) || return @sprintf("rgba(82,82,82,%.3f)", alpha)
    x = dmax > dmin ? clamp((distance - dmin) / (dmax - dmin), 0.0, 1.0) : 0.5
    anchors = ((68, 1, 84), (59, 82, 139), (33, 145, 140),
        (94, 201, 98), (253, 231, 37))
    pos = x * (length(anchors) - 1)
    i = min(floor(Int, pos) + 1, length(anchors) - 1)
    f = pos - (i - 1)
    rgb = round.(Int, (1 - f) .* collect(anchors[i]) .+ f .* collect(anchors[i + 1]))
    return @sprintf("rgba(%d,%d,%d,%.3f)", rgb[1], rgb[2], rgb[3], alpha)
end

# ╔═╡ 8eae1de6-81a8-4123-a538-b7e1dc681b8d
# Candidate selection only needs the pick-cloud branches. Geometry is required
# later when selected rows are converted to DSurfTomo source/receiver rows.
function _selection_inputs(consensus_by_pair)
    [(; label=pair_label, consensus=consensus_by_pair[pair_label])
     for pair_label in sort(collect(keys(consensus_by_pair)))]
end

# ╔═╡ c1000012-0000-0000-0000-000000000001
function plot_consensus(consensus_by_pair, pair_label; title,
        background_branch_rows=NamedTuple[], background_pick_rows=NamedTuple[],
        background_pick_label="Saved pick cloud")
    haskey(consensus_by_pair, pair_label) || return md"No consensus candidates for $(pair_label)."
    c = consensus_by_pair[pair_label]
    traces = AbstractTrace[]
    for (branch, color, symbol) in (
            ("causal", "#2166ac", "circle"),
            ("acausal", "#b2182b", "diamond"))
        rows = [r for r in background_branch_rows
            if r.pair_label == pair_label && lowercase(String(r.branch)) == branch &&
               isfinite(r.period) && r.period > 0 &&
               isfinite(r.group_velocity) && r.group_velocity > 0]
        isempty(rows) && continue
        push!(traces, PlutoPlotly.scatter(
            x=[r.period for r in rows],
            y=[r.group_velocity for r in rows],
            mode="markers",
            name="Global mean $(branch) picks",
            marker=PlutoPlotly.attr(size=6, color=color, symbol=symbol,
                opacity=0.30, line=PlutoPlotly.attr(color=color, width=0.5)),
            hovertemplate="Global mean $(branch)<br>Period: %{x:.3g} s<br>v_g: %{y:.3f} km/s<extra></extra>",
        ))
    end
    pick_rows = [r for r in background_pick_rows
        if r.pair_label == pair_label &&
           isfinite(r.period) && r.period > 0 &&
           isfinite(r.group_velocity) && r.group_velocity > 0]
    if !isempty(pick_rows)
        push!(traces, PlutoPlotly.scatter(
            x=[r.period for r in pick_rows],
            y=[r.group_velocity for r in pick_rows],
            mode="markers",
            name=background_pick_label,
            marker=PlutoPlotly.attr(size=6, color="#525252", symbol="x",
                opacity=0.35, line=PlutoPlotly.attr(color="#525252", width=0.8)),
            hovertemplate="$(background_pick_label)<br>Period: %{x:.3g} s<br>v_g: %{y:.3f} km/s<extra></extra>",
        ))
    end
    first_candidate = true
    for ic in axes(c.candidate_group_velocities, 2)
        valid = findall(v -> isfinite(v) && v > 0, c.candidate_group_velocities[:, ic])
        isempty(valid) && continue
        push!(traces, PlutoPlotly.scatter(x=c.periods[valid],
            y=c.candidate_group_velocities[valid, ic], mode="lines+markers",
            name="Candidate $(ic)",
            marker=PlutoPlotly.attr(size=6 .+ c.candidate_support[valid, ic],
                color=c.candidate_confidence[valid, ic], colorscale="Viridis",
                showscale=first_candidate, colorbar=PlutoPlotly.attr(title="Confidence"))))
        first_candidate = false
    end
    first_candidate && return md"No averaged pick-cloud candidate is available with the current controls."
    return WideCell(PlutoPlotly.plot(traces, PlutoPlotly.Layout(title=title,
        xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
        yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"), width=1050, height=520,
        legend=PlutoPlotly.attr(orientation="h", x=0.0, y=-0.18))))
end

# ╔═╡ c1000015-0000-0000-0000-000000000001
function selection_table(selection; title)
    isempty(selection.pair_diagnostics) && return md"No candidates were selected for **$(title)**."
    lines = ["| Pair | Candidate | Periods | Mean velocity | Outliers | Fraction | Mean confidence | Mean support | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in selection.pair_diagnostics
        push!(lines, @sprintf("| %s | %d | %d | %.3f | %d | %.3f | %.3f | %.2f | %s |",
            row.pair_label, row.selected_candidate, row.coverage, row.mean_velocity, row.n_outliers,
            row.outlier_fraction, row.mean_confidence, row.mean_support, row.status))
    end
    return Markdown.parse("### $(title)\n\n" * join(lines, "\n") *
        "\n\nIterations: **$(selection.iterations)**. Converged: **$(selection.converged)**.")
end

# ╔═╡ c1000016-0000-0000-0000-000000000001
function plot_selected_candidates(selection; title)
    rows = selection.selected_rows
    isempty(rows) && return md"No selected candidate rows are available."
    traces = AbstractTrace[]
    row_distances = [_pair_distance_km(row.pair_label) for row in rows]
    finite_distances = [d for d in row_distances if isfinite(d)]
    has_distances = !isempty(finite_distances)
    dmin = has_distances ? minimum(finite_distances) : NaN
    dmax = has_distances ? maximum(finite_distances) : NaN
    ps = sort(collect(keys(selection.period_stats)))
    μ = [selection.period_stats[p].mean for p in ps]
    σ = [selection.period_stats[p].std for p in ps]
    push!(traces, PlutoPlotly.scatter(x=vcat(ps, reverse(ps)),
        y=vcat(μ .+ Float64(ui_outlier_nsigma) .* σ,
            reverse(μ .- Float64(ui_outlier_nsigma) .* σ)),
        fill="toself", line=PlutoPlotly.attr(color="transparent"),
        fillcolor="rgba(0,0,0,0.08)", name="Final sigma band"))
    push!(traces, PlutoPlotly.scatter(x=ps, y=μ, mode="lines+markers",
        name="Selected mean", line=PlutoPlotly.attr(color="black", width=2),
        marker=PlutoPlotly.attr(size=5, color="black")))

    for pair_label in sort(unique(row.pair_label for row in rows))
        pair_rows = sort([row for row in rows if row.pair_label == pair_label], by=row -> row.period)
        candidate = first(pair_rows).candidate
        distance = _pair_distance_km(pair_label)
        distance_label = isfinite(distance) ? @sprintf("%.1f km", distance) : "distance unavailable"
        push!(traces, PlutoPlotly.scatter(x=[row.period for row in pair_rows],
            y=[row.group_velocity for row in pair_rows], mode="lines",
            name="$(pair_label) candidate $(candidate)",
            text=["$(pair_label) candidate $(candidate)<br>Distance: $(distance_label)"
                for _ in pair_rows],
            hovertemplate="%{text}<br>Period: %{x:.3g} s<br>v_g: %{y:.3f} km/s<extra></extra>",
            showlegend=false,
            line=PlutoPlotly.attr(width=1.25,
                color=_distance_color(distance, dmin, dmax; alpha=0.48))))
    end

    kept = [row for row in rows if row.status == "kept"]
    outliers = [row for row in rows if row.status == "outlier"]
    if !isempty(kept)
        kept_distances = [_pair_distance_km(row.pair_label) for row in kept]
        push!(traces, PlutoPlotly.scatter(x=[row.period for row in kept],
            y=[row.group_velocity for row in kept], mode="markers",
            name="Selected kept picks",
            text=["$(row.pair_label) candidate $(row.candidate)<br>Distance: " *
                (isfinite(distance) ? @sprintf("%.1f km", distance) : "unavailable")
                for (row, distance) in zip(kept, kept_distances)],
            marker=has_distances ?
                PlutoPlotly.attr(size=6, color=kept_distances, opacity=0.78,
                    colorscale="Viridis", cmin=dmin, cmax=dmax, showscale=true,
                    colorbar=PlutoPlotly.attr(title="Interstation<br>distance (km)"),
                    line=PlutoPlotly.attr(color="white", width=0.5)) :
                PlutoPlotly.attr(size=6, color="#1b9e77", opacity=0.72,
                    line=PlutoPlotly.attr(color="white", width=0.5)),
            hovertemplate="%{text}<br>Period: %{x:.3g} s<br>v_g: %{y:.3f} km/s<br>kept<extra></extra>"))
    end
    if !isempty(outliers)
        outlier_distances = [_pair_distance_km(row.pair_label) for row in outliers]
        push!(traces, PlutoPlotly.scatter(x=[row.period for row in outliers],
            y=[row.group_velocity for row in outliers], mode="markers",
            name="Selected outliers",
            text=["$(row.pair_label) candidate $(row.candidate)<br>Distance: " *
                (isfinite(distance) ? @sprintf("%.1f km", distance) : "unavailable")
                for (row, distance) in zip(outliers, outlier_distances)],
            marker=has_distances ?
                PlutoPlotly.attr(size=12, color=outlier_distances, symbol="x",
                    colorscale="Viridis", cmin=dmin, cmax=dmax, showscale=false,
                    line=PlutoPlotly.attr(width=1.8)) :
                PlutoPlotly.attr(size=11, color="#d95f02", symbol="x",
                    line=PlutoPlotly.attr(color="#d95f02", width=1.5)),
            hovertemplate="%{text}<br>Period: %{x:.3g} s<br>v_g: %{y:.3f} km/s<br>outlier<extra></extra>"))
    end

    WideCell(PlutoPlotly.plot(traces, PlutoPlotly.Layout(title=title,
        xaxis=PlutoPlotly.attr(title="Period (s)", type="log"),
        yaxis=PlutoPlotly.attr(title="Group velocity (km/s)"), width=1200, height=650,
        legend=PlutoPlotly.attr(orientation="h", x=0.0, y=-0.16))))
end

# ╔═╡ 6bb11d90-1745-4d9f-8662-d2c54480ec0b
begin
	global_dsurftomo_path = _dsurftomo_output_controls.global_path
	    best_mix_dsurftomo_path = _dsurftomo_output_controls.best_path
	    dsurftomo_count_header = _dsurftomo_output_controls.count_header
end

# ╔═╡ a401e664-077c-43b6-a864-3a3d08903b22
begin
	   write_global_dsurftomo = _dsurftomo_write_controls.write_global
	    write_best_mix_dsurftomo = _dsurftomo_write_controls.write_best_mix
end

# ╔═╡ c1000024-0000-0000-0000-000000000001
function _artifact_setting(name::Symbol, default)
    isnothing(artifact_settings) && return default
    hasproperty(artifact_settings, name) && return getproperty(artifact_settings, name)
    artifact_settings isa AbstractDict && haskey(artifact_settings, name) && return artifact_settings[name]
    artifact_settings isa AbstractDict && haskey(artifact_settings, String(name)) && return artifact_settings[String(name)]
    return default
end

# ╔═╡ e90a4b9e-b76f-4df1-b9c3-65a7ccd14a83
station_csv_paths = [strip(path) for path in split(_pick_dataset_controls.station_csvs, ";")
    if !isempty(strip(path))]

# ╔═╡ 4f1fd31f-3856-4b90-a615-f49d5a3d968e
function _station_coordinate_lookup(paths)
    lookup = Dict{String,NamedTuple}()
    for path in paths
        isfile(path) || continue
        lines = readlines(path)
        isempty(lines) && continue
        header = split(first(lines), ",")
        code_idx = findfirst(h -> strip(h) in ("Station Code", "Station", "station"), header)
        lat_idx = findfirst(h -> strip(h) in ("Latitude", "lat", "Lat"), header)
        lon_idx = findfirst(h -> strip(h) in ("Longitude", "lon", "Lon", "Longitude "), header)
        (isnothing(code_idx) || isnothing(lat_idx) || isnothing(lon_idx)) && continue
        for line in Iterators.drop(lines, 1)
            parts = split(line, ",")
            length(parts) >= maximum((code_idx, lat_idx, lon_idx)) || continue
            code = strip(parts[code_idx])
            isempty(code) && continue
            lat = tryparse(Float64, strip(parts[lat_idx]))
            lon = tryparse(Float64, strip(parts[lon_idx]))
            (isnothing(lat) || isnothing(lon)) && continue
            isfinite(lat) && isfinite(lon) || continue
            lookup[code] = (; lat, lon)
        end
    end
    lookup
end

# ╔═╡ fcb27df8-a296-42bf-9915-24de8b845e23
station_coordinate_lookup = _station_coordinate_lookup(station_csv_paths)

# ╔═╡ 38996efe-71bc-4ea3-9119-ca6bd61bc722
function _geometry_for(pair_label)
    idx = findfirst(g -> g.pair_label == pair_label, pair_geometry_rows)
    idx === nothing && return nothing
    g = pair_geometry_rows[idx]
    if all(isfinite, (g.lat1, g.lat2, g.lon1, g.lon2, g.distance))
        return g
    end
    s1 = get(station_coordinate_lookup, String(g.station1), nothing)
    s2 = get(station_coordinate_lookup, String(g.station2), nothing)
    (isnothing(s1) || isnothing(s2) || !isfinite(g.distance) || g.distance <= 0) && return nothing
    return (; pair_label=String(g.pair_label), station1=String(g.station1),
        station2=String(g.station2), distance=Float64(g.distance),
        lat1=Float64(s1.lat), lon1=Float64(s1.lon),
        lat2=Float64(s2.lat), lon2=Float64(s2.lon))
end

# ╔═╡ 2c1b166c-0a81-457d-a56a-2825292a3c2e
function _tomography_inputs(consensus_by_pair)
    inputs = tomo.PairConsensusForTomography[]
    for pair_label in sort(collect(keys(consensus_by_pair)))
        g = _geometry_for(pair_label)
        isnothing(g) && continue
        push!(inputs, tomo.tomography_pair_consensus((g.station1, g.station2),
            consensus_by_pair[pair_label]; label=pair_label,
            latitudes=[g.lat1, g.lat2], longitudes=[g.lon1, g.lon2], distance=g.distance))
    end
    inputs
end

# ╔═╡ c1000025-0000-0000-0000-000000000001
function _pair_period_grid(pair_label)
    periods = _artifact_setting(:mft_periods, nothing)
    isnothing(periods) && return nothing
    g = _geometry_for(pair_label)
    isnothing(g) && return Float64.(periods)
    wavelength_ref_velocity = Float64(_artifact_setting(:wavelength_ref_velocity, 2.0))
    wavelength_fraction = Float64(_artifact_setting(:wavelength_fraction, 0.33))
    return [Float64(period) for period in periods
        if tomo.wavelength_valid_period(period, g.distance;
            wavelength_ref_velocity, wavelength_fraction)]
end

# ╔═╡ c100000b-0000-0000-0000-000000000001
begin
    global_consensus_by_pair = Dict{String,Any}()
    for pair_label in sort(unique(r.pair_label for r in global_branch_pick_rows))
        rows = [r for r in global_branch_pick_rows if r.pair_label == pair_label]
        global_consensus_by_pair[pair_label] = tomo.pick_cloud_consensus_from_rows(rows;
            periods=_pair_period_grid(pair_label),
            cluster_tolerance_fraction=Float64(ui_candidate_tolerance),
            min_candidate_periods=Int(ui_candidate_min_periods),
            max_gap_periods=Int(ui_candidate_max_gap_periods))
    end
end

# ╔═╡ c1000013-0000-0000-0000-000000000001
plot_consensus(global_consensus_by_pair, selected_prepare_pair;
    title="Global-average consensus candidates: $(selected_prepare_pair)",
    background_branch_rows=global_branch_pick_rows)

# ╔═╡ c100000c-0000-0000-0000-000000000001
best_mix_consensus_by_pair = let
    out = Dict{String,Any}()
    for pair_label in sort(unique(r.pair_label for r in top_best_mix_pick_rows))
        rows = [r for r in top_best_mix_pick_rows if r.pair_label == pair_label]
        out[pair_label] = tomo.pick_cloud_consensus_from_rows(rows;
            periods=_pair_period_grid(pair_label),
            cluster_tolerance_fraction=Float64(ui_candidate_tolerance),
            min_candidate_periods=Int(ui_candidate_min_periods),
            max_gap_periods=Int(ui_candidate_max_gap_periods))
    end
    out
end

# ╔═╡ c1000014-0000-0000-0000-000000000001
plot_consensus(best_mix_consensus_by_pair, selected_prepare_pair;
    title="Top best-mix consensus candidates: $(selected_prepare_pair)",
    background_pick_rows=top_best_mix_pick_rows,
    background_pick_label="Winning best-mix picks")

# ╔═╡ c100000e-0000-0000-0000-000000000001
begin
    global_selection_inputs = _selection_inputs(global_consensus_by_pair)
    best_mix_selection_inputs = _selection_inputs(best_mix_consensus_by_pair)
    global_tomography_inputs = _tomography_inputs(global_consensus_by_pair)
    best_mix_tomography_inputs = _tomography_inputs(best_mix_consensus_by_pair)
    _candidate_pair_count(inputs) = count(input ->
        size(input.consensus.candidate_group_velocities, 2) > 0, inputs)
    md"""
    | Prepared path | Candidate-pair inputs | Pairs with candidate rows | Export-ready geometry pairs |
    |:---|---:|---:|---:|
    | Global average | $(length(global_selection_inputs)) | $(_candidate_pair_count(global_selection_inputs)) | $(length(global_tomography_inputs)) |
    | Best mix | $(length(best_mix_selection_inputs)) | $(_candidate_pair_count(best_mix_selection_inputs)) | $(length(best_mix_tomography_inputs)) |

    Selection and selected-candidate plots use candidate rows directly. Export-ready
    geometry pairs require finite station latitude and longitude from the saved artifact
    or station CSV fallback. Loaded station coordinates: **$(length(station_coordinate_lookup))**.
    """
end

# ╔═╡ c100000f-0000-0000-0000-000000000001
begin
    global_candidate_selection = tomo.select_outlier_candidates(global_selection_inputs;
        nsigma=Float64(ui_outlier_nsigma),
        candidate_mean_velocity_min=Float64(ui_candidate_mean_velocity_min),
        candidate_mean_velocity_max=Float64(ui_candidate_mean_velocity_max))
    best_mix_candidate_selection = tomo.select_outlier_candidates(best_mix_selection_inputs;
        nsigma=Float64(ui_outlier_nsigma),
        candidate_mean_velocity_min=Float64(ui_candidate_mean_velocity_min),
        candidate_mean_velocity_max=Float64(ui_candidate_mean_velocity_max))
end

# ╔═╡ c100001d-0000-0000-0000-000000000001
plot_selected_candidates(global_candidate_selection;
    title="Selected global-average candidates and final outlier band")

# ╔═╡ c100001e-0000-0000-0000-000000000001
plot_selected_candidates(best_mix_candidate_selection;
    title="Selected best-mix candidates and final outlier band")

# ╔═╡ c100001a-0000-0000-0000-000000000001
selection_table(global_candidate_selection; title="Global-average selected candidates")

# ╔═╡ c100001b-0000-0000-0000-000000000001
selection_table(best_mix_candidate_selection; title="Best-mix selected candidates")

# ╔═╡ c1000020-0000-0000-0000-000000000001
begin
    global_dsurftomo_rows = tomo.selected_dsurftomo_rows(global_candidate_selection,
        global_tomography_inputs)
    best_mix_dsurftomo_rows = tomo.selected_dsurftomo_rows(best_mix_candidate_selection,
        best_mix_tomography_inputs)
    (; global_rows=length(global_dsurftomo_rows), best_mix_rows=length(best_mix_dsurftomo_rows))
end

# ╔═╡ c1000022-0000-0000-0000-000000000001
let
    if write_global_dsurftomo > 0
        isempty(global_dsurftomo_rows) && return md"Cannot write global-average DSurfTomo file: no export rows have valid station geometry."
        tomo.write_dsurftomo_rows(global_dsurftomo_path, global_dsurftomo_rows;
            include_count_header=Bool(dsurftomo_count_header))
        md"Wrote $(length(global_dsurftomo_rows)) global-average tomography rows to `$(global_dsurftomo_path)`."
    else
        md"Global-average DSurfTomo rows ready: **$(length(global_dsurftomo_rows))**."
    end
end

# ╔═╡ c1000023-0000-0000-0000-000000000001
let
    if write_best_mix_dsurftomo > 0
        isempty(best_mix_dsurftomo_rows) && return md"Cannot write best-mix DSurfTomo file: no export rows have valid station geometry."
        tomo.write_dsurftomo_rows(best_mix_dsurftomo_path, best_mix_dsurftomo_rows;
            include_count_header=Bool(dsurftomo_count_header))
        md"Wrote $(length(best_mix_dsurftomo_rows)) best-mix tomography rows to `$(best_mix_dsurftomo_path)`."
    else
        md"Best-mix DSurfTomo rows ready: **$(length(best_mix_dsurftomo_rows))**."
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
JLD2 = "~0.6.4"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "a0bbd2233af671e233ddeec7d3c7e027c67c006a"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

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

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

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

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

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

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

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

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

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

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

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

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─c1000003-0000-0000-0000-000000000001
# ╟─c1000004-0000-0000-0000-000000000001
# ╟─c1000009-0000-0000-0000-000000000001
# ╟─c1000011-0000-0000-0000-000000000001
# ╟─c1000013-0000-0000-0000-000000000001
# ╟─c1000010-0000-0000-0000-000000000001
# ╠═c1000014-0000-0000-0000-000000000001
# ╟─c100001f-0000-0000-0000-000000000001
# ╟─c1000021-0000-0000-0000-000000000001
# ╟─c100001d-0000-0000-0000-000000000001
# ╟─c100001e-0000-0000-0000-000000000001
# ╠═c1000001-0000-0000-0000-000000000001
# ╠═c1000002-0000-0000-0000-000000000001
# ╠═0e9a7ae1-09a6-4fa9-a5e7-a2c869c0c592
# ╠═c1000005-0000-0000-0000-000000000001
# ╠═c1000006-0000-0000-0000-000000000001
# ╠═c1000007-0000-0000-0000-000000000001
# ╠═c1000008-0000-0000-0000-000000000001
# ╠═7839327c-bdb3-4ee0-a34f-c3530d4e9cf2
# ╠═38996efe-71bc-4ea3-9119-ca6bd61bc722
# ╠═b3c368be-ac3d-4327-a68d-f11a3abb3aa2
# ╠═53921daa-6935-45a3-8e4a-eaa9ccb9c7f1
# ╠═c100000b-0000-0000-0000-000000000001
# ╠═c100000c-0000-0000-0000-000000000001
# ╠═2c1b166c-0a81-457d-a56a-2825292a3c2e
# ╠═8eae1de6-81a8-4123-a538-b7e1dc681b8d
# ╠═c100000e-0000-0000-0000-000000000001
# ╠═c100000f-0000-0000-0000-000000000001
# ╠═c1000012-0000-0000-0000-000000000001
# ╠═c1000015-0000-0000-0000-000000000001
# ╠═c1000016-0000-0000-0000-000000000001
# ╠═c100001a-0000-0000-0000-000000000001
# ╠═c100001b-0000-0000-0000-000000000001
# ╠═6bb11d90-1745-4d9f-8662-d2c54480ec0b
# ╠═c1000020-0000-0000-0000-000000000001
# ╠═a401e664-077c-43b6-a864-3a3d08903b22
# ╠═c1000022-0000-0000-0000-000000000001
# ╠═c1000023-0000-0000-0000-000000000001
# ╠═c1000024-0000-0000-0000-000000000001
# ╠═c1000025-0000-0000-0000-000000000001
# ╠═e90a4b9e-b76f-4df1-b9c3-65a7ccd14a83
# ╠═4f1fd31f-3856-4b90-a615-f49d5a3d968e
# ╠═fcb27df8-a296-42bf-9915-24de8b845e23
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
