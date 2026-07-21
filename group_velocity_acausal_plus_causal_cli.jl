#!/usr/bin/env julia

include(joinpath(@__DIR__, "triplet_analysis_cli.jl"))

using CSV
using DataFrames
using Printf
using Statistics

const GV_DEFAULT_CANDIDATE_TRIPLET = "SN43-SN53-SN61"
const GV_APC_FAMILY = "selected_state_acausal_plus_causal"
const GV_DISABLED_SPLIT_OPTIONS = (;
    enabled=false,
    split_period=NaN,
    split_overlap=1.0,
    split_order=2,
    causal_period_cutoff=NaN,
    acausal_period_cutoff=NaN)

function gv_usage()
    println("""
    Usage:
      julia --startup-file=no group_velocity_acausal_plus_causal_cli.jl \\
          --config triplet_mft_defaults.toml \\
          --saved-root DIR \\
          --raw-data-dir DIR \\
          --triplets-csv station_triplet_csvs/all_station_triplets_unfiltered.csv \\
          --candidate-triplet SN43-SN53-SN61 \\
          --output-dir DIR [options]

    Required:
      --saved-root DIR              Root containing trained/transferred source-state artifacts
      --raw-data-dir DIR            Raw pair directory used for discovery/reporting
      --output-dir DIR              Destination for CSVs and plots

    Core options:
      --config FILE                 TOML file with MFT/triplet-analysis defaults
      --triplets-csv FILE           Unfiltered triplet CSV. Default: $(DEFAULT_TRIPLETS_CSV)
      --candidate-triplet A-B-C     Reference triplet used to choose candidates. Default: $(GV_DEFAULT_CANDIDATE_TRIPLET)
      --top-ranks N                 Candidate ranks per selected family. Default: 5
      --max-candidates N            Candidate states per pair before triplet ranking. Default: 25
      --apply-acausal-plus-causal-split-filter true|false
                                    Apply split filter only to selected_state_acausal_plus_causal. Default: false
      --split-period X              Split period in seconds. Required when split filtering is true
      --split-overlap X             Overlap in seconds around split period. Default: 1.0, minimum: 1.0
      --split-filter-order N        Zero-phase Butterworth order. Default: 2

    Geometry filters:
      --max-delta-az X              Maximum azimuth spread in degrees. Default: 0.7
      --max-delta-d X               Maximum abs distance-closure error in km. Default: 0.1
      --min-segment-distance X      Minimum Dab and Dbc segment distance in km. Default: 45

    MFT / filtering options:
      --bandpass none|TMIN:TMAX     Zero-phase period bandpass before MFT. Default: 10:30
      --dt X                        Waveform sample interval in seconds. Default: 1
      --period-min X                Minimum MFT period in seconds. Default: 10
      --period-max X                Maximum MFT period in seconds. Default: 30
      --nperiods N                  Number of log-spaced MFT periods. Default: 100
      --velocity-min X              Minimum MFT group/filter velocity. Default: 1
      --velocity-max X              Maximum MFT group/filter velocity. Default: 8
      --phase-velocity-min X        Minimum phase velocity search value. Default: 1
      --phase-velocity-max X        Maximum phase velocity search value. Default: 6
      --wavelength-ref-velocity X   Reference velocity for wavelength period filtering. Default: 2
      --wavelength-fraction X       Distance fraction for wavelength period filtering. Default: 0.33
      --bandwidth-factor X          MFT bandwidth factor. Default: 0.15
      --zero-pad-factor N           MFT zero-pad factor. Default: 2
      --upsample-factor X           MFT upsample factor. Default: 2
      --precision Float32|Float64   MFT numeric precision. Default: Float32
      --use-phtovel true|false      Use phase-to-velocity branch. Default: true

    Flags:
      --overwrite                   Replace existing output files
      --dry-run                     Print discovered work without running MFT or writing outputs
      --help                        Show this message
    """)
end

function gv_extract_apc_split_options(argv)
    cleaned = String[]
    enabled = false
    split_period = NaN
    split_overlap = 1.0
    split_order = 2
    i = 1
    while i <= length(argv)
        arg = String(argv[i])
        if arg == "--apply-acausal-plus-causal-split-filter"
            i == length(argv) && error("Missing value for --apply-acausal-plus-causal-split-filter")
            enabled = parse_bool_option(argv[i + 1], arg)
            i += 2
        elseif arg == "--split-period"
            i == length(argv) && error("Missing value for --split-period")
            parsed = tryparse(Float64, String(argv[i + 1]))
            isnothing(parsed) && error("--split-period must be numeric; got $(argv[i + 1])")
            split_period = parsed
            i += 2
        elseif arg == "--split-overlap"
            i == length(argv) && error("Missing value for --split-overlap")
            parsed = tryparse(Float64, String(argv[i + 1]))
            isnothing(parsed) && error("--split-overlap must be numeric; got $(argv[i + 1])")
            split_overlap = parsed
            i += 2
        elseif arg == "--split-filter-order"
            i == length(argv) && error("Missing value for --split-filter-order")
            parsed = tryparse(Int, String(argv[i + 1]))
            isnothing(parsed) && error("--split-filter-order must be an integer; got $(argv[i + 1])")
            split_order = parsed
            i += 2
        else
            push!(cleaned, arg)
            i += 1
        end
    end
    split_overlap >= 1.0 || error("--split-overlap must be >= 1.0")
    split_order >= 1 || error("--split-filter-order must be >= 1")
    if enabled
        isfinite(split_period) || error("--split-period is required when --apply-acausal-plus-causal-split-filter true")
        split_period > split_overlap || error("--split-period must be greater than --split-overlap")
    end
    causal_cutoff = enabled ? split_period + split_overlap : NaN
    acausal_cutoff = enabled ? split_period - split_overlap : NaN
    (; argv=cleaned,
        enabled,
        split_period=Float64(split_period),
        split_overlap=Float64(split_overlap),
        split_order=Int(split_order),
        causal_period_cutoff=Float64(causal_cutoff),
        acausal_period_cutoff=Float64(acausal_cutoff))
end

function gv_apc_split_metadata(split_opts; applied::Union{Nothing,Bool}=nothing)
    applied_flag = isnothing(applied) ? Bool(split_opts.enabled) : Bool(applied)
    (; split_filter_applied=applied_flag,
        split_period_s=Float64(split_opts.split_period),
        split_overlap_s=Float64(split_opts.split_overlap),
        causal_period_cutoff_s=Float64(split_opts.causal_period_cutoff),
        acausal_period_cutoff_s=Float64(split_opts.acausal_period_cutoff),
        split_filter_order=Int(split_opts.split_order))
end

function gv_parse_triplet_label(label::AbstractString)
    parts = split(strip(String(label)), "-")
    length(parts) == 3 || error("--candidate-triplet must look like SN43-SN53-SN61; got $(label)")
    Tuple(String.(strip.(parts)))
end

function gv_triplet_row(df::DataFrame, label::AbstractString)
    idx = gv_triplet_row_index(df, label)
    df[idx, :]
end

function gv_triplet_row_index(df::DataFrame, label::AbstractString)
    a, b, c = gv_parse_triplet_label(label)
    for (i, row) in enumerate(eachrow(df))
        if String(row.station_a) == a && String(row.station_b) == b && String(row.station_c) == c
            return i
        end
        if String(row.triplet) == String(label)
            return i
        end
    end
    error("Candidate triplet $(label) was not found among triplets passing the geometry criteria")
end

function gv_output_guard(path::AbstractString, overwrite::Bool)
    isfile(path) && !overwrite && error("Refusing to overwrite $(path); pass --overwrite")
    mkpath(dirname(path))
    path
end

function gv_base_config(opts)
    _mft_config(; dt=Float64(opts["dt"]),
        period_min=Float64(opts["period-min"]),
        period_max=Float64(opts["period-max"]),
        nperiods=Int(opts["nperiods"]),
        wavelength_ref_velocity=Float64(opts["wavelength-ref-velocity"]),
        wavelength_fraction=Float64(opts["wavelength-fraction"]),
        velocity_range=(Float64(opts["velocity-min"]), Float64(opts["velocity-max"])),
        bandwidth_factor=Float64(opts["bandwidth-factor"]),
        zero_pad_factor=Int(opts["zero-pad-factor"]),
        upsample_factor=Float64(opts["upsample-factor"]),
        precision=String(opts["precision"]),
        max_modes=Int(opts["max-modes"]),
        phase_velocity_range=(Float64(opts["phase-velocity-min"]), Float64(opts["phase-velocity-max"])),
        use_phtovel=Bool(opts["use-phtovel"]),
        phvel_correction=0.0)
end

function maybe_bandpass_vector(x::AbstractVector, bp, dt::Real)
    y = Float32.(vec(x))
    isnothing(bp) && return y
    fs = 1.0 / Float64(dt)
    f1 = 1.0 / Float64(bp.period_max)
    f2 = 1.0 / Float64(bp.period_min)
    nyq = fs / 2
    f2 < nyq || error("Bandpass high frequency $(f2) Hz exceeds Nyquist $(nyq) Hz for dt=$(dt)")
    response = Base.invokelatest(DSP.Bandpass, f1, f2; fs=fs)
    design = Base.invokelatest(DSP.Butterworth, 4)
    filt = Base.invokelatest(DSP.digitalfilter, response, design)
    Float32.(Base.invokelatest(DSP.filtfilt, filt, Float64.(y)))
end

function gv_period_lowpass_vector(x::AbstractVector, cutoff_period::Real, dt::Real, order::Integer)
    y = Float32.(vec(x))
    fs = 1.0 / Float64(dt)
    cutoff_hz = 1.0 / Float64(cutoff_period)
    nyq = fs / 2
    0 < cutoff_hz < nyq || error("Period-lowpass cutoff frequency $(cutoff_hz) Hz must be below Nyquist $(nyq) Hz")
    response = Base.invokelatest(DSP.Highpass, cutoff_hz; fs=fs)
    design = Base.invokelatest(DSP.Butterworth, Int(order))
    filt = Base.invokelatest(DSP.digitalfilter, response, design)
    Float32.(Base.invokelatest(DSP.filtfilt, filt, Float64.(y)))
end

function gv_period_highpass_vector(x::AbstractVector, cutoff_period::Real, dt::Real, order::Integer)
    y = Float32.(vec(x))
    fs = 1.0 / Float64(dt)
    cutoff_hz = 1.0 / Float64(cutoff_period)
    nyq = fs / 2
    0 < cutoff_hz < nyq || error("Period-highpass cutoff frequency $(cutoff_hz) Hz must be below Nyquist $(nyq) Hz")
    response = Base.invokelatest(DSP.Lowpass, cutoff_hz; fs=fs)
    design = Base.invokelatest(DSP.Butterworth, Int(order))
    filt = Base.invokelatest(DSP.digitalfilter, response, design)
    Float32.(Base.invokelatest(DSP.filtfilt, filt, Float64.(y)))
end

function gv_period_lowpass_matrix(X::AbstractMatrix, cutoff_period::Real, dt::Real, order::Integer)
    Y = Float32.(X)
    out = similar(Y)
    for j in axes(Y, 2)
        out[:, j] .= gv_period_lowpass_vector(@view(Y[:, j]), cutoff_period, dt, order)
    end
    out
end

function gv_period_highpass_matrix(X::AbstractMatrix, cutoff_period::Real, dt::Real, order::Integer)
    Y = Float32.(X)
    out = similar(Y)
    for j in axes(Y, 2)
        out[:, j] .= gv_period_highpass_vector(@view(Y[:, j]), cutoff_period, dt, order)
    end
    out
end

function gv_mean_cols_with_optional_split(A::AbstractArray, B::AbstractArray, split_opts, dt::Real)
    n = min(size(A, 1), size(B, 1))
    m = min(size(A, 2), size(B, 2))
    n == 0 || m == 0 ? Float32[;;] : begin
        Ac = Float32.(A[1:n, 1:m])
        Bac = Float32.(B[1:n, 1:m])
        if split_opts.enabled
            Ac = gv_period_lowpass_matrix(Ac, split_opts.causal_period_cutoff, dt, split_opts.split_order)
            Bac = gv_period_highpass_matrix(Bac, split_opts.acausal_period_cutoff, dt, split_opts.split_order)
        end
        Float32.(0.5 .* (Ac .+ Bac))
    end
end

function gv_mean_vec_with_optional_split(a::AbstractVector, b::AbstractVector, split_opts, dt::Real)
    n = min(length(a), length(b))
    n == 0 ? Float32[] : begin
        ca = Float32.(a[1:n])
        ac = Float32.(b[1:n])
        if split_opts.enabled
            ca = gv_period_lowpass_vector(ca, split_opts.causal_period_cutoff, dt, split_opts.split_order)
            ac = gv_period_highpass_vector(ac, split_opts.acausal_period_cutoff, dt, split_opts.split_order)
        end
        Float32.(0.5 .* (ca .+ ac))
    end
end

function gv_acausal_plus_causal_item(item, split_opts, dt::Real)
    c = gv_mean_cols_with_optional_split(item.causal, item.acausal, split_opts, dt)
    s1 = gv_mean_cols_with_optional_split(item.marginal_stage1_c, item.marginal_stage1_ac, split_opts, dt)
    s2 = gv_mean_cols_with_optional_split(item.marginal_stage2_c, item.marginal_stage2_ac, split_opts, dt)
    g = gv_mean_vec_with_optional_split(item.global_avg_c, item.global_avg_ac, split_opts, dt)
    merge(item, (;
        causal=c, acausal=c,
        marginal_stage1_c=s1, marginal_stage1_ac=s1,
        marginal_stage2_c=s2, marginal_stage2_ac=s2,
        global_avg_c=g, global_avg_ac=g))
end

function gv_selected_scores_for_family(items, cfg, family::AbstractString, split_opts, dt::Real;
        score_method="geomean", huber_delta=0.10)
    if family == GV_APC_FAMILY
        return score_state_items([gv_acausal_plus_causal_item(item, split_opts, dt) for item in items],
            cfg; branch_filter="causal", branch_label="acausal_plus_causal", score_method, huber_delta)
    end
    selected_scores_for_family(items, cfg, family; score_method, huber_delta)
end

function gv_needed_pairs(triplets::DataFrame)
    sort(unique(vcat(String.(triplets.pair_ab), String.(triplets.pair_bc), String.(triplets.pair_ac))))
end

function gv_load_items_for_pairs(runs, pairs::AbstractVector{<:AbstractString}, bp, dt::Real)
    wanted = Set(String.(pairs))
    selected_runs = [run for run in runs if String(run.pair_label) in wanted]
    [bandpass_item(load_source_state_artifact(run), bp, dt) for run in selected_runs]
end

function gv_candidate_row(; candidate_id, family, rank, combo, triplet_label,
        pair_ab="", pair_bc="", pair_ac="", split_meta=NamedTuple())
    state_label = _logical_state_display(String(combo.label_ab))
    merge((; candidate_id, candidate_family=family, rank,
        selected_state_label=state_label,
        reference_triplet=triplet_label,
        reference_group_rms_abs_vabc_minus_vac=Float64(combo.rms_vdiff),
        reference_n_periods=Int(combo.n_periods),
        pair_ab=String(pair_ab),
        pair_bc=String(pair_bc),
        pair_ac=String(pair_ac),
        ab_label=String(combo.label_ab),
        bc_label=String(combo.label_bc),
        ac_label=String(combo.label_ac),
        ab_uc_score=Float64(combo.spec_ab.uc_score),
        bc_uc_score=Float64(combo.spec_bc.uc_score),
        ac_uc_score=Float64(combo.spec_ac.uc_score)), split_meta)
end

function gv_choose_reference_candidates(candidate_row, items, cfg_selected, cfg_selected_mean;
        top_ranks::Int, max_candidates::Int, split_opts=GV_DISABLED_SPLIT_OPTIONS, dt::Real=1.0)
    triplet_label = String(candidate_row.triplet)
    pairs = (String(candidate_row.pair_ab), String(candidate_row.pair_bc), String(candidate_row.pair_ac))
    geom = geom_from_triplet_row(candidate_row, pairs)
    trip_items = items_for_pairs(items, pairs)
    families = (
        ("selected_state_causal", cfg_selected, "causal"),
        ("selected_state_acausal", cfg_selected, "acausal"),
        (GV_APC_FAMILY, cfg_selected_mean, "acausal_plus_causal"),
    )
    selected = NamedTuple[]
    candidate_rows = NamedTuple[]
    for (family, cfg, short) in families
        split_meta = gv_apc_split_metadata(split_opts; applied=(family == GV_APC_FAMILY && split_opts.enabled))
        println("  ranking $(family) on $(triplet_label)")
        scored = gv_selected_scores_for_family(trip_items, cfg, family, split_opts, dt)
        ranked = _rank_station_triple_state_combinations(geom, scored;
            max_candidates, velocity_field=:group_velocity)
        for (irank, combo) in enumerate(ranked[1:min(top_ranks, length(ranked))])
            candidate_id = "$(short)_rank_$(irank)"
            push!(selected, (; candidate_id, family, rank=irank,
                template=combo.spec_ab, reference_combo=combo))
            push!(candidate_rows, gv_candidate_row(; candidate_id, family, rank=irank,
                combo, triplet_label,
                pair_ab=pairs[1], pair_bc=pairs[2], pair_ac=pairs[3], split_meta))
        end
    end
    selected, DataFrame(candidate_rows)
end

function gv_candidate_time_window_counts(items, cand)
    target = _logical_state_display(String(cand.template.display))
    causal_count = 0
    acausal_count = 0
    causal_windows_total = 0
    acausal_windows_total = 0
    matched_pairs = Set{String}()
    for item in items
        item_causal_windows = sum(Int.(item.counts_c))
        item_acausal_windows = sum(Int.(item.counts_ac))
        prefix = String(item.artifact_kind) == "selected_state_transfer" ?
            "$(item.pair_label) selected transfer from $(item.reference_pair_label) seed $(item.seed)" :
            "$(item.pair_label) seed $(item.seed)"
        nstates = min(length(item.combo_labels), length(item.counts_c), length(item.counts_ac))
        for i in 1:nstates
            base = "$(prefix) | $(item.combo_labels[i])"
            display = _logical_state_display(String(_score_display_from_state_label(base, "state").display))
            display == target || continue
            causal_count += Int(item.counts_c[i])
            acausal_count += Int(item.counts_ac[i])
            causal_windows_total += item_causal_windows
            acausal_windows_total += item_acausal_windows
            push!(matched_pairs, String(item.pair_label))
        end
        for (counts_c, counts_ac, labels, stage) in (
                (item.marginal_stage1_counts_c, item.marginal_stage1_counts_ac, item.marginal_stage1_labels, "S1"),
                (item.marginal_stage2_counts_c, item.marginal_stage2_counts_ac, item.marginal_stage2_labels, "S2"))
            kmax = min(length(counts_c), length(counts_ac), length(labels))
            for k in 1:kmax
                base = "$(prefix) | $(stage) $(labels[k])"
                display = _logical_state_display(String(_score_display_from_state_label(base, "state").display))
                display == target || continue
                causal_count += Int(counts_c[k])
                acausal_count += Int(counts_ac[k])
                causal_windows_total += item_causal_windows
                acausal_windows_total += item_acausal_windows
                push!(matched_pairs, String(item.pair_label))
            end
        end
    end
    (; causal_count, acausal_count, causal_windows_total, acausal_windows_total,
        n_pairs=length(matched_pairs))
end

function gv_print_candidate_time_window_allocations(selected_candidates, items)
    isempty(selected_candidates) && return
    println()
    println("Selected candidate time-window allocations:")
    for cand in selected_candidates
        counts = gv_candidate_time_window_counts(items, cand)
        label = String(cand.template.display)
        if counts.causal_windows_total == 0 && counts.acausal_windows_total == 0
            println("  $(cand.candidate_id) | $(cand.family) rank $(cand.rank) | $(label): no matching time-window counts found")
            continue
        end
        causal_pct = counts.causal_windows_total == 0 ? 0.0 :
            100.0 * counts.causal_count / counts.causal_windows_total
        acausal_pct = counts.acausal_windows_total == 0 ? 0.0 :
            100.0 * counts.acausal_count / counts.acausal_windows_total
        println(@sprintf("  %s | %s rank %d | %s: causal %.1f%% (%d/%d), acausal %.1f%% (%d/%d), matched pairs=%d",
            String(cand.candidate_id), String(cand.family), Int(cand.rank), label,
            causal_pct, counts.causal_count, counts.causal_windows_total,
            acausal_pct, counts.acausal_count, counts.acausal_windows_total,
            counts.n_pairs))
    end
end

function gv_phase_candidate_branch(family::AbstractString)
    fam = String(family)
    fam == "selected_state_causal" && return "causal"
    fam == "selected_state_acausal" && return "acausal"
    fam == GV_APC_FAMILY && return "acausal_plus_causal"
    error("Cannot convert candidate family $(fam) to phase-sweep branch")
end

function gv_strip_candidate_branch(label::AbstractString)
    parts = strip.(split(String(label), "|"))
    isempty(parts) && return ""
    if lowercase(parts[end]) in ("causal", "acausal", "mean", "causal+acausal",
            "causal + acausal", "average", "acausal_plus_causal",
            "selected_state_acausal_plus_causal")
        return join(parts[1:end-1], " | ")
    end
    String(label)
end

function gv_phase_candidate_sweep_candidates(candidate_df::DataFrame)
    isempty(candidate_df) && return ""
    rows = :reference_group_rms_abs_vabc_minus_vac in Symbol.(names(candidate_df)) ?
        sort(candidate_df, [:reference_group_rms_abs_vabc_minus_vac, :candidate_family, :rank, :candidate_id]) :
        sort(candidate_df, [:candidate_family, :rank, :candidate_id])
    labels = String[]
    seen = Set{String}()
    for row in eachrow(rows)
        base_label = gv_strip_candidate_branch(String(row.selected_state_label))
        label = "$(base_label) | $(gv_phase_candidate_branch(String(row.candidate_family)))"
        label in seen && continue
        push!(labels, label)
        push!(seen, label)
    end
    join(labels, "; ")
end

function gv_shell_quote(text::AbstractString)
    s = String(text)
    "'" * replace(s, "'" => "'\"'\"'") * "'"
end

function gv_followup_phase_output_dir(output_dir::AbstractString)
    out = String(output_dir)
    replaced = replace(out, "group_velocity_acausal_plus_causal" => "phase_candidate_sweep_acausal_plus_causal")
    replaced == out ? out * "_phase_candidate_sweep_acausal_plus_causal" : replaced
end

function gv_print_phase_candidate_sweep_followup(candidate_df::DataFrame; config_path::AbstractString,
        saved_root::AbstractString, raw_data_dir::AbstractString, triplets_csv::AbstractString,
        candidate_triplet::AbstractString, output_dir::AbstractString, split_opts)
    candidates = gv_phase_candidate_sweep_candidates(candidate_df)
    isempty(candidates) && return
    phase_output_dir = gv_followup_phase_output_dir(output_dir)
    println()
    println("Follow-up phase candidate sweep command:")
    lines = [
        "julia --startup-file=no phase_candidate_sweep_acausal_plus_causal_cli.jl",
        "  --config $(gv_shell_quote(config_path))",
        "  --saved-root $(gv_shell_quote(saved_root))",
        "  --raw-data-dir $(gv_shell_quote(raw_data_dir))",
        "  --triplets-csv $(gv_shell_quote(triplets_csv))",
        "  --candidates $(gv_shell_quote(candidates))",
        "  --d-values \"2:0.2:12\"",
        "  --output-dir $(gv_shell_quote(phase_output_dir))",
        "  --apply-acausal-plus-causal-split-filter $(split_opts.enabled)",
        "  --split-period $(split_opts.split_period)",
        "  --split-overlap $(split_opts.split_overlap)",
        "  --split-filter-order $(split_opts.split_order)",
        "  --overwrite",
        "  --reference-triplet $(gv_shell_quote(candidate_triplet))",
    ]
    println(join(lines, " \\\n"))
end

function gv_combo_from_template(scored_by_pair, geom, template)
    spec_ab = matching_spec(scored_by_pair, geom.pair_ab, template)
    spec_bc = matching_spec(scored_by_pair, geom.pair_bc, template)
    spec_ac = matching_spec(scored_by_pair, geom.pair_ac, template)
    any(isnothing, (spec_ab, spec_bc, spec_ac)) && return nothing
    (; spec_ab, spec_bc, spec_ac,
        label_ab="$(spec_ab.display) | $(spec_ab.branch)",
        label_bc="$(spec_bc.display) | $(spec_bc.branch)",
        label_ac="$(spec_ac.display) | $(spec_ac.branch)")
end

function gv_rows_dataframe(rows)
    isempty(rows) && return DataFrame()
    sorted_rows = sort(collect(rows), by=r -> (String(r.triplet), String(r.candidate_family),
        Int(r.rank), Float64(r.period), String(r.source)))
    DataFrame(
        triplet=String[String(r.triplet) for r in sorted_rows],
        candidate_id=String[String(r.candidate_id) for r in sorted_rows],
        candidate_family=String[String(r.candidate_family) for r in sorted_rows],
        rank=Int[Int(r.rank) for r in sorted_rows],
        selected_state_label=String[String(r.selected_state_label) for r in sorted_rows],
        source=String[String(r.source) for r in sorted_rows],
        period_s=Float64[Float64(r.period) for r in sorted_rows],
        pair_ab=String[String(r.pair_ab) for r in sorted_rows],
        pair_bc=String[String(r.pair_bc) for r in sorted_rows],
        pair_ac=String[String(r.pair_ac) for r in sorted_rows],
        Vab_km_s=Float64[Float64(r.vab) for r in sorted_rows],
        Vbc_km_s=Float64[Float64(r.vbc) for r in sorted_rows],
        Vac_km_s=Float64[Float64(r.vac) for r in sorted_rows],
        Vabc_km_s=Float64[Float64(r.vabc) for r in sorted_rows],
        Vabc_minus_Vac_km_s=Float64[Float64(r.vdif) for r in sorted_rows],
        abs_Vabc_minus_Vac_km_s=Float64[abs(Float64(r.vdif)) for r in sorted_rows],
        Tdiff_s=Float64[Float64(r.tdif) for r in sorted_rows],
        Tdiff_percent=Float64[Float64(r.tpdif) for r in sorted_rows],
        Delta_az_deg=Float64[Float64(r.Delta_az_deg) for r in sorted_rows],
        Delta_D_km=Float64[Float64(r.Delta_D_km) for r in sorted_rows],
        split_filter_applied=Bool[hasproperty(r, :split_filter_applied) ? Bool(r.split_filter_applied) : false for r in sorted_rows],
        split_period_s=Float64[hasproperty(r, :split_period_s) ? Float64(r.split_period_s) : NaN for r in sorted_rows],
        split_overlap_s=Float64[hasproperty(r, :split_overlap_s) ? Float64(r.split_overlap_s) : NaN for r in sorted_rows],
        causal_period_cutoff_s=Float64[hasproperty(r, :causal_period_cutoff_s) ? Float64(r.causal_period_cutoff_s) : NaN for r in sorted_rows],
        acausal_period_cutoff_s=Float64[hasproperty(r, :acausal_period_cutoff_s) ? Float64(r.acausal_period_cutoff_s) : NaN for r in sorted_rows],
        split_filter_order=Int[hasproperty(r, :split_filter_order) ? Int(r.split_filter_order) : 0 for r in sorted_rows])
end

function gv_rmse_dataframe(rows_df::DataFrame)
    isempty(rows_df) && return DataFrame()
    out = NamedTuple[]
    keys = [:candidate_id, :candidate_family, :rank, :selected_state_label, :triplet]
    for sub in groupby(rows_df, keys)
        vals = Float64.(sub.abs_Vabc_minus_Vac_km_s)
        push!(out, (; scope="triplet",
            candidate_id=String(first(sub.candidate_id)),
            candidate_family=String(first(sub.candidate_family)),
            rank=Int(first(sub.rank)),
            selected_state_label=String(first(sub.selected_state_label)),
            triplet=String(first(sub.triplet)),
            n_periods=length(vals),
            rmse_abs_vabc_minus_vac=sqrt(mean(vals .^ 2)),
            median_abs_vabc_minus_vac=median(vals),
            max_abs_vabc_minus_vac=maximum(vals),
            split_filter_applied=Bool(first(sub.split_filter_applied)),
            split_period_s=Float64(first(sub.split_period_s)),
            split_overlap_s=Float64(first(sub.split_overlap_s)),
            causal_period_cutoff_s=Float64(first(sub.causal_period_cutoff_s)),
            acausal_period_cutoff_s=Float64(first(sub.acausal_period_cutoff_s)),
            split_filter_order=Int(first(sub.split_filter_order))))
    end
    keys2 = [:candidate_id, :candidate_family, :rank, :selected_state_label]
    for sub in groupby(rows_df, keys2)
        vals = Float64.(sub.abs_Vabc_minus_Vac_km_s)
        push!(out, (; scope="overall",
            candidate_id=String(first(sub.candidate_id)),
            candidate_family=String(first(sub.candidate_family)),
            rank=Int(first(sub.rank)),
            selected_state_label=String(first(sub.selected_state_label)),
            triplet="ALL",
            n_periods=length(vals),
            rmse_abs_vabc_minus_vac=sqrt(mean(vals .^ 2)),
            median_abs_vabc_minus_vac=median(vals),
            max_abs_vabc_minus_vac=maximum(vals),
            split_filter_applied=Bool(first(sub.split_filter_applied)),
            split_period_s=Float64(first(sub.split_period_s)),
            split_overlap_s=Float64(first(sub.split_overlap_s)),
            causal_period_cutoff_s=Float64(first(sub.causal_period_cutoff_s)),
            acausal_period_cutoff_s=Float64(first(sub.acausal_period_cutoff_s)),
            split_filter_order=Int(first(sub.split_filter_order))))
    end
    sort!(DataFrame(out), [:scope, :candidate_family, :rank, :candidate_id, :triplet])
end

function gv_color(index::Int)
    palette = ["#2563eb", "#dc2626", "#4b5563", "#16a34a", "#f97316", "#7c3aed",
        "#0891b2", "#be123c", "#65a30d", "#a16207", "#0f766e", "#9333ea",
        "#ea580c", "#475569", "#15803d", "#111827", "#60a5fa", "#f87171"]
    palette[mod1(index, length(palette))]
end

function gv_pretty_candidate_id(id::AbstractString)
    s = String(id)
    s == "global_average_mean" && return "global mean (causal+acausal)"
    s == "global_average_causal" && return "global causal"
    s == "global_average_acausal" && return "global acausal"
    replace(s, "_" => " ")
end

function gv_plot_name(row)
    id = String(row.candidate_id)
    fam = String(row.candidate_family)
    rank = Int(row.rank)
    label = String(row.selected_state_label)
    startswith(id, "global") ? gv_pretty_candidate_id(id) : "$(fam) rank $(rank): $(label)"
end

function gv_write_reference_html(path::AbstractString, rows_df::DataFrame, candidate_triplet::AbstractString; overwrite::Bool)
    gv_output_guard(path, overwrite)
    sub = rows_df[String.(rows_df.triplet) .== String(candidate_triplet), :]
    traces = String[]
    ids = sort(unique(String.(sub.candidate_id)))
    for (i, id) in enumerate(ids)
        ss = sub[String.(sub.candidate_id) .== id, :]
        isempty(ss) && continue
        firstrow = first(eachrow(ss))
        name = html_escape(gv_plot_name(firstrow))
        color = gv_color(i)
        for (col, suffix, dash) in ((:Vab_km_s, "ab", "solid"), (:Vbc_km_s, "bc", "dash"), (:Vac_km_s, "ac", "dot"))
            xs = join(string.(Float64.(ss.period_s)), ",")
            ys = join(string.(Float64.(ss[!, col])), ",")
            push!(traces, "{x:[$xs], y:[$ys], mode:'lines+markers', type:'scatter', xaxis:'x', yaxis:'y', name:'$(name) $(suffix)', legendgroup:'$(html_escape(id))', line:{color:'$(color)', dash:'$(dash)'}, marker:{size:5}}")
        end
        xs = join(string.(Float64.(ss.period_s)), ",")
        ys = join(string.(Float64.(ss.abs_Vabc_minus_Vac_km_s)), ",")
        push!(traces, "{x:[$xs], y:[$ys], mode:'lines+markers', type:'scatter', xaxis:'x2', yaxis:'y2', name:'$(name) abs diff', legendgroup:'$(html_escape(id))', line:{color:'$(color)', width:2}, marker:{symbol:'star', size:7}}")
    end
    body = """
    <!doctype html>
    <html><head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
    <body><div id="plot" style="width:1300px;height:980px;"></div>
    <script>
    const data = [$(join(traces, ","))];
    const layout = {
      title: 'Group velocity triplet: $(html_escape(String(candidate_triplet)))',
      grid: {rows: 2, columns: 1, pattern: 'independent'},
      xaxis: {title: 'Period (s)'},
      yaxis: {title: 'Group velocity (km/s)'},
      xaxis2: {title: 'Period (s)'},
      yaxis2: {title: 'abs(Vabc - Vac) (km/s)'},
      hovermode: 'closest',
      legend: {orientation: 'h'}
    };
    Plotly.newPlot('plot', data, layout, {responsive: true});
    </script></body></html>
    """
    open(path, "w") do io
        write(io, body)
    end
    path
end

function gv_write_reference_png(path::AbstractString, rows_df::DataFrame, candidate_triplet::AbstractString; overwrite::Bool)
    gv_output_guard(path, overwrite)
    tmp = tempname() * ".csv"
    CSV.write(tmp, rows_df[String.(rows_df.triplet) .== String(candidate_triplet), :])
    py = raw"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, out_path, triplet = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_csv(csv_path)
fig, axes = plt.subplots(2, 1, figsize=(13, 10), dpi=170, sharex=True)
def pretty(cid):
    cid = str(cid)
    if cid == "global_average_mean":
        return "global mean (causal+acausal)"
    if cid == "global_average_causal":
        return "global causal"
    if cid == "global_average_acausal":
        return "global acausal"
    return cid.replace("_", " ")
for idx, (cid, sub) in enumerate(df.groupby("candidate_id", sort=True)):
    label = pretty(cid)
    state_label = sub["selected_state_label"].iloc[0]
    if not str(cid).startswith("global_average_") and pd.notna(state_label) and str(state_label).strip():
        label = f'{sub["candidate_family"].iloc[0]} r{sub["rank"].iloc[0]}'
    axes[0].plot(sub["period_s"], sub["Vab_km_s"], linewidth=1.0, alpha=0.55)
    axes[0].plot(sub["period_s"], sub["Vbc_km_s"], linewidth=1.0, alpha=0.55, linestyle="--")
    axes[0].plot(sub["period_s"], sub["Vac_km_s"], linewidth=1.0, alpha=0.55, linestyle=":")
    axes[1].plot(sub["period_s"], sub["abs_Vabc_minus_Vac_km_s"], marker="o", linewidth=1.3, markersize=3, label=label)
axes[0].set_title(f"Group velocity triplet: {triplet}")
axes[0].set_ylabel("Group velocity (km/s)")
axes[1].set_ylabel("abs(Vabc - Vac) (km/s)")
axes[1].set_xlabel("Period (s)")
for ax in axes:
    ax.grid(True, alpha=0.3)
axes[1].legend(fontsize=7, ncol=3)
fig.tight_layout()
fig.savefig(out_path)
"""
    try
        run(`python3 -c $py $tmp $path $candidate_triplet`)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    path
end

function gv_write_rmse_html(path::AbstractString, rmse_df::DataFrame; overwrite::Bool)
    gv_output_guard(path, overwrite)
    overall = rmse_df[String.(rmse_df.scope) .== "overall", :]
    sort!(overall, [:rmse_abs_vabc_minus_vac])
    xs = join(["\"" * html_escape(gv_pretty_candidate_id(String(row.candidate_id))) * "\"" for row in eachrow(overall)], ",")
    ys = join(string.(Float64.(overall.rmse_abs_vabc_minus_vac)), ",")
    text = join(["\"" * html_escape(String(row.selected_state_label)) * "\"" for row in eachrow(overall)], ",")
    colors = join(["\"" * gv_color(i) * "\"" for i in 1:nrow(overall)], ",")
    body = """
    <!doctype html>
    <html><head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
    <body><div id="plot" style="width:1200px;height:760px;"></div>
    <script>
    const data = [{x:[$xs], y:[$ys], text:[$text], type:'scatter', mode:'markers',
      marker:{color:[$colors], size:13, line:{color:'#111827', width:1}}}];
    const layout = {
      title: 'Overall group-velocity closure RMSE across analyzed triplets',
      xaxis: {title: 'candidate'},
      yaxis: {title: 'RMSE abs(Vabc - Vac) (km/s)'},
      hovermode: 'closest'
    };
    Plotly.newPlot('plot', data, layout, {responsive: true});
    </script></body></html>
    """
    open(path, "w") do io
        write(io, body)
    end
    path
end

function gv_write_rmse_png(path::AbstractString, rmse_df::DataFrame; overwrite::Bool)
    gv_output_guard(path, overwrite)
    tmp = tempname() * ".csv"
    CSV.write(tmp, rmse_df)
    py = raw"""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)
overall = df[df["scope"] == "overall"].sort_values("rmse_abs_vabc_minus_vac")
fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
ax.scatter(range(len(overall)), overall["rmse_abs_vabc_minus_vac"], s=80, edgecolors="black", linewidths=0.7)
def pretty(cid):
    cid = str(cid)
    if cid == "global_average_mean":
        return "global mean\n(causal+acausal)"
    if cid == "global_average_causal":
        return "global causal"
    if cid == "global_average_acausal":
        return "global acausal"
    return cid
ax.set_xticks(range(len(overall)))
ax.set_xticklabels([pretty(x) for x in overall["candidate_id"]], rotation=45, ha="right", fontsize=8)
ax.set_title("Overall group-velocity closure RMSE across analyzed triplets")
ax.set_ylabel("RMSE abs(Vabc - Vac) (km/s)")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out_path)
"""
    try
        run(`python3 -c $py $tmp $path`)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    path
end

function gv_state_branch_from_combo_label(label::AbstractString)
    parts = strip.(split(String(label), " | "))
    isempty(parts) && return ("", "")
    branch = lowercase(String(parts[end]))
    if branch in ("causal", "acausal", "mean", "causal+acausal", "acausal_plus_causal")
        return (_logical_state_display(join(parts[1:end-1], " | ")), branch)
    end
    (_logical_state_display(String(label)), "")
end

function gv_time_axis_for_waveform(n::Integer, dt::Real)
    center = (Float64(n) + 1.0) / 2.0
    (collect(1:Int(n)) .- center) .* Float64(dt)
end

function gv_waveform_rows_for_reference_candidates(candidate_df::DataFrame, items, split_opts, dt::Real)
    isempty(candidate_df) && return DataFrame()
    lookup = state_waveform_lookup(items)
    rows = NamedTuple[]
    pair_cols = ((:ab_label, "ab"), (:bc_label, "bc"), (:ac_label, "ac"))
    for cand in eachrow(candidate_df)
        family = String(cand.candidate_family)
        for (label_col, pair_role) in pair_cols
            pair_col = Symbol("pair_$(pair_role)")
            hasproperty(cand, pair_col) || continue
            pair = String(getproperty(cand, pair_col))
            state_label, _branch = gv_state_branch_from_combo_label(String(getproperty(cand, label_col)))
            state_label = _logical_state_display(state_label)
            causal = get(lookup, (pair, state_label, "causal"), nothing)
            acausal = get(lookup, (pair, state_label, "acausal"), nothing)
            isnothing(causal) && isnothing(acausal) && continue

            components = Pair{String,Vector{Float32}}[]
            if family == "selected_state_causal"
                !isnothing(causal) && push!(components, "selected causal" => Float32.(causal))
            elseif family == "selected_state_acausal"
                !isnothing(acausal) && push!(components, "selected acausal" => Float32.(acausal))
            elseif family == GV_APC_FAMILY
                if !isnothing(causal)
                    push!(components, "causal input" => Float32.(causal))
                end
                if !isnothing(acausal)
                    push!(components, "acausal input" => Float32.(acausal))
                end
                if !isnothing(causal) && !isnothing(acausal)
                    push!(components, "selected acausal+causal" =>
                        gv_mean_vec_with_optional_split(causal, acausal, split_opts, dt))
                end
            end

            for (component, wf) in components
                t = gv_time_axis_for_waveform(length(wf), dt)
                for i in eachindex(wf)
                    push!(rows, (; candidate_id=String(cand.candidate_id),
                        candidate_family=family,
                        rank=Int(cand.rank),
                        selected_state_label=String(cand.selected_state_label),
                        reference_triplet=String(cand.reference_triplet),
                        pair_role,
                        pair,
                        state_label,
                        component,
                        sample=i,
                        time_lag_s=Float64(t[i]),
                        amplitude=Float64(wf[i]),
                        split_filter_applied=family == GV_APC_FAMILY && Bool(split_opts.enabled),
                        split_period_s=Float64(split_opts.split_period),
                        split_overlap_s=Float64(split_opts.split_overlap),
                        causal_period_cutoff_s=Float64(split_opts.causal_period_cutoff),
                        acausal_period_cutoff_s=Float64(split_opts.acausal_period_cutoff),
                        split_filter_order=Int(split_opts.split_order)))
                end
            end
        end
    end
    isempty(rows) ? DataFrame() : DataFrame(rows)
end

function gv_write_waveform_panels_png(path::AbstractString, waveform_df::DataFrame; overwrite::Bool)
    gv_output_guard(path, overwrite)
    tmp = tempname() * ".csv"
    CSV.write(tmp, waveform_df)
    py = raw"""
import math
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv_path)
if df.empty:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=170)
    ax.text(0.5, 0.5, "No selected-state waveforms found", ha="center", va="center")
    ax.axis("off")
    fig.savefig(out_path)
    raise SystemExit(0)

keys = (df[["candidate_id", "candidate_family", "rank", "selected_state_label"]]
    .drop_duplicates()
    .sort_values(["candidate_family", "rank", "candidate_id"]))
n = len(keys)
ncols = 3 if n > 2 else n
nrows = int(math.ceil(n / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 2.8 * nrows), dpi=170, squeeze=False)
pair_colors = {"ab": "#2563eb", "bc": "#dc2626", "ac": "#4b5563"}
component_styles = {
    "selected causal": "-",
    "selected acausal": "-",
    "selected acausal+causal": "-",
    "causal input": "--",
    "acausal input": ":",
}
for ax in axes.ravel():
    ax.axis("off")

for panel_idx, (_, row) in enumerate(keys.iterrows()):
    ax = axes.ravel()[panel_idx]
    ax.axis("on")
    mask = (
        (df["candidate_id"] == row["candidate_id"]) &
        (df["candidate_family"] == row["candidate_family"]) &
        (df["rank"] == row["rank"])
    )
    sub = df[mask]
    for (pair_role, component), ss in sub.groupby(["pair_role", "component"], sort=True):
        ss = ss.sort_values("sample")
        color = pair_colors.get(str(pair_role), "#111827")
        style = component_styles.get(str(component), "-")
        lw = 1.7 if str(component).startswith("selected") else 0.9
        alpha = 0.95 if str(component).startswith("selected") else 0.45
        label = f"{pair_role} {component}"
        amp = ss["amplitude"].astype(float)
        scale = float(amp.abs().max()) if len(amp) else 0.0
        y = amp / scale if scale > 0 else amp
        ax.plot(ss["time_lag_s"], y, linestyle=style, color=color, linewidth=lw, alpha=alpha, label=label)
    title = f'{row["candidate_family"]} r{int(row["rank"])}\n{row["selected_state_label"]}'
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Lag (s)", fontsize=7)
    ax.set_ylabel("Normalized amplitude", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=5.7, ncol=2, loc="best")

fig.suptitle("Selected-state waveforms for reference-triplet candidates", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(out_path)
"""
    try
        run(`python3 -c $py $tmp $path`)
    finally
        isfile(tmp) && rm(tmp; force=true)
    end
    path
end

function gv_write_waveform_panels_html(path::AbstractString, png_name::AbstractString; overwrite::Bool)
    gv_output_guard(path, overwrite)
    body = """
    <!doctype html>
    <html><head><meta charset="utf-8"><title>Selected-state waveforms</title></head>
    <body style="font-family: sans-serif;">
      <h2>Selected-state waveforms for reference-triplet candidates</h2>
      <img src="$(html_escape(png_name))" style="max-width:100%;height:auto;" />
    </body></html>
    """
    open(path, "w") do io
        write(io, body)
    end
    path
end

function gv_print_rmse_ranking(rmse_df::DataFrame)
    isempty(rmse_df) && return
    overall = rmse_df[(String.(rmse_df.scope) .== "overall") .&
        .!startswith.(String.(rmse_df.candidate_id), "global_average_"), :]
    isempty(overall) && return
    sort!(overall, [:rmse_abs_vabc_minus_vac])
    println()
    println("Selected-state candidates ranked by group-velocity RMSE:")
    for (i, row) in enumerate(eachrow(overall))
        @printf("  %2d. RMSE %.6g | median %.6g | max %.6g | n=%d | %s rank %d | %s\n",
            i,
            Float64(row.rmse_abs_vabc_minus_vac),
            Float64(row.median_abs_vabc_minus_vac),
            Float64(row.max_abs_vabc_minus_vac),
            Int(row.n_periods),
            String(row.candidate_family),
            Int(row.rank),
            String(row.selected_state_label))
    end
    globals = rmse_df[(String.(rmse_df.scope) .== "overall") .&
        startswith.(String.(rmse_df.candidate_id), "global_average_"), :]
    isempty(globals) && return
    sort!(globals, [:rmse_abs_vabc_minus_vac])
    println()
    println("Global references ranked by group-velocity RMSE:")
    for (i, row) in enumerate(eachrow(globals))
        @printf("  %2d. RMSE %.6g | median %.6g | max %.6g | n=%d | %s\n",
            i,
            Float64(row.rmse_abs_vabc_minus_vac),
            Float64(row.median_abs_vabc_minus_vac),
            Float64(row.max_abs_vabc_minus_vac),
            Int(row.n_periods),
            gv_pretty_candidate_id(String(row.candidate_id)))
    end
end

function gv_evaluate_rows(triplets::DataFrame, selected_candidates, scored_by_family,
        global_rows_by_family; max_candidates::Int, split_opts=GV_DISABLED_SPLIT_OPTIONS)
    all_rows = NamedTuple[]
    global_split_meta = gv_apc_split_metadata(split_opts; applied=false)
    for (itrip, trow) in enumerate(eachrow(triplets))
        triplet_label = String(trow.triplet)
        @printf("[%d/%d] evaluating %s\n", itrip, nrow(triplets), triplet_label)
        pairs = (String(trow.pair_ab), String(trow.pair_bc), String(trow.pair_ac))
        geom = geom_from_triplet_row(trow, pairs)
        for family in ("global_average_mean", "global_average_causal", "global_average_acausal")
            rows = rows_for_global_triplet(geom, global_rows_by_family[family]; velocity_field=:group_velocity)
            append!(all_rows, [merge(r, (; triplet=triplet_label,
                candidate_id=family,
                candidate_family="global",
                rank=0,
                selected_state_label=""), global_split_meta) for r in rows])
        end
        for cand in selected_candidates
            cand_split_meta = gv_apc_split_metadata(split_opts; applied=(String(cand.family) == GV_APC_FAMILY && split_opts.enabled))
            scored = scored_by_family[String(cand.family)]
            combo = gv_combo_from_template(scored, geom, cand.template)
            isnothing(combo) && continue
            label = _logical_state_display(String(combo.label_ab))
            rows = rows_for_selected_combo(geom, combo; velocity_field=:group_velocity,
                source_label="$(cand.family) rank $(cand.rank)")
            append!(all_rows, [merge(r, (; triplet=triplet_label,
                candidate_id=String(cand.candidate_id),
                candidate_family=String(cand.family),
                rank=Int(cand.rank),
                selected_state_label=label), cand_split_meta) for r in rows])
        end
    end
    all_rows
end

function gv_main(argv=ARGS)
    split_opts = gv_extract_apc_split_options(argv)
    opts = parse_args(split_opts.argv)
    if get(opts, "help", false)
        gv_usage()
        return 0
    end
    get!(opts, "candidate-triplet", GV_DEFAULT_CANDIDATE_TRIPLET)
    explicit_keys = get(opts, "__explicit_keys__", Set{String}())
    "top-ranks" in explicit_keys || (opts["top-ranks"] = 5)
    apply_toml_config!(opts)
    validate_resolved_options!(opts)

    saved_root = require_arg(opts, "saved-root")
    raw_data_dir = require_arg(opts, "raw-data-dir")
    output_dir = require_arg(opts, "output-dir")
    config_path = String(get(opts, "config", ""))
    candidate_triplet = String(opts["candidate-triplet"])
    triplets_csv = String(opts["triplets-csv"])
    overwrite = Bool(opts["overwrite"])
    dry_run = Bool(opts["dry-run"])
    top_ranks = Int(opts["top-ranks"])
    max_candidates = Int(opts["max-candidates"])
    dt = Float64(opts["dt"])
    bp = parse_bandpass(String(opts["bandpass"]))
    bandpass_label = isnothing(bp) ? "none" : "$(bp.period_min):$(bp.period_max)"
    if split_opts.enabled
        nyq = 0.5 / dt
        causal_cutoff_hz = 1.0 / split_opts.causal_period_cutoff
        acausal_cutoff_hz = 1.0 / split_opts.acausal_period_cutoff
        causal_cutoff_hz < nyq || error("Causal split cutoff $(causal_cutoff_hz) Hz exceeds Nyquist $(nyq) Hz")
        acausal_cutoff_hz < nyq || error("Acausal split cutoff $(acausal_cutoff_hz) Hz exceeds Nyquist $(nyq) Hz")
    end

    triplets_all = load_triplets(triplets_csv;
        max_delta_az=opts["max-delta-az"],
        max_delta_d=opts["max-delta-d"],
        min_segment_distance=opts["min-segment-distance"],
        max_triplets=0)
    candidate_idx = gv_triplet_row_index(triplets_all, candidate_triplet)
    candidate_row = triplets_all[candidate_idx, :]
    triplets = triplets_all[candidate_idx:candidate_idx, :]
    candidate_pairs = String[String(candidate_row.pair_ab), String(candidate_row.pair_bc), String(candidate_row.pair_ac)]
    needed_pairs = sort(unique(candidate_pairs))
    raw_pairs = list_raw_pairs(raw_data_dir)
    runs = discover_all_vqvae_runs(saved_root; transfer_root=saved_root)
    run_pairs = sort(unique(String.(getproperty.(runs, :pair_label))))
    available_needed = intersect(needed_pairs, run_pairs)

    println("Group-velocity acausal+causal split-filter CLI")
    println("  candidate triplet: $(candidate_triplet)")
    println("  triplets CSV: $(triplets_csv)")
    println("  triplets passing criteria: $(nrow(triplets_all))")
    println("  triplets to evaluate: 1 (reference triplet only)")
    println("  raw pairs discovered: $(length(raw_pairs))")
    println("  artifact runs discovered: $(length(runs))")
    println("  needed pairs with artifacts: $(length(available_needed)) / $(length(needed_pairs))")
    println("  top ranks per selected family: $(top_ranks)")
    println("  bandpass: $(bandpass_label)")
    println("  acausal+causal split filter: $(split_opts.enabled)")
    if split_opts.enabled
        println("  split period: $(split_opts.split_period) s")
        println("  split overlap: $(split_opts.split_overlap) s")
        println("  causal period cutoff: $(split_opts.causal_period_cutoff) s")
        println("  acausal period cutoff: $(split_opts.acausal_period_cutoff) s")
        println("  split filter order: $(split_opts.split_order)")
    end
    println("  MFT period band: $(opts["period-min"])-$(opts["period-max"]) s ($(opts["nperiods"]) periods)")
    println("  MFT velocity range: $(opts["velocity-min"])-$(opts["velocity-max"]) km/s")
    println("  wavelength filter: ref velocity=$(opts["wavelength-ref-velocity"]) km/s, fraction=$(opts["wavelength-fraction"])")

    if dry_run
        println("Dry run only. Expected outputs:")
        for name in ("reference_triplet_top_candidates.csv",
                "reference_triplet_group_velocity_rows.csv",
                "reference_triplet_group_velocity_rmse.csv",
                "reference_triplet_selected_state_waveforms.csv",
                "plots/reference_triplet_group_velocity.html",
                "plots/reference_triplet_group_velocity.png",
                "plots/reference_triplet_rmse.html",
                "plots/reference_triplet_rmse.png",
                "plots/reference_triplet_selected_state_waveforms.html",
                "plots/reference_triplet_selected_state_waveforms.png")
            println("  ", joinpath(output_dir, name))
        end
        return 0
    end

    isempty(runs) && error("No source-state artifacts found under $(saved_root)")
    isempty(triplets_all) && error("No triplets pass the requested criteria")
    length(available_needed) == length(needed_pairs) ||
        @warn "Some needed pairs do not have source-state artifacts" missing=setdiff(needed_pairs, available_needed)

    cfg_base = gv_base_config(opts)
    cfg_selected = merge(cfg_base, (; phvel_correction=correction_from_config_denominator(opts, "selected-phvel-correction-denominator"), cache=Dict{Any,Any}()))
    cfg_causal = merge(cfg_base, (; phvel_correction=correction_from_config_denominator(opts, "causal-phvel-correction-denominator"), cache=Dict{Any,Any}()))
    cfg_selected_mean = merge(cfg_base, (; phvel_correction=correction_from_config_denominator(opts, "selected-mean-phvel-correction-denominator"), cache=Dict{Any,Any}()))

    activate_ftan_project!()

    println("Loading candidate-triplet artifacts...")
    candidate_items = gv_load_items_for_pairs(runs, candidate_pairs, bp, dt)
    selected_candidates, candidate_df = gv_choose_reference_candidates(candidate_row, candidate_items,
        cfg_selected, cfg_selected_mean; top_ranks, max_candidates, split_opts, dt)
    isempty(selected_candidates) && error("No selected-state candidates could be ranked for $(candidate_triplet)")
    gv_print_candidate_time_window_allocations(selected_candidates, candidate_items)

    println("Loading artifacts for reference triplet pairs...")
    items = gv_load_items_for_pairs(runs, needed_pairs, bp, dt)
    global_items = collect(values(latest_trained_items_by_pair(items)))

    println("Running selected-state MFT batches for reference triplet pairs...")
    scored_by_family = Dict{String,Any}()
    scored_by_family["selected_state_causal"] = selected_scores_for_family(items, cfg_selected, "selected_state_causal")
    scored_by_family["selected_state_acausal"] = selected_scores_for_family(items, cfg_selected, "selected_state_acausal")
    scored_by_family[GV_APC_FAMILY] = gv_selected_scores_for_family(items, cfg_selected_mean, GV_APC_FAMILY, split_opts, dt)

    println("Running global MFT batches for reference triplet pairs...")
    global_rows_by_family = Dict{String,Any}()
    global_rows_by_family["global_average_mean"] = analyze_global_rows(global_items, cfg_base, "global_average_mean")
    global_rows_by_family["global_average_acausal"] = analyze_global_rows(global_items, cfg_base, "global_average_acausal")
    global_rows_by_family["global_average_causal"] = analyze_global_rows(global_items, cfg_causal, "global_average_causal")

    group_rows = gv_evaluate_rows(triplets, selected_candidates, scored_by_family,
        global_rows_by_family; max_candidates, split_opts)
    rows_df = gv_rows_dataframe(group_rows)
    rmse_df = gv_rmse_dataframe(rows_df)
    waveform_df = gv_waveform_rows_for_reference_candidates(candidate_df, items, split_opts, dt)
    gv_print_rmse_ranking(rmse_df)

    mkpath(output_dir)
    plots_dir = joinpath(output_dir, "plots")
    write_csv(joinpath(output_dir, "reference_triplet_top_candidates.csv"), candidate_df; overwrite)
    write_csv(joinpath(output_dir, "reference_triplet_group_velocity_rows.csv"), rows_df; overwrite)
    write_csv(joinpath(output_dir, "reference_triplet_group_velocity_rmse.csv"), rmse_df; overwrite)
    write_csv(joinpath(output_dir, "reference_triplet_selected_state_waveforms.csv"), waveform_df; overwrite)
    gv_write_reference_html(joinpath(plots_dir, "reference_triplet_group_velocity.html"),
        rows_df, candidate_triplet; overwrite)
    gv_write_reference_png(joinpath(plots_dir, "reference_triplet_group_velocity.png"),
        rows_df, candidate_triplet; overwrite)
    gv_write_rmse_html(joinpath(plots_dir, "reference_triplet_rmse.html"), rmse_df; overwrite)
    gv_write_rmse_png(joinpath(plots_dir, "reference_triplet_rmse.png"), rmse_df; overwrite)
    gv_write_waveform_panels_png(joinpath(plots_dir, "reference_triplet_selected_state_waveforms.png"),
        waveform_df; overwrite)
    gv_write_waveform_panels_html(joinpath(plots_dir, "reference_triplet_selected_state_waveforms.html"),
        "reference_triplet_selected_state_waveforms.png"; overwrite)

    println()
    println("Wrote:")
    for name in ("reference_triplet_top_candidates.csv",
            "reference_triplet_group_velocity_rows.csv",
            "reference_triplet_group_velocity_rmse.csv",
            "reference_triplet_selected_state_waveforms.csv",
            "plots/reference_triplet_group_velocity.html",
            "plots/reference_triplet_group_velocity.png",
            "plots/reference_triplet_rmse.html",
            "plots/reference_triplet_rmse.png",
            "plots/reference_triplet_selected_state_waveforms.html",
            "plots/reference_triplet_selected_state_waveforms.png")
        println("  ", joinpath(output_dir, name))
    end
    gv_print_phase_candidate_sweep_followup(candidate_df;
        config_path, saved_root, raw_data_dir, triplets_csv, candidate_triplet,
        output_dir, split_opts)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(gv_main())
    catch err
        showerror(stderr, err)
        println(stderr)
        exit(1)
    end
end
