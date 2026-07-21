#!/usr/bin/env julia

include(joinpath(@__DIR__, "pair_export_selected_candidate_cli.jl"))
include(joinpath(@__DIR__, "group_velocity_acausal_plus_causal_cli.jl"))
include(joinpath(@__DIR__, "acausal_plus_causal_downstream_helpers.jl"))

function pex_apc_usage()
    pex_usage()
    println("""

    Acausal+causal split options:
      --apply-acausal-plus-causal-split-filter true|false
      --split-period X
      --split-overlap X
      --split-filter-order N

    Combined candidate branch aliases:
      mean, causal+acausal, acausal_plus_causal, selected_state_acausal_plus_causal
    """)
end

function apc_pex_selected_waveform_lookup(items, candidate, split_opts, dt::Real)
    lookup = apc_state_waveform_lookup(items, split_opts, dt)
    target = _logical_state_display(String(candidate.base_label))
    out = Dict{String,Vector{Float32}}()
    for (pair, label, branch) in keys(lookup)
        label == target && branch == String(candidate.branch) || continue
        out[String(pair)] = lookup[(pair, label, branch)]
    end
    out
end

function apc_pex_selected_waveform_sides_lookup(items, candidate, split_opts, dt::Real)
    lookup = apc_state_waveform_lookup(items, split_opts, dt)
    target = _logical_state_display(String(candidate.base_label))
    out = Dict{String,NamedTuple}()
    for (pair, label, _branch) in keys(lookup)
        label == target || continue
        pair_s = String(pair)
        causal = get(lookup, (pair_s, target, "causal"), nothing)
        acausal = get(lookup, (pair_s, target, "acausal"), nothing)
        combined = get(lookup, (pair_s, target, "acausal_plus_causal"), nothing)
        isnothing(causal) || isnothing(acausal) || isnothing(combined) || begin
            out[pair_s] = (;
                causal=Float32.(causal),
                acausal=Float32.(acausal),
                acausal_plus_causal=Float32.(combined))
        end
    end
    out
end

function apc_pex_best_selected_wave(sides, candidate)
    branch = String(candidate.branch)
    branch == "causal" && return sides.causal
    branch == "acausal" && return sides.acausal
    sides.acausal_plus_causal
end

function apc_pex_waveform_csv_dataframe(pair::AbstractString, sides, global_item, candidate, bp, dt::Real)
    selected_best = apc_pex_best_selected_wave(sides, candidate)
    n = minimum((
        length(selected_best),
        length(sides.causal),
        length(sides.acausal),
        length(sides.acausal_plus_causal),
        length(global_item.global_avg_ac),
        length(global_item.global_avg_c)))
    n > 0 || return nothing
    global_ac = Float64.(global_item.global_avg_ac[1:n])
    global_c = Float64.(global_item.global_avg_c[1:n])
    global_mean = 0.5 .* (global_ac .+ global_c)
    selected_causal = Float64.(sides.causal[1:n])
    selected_acausal = Float64.(sides.acausal[1:n])
    selected_combined = Float64.(sides.acausal_plus_causal[1:n])
    selected = Float64.(selected_best[1:n])
    DataFrame(
        time_lag_s=(1:n) .* Float64(dt),
        global_average_ac=global_ac,
        global_average_c=global_c,
        global_average_mean=global_mean,
        best_selected_state=selected,
        selected_state_causal=selected_causal,
        selected_state_acausal=selected_acausal,
        selected_state_acausal_plus_causal=selected_combined,
        filtered_global_average_ac=pex_maybe_filtered(global_ac, bp, dt),
        filtered_global_average_c=pex_maybe_filtered(global_c, bp, dt),
        filtered_global_average_mean=pex_maybe_filtered(global_mean, bp, dt),
        filtered_best_selected_state=pex_maybe_filtered(selected, bp, dt),
        filtered_selected_state_causal=pex_maybe_filtered(selected_causal, bp, dt),
        filtered_selected_state_acausal=pex_maybe_filtered(selected_acausal, bp, dt),
        filtered_selected_state_acausal_plus_causal=pex_maybe_filtered(selected_combined, bp, dt),
        pair=fill(String(pair), n),
        selected_state=fill(String(candidate.base_label), n),
        selected_branch=fill(String(candidate.branch), n),
        selected_kind=fill(String(candidate.family), n),
        selected_waveform_sides=fill("causal,acausal,acausal_plus_causal", n))
end

function apc_pex_write_waveform_csvs(output_dir::AbstractString, pairs, items, candidate, bp, dt::Real,
        split_opts; overwrite::Bool)
    out_dir = joinpath(output_dir, "waveform_csvs")
    mkpath(out_dir)
    global_by_pair = latest_trained_items_by_pair(items)
    selected_by_pair = apc_pex_selected_waveform_sides_lookup(items, candidate, split_opts, dt)
    rows = NamedTuple[]
    for pair in sort(String.(pairs))
        haskey(global_by_pair, pair) && haskey(selected_by_pair, pair) || continue
        df = apc_pex_waveform_csv_dataframe(pair, selected_by_pair[pair], global_by_pair[pair], candidate, bp, dt)
        isnothing(df) && continue
        apc_add_split_columns!(df, split_opts; family_col=:selected_kind)
        path = joinpath(out_dir, "$(safe_name(pair))_$(safe_name(candidate.display_label)).csv")
        write_csv(path, df; overwrite)
        push!(rows, (; pair, path, nrows=nrow(df)))
    end
    DataFrame(rows)
end

function apc_pex_print_dispersion_pair_counts(df::DataFrame)
    if isempty(df) || !(:curve_label in Symbol.(names(df))) || !(:pair in Symbol.(names(df)))
        println()
        println("Dispersion curve pair counts: no dispersion rows available")
        return
    end
    println()
    println("Dispersion curve pair counts:")
    for sub in groupby(df, [:curve_label, :curve_family, :d_schedule_id])
        label = String(first(sub.curve_label))
        dlabel = (:d_schedule_label in Symbol.(names(sub))) ?
            String(first(sub.d_schedule_label)) : "d=$(first(sub.denominator))"
        all_pairs = unique(String.(sub.pair))
        group_pairs = (:group_velocity_threshold_keep in Symbol.(names(sub))) ?
            unique(String.(sub.pair[Bool.(sub.group_velocity_threshold_keep)])) : all_pairs
        phase_pairs = (:phase_velocity_threshold_keep in Symbol.(names(sub))) ?
            unique(String.(sub.pair[Bool.(sub.phase_velocity_threshold_keep)])) : all_pairs
        periods = unique(Float64.(sub.period_s))
        @printf("  %s | %s: pairs=%d, group kept=%d, phase kept=%d, periods=%d\n",
            label, dlabel, length(all_pairs), length(group_pairs), length(phase_pairs), length(periods))
    end
end

function apc_pex_write_dispersion_outputs(output_dir::AbstractString, items, candidate, d_by_curve, cfg_base,
        split_opts, dt::Real; overwrite::Bool, threshold_enabled::Bool=false,
        threshold_fraction::Real=10.0, threshold_mode::AbstractString="median")
    curve_specs_all = apc_curve_specs(items, collect(values(latest_trained_items_by_pair(items))), (candidate,), split_opts, dt)
    curve_meta = unique([(String(spec.curve_label), String(spec.curve_family)) for spec in curve_specs_all])
    labels_by_spec = Dict{String,Vector{Tuple{String,String}}}()
    spec_by_id = Dict{String,NamedTuple}()
    for (label, family) in curve_meta
        key = family in ("global_average_mean", "global_average_causal", "global_average_acausal") ?
            family : "selected_candidate"
        for dspec in get(d_by_curve, key, NamedTuple[])
            spec_by_id[String(dspec.id)] = dspec
            push!(get!(labels_by_spec, String(dspec.id), Tuple{String,String}[]), (label, family))
        end
    end
    csv_dir = joinpath(output_dir, "dispersion_csvs")
    txt_dir = joinpath(output_dir, "dispersion_txts")
    labeled_csv_dir = joinpath(output_dir, "dispersion_labeled_csvs")
    mkpath(csv_dir); mkpath(txt_dir); mkpath(labeled_csv_dir)
    all_rows = DataFrame()
    for spec_id in sort(collect(keys(labels_by_spec)))
        dspec = spec_by_id[spec_id]
        labels_for_d = labels_by_spec[spec_id]
        wanted = Set(labels_for_d)
        specs = [spec for spec in curve_specs_all if (String(spec.curve_label), String(spec.curve_family)) in wanted]
        println("Running all-pair dispersion MFT $(dspec.label) for $(length(labels_for_d)) curve(s), $(length(specs)) pair waveforms")
        cfg = pfa_config_with_spec(cfg_base, dspec)
        analyzed = pcs_analyze_curve_specs(specs, cfg)
        df = pex_dispersion_dataframe(analyzed, specs, dspec, cfg)
        isempty(df) && continue
        append!(all_rows, df; cols=:union)
    end
    all_rows = pex_apply_period_threshold(all_rows; enabled=threshold_enabled, fraction=threshold_fraction, mode=threshold_mode)
    apc_add_split_columns!(all_rows, split_opts)
    apc_pex_print_dispersion_pair_counts(all_rows)
    csv_path = joinpath(csv_dir, "all_pair_dispersion_phase_group.csv")
    write_csv(csv_path, all_rows; overwrite)
    txt_written = String[]
    txt_manifest_rows = NamedTuple[]
    if !isempty(all_rows)
        for sub in groupby(all_rows, [:curve_id, :d_schedule_id])
            curve_id = String(first(sub.curve_id))
            dtag = safe_name(String(first(sub.d_schedule_id)))
            dspec = spec_by_id[String(first(sub.d_schedule_id))]
            group_sub = sub[Bool.(sub.group_velocity_threshold_keep), :]
            phase_sub = sub[Bool.(sub.phase_velocity_threshold_keep), :]
            group_rows = pex_txt_rows(group_sub, :group_velocity_km_s)
            phase_rows = pex_txt_rows(phase_sub, :phase_velocity_km_s)
            group_labeled_rows = pex_labeled_dispersion_csv_rows(sub,
                :group_velocity_km_s, :group_velocity_threshold_keep)
            phase_labeled_rows = pex_labeled_dispersion_csv_rows(sub,
                :phase_velocity_km_s, :phase_velocity_threshold_keep)
            gpath = joinpath(txt_dir, "group_velocity_$(curve_id)_d=$(dtag)_for_dsurftomo.txt")
            ppath = joinpath(txt_dir, "phase_velocity_$(curve_id)_d=$(dtag)_for_dsurftomo.txt")
            gcsv_path = joinpath(labeled_csv_dir, "group_velocity_$(curve_id)_d=$(dtag)_for_dsurftomo.csv")
            pcsv_path = joinpath(labeled_csv_dir, "phase_velocity_$(curve_id)_d=$(dtag)_for_dsurftomo.csv")
            pex_write_pdsurftomo_txt(gpath, group_rows; overwrite)
            pex_write_pdsurftomo_txt(ppath, phase_rows; overwrite)
            pex_write_labeled_dispersion_csv(gcsv_path, group_labeled_rows; overwrite)
            pex_write_labeled_dispersion_csv(pcsv_path, phase_labeled_rows; overwrite)
            push!(txt_manifest_rows, pex_txt_manifest_row(gpath, group_rows, curve_id, "group_velocity", dspec))
            push!(txt_manifest_rows, pex_txt_manifest_row(ppath, phase_rows, curve_id, "phase_velocity", dspec))
            push!(txt_written, gpath); push!(txt_written, ppath)
        end
    end
    plot_written = pex_write_txt_dispersion_plots(txt_written; output_dir, overwrite,
        dispersion_df=all_rows, threshold_enabled, threshold_fraction, threshold_mode)
    manifest_df = isempty(txt_manifest_rows) ? DataFrame() : DataFrame(txt_manifest_rows)
    csv_path, txt_written, plot_written, all_rows, manifest_df
end

function apc_pex_write_metadata(output_dir, opts, candidate, d_by_curve, pair_scope, pairs, waveform_df, dispersion_df,
        txt_manifest_df; bandpass_label, overwrite, threshold_enabled::Bool=false,
        threshold_fraction::Real=10.0, threshold_mode::AbstractString="median", split_opts)
    mkpath(joinpath(output_dir, "metadata"))
    rows = [
        (; key="candidate", value=String(candidate.display_label)),
        (; key="pair_scope", value=String(pair_scope)),
        (; key="n_pairs_requested", value=string(length(pairs))),
        (; key="n_waveform_csvs", value=string(nrow(waveform_df))),
        (; key="waveform_csv_selected_state_sides", value="causal,acausal,acausal_plus_causal"),
        (; key="n_dispersion_rows", value=string(nrow(dispersion_df))),
        (; key="bandpass", value=String(bandpass_label)),
        (; key="period_min", value=string(opts["period-min"])),
        (; key="period_max", value=string(opts["period-max"])),
        (; key="nperiods", value=string(opts["nperiods"])),
        (; key="global_mean_d", value=pfa_specs_label(d_by_curve["global_average_mean"])),
        (; key="global_causal_d", value=pfa_specs_label(d_by_curve["global_average_causal"])),
        (; key="global_acausal_d", value=pfa_specs_label(d_by_curve["global_average_acausal"])),
        (; key="candidate_d", value=pfa_specs_label(d_by_curve["selected_candidate"])),
        (; key="threshold_by_period", value=string(Bool(threshold_enabled))),
        (; key="threshold_mode", value=String(threshold_mode)),
        (; key="threshold_fraction", value=string(Float64(threshold_fraction))),
        (; key="split_filter_applied", value=string(Bool(split_opts.enabled && String(candidate.family) == GV_APC_FAMILY))),
        (; key="split_period_s", value=string(Float64(split_opts.split_period))),
        (; key="split_overlap_s", value=string(Float64(split_opts.split_overlap))),
        (; key="causal_period_cutoff_s", value=string(Float64(split_opts.causal_period_cutoff))),
        (; key="acausal_period_cutoff_s", value=string(Float64(split_opts.acausal_period_cutoff))),
        (; key="split_filter_order", value=string(Int(split_opts.split_order))),
    ]
    write_csv(joinpath(output_dir, "metadata", "export_metadata.csv"), DataFrame(rows); overwrite)
    write_csv(joinpath(output_dir, "metadata", "dispersion_txt_manifest.csv"), txt_manifest_df; overwrite)
end

function pex_apc_main(argv=ARGS)
    argv2, threshold_enabled, threshold_fraction, threshold_mode = pex_extract_threshold_options(argv)
    split_opts = gv_extract_apc_split_options(argv2)
    argv2 = split_opts.argv
    argv2, pair_scope = pex_extract_pair_scope(argv2)
    argv2, global_mean_d_raw = pfa_extract_value_option(argv2, "--global-mean-d")
    argv2, global_causal_d_raw = pfa_extract_value_option(argv2, "--global-causal-d")
    argv2, global_acausal_d_raw = pfa_extract_value_option(argv2, "--global-acausal-d")
    argv2, candidate_d_raw = pfa_extract_value_option(argv2, "--candidate-d")
    argv2, candidate_d_schedule_files = pfa_extract_value_option(argv2, "--candidate-d-schedules")
    opts = parse_args(argv2)
    if get(opts, "help", false)
        pex_apc_usage()
        return 0
    end
    get!(opts, "d-values", "")
    apply_toml_config!(opts)
    validate_resolved_options!(opts)

    saved_root = require_arg(opts, "saved-root")
    raw_data_dir = require_arg(opts, "raw-data-dir")
    output_dir = require_arg(opts, "output-dir")
    candidate_text = require_arg(opts, "final-candidate")
    candidates = apc_parse_candidates(candidate_text)
    length(candidates) == 1 || error("--final-candidate must contain exactly one candidate")
    candidate = first(candidates)
    fallback_d_values = haskey(opts, "d-values") && !isempty(strip(String(opts["d-values"]))) ?
        parse_denominators(String(opts["d-values"])) : nothing
    d_by_curve = pfa_curve_d_specs(;
        fallback=fallback_d_values,
        global_mean=pfa_parse_single_denominator(global_mean_d_raw, "--global-mean-d"),
        global_causal=pfa_parse_single_denominator(global_causal_d_raw, "--global-causal-d"),
        global_acausal=pfa_parse_single_denominator(global_acausal_d_raw, "--global-acausal-d"),
        candidate=pfa_parse_candidate_d_specs(candidate_d_raw, candidate_d_schedule_files;
            allow_multiple_scalars=true))
    dt = Float64(opts["dt"])
    bp = parse_bandpass(String(opts["bandpass"]))
    bandpass_label = isnothing(bp) ? "none" : "$(bp.period_min):$(bp.period_max)"
    overwrite = Bool(opts["overwrite"])
    dry_run = Bool(opts["dry-run"])

    triplets = load_triplets(String(opts["triplets-csv"]);
        max_delta_az=opts["max-delta-az"], max_delta_d=opts["max-delta-d"],
        min_segment_distance=opts["min-segment-distance"], max_triplets=Int(opts["max-triplets"]))
    runs = discover_all_vqvae_runs(saved_root; transfer_root=saved_root)
    raw_pairs = list_raw_pairs(raw_data_dir)
    pairs = pex_pair_scope_pairs(pair_scope, runs, triplets)
    available_pairs = intersect(pairs, sort(unique(String.(getproperty.(runs, :pair_label)))))

    println("Pair export acausal+causal selected-candidate CLI")
    println("  candidate: $(candidate.display_label)")
    println("  pair scope: $(pair_scope)")
    println("  pairs requested: $(length(pairs))")
    println("  pairs with artifacts: $(length(available_pairs)) / $(length(pairs))")
    println("  raw pairs discovered: $(length(raw_pairs))")
    println("  bandpass: $(bandpass_label)")
    apc_print_split_options(split_opts)
    println("  d global mean (causal+acausal): $(pfa_specs_label(d_by_curve["global_average_mean"]))")
    println("  d global causal: $(pfa_specs_label(d_by_curve["global_average_causal"]))")
    println("  d global acausal: $(pfa_specs_label(d_by_curve["global_average_acausal"]))")
    println("  d selected candidate: $(pfa_specs_label(d_by_curve["selected_candidate"]))")
    println("  threshold by period: $(threshold_enabled) (mode=$(threshold_mode), fraction=$(threshold_fraction))")

    if dry_run
        println("Dry run only. Expected output folders:")
        for dir in ("waveform_csvs", "dispersion_txts", "dispersion_labeled_csvs", "dispersion_csvs", "plots/dispersion_txts", "metadata")
            println("  ", joinpath(output_dir, dir))
        end
        return 0
    end

    cfg_base = gv_base_config(opts)
    activate_ftan_project!()
    items = gv_load_items_for_pairs(runs, available_pairs, bp, dt)
    isempty(items) && error("No loadable artifacts for requested pair scope")
    waveform_df = apc_pex_write_waveform_csvs(output_dir, available_pairs, items, candidate, bp, dt, split_opts; overwrite)
    dispersion_csv, txt_written, plot_written, dispersion_df, txt_manifest_df =
        apc_pex_write_dispersion_outputs(output_dir, items, candidate, d_by_curve, cfg_base, split_opts, dt;
            overwrite, threshold_enabled, threshold_fraction, threshold_mode)
    apc_pex_write_metadata(output_dir, opts, candidate, d_by_curve, pair_scope, available_pairs,
        waveform_df, dispersion_df, txt_manifest_df; bandpass_label, overwrite,
        threshold_enabled, threshold_fraction, threshold_mode, split_opts)

    println()
    println("Wrote:")
    println("  waveform CSVs: ", nrow(waveform_df), " files in ", joinpath(output_dir, "waveform_csvs"))
    println("  dispersion CSV: ", dispersion_csv)
    println("  dispersion TXTs: ", length(txt_written), " files in ", joinpath(output_dir, "dispersion_txts"))
    println("  dispersion plots: ", length(plot_written), " files in ", joinpath(output_dir, "plots", "dispersion_txts"))
    println("  metadata: ", joinpath(output_dir, "metadata", "export_metadata.csv"))
    println("  TXT manifest: ", joinpath(output_dir, "metadata", "dispersion_txt_manifest.csv"))
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(pex_apc_main())
    catch err
        showerror(stderr, err)
        println(stderr)
        exit(1)
    end
end
