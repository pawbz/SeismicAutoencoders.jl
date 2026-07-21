#!/usr/bin/env julia

include(joinpath(@__DIR__, "phase_candidate_sweep_cli.jl"))
include(joinpath(@__DIR__, "group_velocity_acausal_plus_causal_cli.jl"))
include(joinpath(@__DIR__, "acausal_plus_causal_downstream_helpers.jl"))

function pcs_apc_usage()
    pcs_usage()
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

function apc_pcs_run_sweep(triplets::DataFrame, items, global_items, candidates, denominators, cfg_base, split_opts, dt::Real)
    curve_specs = apc_curve_specs(items, global_items, candidates, split_opts, dt)
    curve_meta = unique([(String(spec.curve_label), String(spec.curve_family)) for spec in curve_specs])
    println("  phase curves batched per d: $(length(curve_meta)) labels, $(length(curve_specs)) pair waveforms")
    summary_rows = NamedTuple[]
    period_rows = NamedTuple[]
    for d in denominators
        @printf("Running phase sweep d=%.4g\n", d)
        cfg = pcs_config_with_d(cfg_base, d)
        analyzed = pcs_analyze_curve_specs(curve_specs, cfg)
        for trow in eachrow(triplets)
            triplet_label = String(trow.triplet)
            pairs = (String(trow.pair_ab), String(trow.pair_bc), String(trow.pair_ac))
            geom = geom_from_triplet_row(trow, pairs)
            for (label, family) in curve_meta
                rows = pcs_triplet_curve_rows(geom, label, analyzed)
                srow = pcs_summary_row(; triplet=triplet_label, pair_ab=geom.pair_ab, pair_bc=geom.pair_bc,
                    pair_ac=geom.pair_ac, label, family, denominator=d, rows)
                isnothing(srow) || push!(summary_rows, srow)
                append!(period_rows, [merge(r, (; triplet=triplet_label, curve_label=label,
                    curve_family=family, denominator=Float64(d), correction=correction_from_denominator(d)))
                    for r in rows])
            end
        end
    end
    period_rows, summary_rows
end

function apc_reference_allocation_pair(reference_triplet::AbstractString)
    a, _, c = gv_parse_triplet_label(reference_triplet)
    "$(a)-$(c)"
end

function pcs_apc_main(argv=ARGS)
    argv2, show_plot = pcs_consume_flag(argv, "--show-plot")
    split_opts = gv_extract_apc_split_options(argv2)
    opts = parse_args(split_opts.argv)
    if get(opts, "help", false)
        pcs_apc_usage()
        return 0
    end
    get!(opts, "reference-triplet", PCS_DEFAULT_REFERENCE_TRIPLET)
    get!(opts, "d-values", "0,2:0.2:12")
    apply_toml_config!(opts)
    validate_resolved_options!(opts)

    saved_root = require_arg(opts, "saved-root")
    raw_data_dir = require_arg(opts, "raw-data-dir")
    output_dir = require_arg(opts, "output-dir")
    candidates = apc_parse_candidates(require_arg(opts, "candidates"))
    denominators = parse_denominators(String(opts["d-values"]))
    dt = Float64(opts["dt"])
    bp = parse_bandpass(String(opts["bandpass"]))
    bandpass_label = isnothing(bp) ? "none" : "$(bp.period_min):$(bp.period_max)"
    overwrite = Bool(opts["overwrite"])
    dry_run = Bool(opts["dry-run"])
    cfg_base = gv_base_config(opts)

    triplets = load_triplets(String(opts["triplets-csv"]);
        max_delta_az=opts["max-delta-az"], max_delta_d=opts["max-delta-d"],
        min_segment_distance=opts["min-segment-distance"], max_triplets=Int(opts["max-triplets"]))
    needed_pairs = gv_needed_pairs(triplets)
    runs = discover_all_vqvae_runs(saved_root; transfer_root=saved_root)
    raw_pairs = list_raw_pairs(raw_data_dir)
    available_needed = intersect(needed_pairs, sort(unique(String.(getproperty.(runs, :pair_label)))))

    println("Phase candidate sweep acausal+causal CLI")
    println("  triplets to evaluate: $(nrow(triplets))")
    println("  reference triplet period plot: $(String(opts["reference-triplet"]))")
    println("  candidates: $(length(candidates))")
    println("  d values: $(length(denominators))")
    println("  bandpass: $(bandpass_label)")
    apc_print_split_options(split_opts)
    println("  raw pairs discovered: $(length(raw_pairs))")
    println("  needed pairs with artifacts: $(length(available_needed)) / $(length(needed_pairs))")
    println("  globals included: global mean (causal+acausal), global causal, global acausal")
    allocation_pair = apc_reference_allocation_pair(String(opts["reference-triplet"]))
    for cand in candidates
        println("  candidate: ", cand.display_label)
    end

    if dry_run
        println("Dry run only. Expected outputs:")
        for name in ("phase_reference_sweep_rows.csv", "phase_reference_sweep_summary.csv",
                "phase_reference_sweep_argmin.csv", "phase_reference_sweep_argmin_mean.csv",
                "phase_reference_sweep_reference_errorbars.csv",
                "reference_pair_averaged_crosscorrelations.csv",
                "reference_triplet_phase_period_plot_rows.csv",
                "plots/reference_triplet_phase_abs_vabc_minus_vac.html",
                "plots/reference_triplet_phase_abs_vabc_minus_vac.png",
                "plots/reference_phase_correction_sweep.html", "plots/reference_phase_correction_sweep.png",
                "plots/reference_phase_correction_errorbars.html", "plots/reference_phase_correction_errorbars.png",
                "plots/reference_pair_averaged_crosscorrelations.html",
                "plots/reference_pair_averaged_crosscorrelations.png")
            println("  ", joinpath(output_dir, name))
        end
        return 0
    end

    activate_ftan_project!()
    items = gv_load_items_for_pairs(runs, needed_pairs, bp, dt)
    apc_print_candidate_time_window_percentages(candidates, items; pair_filter=allocation_pair)
    waveform_df = apc_reference_pair_waveform_dataframe(candidates, items, allocation_pair, split_opts, dt)
    global_items = collect(values(latest_trained_items_by_pair(items)))
    period_rows, summary_rows = apc_pcs_run_sweep(triplets, items, global_items, candidates, denominators, cfg_base, split_opts, dt)
    rows_df = pcs_rows_dataframe(period_rows)
    apc_add_split_columns!(rows_df, split_opts)
    summary_df = isempty(summary_rows) ? DataFrame() : DataFrame(summary_rows)
    apc_add_split_columns!(summary_df, split_opts)
    argmin_df = pcs_argmin_dataframe(summary_df)
    apc_add_split_columns!(argmin_df, split_opts)
    agg_df = pcs_aggregate_argmin_dataframe(argmin_df)
    apc_add_split_columns!(agg_df, split_opts)
    best_d_df = pcs_best_d_by_state_dataframe(summary_df)
    reference_errorbar_df = pcs_reference_errorbar_dataframe(summary_df, String(opts["reference-triplet"]))
    apc_add_split_columns!(reference_errorbar_df, split_opts)
    reference_plot_df = pcs_reference_period_plot_dataframe(rows_df, argmin_df, String(opts["reference-triplet"]);
        global_denominator=4.0)
    apc_add_split_columns!(reference_plot_df, split_opts)
    pcs_print_argmin(agg_df)
    pcs_print_best_d_by_state(summary_df)

    mkpath(output_dir)
    plots_dir = joinpath(output_dir, "plots")
    write_csv(joinpath(output_dir, "phase_reference_sweep_rows.csv"), rows_df; overwrite)
    write_csv(joinpath(output_dir, "phase_reference_sweep_summary.csv"), summary_df; overwrite)
    write_csv(joinpath(output_dir, "phase_reference_sweep_argmin.csv"), argmin_df; overwrite)
    write_csv(joinpath(output_dir, "phase_reference_sweep_argmin_mean.csv"), agg_df; overwrite)
    write_csv(joinpath(output_dir, "phase_reference_sweep_reference_errorbars.csv"), reference_errorbar_df; overwrite)
    write_csv(joinpath(output_dir, "reference_pair_averaged_crosscorrelations.csv"), waveform_df; overwrite)
    write_csv(joinpath(output_dir, "reference_triplet_phase_period_plot_rows.csv"), reference_plot_df; overwrite)
    if isempty(waveform_df) || !(:legend_label in Symbol.(names(waveform_df)))
        @warn "No reference-pair averaged waveform rows were produced; skipping waveform plot" pair=allocation_pair
    else
        apc_write_reference_pair_waveforms_html(joinpath(plots_dir, "reference_pair_averaged_crosscorrelations.html"),
            waveform_df; overwrite)
        wf_png_path = apc_write_reference_pair_waveforms_png(joinpath(plots_dir, "reference_pair_averaged_crosscorrelations.png"),
            waveform_df; overwrite)
        show_plot && pcs_show_png(wf_png_path)
    end
    if isempty(argmin_df) || !(:curve_label in Symbol.(names(argmin_df)))
        @warn "No valid phase argmin rows were produced; skipping argmin plots"
    else
        pcs_write_argmin_html(joinpath(plots_dir, "reference_phase_correction_sweep.html"), argmin_df, agg_df; overwrite, best_df=best_d_df)
        png_path = pcs_write_argmin_png(joinpath(plots_dir, "reference_phase_correction_sweep.png"), argmin_df, agg_df; overwrite, best_df=best_d_df)
        show_plot && pcs_show_png(png_path)
    end
    if isempty(reference_errorbar_df) || !(:curve_label in Symbol.(names(reference_errorbar_df)))
        @warn "No reference triplet errorbar rows were produced; skipping reference errorbar plots" reference_triplet=String(opts["reference-triplet"])
    else
        pcs_write_reference_errorbar_html(joinpath(plots_dir, "reference_phase_correction_errorbars.html"),
            reference_errorbar_df, String(opts["reference-triplet"]); overwrite, best_df=best_d_df)
        err_png_path = pcs_write_reference_errorbar_png(joinpath(plots_dir, "reference_phase_correction_errorbars.png"),
            reference_errorbar_df, String(opts["reference-triplet"]); overwrite, best_df=best_d_df)
        show_plot && pcs_show_png(err_png_path)
    end
    if isempty(reference_plot_df)
        @warn "No reference triplet period plot rows were produced" reference_triplet=String(opts["reference-triplet"])
    else
        pcs_write_reference_period_html(joinpath(plots_dir, "reference_triplet_phase_abs_vabc_minus_vac.html"),
            reference_plot_df, String(opts["reference-triplet"]); overwrite)
        ref_png_path = pcs_write_reference_period_png(joinpath(plots_dir, "reference_triplet_phase_abs_vabc_minus_vac.png"),
            reference_plot_df, String(opts["reference-triplet"]); overwrite)
        show_plot && pcs_show_png(ref_png_path)
    end

    println()
    println("Wrote:")
    for name in ("phase_reference_sweep_rows.csv", "phase_reference_sweep_summary.csv",
            "phase_reference_sweep_argmin.csv", "phase_reference_sweep_argmin_mean.csv",
            "phase_reference_sweep_reference_errorbars.csv",
            "reference_pair_averaged_crosscorrelations.csv",
            "reference_triplet_phase_period_plot_rows.csv")
        println("  ", joinpath(output_dir, name))
    end
    if !isempty(waveform_df) && (:legend_label in Symbol.(names(waveform_df)))
        println("  ", joinpath(plots_dir, "reference_pair_averaged_crosscorrelations.html"))
        println("  ", joinpath(plots_dir, "reference_pair_averaged_crosscorrelations.png"))
    end
    if !isempty(argmin_df) && (:curve_label in Symbol.(names(argmin_df)))
        println("  ", joinpath(plots_dir, "reference_phase_correction_sweep.html"))
        println("  ", joinpath(plots_dir, "reference_phase_correction_sweep.png"))
    end
    if !isempty(reference_errorbar_df) && (:curve_label in Symbol.(names(reference_errorbar_df)))
        println("  ", joinpath(plots_dir, "reference_phase_correction_errorbars.html"))
        println("  ", joinpath(plots_dir, "reference_phase_correction_errorbars.png"))
    end
    if !isempty(reference_plot_df)
        println("  ", joinpath(plots_dir, "reference_triplet_phase_abs_vabc_minus_vac.html"))
        println("  ", joinpath(plots_dir, "reference_triplet_phase_abs_vabc_minus_vac.png"))
    end
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(pcs_apc_main())
    catch err
        showerror(stderr, err)
        println(stderr)
        exit(1)
    end
end
