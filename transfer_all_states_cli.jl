#!/usr/bin/env julia

using JLD2
using Statistics

const SUMMARY_COLUMNS = [
    "status", "reference_pair", "target_pair", "reference_run_tag",
    "reference_artifact_path", "target_raw_path",
    "n_reference_windows", "n_target_windows", "n_matched_windows",
    "n_missing_reference_windows",
    "n_reference_causal_windows", "n_reference_acausal_windows",
    "n_target_causal_windows", "n_target_acausal_windows",
    "n_matched_causal_windows", "n_matched_acausal_windows",
    "n_causal_reference_assignments", "n_acausal_reference_assignments",
    "n_causal_matched_assignments", "n_acausal_matched_assignments",
    "output_path", "error",
]

const MATCHED_COLUMNS = [
    "reference_pair", "target_pair", "reference_run_tag", "timestamp_label",
    "reference_index", "target_index", "reference_header", "target_header",
    "source_state_c", "source_state_ac", "stage1_c", "stage1_ac",
    "stage2_c", "stage2_ac",
]

const MISSING_COLUMNS = [
    "reference_pair", "target_pair", "reference_run_tag", "timestamp_label",
    "reference_index", "reference_header",
]

function usage()
    println("""
    Usage:
      julia transfer_all_states_cli.jl --saved-root DIR --reference-pair SN43-SN63 \\
          --raw-data-dir DIR --output-dir DIR [options]

    Required:
      --saved-root DIR        Root containing <reference_pair>/<run>/source_state_averages.jld2
      --reference-pair PAIR   Reference trained pair, e.g. SN43-SN63 or SN43_SN63
      --raw-data-dir DIR      Directory containing raw pair JLD2 files named like SN43_SN57-*.jld2
      --output-dir DIR        Destination folder for transferred artifacts and CSV audits

    Options:
      --target-pairs LIST     Comma/space-separated target pairs. Default: all raw pairs except source
      --min-overlap N         Minimum matched timestamps required per target. Default: 1
      --overwrite             Replace existing artifacts and recreate audit CSVs
      --dry-run               Print discovered work without writing files
      --help                  Show this message
    """)
end

function parse_args(argv)
    opts = Dict{String,Any}(
        "target-pairs" => "",
        "min-overlap" => 1,
        "overwrite" => false,
        "dry-run" => false,
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--help" || arg == "-h"
            opts["help"] = true
            i += 1
        elseif arg in ("--overwrite", "--dry-run")
            opts[arg[3:end]] = true
            i += 1
        elseif startswith(arg, "--")
            key = arg[3:end]
            i == length(argv) && error("Missing value for $(arg)")
            value = argv[i + 1]
            if key == "min-overlap"
                parsed = tryparse(Int, value)
                isnothing(parsed) && error("--min-overlap must be an integer; got $(value)")
                opts[key] = parsed
            else
                opts[key] = value
            end
            i += 2
        else
            error("Unexpected argument: $(arg)")
        end
    end
    opts
end

function require_arg(opts, key)
    value = get(opts, key, "")
    isempty(strip(String(value))) && error("Missing required --$(key)")
    String(value)
end

function parse_seed_timestamp(run_dir::String)
    name = basename(run_dir)
    m = match(r"^seed([0-9]+)_(.+)$", name)
    isnothing(m) && return (; seed=missing, timestamp=name)
    (; seed=parse(Int, m.captures[1]), timestamp=m.captures[2])
end

function parse_pair_label(s::AbstractString)
    parts = occursin("-", String(s)) ? split(String(s), "-"; limit=2) :
        split(String(s), "_"; limit=2)
    length(parts) == 2 || error("Pair must look like SN43-SN63 or SN43_SN63; got $(s)")
    (String(strip(parts[1])), String(strip(parts[2])))
end

pair_dir_label(pair::Tuple{String,String}) = "$(pair[1])_$(pair[2])"
pair_display_label(pair::Tuple{String,String}) = "$(pair[1])-$(pair[2])"

function header_time_label(header::AbstractString)
    m = match(r"^(\d{4})\.(\d{3})\.(\d{4})\.(\d{4})", String(header))
    isnothing(m) ? String(header) : join(m.captures, ".")
end

function safe_filename(s::AbstractString)
    cleaned = replace(strip(String(s)), r"[^A-Za-z0-9_.=-]+" => "_")
    isempty(cleaned) ? "none" : cleaned
end

function data_file_for_pair(pair::Tuple{String,String}, data_dir::String)
    prefix = pair_dir_label(pair)
    files = sort(filter(readdir(data_dir, join=true)) do path
        startswith(basename(path), "$(prefix)-") && endswith(path, ".jld2")
    end)
    isempty(files) && error("No raw JLD2 file found for $(prefix) in $(data_dir)")
    first(files)
end

function list_raw_pairs(data_dir::String)
    isdir(data_dir) || return Tuple{String,String}[]
    pairs = Set{Tuple{String,String}}()
    for file in readdir(data_dir)
        m = match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)-", file)
        isnothing(m) || push!(pairs, (m.captures[1], m.captures[2]))
    end
    sort(collect(pairs), by=p -> (p[1], p[2]))
end

function parse_target_pairs(text::AbstractString, data_dir::String, reference_pair::Tuple{String,String})
    raw = strip(String(text))
    pairs = lowercase(raw) == "all" || isempty(raw) ? list_raw_pairs(data_dir) :
        [parse_pair_label(part) for part in split(raw, r"[, ]+") if !isempty(strip(part))]
    [p for p in pairs if p != reference_pair]
end

function normalise_cols(X::AbstractMatrix)
    Y = Float32.(X)
    scale = sqrt.(sum(abs2, Y; dims=1))
    scale .= max.(scale, eps(Float32))
    Y ./ scale
end

function split_causal_acausal(X::AbstractMatrix)
    nt = size(X, 1)
    isodd(nt) || error("Expected odd nt for zero-lag centered correlations, got $(nt)")
    center = div(nt + 1, 2)
    ac = reverse(X[1:center-1, :], dims=1)
    c = X[center+1:end, :]
    vcat(zeros(Float32, 1, size(X, 2)), Float32.(ac)),
        vcat(zeros(Float32, 1, size(X, 2)), Float32.(c))
end

function preprocess_pair(pair::Tuple{String,String}, data_dir::String)
    path = data_file_for_pair(pair, data_dir)
    d = load(path)
    X = normalise_cols(d["correlations"])
    ac0, c0 = split_causal_acausal(X)
    ac = normalise_cols(ac0[2:end, :])
    c = normalise_cols(c0[2:end, :])
    raw_headers = String.(d["headers"])
    nwin = minimum((length(raw_headers), size(ac, 2), size(c, 2)))
    (; pair, path, headers=raw_headers[1:nwin],
        acausal=ac[:, 1:nwin], causal=c[:, 1:nwin],
        distance=Float64(d["dist"]),
        latitudes=haskey(d, "latitudes") ? Float64.(d["latitudes"]) : Float64[],
        longitudes=haskey(d, "longitudes") ? Float64.(d["longitudes"]) : Float64[])
end

function reference_artifacts(saved_root::AbstractString, reference_pair::Tuple{String,String})
    pair_dir = joinpath(String(saved_root), pair_dir_label(reference_pair))
    isdir(pair_dir) || return String[]
    sort([joinpath(run_dir, "source_state_averages.jld2")
        for run_dir in filter(isdir, readdir(pair_dir, join=true))
        if isfile(joinpath(run_dir, "source_state_averages.jld2"))])
end

function load_reference_transfer(path::String)
    d = load(path)
    required = ("window_time_labels", "source_state_ac", "source_state_c",
        "stage_assignments_ac", "stage_assignments_c", "combo_labels")
    for key in required
        haskey(d, key) || error("Reference artifact $(path) is missing key $(key)")
    end
    n = minimum((length(d["window_time_labels"]), length(d["source_state_ac"]), length(d["source_state_c"])))
    run_dir = dirname(path)
    parsed = parse_seed_timestamp(run_dir)
    pair = haskey(d, "pair") ? Tuple(String.(d["pair"])) : parse_pair_label(basename(dirname(run_dir)))
    (; artifact_path=path, run_dir, run_tag=basename(run_dir),
        seed=parsed.seed, timestamp=parsed.timestamp, pair,
        analysis_settings=haskey(d, "analysis_settings") ? d["analysis_settings"] : (; period_min=3.0, period_max=10.0),
        combo_labels=String.(d["combo_labels"]),
        window_headers=haskey(d, "window_headers") ? String.(d["window_headers"])[1:n] : String.(d["window_time_labels"])[1:n],
        window_time_labels=String.(d["window_time_labels"])[1:n],
        source_state_ac=Int.(d["source_state_ac"])[1:n],
        source_state_c=Int.(d["source_state_c"])[1:n],
        stage_assignments_ac=Int.(d["stage_assignments_ac"][:, 1:n]),
        stage_assignments_c=Int.(d["stage_assignments_c"][:, 1:n]),
        n_reference_windows=n)
end

function match_by_time_labels(ref, headers::AbstractVector{<:AbstractString})
    by_label = Dict{String,Int}()
    target_labels = header_time_label.(headers)
    for (i, label) in enumerate(target_labels)
        haskey(by_label, label) || (by_label[label] = i)
    end
    ref_inds = Int[]; target_inds = Int[]; labels = String[]
    missing_inds = Int[]; missing_labels = String[]
    for (i, label) in enumerate(header_time_label.(ref.window_time_labels))
        j = get(by_label, label, 0)
        if j == 0
            push!(missing_inds, i)
            push!(missing_labels, label)
        else
            push!(ref_inds, i)
            push!(target_inds, j)
            push!(labels, label)
        end
    end
    (; ref_inds, target_inds, labels, missing_inds, missing_labels)
end

function state_averages(X::AbstractMatrix{Float32}, states::AbstractVector{<:Integer}, nstates::Int)
    out = zeros(Float32, size(X, 1), max(nstates, 0))
    counts = zeros(Int, max(nstates, 0))
    for j in 1:min(size(X, 2), length(states))
        k = Int(states[j])
        1 <= k <= nstates || continue
        out[:, k] .+= X[:, j]
        counts[k] += 1
    end
    for k in 1:nstates
        counts[k] > 0 && (out[:, k] ./= counts[k])
    end
    (; averages=out, counts)
end

function stage_averages(X::AbstractMatrix{Float32}, stages::AbstractMatrix{<:Integer}, stage::Int)
    size(stages, 1) >= stage || return (; averages=zeros(Float32, size(X, 1), 0), counts=Int[])
    labels = Int.(stages[stage, :])
    isempty(labels) && return (; averages=zeros(Float32, size(X, 1), 0), counts=Int[])
    kmax = maximum(labels)
    kmax <= 0 && return (; averages=zeros(Float32, size(X, 1), 0), counts=Int[])
    state_averages(X, labels, kmax)
end

function transfer_full_state_artifact(ref, target, output_root::AbstractString;
        min_overlap::Int=1, overwrite::Bool=false)
    match = match_by_time_labels(ref, target.headers)
    nmatched = length(match.target_inds)
    out_dir = joinpath(String(output_root), pair_dir_label(target.pair), ref.run_tag)
    out_path = joinpath(out_dir, "transferred_source_state_averages.jld2")
    if isfile(out_path) && !overwrite
        return (; status="skipped", ref, target, match, output_path=out_path,
            error="output exists; pass --overwrite to replace")
    end
    if nmatched < min_overlap
        error("Only $(nmatched) matched windows for $(pair_display_label(target.pair)); min_overlap=$(min_overlap)")
    end

    nstates = length(ref.combo_labels)
    Xac = target.acausal[:, match.target_inds]
    Xc = target.causal[:, match.target_inds]
    global_avg_ac = Float32.(vec(mean(target.acausal; dims=2)))
    global_avg_c = Float32.(vec(mean(target.causal; dims=2)))
    states_ac = ref.source_state_ac[match.ref_inds]
    states_c = ref.source_state_c[match.ref_inds]
    ac = state_averages(Xac, states_ac, nstates)
    c = state_averages(Xc, states_c, nstates)
    stage_ac = ref.stage_assignments_ac[:, match.ref_inds]
    stage_c = ref.stage_assignments_c[:, match.ref_inds]
    s1_ac = stage_averages(Xac, stage_ac, 1)
    s1_c = stage_averages(Xc, stage_c, 1)
    s2_ac = stage_averages(Xac, stage_ac, 2)
    s2_c = stage_averages(Xc, stage_c, 2)
    headers = target.headers[match.target_inds]
    mkpath(out_dir)
    jldsave(out_path;
        acausal=Float32.(ac.averages),
        causal=Float32.(c.averages),
        counts_ac=Int.(ac.counts),
        counts_c=Int.(c.counts),
        combo_labels=ref.combo_labels,
        marginal_stage1_ac=Float32.(s1_ac.averages),
        marginal_stage1_c=Float32.(s1_c.averages),
        marginal_stage2_ac=Float32.(s2_ac.averages),
        marginal_stage2_c=Float32.(s2_c.averages),
        marginal_stage1_labels=["s1=$(k)" for k in 1:size(s1_ac.averages, 2)],
        marginal_stage2_labels=["s2=$(k)" for k in 1:size(s2_ac.averages, 2)],
        marginal_stage1_counts_ac=Int.(s1_ac.counts),
        marginal_stage1_counts_c=Int.(s1_c.counts),
        marginal_stage2_counts_ac=Int.(s2_ac.counts),
        marginal_stage2_counts_c=Int.(s2_c.counts),
        global_avg_ac=global_avg_ac,
        global_avg_c=global_avg_c,
        window_headers=headers,
        window_time_labels=match.labels,
        reference_window_headers=ref.window_headers[match.ref_inds],
        reference_window_time_labels=ref.window_time_labels[match.ref_inds],
        source_state_ac=states_ac,
        source_state_c=states_c,
        stage_assignments_ac=stage_ac,
        stage_assignments_c=stage_c,
        analysis_settings=ref.analysis_settings,
        distance=target.distance,
        latitudes=target.latitudes,
        longitudes=target.longitudes,
        pair=target.pair,
        target_pair=target.pair,
        reference_pair=ref.pair,
        selected_from_pair=ref.pair,
        reference_run_dir=ref.run_dir,
        reference_artifact_path=ref.artifact_path,
        reference_run_tag=ref.run_tag,
        n_reference_windows=ref.n_reference_windows,
        n_target_windows=length(target.headers),
        n_reference_causal_windows=length(ref.source_state_c),
        n_reference_acausal_windows=length(ref.source_state_ac),
        n_target_causal_windows=size(target.causal, 2),
        n_target_acausal_windows=size(target.acausal, 2),
        n_matched_windows=nmatched,
        n_matched_causal_windows=length(states_c),
        n_matched_acausal_windows=length(states_ac),
        n_missing_reference_windows=length(match.missing_labels),
        missing_reference_time_labels=match.missing_labels,
        missing_reference_indices=match.missing_inds,
        match_mode="exact_time_label",
        artifact_kind="transferred_source_state_averages")
    (; status="ok", ref, target, match, output_path=out_path, error="")
end

csv_escape(x) = begin
    s = x === missing ? "" : String(x)
    occursin(r"[,\n\"]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
end

function append_csv(path::AbstractString, columns::Vector{String}, rows::Vector{<:AbstractDict})
    isempty(rows) && return
    write_header = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        write_header && println(io, join(columns, ","))
        for row in rows
            println(io, join([csv_escape(get(row, col, "")) for col in columns], ","))
        end
    end
end

function summary_row(result)
    ref = result.ref
    target = result.target
    match = result.match
    Dict(
        "status" => result.status,
        "reference_pair" => pair_display_label(ref.pair),
        "target_pair" => pair_display_label(target.pair),
        "reference_run_tag" => ref.run_tag,
        "reference_artifact_path" => ref.artifact_path,
        "target_raw_path" => target.path,
        "n_reference_windows" => string(ref.n_reference_windows),
        "n_target_windows" => string(length(target.headers)),
        "n_matched_windows" => string(length(match.target_inds)),
        "n_missing_reference_windows" => string(length(match.missing_labels)),
        "n_reference_causal_windows" => string(length(ref.source_state_c)),
        "n_reference_acausal_windows" => string(length(ref.source_state_ac)),
        "n_target_causal_windows" => string(size(target.causal, 2)),
        "n_target_acausal_windows" => string(size(target.acausal, 2)),
        "n_matched_causal_windows" => string(length(match.ref_inds)),
        "n_matched_acausal_windows" => string(length(match.ref_inds)),
        "n_causal_reference_assignments" => string(length(ref.source_state_c)),
        "n_acausal_reference_assignments" => string(length(ref.source_state_ac)),
        "n_causal_matched_assignments" => string(length(match.ref_inds)),
        "n_acausal_matched_assignments" => string(length(match.ref_inds)),
        "output_path" => result.output_path,
        "error" => result.error,
    )
end

function error_summary_row(ref, target_pair, target_raw_path, output_path, err)
    Dict(
        "status" => "error",
        "reference_pair" => pair_display_label(ref.pair),
        "target_pair" => pair_display_label(target_pair),
        "reference_run_tag" => ref.run_tag,
        "reference_artifact_path" => ref.artifact_path,
        "target_raw_path" => target_raw_path,
        "n_reference_windows" => string(ref.n_reference_windows),
        "n_target_windows" => "",
        "n_matched_windows" => "0",
        "n_missing_reference_windows" => "",
        "n_reference_causal_windows" => string(length(ref.source_state_c)),
        "n_reference_acausal_windows" => string(length(ref.source_state_ac)),
        "n_target_causal_windows" => "",
        "n_target_acausal_windows" => "",
        "n_matched_causal_windows" => "0",
        "n_matched_acausal_windows" => "0",
        "n_causal_reference_assignments" => string(length(ref.source_state_c)),
        "n_acausal_reference_assignments" => string(length(ref.source_state_ac)),
        "n_causal_matched_assignments" => "0",
        "n_acausal_matched_assignments" => "0",
        "output_path" => output_path,
        "error" => sprint(showerror, err),
    )
end

function matched_rows(result)
    result.status == "ok" || return Dict{String,String}[]
    ref = result.ref
    target = result.target
    match = result.match
    rows = Dict{String,String}[]
    for k in eachindex(match.ref_inds)
        ri = match.ref_inds[k]
        ti = match.target_inds[k]
        push!(rows, Dict(
            "reference_pair" => pair_display_label(ref.pair),
            "target_pair" => pair_display_label(target.pair),
            "reference_run_tag" => ref.run_tag,
            "timestamp_label" => match.labels[k],
            "reference_index" => string(ri),
            "target_index" => string(ti),
            "reference_header" => ref.window_headers[ri],
            "target_header" => target.headers[ti],
            "source_state_c" => string(ref.source_state_c[ri]),
            "source_state_ac" => string(ref.source_state_ac[ri]),
            "stage1_c" => size(ref.stage_assignments_c, 1) >= 1 ? string(ref.stage_assignments_c[1, ri]) : "",
            "stage1_ac" => size(ref.stage_assignments_ac, 1) >= 1 ? string(ref.stage_assignments_ac[1, ri]) : "",
            "stage2_c" => size(ref.stage_assignments_c, 1) >= 2 ? string(ref.stage_assignments_c[2, ri]) : "",
            "stage2_ac" => size(ref.stage_assignments_ac, 1) >= 2 ? string(ref.stage_assignments_ac[2, ri]) : "",
        ))
    end
    rows
end

function missing_rows(result)
    result.status == "ok" || return Dict{String,String}[]
    ref = result.ref
    target = result.target
    match = result.match
    rows = Dict{String,String}[]
    for k in eachindex(match.missing_inds)
        ri = match.missing_inds[k]
        push!(rows, Dict(
            "reference_pair" => pair_display_label(ref.pair),
            "target_pair" => pair_display_label(target.pair),
            "reference_run_tag" => ref.run_tag,
            "timestamp_label" => match.missing_labels[k],
            "reference_index" => string(ri),
            "reference_header" => ref.window_headers[ri],
        ))
    end
    rows
end

function reset_csvs(output_dir::AbstractString)
    for name in ("transfer_summary.csv", "matched_windows.csv", "missing_reference_windows.csv")
        path = joinpath(String(output_dir), name)
        isfile(path) && rm(path)
    end
end

function main(argv=ARGS)
    opts = parse_args(argv)
    if get(opts, "help", false)
        usage()
        return 0
    end
    saved_root = require_arg(opts, "saved-root")
    raw_data_dir = require_arg(opts, "raw-data-dir")
    output_dir = require_arg(opts, "output-dir")
    reference_pair = parse_pair_label(require_arg(opts, "reference-pair"))
    min_overlap = Int(opts["min-overlap"])
    min_overlap >= 0 || error("--min-overlap must be >= 0")
    overwrite = Bool(opts["overwrite"])
    dry_run = Bool(opts["dry-run"])

    isdir(saved_root) || error("--saved-root does not exist: $(saved_root)")
    isdir(raw_data_dir) || error("--raw-data-dir does not exist: $(raw_data_dir)")

    refs = reference_artifacts(saved_root, reference_pair)
    isempty(refs) && error("No reference artifacts found under $(joinpath(saved_root, pair_dir_label(reference_pair)))")
    targets = parse_target_pairs(String(opts["target-pairs"]), raw_data_dir, reference_pair)
    isempty(targets) && error("No target pairs selected/found in $(raw_data_dir)")

    if dry_run
        println("Dry run: no files will be written.")
        println("Reference pair: $(pair_display_label(reference_pair))")
        println("Reference runs: $(length(refs))")
        for ref in refs
            println("  - $(ref)")
        end
        println("Target pairs: $(length(targets))")
        for pair in targets
            println("  - $(pair_display_label(pair)) -> $(joinpath(output_dir, pair_dir_label(pair), "<reference_run_tag>", "transferred_source_state_averages.jld2"))")
        end
        return 0
    end

    mkpath(output_dir)
    overwrite && reset_csvs(output_dir)
    summary_path = joinpath(output_dir, "transfer_summary.csv")
    matched_path = joinpath(output_dir, "matched_windows.csv")
    missing_path = joinpath(output_dir, "missing_reference_windows.csv")

    completed = 0
    skipped = 0
    failed = 0
    for artifact in refs
        ref = try
            load_reference_transfer(artifact)
        catch err
            failed += length(targets)
            rows = Dict{String,String}[]
            for pair in targets
                out_path = joinpath(output_dir, pair_dir_label(pair), basename(dirname(artifact)), "transferred_source_state_averages.jld2")
                push!(rows, Dict(
                    "status" => "error",
                    "reference_pair" => pair_display_label(reference_pair),
                    "target_pair" => pair_display_label(pair),
                    "reference_run_tag" => basename(dirname(artifact)),
                    "reference_artifact_path" => artifact,
                    "target_raw_path" => "",
                    "output_path" => out_path,
                    "error" => sprint(showerror, err),
                ))
            end
            append_csv(summary_path, SUMMARY_COLUMNS, rows)
            continue
        end

        for pair in targets
            out_path = joinpath(output_dir, pair_dir_label(pair), ref.run_tag, "transferred_source_state_averages.jld2")
            try
                target = preprocess_pair(pair, raw_data_dir)
                result = transfer_full_state_artifact(ref, target, output_dir;
                    min_overlap, overwrite)
                result.status == "ok" && (completed += 1)
                result.status == "skipped" && (skipped += 1)
                append_csv(summary_path, SUMMARY_COLUMNS, [summary_row(result)])
                append_csv(matched_path, MATCHED_COLUMNS, matched_rows(result))
                append_csv(missing_path, MISSING_COLUMNS, missing_rows(result))
                println("$(result.status): $(pair_display_label(ref.pair)) $(ref.run_tag) -> $(pair_display_label(pair)) ($(length(result.match.target_inds)) matched)")
            catch err
                failed += 1
                target_path = try
                    data_file_for_pair(pair, raw_data_dir)
                catch
                    ""
                end
                row = error_summary_row(ref, pair, target_path, out_path, err)
                append_csv(summary_path, SUMMARY_COLUMNS, [row])
                println("error: $(pair_display_label(ref.pair)) $(ref.run_tag) -> $(pair_display_label(pair)): $(sprint(showerror, err))")
            end
        end
    end

    println("Done. completed=$(completed), skipped=$(skipped), failed=$(failed)")
    println("Summary CSV: $(summary_path)")
    println("Matched windows CSV: $(matched_path)")
    println("Missing reference windows CSV: $(missing_path)")
    completed > 0 || skipped > 0 ? 0 : 2
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch err
        println(stderr, "ERROR: $(sprint(showerror, err))")
        exit(1)
    end
end
