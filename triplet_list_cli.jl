#!/usr/bin/env julia

using CSV
using DataFrames
using Printf

const DEFAULT_STATION_CSV = "/home/sanket/Desktop/Stations_XI_2011_13_SN.csv"
const DEFAULT_OUTPUT_CSV = joinpath(@__DIR__, "available_station_triplets.csv")

const DEFAULT_OPTIONS = Dict{String,Any}(
    "station-csv" => DEFAULT_STATION_CSV,
    "raw-data-dir" => "",
    "output" => DEFAULT_OUTPUT_CSV,
    "max-delta-az" => 0.5,
    "max-delta-d" => 0.1,
    "min-segment-distance" => 45.0,
    "min-ratio" => 0.0,
    "max-ratio" => Inf,
    "stations" => "",
    "all-triplets" => false,
    "include-all" => false,
    "require-raw-pairs" => false,
    "overwrite" => false,
)

const FLOAT_OPTION_KEYS = Set([
    "max-delta-az",
    "max-delta-d",
    "min-segment-distance",
    "min-ratio",
    "max-ratio",
])

function usage()
    println("""
    Usage:
      julia --startup-file=no triplet_list_cli.jl --station-csv FILE --output FILE [options]

    Writes a CSV of station triplets. By default, geometry criteria are applied.

    Options:
      --station-csv FILE            Station CSV with Station Code, Latitude, Longitude.
                                    Default: $(DEFAULT_STATION_CSV)
      --raw-data-dir DIR            Directory containing raw pair files named like SN43_SN63-*.jld2.
      --require-raw-pairs           Keep only triplets where AB, BC, and AC raw pair files exist.
      --output FILE                 Destination CSV. Default: $(DEFAULT_OUTPUT_CSV)
      --max-delta-az X              Maximum azimuth spread in degrees. Default: 0.5
      --max-delta-d X               Maximum abs distance-closure error in km. Default: 0.1
      --min-segment-distance X      Minimum Dab and Dbc segment distance in km. Default: 45
      --min-ratio X                 Minimum Dab/Dbc ratio. Default: 0
      --max-ratio X                 Maximum Dab/Dbc ratio. Default: Inf
      --stations LIST               Optional comma-separated station subset, e.g. SN43,SN44,SN63.
      --all-triplets                Do not filter by geometry criteria. With --require-raw-pairs,
                                    this writes all triplets available in the raw dataset.
      --include-all                 Write all computed triplets with passes_criteria instead of only passing rows.
      --overwrite                   Replace an existing output CSV.
      --help                        Show this message.
    """)
end

function parse_bool_flag!(opts, key::String)
    opts[key] = true
end

function parse_args(argv)
    opts = copy(DEFAULT_OPTIONS)
    i = 1
    while i <= length(argv)
        arg = String(argv[i])
        if arg == "--help" || arg == "-h"
            usage()
            exit(0)
        elseif arg == "--include-all"
            parse_bool_flag!(opts, "include-all")
            i += 1
        elseif arg == "--all-triplets"
            parse_bool_flag!(opts, "all-triplets")
            i += 1
        elseif arg == "--require-raw-pairs"
            parse_bool_flag!(opts, "require-raw-pairs")
            i += 1
        elseif arg == "--overwrite"
            parse_bool_flag!(opts, "overwrite")
            i += 1
        elseif startswith(arg, "--")
            key = String(arg[3:end])
            haskey(opts, key) || error("Unknown option $(arg). Use --help for usage.")
            i == length(argv) && error("Missing value for $(arg)")
            value = String(argv[i + 1])
            if key in FLOAT_OPTION_KEYS
                parsed = tryparse(Float64, value)
                isnothing(parsed) && error("$(arg) must be numeric; got $(value)")
                opts[key] = parsed
            else
                opts[key] = value
            end
            i += 2
        else
            error("Unexpected positional argument $(arg). Use --help for usage.")
        end
    end
    validate_options(opts)
    opts
end

function validate_options(opts)
    station_csv = String(opts["station-csv"])
    isfile(station_csv) || error("--station-csv not found: $(station_csv)")
    if Bool(opts["require-raw-pairs"])
        raw_data_dir = String(opts["raw-data-dir"])
        !isempty(strip(raw_data_dir)) || error("--require-raw-pairs needs --raw-data-dir")
        isdir(raw_data_dir) || error("--raw-data-dir not found: $(raw_data_dir)")
    elseif !isempty(strip(String(opts["raw-data-dir"])))
        isdir(String(opts["raw-data-dir"])) || error("--raw-data-dir not found: $(opts["raw-data-dir"])")
    end
    Float64(opts["max-delta-az"]) >= 0 || error("--max-delta-az must be non-negative")
    Float64(opts["max-delta-d"]) >= 0 || error("--max-delta-d must be non-negative")
    Float64(opts["min-segment-distance"]) >= 0 || error("--min-segment-distance must be non-negative")
    Float64(opts["min-ratio"]) >= 0 || error("--min-ratio must be non-negative")
    Float64(opts["max-ratio"]) >= Float64(opts["min-ratio"]) ||
        error("--max-ratio must be >= --min-ratio")
end

function station_column(df::DataFrame, candidates)
    nameset = Set(String.(names(df)))
    for name in candidates
        String(name) in nameset && return Symbol(name)
    end
    error("Station CSV must contain one of these columns: $(join(String.(candidates), ", "))")
end

function read_station_table(path::AbstractString)
    df = CSV.read(path, DataFrame)
    code_col = station_column(df, ["Station Code", "station", "station_code", "code", "Station"])
    lat_col = station_column(df, ["Latitude", "latitude", "lat"])
    lon_col = station_column(df, ["Longitude", "longitude", "lon", "long"])
    rows = NamedTuple[]
    seen = Set{String}()
    for row in eachrow(df)
        code = strip(String(row[code_col]))
        isempty(code) && continue
        code in seen && continue
        push!(seen, code)
        push!(rows, (; code, lat=Float64(row[lat_col]), lon=Float64(row[lon_col])))
    end
    isempty(rows) && error("No stations found in $(path)")
    DataFrame(rows)
end

deg2radf(x) = Float64(x) * pi / 180.0
rad2degf(x) = Float64(x) * 180.0 / pi

function haversine_km(lat1, lon1, lat2, lon2)
    r = 6371.0
    phi1, phi2 = deg2radf(lat1), deg2radf(lat2)
    dphi = deg2radf(Float64(lat2) - Float64(lat1))
    dlambda = deg2radf(Float64(lon2) - Float64(lon1))
    a = sin(dphi / 2)^2 + cos(phi1) * cos(phi2) * sin(dlambda / 2)^2
    2r * atan(sqrt(a), sqrt(max(0.0, 1.0 - a)))
end

function azimuth_deg(lat1, lon1, lat2, lon2)
    phi1, phi2 = deg2radf(lat1), deg2radf(lat2)
    dlambda = deg2radf(Float64(lon2) - Float64(lon1))
    y = sin(dlambda) * cos(phi2)
    x = cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(dlambda)
    mod(rad2degf(atan(y, x)) + 360.0, 360.0)
end

function azimuth_spread_deg(azimuths)
    vals = Float64[a for a in azimuths if isfinite(Float64(a))]
    length(vals) < 2 && return NaN
    maximum(min(abs(a - b), 360.0 - abs(a - b)) for a in vals for b in vals)
end

canonical_pair(a::AbstractString, b::AbstractString) = join(sort(String[String(a), String(b)]), "-")

function pair_labels_from_raw_dir(raw_data_dir::AbstractString)
    isempty(strip(String(raw_data_dir))) && return Set{String}()
    labels = Set{String}()
    for entry in readdir(raw_data_dir)
        m = match(r"^([A-Za-z0-9]+)[_-]([A-Za-z0-9]+)(?:[-_.].*)?$", entry)
        isnothing(m) && continue
        push!(labels, canonical_pair(m.captures[1], m.captures[2]))
    end
    labels
end

function station_subset(stations::DataFrame, spec::AbstractString)
    raw = strip(String(spec))
    isempty(raw) && return stations
    wanted = Set(String.(strip.(String.(split(raw, ",")))))
    isempty(wanted) && return stations
    selected = stations[in.(String.(stations.code), Ref(wanted)), :]
    missing_codes = sort(collect(setdiff(wanted, Set(String.(selected.code)))))
    isempty(missing_codes) || error("--stations contains code(s) not present in station CSV: $(join(missing_codes, ", "))")
    nrow(selected) >= 3 || error("At least three stations are required after --stations filtering")
    selected
end

function triplet_geometry(latlon, a::AbstractString, b::AbstractString, c::AbstractString)
    pa, pb, pc = latlon[String(a)], latlon[String(b)], latlon[String(c)]
    dab = haversine_km(pa.lat, pa.lon, pb.lat, pb.lon)
    dbc = haversine_km(pb.lat, pb.lon, pc.lat, pc.lon)
    dac = haversine_km(pa.lat, pa.lon, pc.lat, pc.lon)
    az_ab = azimuth_deg(pa.lat, pa.lon, pb.lat, pb.lon)
    az_bc = azimuth_deg(pb.lat, pb.lon, pc.lat, pc.lon)
    az_ac = azimuth_deg(pa.lat, pa.lon, pc.lat, pc.lon)
    delta_az = azimuth_spread_deg((az_ab, az_bc, az_ac))
    delta_d = dab + dbc - dac
    ratio = isfinite(dab) && isfinite(dbc) && dbc > 0.0 ? dab / dbc : NaN
    (; triplet="$(a)-$(b)-$(c)", station_a=String(a), station_b=String(b), station_c=String(c),
        pair_ab=canonical_pair(a, b), pair_bc=canonical_pair(b, c), pair_ac=canonical_pair(a, c),
        dab_km=dab, dbc_km=dbc, dac_km=dac,
        az_ab_deg=az_ab, az_bc_deg=az_bc, az_ac_deg=az_ac,
        delta_az_deg=delta_az, delta_d_km=delta_d,
        abs_delta_d_km=abs(delta_d), dab_dbc_ratio=ratio)
end

function ordered_triplet_candidates(a::AbstractString, b::AbstractString, c::AbstractString, latlon)
    [
        triplet_geometry(latlon, a, b, c),
        triplet_geometry(latlon, a, c, b),
        triplet_geometry(latlon, b, a, c),
        triplet_geometry(latlon, b, c, a),
        triplet_geometry(latlon, c, a, b),
        triplet_geometry(latlon, c, b, a),
    ]
end

function geometry_criteria_pass(row, opts)
    isfinite(row.delta_az_deg) &&
        isfinite(row.delta_d_km) &&
        isfinite(row.dab_km) &&
        isfinite(row.dbc_km) &&
        row.delta_az_deg <= Float64(opts["max-delta-az"]) &&
        row.abs_delta_d_km <= Float64(opts["max-delta-d"]) &&
        row.dab_km >= Float64(opts["min-segment-distance"]) &&
        row.dbc_km >= Float64(opts["min-segment-distance"]) &&
        row.dab_dbc_ratio >= Float64(opts["min-ratio"]) &&
        row.dab_dbc_ratio <= Float64(opts["max-ratio"])
end

function raw_pairs_pass(row, raw_pairs::Set{String}, opts)
    !Bool(opts["require-raw-pairs"]) ||
        (row.pair_ab in raw_pairs && row.pair_bc in raw_pairs && row.pair_ac in raw_pairs)
end

function selected_for_output(row, raw_pairs::Set{String}, opts)
    raw_ok = raw_pairs_pass(row, raw_pairs, opts)
    Bool(opts["all-triplets"]) ? raw_ok : (geometry_criteria_pass(row, opts) && raw_ok)
end

function ranked_triplet_dataframe(stations::DataFrame, raw_pairs::Set{String}, opts)
    latlon = Dict(String(row.code) => (; lat=Float64(row.lat), lon=Float64(row.lon)) for row in eachrow(stations))
    codes = sort(String.(stations.code))
    rows = NamedTuple[]
    for i in 1:(length(codes) - 2), j in (i + 1):(length(codes) - 1), k in (j + 1):length(codes)
        candidates = ordered_triplet_candidates(codes[i], codes[j], codes[k], latlon)
        scored = map(candidates) do row
            passes_geometry = geometry_criteria_pass(row, opts)
            passes_raw = raw_pairs_pass(row, raw_pairs, opts)
            selected = selected_for_output(row, raw_pairs, opts)
            raw_pair_count = count(pair -> pair in raw_pairs, (row.pair_ab, row.pair_bc, row.pair_ac))
            (; row..., raw_pair_count,
                has_pair_ab=row.pair_ab in raw_pairs,
                has_pair_bc=row.pair_bc in raw_pairs,
                has_pair_ac=row.pair_ac in raw_pairs,
                passes_criteria=passes_geometry,
                passes_raw_pairs=passes_raw,
                selected_for_output=selected,
                criterion_max_delta_az_deg=Float64(opts["max-delta-az"]),
                criterion_max_abs_delta_d_km=Float64(opts["max-delta-d"]),
                criterion_min_segment_distance_km=Float64(opts["min-segment-distance"]),
                criterion_min_dab_dbc_ratio=Float64(opts["min-ratio"]),
                criterion_max_dab_dbc_ratio=Float64(opts["max-ratio"]),
                criterion_apply_geometry=!Bool(opts["all-triplets"]),
                criterion_require_raw_pairs=Bool(opts["require-raw-pairs"]),
                source_station_csv=String(opts["station-csv"]),
                source_raw_data_dir=String(opts["raw-data-dir"]))
        end
        sort!(scored; by=r -> (
            !r.selected_for_output,
            isfinite(r.dab_dbc_ratio) ? abs(log(r.dab_dbc_ratio)) : Inf,
            isfinite(r.delta_az_deg) ? r.delta_az_deg : Inf,
            isfinite(r.abs_delta_d_km) ? r.abs_delta_d_km : Inf,
            -r.raw_pair_count,
            r.triplet))
        push!(rows, first(scored))
    end
    df = DataFrame(rows)
    isempty(df) && return df
    sort!(df, [:selected_for_output, :delta_az_deg, :abs_delta_d_km, :triplet], rev=[true, false, false, false])
    df.rank = collect(1:nrow(df))
    select!(df, [:rank, :triplet, :station_a, :station_b, :station_c,
        :pair_ab, :pair_bc, :pair_ac,
        :has_pair_ab, :has_pair_bc, :has_pair_ac, :raw_pair_count,
        :dab_km, :dbc_km, :dac_km,
        :az_ab_deg, :az_bc_deg, :az_ac_deg,
        :delta_az_deg, :delta_d_km, :abs_delta_d_km, :dab_dbc_ratio,
        :passes_criteria, :passes_raw_pairs, :selected_for_output,
        :criterion_max_delta_az_deg, :criterion_max_abs_delta_d_km,
        :criterion_min_segment_distance_km,
        :criterion_min_dab_dbc_ratio, :criterion_max_dab_dbc_ratio,
        :criterion_apply_geometry, :criterion_require_raw_pairs,
        :source_station_csv, :source_raw_data_dir])
    Bool(opts["include-all"]) ? df : df[Bool.(df.selected_for_output), :]
end

function write_csv_atomic(path::AbstractString, df::DataFrame; overwrite::Bool=false)
    output_path = abspath(path)
    if isfile(output_path) && !overwrite
        error("Output exists: $(output_path). Pass --overwrite to replace it.")
    end
    mkpath(dirname(output_path))
    tmp = joinpath(dirname(output_path), ".$(basename(output_path)).tmp")
    try
        CSV.write(tmp, df)
        Base.Filesystem.rename(tmp, output_path)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    output_path
end

function main(argv=ARGS)
    opts = parse_args(argv)
    stations = station_subset(read_station_table(String(opts["station-csv"])), String(opts["stations"]))
    raw_pairs = pair_labels_from_raw_dir(String(opts["raw-data-dir"]))
    df = ranked_triplet_dataframe(stations, raw_pairs, opts)
    output_path = write_csv_atomic(String(opts["output"]), df; overwrite=Bool(opts["overwrite"]))

    println("Wrote $(nrow(df)) triplets to $(output_path)")
    if Bool(opts["all-triplets"])
        println("Geometry criteria: not applied (--all-triplets)")
    else
        println(@sprintf("Criteria: delta_az <= %.6g deg, abs_delta_d <= %.6g km, Dab/Dbc >= %.6g, Dab/Dbc <= %.6g, Dab/Dbc segments >= %.6g km",
            Float64(opts["max-delta-az"]),
            Float64(opts["max-delta-d"]),
            Float64(opts["min-ratio"]),
            Float64(opts["max-ratio"]),
            Float64(opts["min-segment-distance"])))
    end
    !isempty(String(opts["raw-data-dir"])) &&
        println("Raw pairs discovered: $(length(raw_pairs)); require raw pairs: $(Bool(opts["require-raw-pairs"]))")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
