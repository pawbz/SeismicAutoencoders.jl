# I/O utilities for file scanning, pair discovery, and data loading
# NO heavy dependencies (no Lux, Reactant, CUDA, Enzyme)

using HDF5

"""
    list_station_pairs(filepath::String)

Scan a directory for .jld2/.h5 files matching the pattern `STATION_STATION*`.
Extract station pair names (uppercase) and return sorted list of unique pairs.

# Arguments
- `filepath::String`: Directory path containing JLD2/HDF5 files

# Returns
Vector of tuples (sta1, sta2) sorted by station name, e.g., [("AP", "BK"), ("SM17", "SM42")]
"""
function list_station_pairs(filepath::String)
    files = readdir(filepath)
    pairs = Set{Tuple{String,String}}()
    for f in files
        (endswith(f, ".jld2") || endswith(f, ".h5")) || continue
        m = match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)", basename(f))
        m === nothing && continue
        sta1 = uppercase(m.captures[1])
        sta2 = uppercase(m.captures[2])
        push!(pairs, (sta1, sta2))
    end
    return sort!(collect(pairs), by=x -> (x[1], x[2]))
end

# Wraps eagerly-read HDF5 datasets/attributes so it exposes the same haskey/getindex
# interface JLD2.load(...)'s Dict return value has, letting jld2_correlations/
# jld2_distance/jld2_headers below work unchanged on either source. Built by fully
# reading the source HDF5.File and closing it immediately (see load_pair_file), so no
# file handle is held open past the load call.
struct H5PairFile
    data::Dict{String,Any}
end

Base.haskey(h5::H5PairFile, key::AbstractString) = haskey(h5.data, key)
Base.getindex(h5::H5PairFile, key::AbstractString) = h5.data[key]
Base.keys(h5::H5PairFile) = keys(h5.data)

"""
    load_pair_file(path::String)

Load a station-pair correlation file, dispatching on extension (`.jld2` or `.h5`).
Returns a dict-like object usable with `jld2_correlations`, `jld2_distance`, and
`jld2_headers`. HDF5 files are read fully and closed immediately (no lingering file
handle), since `list_station_pairs`-driven training loops may open hundreds to
thousands of files across a run.
"""
function load_pair_file(path::String)
    if endswith(path, ".h5")
        data = Dict{String,Any}()
        HDF5.h5open(path, "r") do file
            for key in keys(file)
                data[key] = read(file[key])
            end
            for key in keys(HDF5.attrs(file))
                data[key] = HDF5.attrs(file)[key]
            end
        end
        return H5PairFile(data)
    end
    return JLD2.load(path)
end

function jld2_key_list(jldfile)
    join(string.(collect(keys(jldfile))), ", ")
end

function _jld2_matrix_from_value(value, key::AbstractString)
    if value isa AbstractMatrix
        return value
    end
    if value isa AbstractArray && ndims(value) == 2
        return value
    end
    if applicable(first, value)
        first_value = try
            first(value)
        catch e
            error("JLD2 key \"$(key)\" exists but its first entry could not be read; got $(typeof(value)): $(e)")
        end
        if first_value isa AbstractMatrix || (first_value isa AbstractArray && ndims(first_value) == 2)
            return first_value
        end
        error("JLD2 key \"$(key)\" exists but is not a matrix and its first entry is not a matrix; got $(typeof(value)) with first entry $(typeof(first_value)).")
    end
    error("JLD2 key \"$(key)\" exists but cannot be interpreted as a waveform matrix; got $(typeof(value)).")
end

function jld2_correlations(jldfile)
    if jldfile isa H5PairFile
        # HDF5.jl reads the on-disk (nt, n_waveforms) dataset transposed as
        # (n_waveforms, nt); permute back to the JLD2 (nt, n_waveforms) convention.
        return permutedims(_jld2_matrix_from_value(jldfile["D"], "D"))
    end
    if haskey(jldfile, "correlations")
        return _jld2_matrix_from_value(jldfile["correlations"], "correlations")
    elseif haskey(jldfile, "D")
        return _jld2_matrix_from_value(jldfile["D"], "D")
    end
    error("Missing waveform data. Supported keys: \"correlations\" or \"D\". Available keys: $(jld2_key_list(jldfile)).")
end

function _jld2_first_scalar(value, key::AbstractString)
    if value isa Number
        return Float64(value)
    end
    if applicable(first, value)
        first_value = try
            first(value)
        catch e
            error("JLD2 key \"$(key)\" exists but its first entry could not be read; got $(typeof(value)): $(e)")
        end
        first_value isa Number && return Float64(first_value)
        error("JLD2 key \"$(key)\" exists but its first entry is not numeric; got $(typeof(first_value)).")
    end
    error("JLD2 key \"$(key)\" exists but cannot be interpreted as a scalar distance; got $(typeof(value)).")
end

function jld2_distance(jldfile)
    if jldfile isa H5PairFile
        return haskey(jldfile, "distance_km") ? _jld2_first_scalar(jldfile["distance_km"], "distance_km") : nothing
    end
    if haskey(jldfile, "dist")
        return _jld2_first_scalar(jldfile["dist"], "dist")
    elseif haskey(jldfile, "Distances")
        return _jld2_first_scalar(jldfile["Distances"], "Distances")
    end
    return nothing
end

jld2_headers(jldfile) = haskey(jldfile, "headers") ? jldfile["headers"] : nothing

"""
    jld2_latlon(jldfile)

Return `(latitudes, longitudes)` as 2-element vectors `[lat1, lat2]`/`[lon1, lon2]`,
or `(nothing, nothing)` if unavailable. For JLD2 files these come from the
`latitudes`/`longitudes` keys; for HDF5 files they are synthesized from the flat
`lat1`/`lon1`/`lat2`/`lon2` attributes.
"""
function jld2_latlon(jldfile)
    if jldfile isa H5PairFile
        if haskey(jldfile, "lat1") && haskey(jldfile, "lon1") && haskey(jldfile, "lat2") && haskey(jldfile, "lon2")
            lats = Float64[jldfile["lat1"], jldfile["lat2"]]
            lons = Float64[jldfile["lon1"], jldfile["lon2"]]
            return lats, lons
        end
        return nothing, nothing
    end
    lats = haskey(jldfile, "latitudes") ? Float64.(jldfile["latitudes"]) : nothing
    lons = haskey(jldfile, "longitudes") ? Float64.(jldfile["longitudes"]) : nothing
    return lats, lons
end

"""
    jld2_dt(jldfile)

Return the sample interval `dt` (seconds) recorded in the file, or `nothing` if not
present. Only HDF5 files carry this (as `sampling_rate`, in Hz — inverted here to
seconds); JLD2 files store no sampling-rate metadata. This is informational only and
does not override the `--dt` CLI flag used for training.
"""
function jld2_dt(jldfile)
    if jldfile isa H5PairFile && haskey(jldfile, "sampling_rate")
        return 1.0 / _jld2_first_scalar(jldfile["sampling_rate"], "sampling_rate")
    end
    return nothing
end
