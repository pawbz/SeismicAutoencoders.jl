# I/O utilities for file scanning, pair discovery, and data loading
# NO heavy dependencies (no Lux, Reactant, CUDA, Enzyme)

"""
    list_station_pairs(filepath::String)

Scan a directory for .jld2 files matching the pattern `STATION_STATION*.jld2`.
Extract station pair names (uppercase) and return sorted list of unique pairs.

# Arguments
- `filepath::String`: Directory path containing JLD2 files

# Returns
Vector of tuples (sta1, sta2) sorted by station name, e.g., [("AP", "BK"), ("SM17", "SM42")]
"""
function list_station_pairs(filepath::String)
    files = readdir(filepath)
    pairs = Set{Tuple{String,String}}()
    for f in files
        endswith(f, ".jld2") || continue
        m = match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)", basename(f))
        m === nothing && continue
        sta1 = uppercase(m.captures[1])
        sta2 = uppercase(m.captures[2])
        push!(pairs, (sta1, sta2))
    end
    return sort!(collect(pairs), by=x -> (x[1], x[2]))
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
    if haskey(jldfile, "dist")
        return _jld2_first_scalar(jldfile["dist"], "dist")
    elseif haskey(jldfile, "Distances")
        return _jld2_first_scalar(jldfile["Distances"], "Distances")
    end
    return nothing
end

jld2_headers(jldfile) = haskey(jldfile, "headers") ? jldfile["headers"] : nothing
