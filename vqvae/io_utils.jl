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
