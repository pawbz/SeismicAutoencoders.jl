### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ a1000001-0000-0000-0000-000000000001
begin
    using 
        MLUtils,
               LinearAlgebra,
    Random, CUDA, BenchmarkTools
     
end

# ╔═╡ a1000003-0000-0000-0000-000000000003
md"## Helpers"

# ╔═╡ 59946319-e22b-41ba-b611-4eed145f3bc1
"""
Fake 2D data matrix: (nt, ntr)
"""
function fake_xcorr(nt=100, ntr=50)
    rand(Float32, nt, ntr)
end

# ╔═╡ 7a34f647-7bf1-4e7b-a9c1-a4c9fa6e8692
is_gpu_array(x) = x isa CUDA.CuArray

# ╔═╡ 0864cdc0-402d-4e2c-b6e9-32f4e9c20b71
"""
Local conditioning helper for benchmarking.
Supports:
  - nothing => unconditioned
  - matrix (cond_dim x ngroups)
"""
has_conditioning_local(::Nothing) = false

# ╔═╡ 91001f0c-9e90-4af4-8d1c-38deb49cdf2f
has_conditioning_local(::AbstractMatrix) = true

# ╔═╡ d4d6acd7-da42-484a-8c1c-7fc60e14923c
function has_conditioning_local(conditioning)
    throw(ArgumentError("conditioning must be nothing or an AbstractMatrix; got $(typeof(conditioning))"))
end

# ╔═╡ 12d4775e-1852-4689-9a25-6fef001d95c7
function get_group_condition_vector_local(conditioning::AbstractMatrix, group_index::Int)
    @views return Float32.(conditioning[:, group_index])
end

# ╔═╡ f9fe0998-0558-4121-b44e-e50d533af854
function validate_conditioning_local(conditioning, ngroups::Int)
    if conditioning === nothing
        return nothing
    end
    if !(conditioning isa AbstractMatrix)
        throw(ArgumentError("conditioning must be nothing or an AbstractMatrix of size (cond_dim, ngroups); got $(typeof(conditioning))"))
    end
    if size(conditioning, 2) != ngroups
        throw(ArgumentError("conditioning size mismatch: expected $(ngroups) columns (one per group), got size $(size(conditioning))"))
    end
    return conditioning
end

# ╔═╡ a100000a-0000-0000-0000-000000000001
md"## 3 – get\_data\_iterator (no conditioning)"

# ╔═╡ a100000c-0000-0000-0000-000000000001
md"## 4 – get\_data\_iterator (with conditioning)"

# ╔═╡ a100000e-0000-0000-0000-000000000001
md"## 5 – get\_data\_iterator\_with\_class\_labels (no conditioning)"

# ╔═╡ a1000010-0000-0000-0000-000000000001
md"## 6 – \_split\_batch\_condition edge cases"

# ╔═╡ a1000020-0000-0000-0000-000000000001
md"## 7 – Benchmarking `get_data_iterator`"

# ╔═╡ a1000021-0000-0000-0000-000000000001
md"""
Benchmark parameters — adjust to match your real training setup.
"""

# ╔═╡ a1000022-0000-0000-0000-000000000001
begin
    bench_nt        = 250    # samples per half-window
    bench_ntr       = 500    # traces per station pair
    bench_ngroups   = 8      # number of matrices in dvec
    bench_ntau      = 100    # traces per mini-sample
    bench_batchsize = 20
    bench_nsteps    = 50     # steps per epoch  => nsamples = 50×20 = 1000

    # Deterministic, decodable values: gid*1_000_000 + tid*1_000 + row
    # This makes it easy to verify group/trace shuffling behavior.
    function make_identifiable_group(nt, ntr, gid)
        out = Matrix{Float32}(undef, nt, ntr)
        for t in 1:ntr
            base = gid * 1_000_000 + t * 1_000
            for r in 1:nt
                out[r, t] = Float32(base + r)
            end
        end
        out
    end

    bench_Dtrain = CUDA.CuArray.([make_identifiable_group(bench_nt, bench_ntr, gid) for gid in 1:bench_ngroups])
end

# ╔═╡ 744e148b-ccbe-4c28-8eeb-8c1d765ba465
bench_Dtrain[1]

# ╔═╡ a1000023-0000-0000-0000-000000000001
md"### Current implementation"

# ╔═╡ a1000025-0000-0000-0000-000000000001
md"### Fast implementation (pre-allocated + multithreaded)"

# ╔═╡ a1000028-0000-0000-0000-000000000001
md"### Correctness: shapes & eltypes match (baseline, fast, lazy)"

# ╔═╡ b1f6364b-c17f-4ded-9b19-fbd8ae9a4c80
md"### MLUtils lazy dataset implementation"

# ╔═╡ 2f69a5b2-8c0e-4d5b-b106-1a499ea1998a
"""
Lazy random dataset for MLUtils.DataLoader.
Sampling happens during iteration instead of pre-materializing a full epoch array.
"""
struct RandomGroupDataset{TD,TC}
    dvec::TD
    ntau::Int
    conditioning::TC
    nsamples::Int
end

# ╔═╡ c858b580-0584-4f83-b33e-86801aee9333
MLUtils.numobs(ds::RandomGroupDataset) = ds.nsamples

# ╔═╡ f1cec787-9aa6-40b9-acf2-8595645531fe
Base.length(ds::RandomGroupDataset) = ds.nsamples

# ╔═╡ 8d237e1a-a6a2-4e3b-842a-da837ae0ead8
"""
Thin iterator that calls MLUtils.getobs(ds, I) once per batch.
Avoids DataLoader's internal scalar-dispatch path that breaks 3D stacking.
"""
struct LazyGroupIterator{TD,TC}
    ds::RandomGroupDataset{TD,TC}
    batchsize::Int
    nsteps::Int
end

# ╔═╡ a9c15d08-3e2c-46a9-9ba2-b0973d609789
Base.length(it::LazyGroupIterator) = it.nsteps

# ╔═╡ fd3bb273-c58b-4698-9ff5-7a0c2f98dee0
"""
Stack a vector of matrices into a 3D array along the 3rd dimension.
Works for CPU arrays and CUDA arrays.
"""
function stack_matrices_dim3(mats)
    @assert !isempty(mats) "Cannot stack an empty collection"
    nt, ntau = size(mats[1])
    out = similar(mats[1], nt, ntau, length(mats))
    for i in eachindex(mats)
        @views out[:, :, i] .= mats[i]
    end
    return out
end

# ╔═╡ 2c123eac-2a79-4f21-81f4-ae32c4b91ff0
"""
Stack a vector of vectors into a matrix along the 2nd dimension.
Works for CPU arrays and CUDA arrays.
"""
function stack_vectors_dim2(vecs)
    @assert !isempty(vecs) "Cannot stack an empty collection"
    m = length(vecs[1])
    out = similar(vecs[1], m, length(vecs))
    for i in eachindex(vecs)
        @views out[:, i] .= vecs[i]
    end
    return out
end

# ╔═╡ a1000008-0000-0000-0000-000000000008
"""
Baseline iterator (serial + stack), copied from the original notebook logic.
"""
function get_data_iterator_baseline(dvec; nsteps=1000, batchsize=256, ntau=20, conditioning=nothing)
    nd = length(dvec)
    conditioning = validate_conditioning_local(conditioning, nd)
    nsamples = nsteps * batchsize
    if !has_conditioning_local(conditioning)
        D = map(1:nsamples) do _
            idx = rand(1:nd)
            randobs(dvec[idx], ntau)
        end
        drepeat = stack_matrices_dim3(D)
        return DataLoader(drepeat, shuffle=true, batchsize=batchsize, partial=false, buffer=true, parallel=true)
    end

    D = map(1:nsamples) do _
        idx = rand(1:nd)
        d1 = randobs(dvec[idx], ntau)
        c1 = get_group_condition_vector_local(conditioning, idx)
        (d1, c1)
    end
    drepeat = stack_matrices_dim3(first.(D))
    condrepeat = stack_vectors_dim2(last.(D))
    return DataLoader((drepeat, condrepeat), shuffle=true, batchsize=batchsize, partial=false, buffer=true, parallel=true)
end

# ╔═╡ a1000009-0000-0000-0000-000000000009
"""
Threaded pre-allocated iterator variant for benchmarking.
"""
function get_data_iterator_threaded(dvec; nsteps=1000, batchsize=256, ntau=20, conditioning=nothing)
    nsamples = nsteps * batchsize
    nd       = length(dvec)
    conditioning = validate_conditioning_local(conditioning, nd)
    nt       = size(dvec[1], 1)
    sample    = dvec[1]
    use_threads = !is_gpu_array(sample)

    if !has_conditioning_local(conditioning)
        drepeat = similar(sample, nt, ntau, nsamples)
        if use_threads
            Threads.@threads for i in 1:nsamples
                g   = dvec[rand(1:nd)]
                ntr = size(g, 2)
                if ntau <= ntr
                    cols = sort!(randperm(ntr)[1:ntau])
                    @views drepeat[:, :, i] .= g[:, cols]
                else
                    @views drepeat[:, 1:ntr, i] .= g
                    for j in (ntr+1):ntau
                        drepeat[:, j, i] .= g[:, rand(1:ntr)]
                    end
                end
            end
        else
            for i in 1:nsamples
                g   = dvec[rand(1:nd)]
                ntr = size(g, 2)
                if ntau <= ntr
                    cols = sort!(randperm(ntr)[1:ntau])
                    @views drepeat[:, :, i] .= g[:, cols]
                else
                    @views drepeat[:, 1:ntr, i] .= g
                    for j in (ntr+1):ntau
                        drepeat[:, j, i] .= g[:, rand(1:ntr)]
                    end
                end
            end
        end
        return DataLoader(drepeat, shuffle=true, batchsize=batchsize, partial=false, buffer=true, parallel=true)
    end

    cond_sample = get_group_condition_vector_local(conditioning, 1)
    cond_dim   = length(cond_sample)
    drepeat    = similar(sample, nt, ntau, nsamples)
    condrepeat = similar(cond_sample, cond_dim, nsamples)
    if use_threads
        Threads.@threads for i in 1:nsamples
            idx = rand(1:nd)
            g   = dvec[idx]
            ntr = size(g, 2)
            if ntau <= ntr
                cols = sort!(randperm(ntr)[1:ntau])
                @views drepeat[:, :, i] .= g[:, cols]
            else
                @views drepeat[:, 1:ntr, i] .= g
                for j in (ntr+1):ntau
                    drepeat[:, j, i] .= g[:, rand(1:ntr)]
                end
            end
            condrepeat[:, i] .= get_group_condition_vector_local(conditioning, idx)
        end
    else
        for i in 1:nsamples
            idx = rand(1:nd)
            g   = dvec[idx]
            ntr = size(g, 2)
            if ntau <= ntr
                cols = sort!(randperm(ntr)[1:ntau])
                @views drepeat[:, :, i] .= g[:, cols]
            else
                @views drepeat[:, 1:ntr, i] .= g
                for j in (ntr+1):ntau
                    drepeat[:, j, i] .= g[:, rand(1:ntr)]
                end
            end
            condrepeat[:, i] .= get_group_condition_vector_local(conditioning, idx)
        end
    end
    return DataLoader((drepeat, condrepeat), shuffle=true, batchsize=batchsize, partial=false, buffer=true, parallel=true)
end

# ╔═╡ a1000026-0000-0000-0000-000000000001
"""
Drop-in replacement for `get_data_iterator` — no external dependencies.
Speedups:
    1. Pre-allocates output array (no intermediate Vector of matrices + stack call).
  2. Fills samples in parallel with `Threads.@threads`.
  3. No GC pressure from temporary matrix allocations in the inner loop.
"""
get_data_iterator_fast = get_data_iterator_threaded

# ╔═╡ 34116a9e-27a1-11f1-b2c6-a3fa2914df4f
"""
Iterator variant that returns class labels along with data.
"""
function get_data_iterator_with_class_labels_local(dvec; nsteps=1000, batchsize=256, ntau=20, conditioning=nothing)
    nd = length(dvec)
    conditioning = validate_conditioning_local(conditioning, nd)
    D = map(1:nsteps*batchsize) do _
        idx = rand(1:nd)
        d1 = randobs(dvec[idx], ntau)
        dlabel = fill(idx, 1, ntau)
        if has_conditioning_local(conditioning)
            c1 = get_group_condition_vector_local(conditioning, idx)
            (d1, dlabel, c1)
        else
            (d1, dlabel)
        end
    end
    if has_conditioning_local(conditioning)
        drepeat = stack_matrices_dim3(first.(D))
        labelrepeat = stack_matrices_dim3(getindex.(D, 2))
        condrepeat = stack_vectors_dim2(getindex.(D, 3))
        return DataLoader((drepeat, labelrepeat, condrepeat), shuffle=true, batchsize=batchsize, partial=false, buffer=true, parallel=true)
    else
        drepeat = stack_matrices_dim3(first.(D))
        labelrepeat = stack_matrices_dim3(last.(D))
        return DataLoader((drepeat, labelrepeat), shuffle=true, batchsize=batchsize, partial=false, buffer=true, parallel=true)
    end
end

# ╔═╡ 34116e2c-27a1-11f1-9903-196d1e833e82
function split_batch_condition_local(batch)
    if !(batch isa Tuple)
        return batch, nothing
    end
    x = batch[1]
    if length(batch) == 2
        second = batch[2]
        if second === nothing || ndims(second) <= 2
            return x, second
        else
            return x, nothing
        end
    elseif length(batch) >= 3
        return x, batch[3]
    end
    return x, nothing
end

# ╔═╡ a100000b-0000-0000-0000-000000000001
let
    nt, ntr   = 100, 80
    ntau      = 10
    batchsize = 8
    nsteps    = 5
    # Dtrain is simply a Vector{Matrix{Float32}} (one matrix per group/half)
    Dtrain = [fake_xcorr(nt, ntr) for _ in 1:8]

    iter  = get_data_iterator_baseline(Dtrain; ntau, batchsize, nsteps)
    batch = first(iter)

    @assert batch isa AbstractArray   "expected plain Array, got $(typeof(batch))"
    @assert ndims(batch) == 3         "expected 3D batch, got $(ndims(batch))D"
    @assert size(batch, 1) == nt      "nt mismatch: $(size(batch,1)) ≠ $nt"
    @assert size(batch, 2) == ntau    "ntau mismatch"
    @assert size(batch, 3) == batchsize "batchsize mismatch"

    # _split_batch_condition on plain array → (x, nothing)
    xb, cond = split_batch_condition_local(batch)
    @assert xb === batch   "xb should be identical to batch"
    @assert cond === nothing "cond should be nothing for unconditioned iterator"

    md"✅ **get_data_iterator (no conditioning)** — all assertions passed"
end

# ╔═╡ a1000011-0000-0000-0000-000000000001
let
    nt, ntau, bs = 100, 10, 8
    x = rand(Float32, nt, ntau, bs)

    # 1. plain array
    xb, c = split_batch_condition_local(x)
    @assert xb === x && c === nothing  "plain array: expected (x, nothing)"

    # 2. (x, nothing) — was crashing before the ndims(::Nothing) fix
    xb2, c2 = split_batch_condition_local((x, nothing))
    @assert xb2 === x && c2 === nothing  "(x, nothing): should not error"

    # 3. (x, 2D condition) → condition is returned
    cond2d = rand(Float32, 6, bs)
    xb3, c3 = split_batch_condition_local((x, cond2d))
    @assert xb3 === x && c3 === cond2d  "(x, 2D cond): cond should pass through"

    # 4. (x, 3D labels) → not a condition, return nothing
    labels = rand(Float32, 1, ntau, bs)
    xb4, c4 = split_batch_condition_local((x, labels))
    @assert xb4 === x && c4 === nothing  "(x, 3D labels): should return nothing"

    # 5. (x, labels, cond) — 3-element tuple
    cond2d_b = rand(Float32, 6, bs)
    xb5, c5 = split_batch_condition_local((x, labels, cond2d_b))
    @assert xb5 === x && c5 === cond2d_b  "3-tuple: cond should be batch[3]"

    md"✅ **_split_batch_condition edge cases** — all assertions passed"
end

# ╔═╡ a100000d-0000-0000-0000-000000000001
let
    nt, ntr   = 100, 80
    ngroups   = 4
    cond_dim  = 6
    ntau      = 10
    batchsize = 8
    nsteps    = 5
    Dtrain = [fake_xcorr(nt, ntr) for _ in 1:ngroups]

    # conditioning matrix: (cond_dim × ngroups)
    conditioning = rand(Float32, cond_dim, ngroups)

    iter  = get_data_iterator_baseline(Dtrain; ntau, batchsize, nsteps, conditioning)
    batch = first(iter)

    @assert batch isa Tuple   "expected Tuple, got $(typeof(batch))"
    @assert length(batch) == 2

    xb, cond = batch
    @assert ndims(xb) == 3               "data should be 3D"
    @assert size(xb, 3) == batchsize     "batchsize mismatch in data"
    @assert ndims(cond) == 2             "condition should be 2D"
    @assert size(cond, 1) == cond_dim    "cond_dim mismatch"
    @assert size(cond, 2) == batchsize   "cond batchsize mismatch"

    # _split_batch_condition
    xb2, cond2 = split_batch_condition_local(batch)
    @assert xb2  === xb    "xb2 should alias xb"
    @assert cond2 === cond "cond2 should alias cond"

    md"✅ **get_data_iterator (with conditioning)** — all assertions passed"
end

# ╔═╡ a100000f-0000-0000-0000-000000000001
let
    nt, ntr   = 100, 80
    ntau      = 10
    batchsize = 8
    nsteps    = 5
    Dtrain = [fake_xcorr(nt, ntr) for _ in 1:8]

    iter  = get_data_iterator_with_class_labels_local(
                Dtrain; ntau, batchsize, nsteps)
    batch = first(iter)

    @assert batch isa Tuple   "expected (data, labels) Tuple"
    @assert length(batch) == 2

    xb, labels = batch
    @assert ndims(xb)     == 3           "data should be 3D"
    @assert ndims(labels) == 3           "labels should be 3D"
    @assert size(labels, 1) == 1         "labels first dim should be 1"
    @assert size(labels, 2) == ntau      "labels ntau mismatch"
    @assert size(labels, 3) == batchsize "labels batchsize mismatch"

    # 3D labels are NOT a condition → _split_batch_condition should return nothing
    xb2, cond2 = split_batch_condition_local(batch)
    @assert xb2  === xb       "xb2 should alias xb"
    @assert cond2 === nothing "class labels should not be treated as condition"

    md"✅ **get_data_iterator_with_class_labels** — all assertions passed"
end

# ╔═╡ 974d93e1-bc0b-4fa1-b75f-1a62a297df75
# Scalar overload: required for DataLoader internals (shuffle, parallel) that call getobs(ds, i::Int)
function MLUtils.getobs(ds::RandomGroupDataset, i::Integer)
    dvec = ds.dvec
    ntau = ds.ntau
    nd = length(dvec)
    idx = rand(1:nd)
    if !has_conditioning_local(ds.conditioning)
        return randobs(dvec[idx], ntau)
    end
    d1 = randobs(dvec[idx], ntau)
    c1 = get_group_condition_vector_local(ds.conditioning, idx)
    return (d1, c1)
end

# ╔═╡ a39fbad7-6259-4f6a-a77c-d0ee3059ad95
function MLUtils.getobs(ds::RandomGroupDataset, I::AbstractVector{<:Integer})
    dvec = ds.dvec
    ntau = ds.ntau
    nd = length(dvec)
    bs = length(I)
    nt = size(dvec[1], 1)
    sample = dvec[1]
    use_threads = !is_gpu_array(sample)

    if !has_conditioning_local(ds.conditioning)
        out = similar(sample, nt, ntau, bs)
        if use_threads
            Threads.@threads for i in 1:bs
                g = dvec[rand(1:nd)]
                ntr = size(g, 2)
                if ntau <= ntr
                    cols = sort!(randperm(ntr)[1:ntau])
                    @views out[:, :, i] .= g[:, cols]
                else
                    @views out[:, 1:ntr, i] .= g
                    for j in (ntr+1):ntau
                        out[:, j, i] .= g[:, rand(1:ntr)]
                    end
                end
            end
        else
            for i in 1:bs
                g = dvec[rand(1:nd)]
                ntr = size(g, 2)
                if ntau <= ntr
                    cols = sort!(randperm(ntr)[1:ntau])
                    @views out[:, :, i] .= g[:, cols]
                else
                    @views out[:, 1:ntr, i] .= g
                    for j in (ntr+1):ntau
                        out[:, j, i] .= g[:, rand(1:ntr)]
                    end
                end
            end
        end
        return out
    end

    cond_dim = size(ds.conditioning, 1)
    out = similar(sample, nt, ntau, bs)
    cond = Matrix{Float32}(undef, cond_dim, bs)
    if use_threads
        Threads.@threads for i in 1:bs
            idx = rand(1:nd)
            g = dvec[idx]
            ntr = size(g, 2)
            if ntau <= ntr
                cols = sort!(randperm(ntr)[1:ntau])
                @views out[:, :, i] .= g[:, cols]
            else
                @views out[:, 1:ntr, i] .= g
                for j in (ntr+1):ntau
                    out[:, j, i] .= g[:, rand(1:ntr)]
                end
            end
            @views cond[:, i] .= get_group_condition_vector_local(ds.conditioning, idx)
        end
    else
        for i in 1:bs
            idx = rand(1:nd)
            g = dvec[idx]
            ntr = size(g, 2)
            if ntau <= ntr
                cols = sort!(randperm(ntr)[1:ntau])
                @views out[:, :, i] .= g[:, cols]
            else
                @views out[:, 1:ntr, i] .= g
                for j in (ntr+1):ntau
                    out[:, j, i] .= g[:, rand(1:ntr)]
                end
            end
            @views cond[:, i] .= get_group_condition_vector_local(ds.conditioning, idx)
        end
    end
    return out, cond
end

# ╔═╡ 26a8bc76-c362-405a-9724-5355900df7e1
Base.eltype(::LazyGroupIterator) = Array{Float32,3}

# ╔═╡ 19d370cb-9399-4336-b2ef-da7b04126772
function Base.iterate(it::LazyGroupIterator, step=1)
    step > it.nsteps && return nothing
    # indices are ignored by getobs (samples randomly), just need the right length
    I = Base.OneTo(it.batchsize)
    batch = MLUtils.getobs(it.ds, I)
    return batch, step + 1
end

# ╔═╡ 35a0a407-b4c6-4808-afd7-187f8b4135f1
"""
Iterator variant built on MLUtils lazy observation access.
"""
function get_data_iterator_lazy(dvec; nsteps=1000, batchsize=256, ntau=20, conditioning=nothing)
    nd = length(dvec)
    conditioning = validate_conditioning_local(conditioning, nd)
    nsamples = nsteps * batchsize
    ds = RandomGroupDataset(dvec, ntau, conditioning, nsamples)
    return LazyGroupIterator(ds, batchsize, nsteps)
end

# ╔═╡ a1000029-0000-0000-0000-000000000001
let
    iter_orig = get_data_iterator_baseline(bench_Dtrain; ntau=bench_ntau,
        batchsize=bench_batchsize, nsteps=bench_nsteps)
    iter_fast = get_data_iterator_fast(bench_Dtrain; ntau=bench_ntau,
        batchsize=bench_batchsize, nsteps=bench_nsteps)
    iter_lazy = get_data_iterator_lazy(bench_Dtrain; ntau=bench_ntau,
        batchsize=bench_batchsize, nsteps=bench_nsteps)
    b_orig = first(iter_orig)
    b_fast = first(iter_fast)
    b_lazy = first(iter_lazy)

    @assert size(b_orig) == size(b_fast) "shape mismatch: baseline $(size(b_orig)) vs fast $(size(b_fast))"
    @assert size(b_orig) == size(b_lazy) "shape mismatch: baseline $(size(b_orig)) vs lazy $(size(b_lazy))"
    @assert eltype(b_orig) == eltype(b_fast) "eltype mismatch: baseline $(eltype(b_orig)) vs fast $(eltype(b_fast))"
    @assert eltype(b_orig) == eltype(b_lazy) "eltype mismatch: baseline $(eltype(b_orig)) vs lazy $(eltype(b_lazy))"

    md"✅ shapes match across baseline/fast/lazy: $(size(b_orig)), eltype $(eltype(b_orig))"
end

# ╔═╡ 0f62a175-ef98-4e8f-9e06-753e7428a584
it_test = get_data_iterator_lazy(bench_Dtrain; ntau=bench_ntau,
            batchsize=bench_batchsize, nsteps=bench_nsteps)

# ╔═╡ 15a49580-3534-4df5-89b7-2702a670dfd7
first(it_test)

# ╔═╡ c1009216-33ff-4db8-98dd-1c563eda2306
it_test_cond = get_data_iterator_lazy(bench_Dtrain; ntau=bench_ntau,
            batchsize=bench_batchsize, nsteps=bench_nsteps, conditioning=randn(2, 8))

# ╔═╡ 57d044c4-b427-49cb-9cfe-985c0b473961
first(it_test_cond)

# ╔═╡ 7a15d5ce-a2b2-4f40-a149-2f0b4d80cc0a
"""
Consume a full iterator once. Used for fair benchmark timing.
"""
function consume_epoch(iter)
    nbatches = 0
    for _ in iter
        nbatches += 1
    end
    return nbatches
end

# ╔═╡ a1000024-0000-0000-0000-000000000001
# ╠═╡ skip_as_script = true
#=╠═╡
let
    # warm-up full epoch
    consume_epoch(get_data_iterator_baseline(bench_Dtrain; ntau=bench_ntau,
        batchsize=bench_batchsize, nsteps=bench_nsteps))
    # timed run for one full epoch
    @btime begin
        it = get_data_iterator_baseline($bench_Dtrain; ntau=$bench_ntau,
            batchsize=$bench_batchsize, nsteps=$bench_nsteps)
        consume_epoch(it)
    end
end
  ╠═╡ =#

# ╔═╡ a1000027-0000-0000-0000-000000000001
# ╠═╡ skip_as_script = true
#=╠═╡
let
    # warm-up full epoch
    consume_epoch(get_data_iterator_fast(bench_Dtrain; ntau=bench_ntau,
        batchsize=bench_batchsize, nsteps=bench_nsteps))
    # timed run for one full epoch
    @btime begin
        it = get_data_iterator_fast($bench_Dtrain; ntau=$bench_ntau,
            batchsize=$bench_batchsize, nsteps=$bench_nsteps)
        consume_epoch(it)
    end
end
  ╠═╡ =#

# ╔═╡ 28c513be-60d1-4030-85c7-3f6605516556
# ╠═╡ skip_as_script = true
#=╠═╡
let
    # warm-up full epoch
    consume_epoch(get_data_iterator_lazy(bench_Dtrain; ntau=bench_ntau,
        batchsize=bench_batchsize, nsteps=bench_nsteps))
    # timed run for one full epoch
    @btime begin
        it = get_data_iterator_lazy($bench_Dtrain; ntau=$bench_ntau,
            batchsize=$bench_batchsize, nsteps=$bench_nsteps)
        consume_epoch(it)
    end
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
BenchmarkTools = "~1.6.3"
CUDA = "~5.11.0"
MLUtils = "~0.4.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "95dbb472436692c8790acc4ee56d5394f448b00c"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "2eeb2c9bef11013efc6f8f97f32ee59b146b09fb"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.44"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "35ea197a51ce46fcd01c4a44befce0578a1aaeca"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b8651b2eb5796a386b0398a20b519a6a6150f75c"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.3"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "e386db8b4753b42caac75ac81d0a4fe161a68a97"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.1"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "ea6a2ab8307059b6c9ea186ff7dfcd032a13b731"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.11.0"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8c19e97de5b7574672e4a7a3abd55714ad66d59a"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.2+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "061f39cc84e99928830aa1005d79f7e99097ba28"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.2.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "af17d37b5b8b4d7525f8902eba1ef6141a9a7d3b"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.21.0+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

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

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "6487601563e4a1d1dab796e88b4548bf5544209e"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.4.1"

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

    [deps.GPUArrays.weakdeps]
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "966946d226e8b676ca6409454718accb18c34c54"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.8.2"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "9e9186b09a13b7f094f87d1a9bb266d8780e1b1c"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.0.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "b3ad4a0255688dcb895a52fafbaae3023b588a90"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.4.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "f2e76d3ced51a2a9e185abc0b97494c7273f649f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.41"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "69e4739502b7ab5176117e97e1664ed181c35036"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.6"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

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

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d4e20500d210247322901841d4eafc7a0c52642d"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.13.1+0"

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

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

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

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "6dc9ffc3a9931e6b988f913b49630d0fb986d0a8"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.33"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NVTX]]
deps = ["JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "a9083c3e469e63cca454d1fc3b19472d9d92c14a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.3"

    [deps.NVTX.extensions]
    NVTXColorsExt = "Colors"

    [deps.NVTX.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

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

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "211530a7dc76ab59087f4d4d1fc3f086fbe87594"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.2.3"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

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

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

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
git-tree-sha1 = "ac4b837d89a58c848e85e698e2a2514e9d59d8f6"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.0"

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

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

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

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

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

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "fa95b3b097bcef5845c142ea2e085f1b2591e92c"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.7.1"

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

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "0f30765c32d66d58e41f4cb5624d4fc8a82ec13b"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.1"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

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
# ╠═a1000001-0000-0000-0000-000000000001
# ╠═a1000003-0000-0000-0000-000000000003
# ╠═59946319-e22b-41ba-b611-4eed145f3bc1
# ╠═fd3bb273-c58b-4698-9ff5-7a0c2f98dee0
# ╠═2c123eac-2a79-4f21-81f4-ae32c4b91ff0
# ╠═7a34f647-7bf1-4e7b-a9c1-a4c9fa6e8692
# ╠═0864cdc0-402d-4e2c-b6e9-32f4e9c20b71
# ╠═91001f0c-9e90-4af4-8d1c-38deb49cdf2f
# ╠═d4d6acd7-da42-484a-8c1c-7fc60e14923c
# ╠═12d4775e-1852-4689-9a25-6fef001d95c7
# ╠═f9fe0998-0558-4121-b44e-e50d533af854
# ╠═a1000008-0000-0000-0000-000000000008
# ╠═a1000009-0000-0000-0000-000000000009
# ╠═34116a9e-27a1-11f1-b2c6-a3fa2914df4f
# ╠═34116e2c-27a1-11f1-9903-196d1e833e82
# ╠═a100000a-0000-0000-0000-000000000001
# ╠═a100000b-0000-0000-0000-000000000001
# ╠═a100000c-0000-0000-0000-000000000001
# ╠═a100000d-0000-0000-0000-000000000001
# ╠═a100000e-0000-0000-0000-000000000001
# ╠═a100000f-0000-0000-0000-000000000001
# ╠═a1000010-0000-0000-0000-000000000001
# ╠═a1000011-0000-0000-0000-000000000001
# ╠═a1000020-0000-0000-0000-000000000001
# ╠═a1000021-0000-0000-0000-000000000001
# ╠═a1000022-0000-0000-0000-000000000001
# ╠═744e148b-ccbe-4c28-8eeb-8c1d765ba465
# ╠═a1000023-0000-0000-0000-000000000001
# ╠═a1000024-0000-0000-0000-000000000001
# ╠═a1000025-0000-0000-0000-000000000001
# ╠═a1000026-0000-0000-0000-000000000001
# ╠═a1000027-0000-0000-0000-000000000001
# ╠═a1000028-0000-0000-0000-000000000001
# ╠═a1000029-0000-0000-0000-000000000001
# ╠═b1f6364b-c17f-4ded-9b19-fbd8ae9a4c80
# ╠═28c513be-60d1-4030-85c7-3f6605516556
# ╠═0f62a175-ef98-4e8f-9e06-753e7428a584
# ╠═15a49580-3534-4df5-89b7-2702a670dfd7
# ╠═c1009216-33ff-4db8-98dd-1c563eda2306
# ╠═57d044c4-b427-49cb-9cfe-985c0b473961
# ╠═2f69a5b2-8c0e-4d5b-b106-1a499ea1998a
# ╠═c858b580-0584-4f83-b33e-86801aee9333
# ╠═f1cec787-9aa6-40b9-acf2-8595645531fe
# ╠═974d93e1-bc0b-4fa1-b75f-1a62a297df75
# ╠═a39fbad7-6259-4f6a-a77c-d0ee3059ad95
# ╠═8d237e1a-a6a2-4e3b-842a-da837ae0ead8
# ╠═a9c15d08-3e2c-46a9-9ba2-b0973d609789
# ╠═26a8bc76-c362-405a-9724-5355900df7e1
# ╠═19d370cb-9399-4336-b2ef-da7b04126772
# ╠═35a0a407-b4c6-4808-afd7-187f8b4135f1
# ╠═7a15d5ce-a2b2-4f40-a149-2f0b4d80cc0a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
