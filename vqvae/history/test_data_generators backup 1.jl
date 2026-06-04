### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ a1000001-0000-0000-0000-000000000001
begin
    using CUDA, cuDNN,
        Flux,
        MLUtils,
        DSP,
        Statistics,
        LinearAlgebra,
        PlutoLinks,
        PlutoUI,
        PlutoHooks,
        Random,
        Optimisers
    CUDA.device!(0)
end

# ╔═╡ a1000002-0000-0000-0000-000000000002
symae = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/SymAE_architecture.jl")

# ╔═╡ a1000003-0000-0000-0000-000000000003
md"## Helpers"

# ╔═╡ a1000004-0000-0000-0000-000000000004
xpu = gpu

# ╔═╡ a1000005-0000-0000-0000-000000000005
"""
Fake symmetric cross-correlation matrix: (nt_odd, ntr)
nt must be odd so that split_causal_acausal does not error.
"""
function fake_xcorr(nt=501, ntr=50)
    rand(Float32, nt, ntr)
end

# ╔═╡ a1000006-0000-0000-0000-000000000006
md"## 1 – split\_causal\_acausal"

# ╔═╡ a1000007-0000-0000-0000-000000000007
let
    nt, ntr = 501, 30
    X    = fake_xcorr(nt, ntr)
    half = div(nt - 1, 2)

    ac, ca = symae.split_causal_acausal(X, false)
    @assert size(ac) == (half, ntr)  "acausal size mismatch: $(size(ac))"
    @assert size(ca) == (half, ntr)  "causal size mismatch: $(size(ca))"

    # zero_lag = true → a zero row is prepended
    ac_zl, ca_zl = symae.split_causal_acausal(X, true)
    @assert size(ac_zl, 1) == half + 1
    @assert all(ac_zl[1, :] .== 0)

    # max_lag truncation
    lag = 80
    ac_lag, ca_lag = symae.split_causal_acausal(X, false, lag)
    @assert size(ac_lag, 1) == lag
    @assert size(ca_lag, 1) == lag

    md"✅ **split_causal_acausal** — all assertions passed"
end

# ╔═╡ a1000008-0000-0000-0000-000000000008
md"## 2 – make\_data"

# ╔═╡ a1000009-0000-0000-0000-000000000009
let
    nt, ntr = 501, 100
    npairs  = 3
    D       = [fake_xcorr(nt, ntr) for _ in 1:npairs]
    half    = div(nt - 1, 2)

    result = symae.make_data(D; at=0.8, shuffle=true)

    @assert length(result.Dtrain) == 2 * npairs  "wrong Dtrain length"
    @assert length(result.Dtest)  == 2 * npairs  "wrong Dtest length"
    @assert length(result.Dall)   == 2 * npairs  "wrong Dall length"

    for d in result.Dtrain
        @assert d isa AbstractMatrix       "Dtrain element not a matrix"
        @assert eltype(d) == Float32       "Dtrain element not Float32"
        @assert size(d, 1) == half         "wrong nt in Dtrain: $(size(d,1)) ≠ $half"
    end

    # with max_lag
    lag        = 80
    result_lag = symae.make_data(D; max_lag=lag)
    for d in result_lag.Dtrain
        @assert size(d, 1) == lag  "wrong nt with max_lag=$(lag): $(size(d,1))"
    end

    md"✅ **make_data** — all assertions passed"
end

# ╔═╡ a100000a-0000-0000-0000-000000000001
md"## 3 – get\_data\_iterator (no conditioning)"

# ╔═╡ a100000b-0000-0000-0000-000000000001
let
    nt, ntr   = 201, 80
    D         = [fake_xcorr(nt, ntr) for _ in 1:4]
    data      = symae.make_data(D)
    half      = div(nt - 1, 2)
    ntau      = 10
    batchsize = 8
    nsteps    = 5

    iter  = symae.get_data_iterator(data.Dtrain; ntau, batchsize, nsteps)
    batch = first(iter)

    @assert batch isa AbstractArray   "expected plain Array, got $(typeof(batch))"
    @assert ndims(batch) == 3         "expected 3D batch, got $(ndims(batch))D"
    @assert size(batch, 1) == half    "nt mismatch: $(size(batch,1)) ≠ $half"
    @assert size(batch, 2) == ntau    "ntau mismatch"
    @assert size(batch, 3) == batchsize "batchsize mismatch"

    # _split_batch_condition on plain array → (x, nothing)
    xb, cond = symae._split_batch_condition(batch)
    @assert xb === batch   "xb should be identical to batch"
    @assert cond === nothing "cond should be nothing for unconditioned iterator"

    md"✅ **get_data_iterator (no conditioning)** — all assertions passed"
end

# ╔═╡ a100000c-0000-0000-0000-000000000001
md"## 4 – get\_data\_iterator (with conditioning)"

# ╔═╡ a100000d-0000-0000-0000-000000000001
let
    nt, ntr   = 201, 80
    ngroups   = 4
    cond_dim  = 6
    D         = [fake_xcorr(nt, ntr) for _ in 1:ngroups]
    data      = symae.make_data(D)
    ntau      = 10
    batchsize = 8
    nsteps    = 5

    # conditioning matrix: (cond_dim × ngroups)
    conditioning = rand(Float32, cond_dim, ngroups)

    iter  = symae.get_data_iterator(data.Dtrain; ntau, batchsize, nsteps, conditioning)
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
    xb2, cond2 = symae._split_batch_condition(batch)
    @assert xb2  === xb    "xb2 should alias xb"
    @assert cond2 === cond "cond2 should alias cond"

    md"✅ **get_data_iterator (with conditioning)** — all assertions passed"
end

# ╔═╡ a100000e-0000-0000-0000-000000000001
md"## 5 – get\_data\_iterator\_with\_class\_labels (no conditioning)"

# ╔═╡ a100000f-0000-0000-0000-000000000001
let
    nt, ntr   = 201, 80
    D         = [fake_xcorr(nt, ntr) for _ in 1:4]
    data      = symae.make_data(D)
    ntau      = 10
    batchsize = 8
    nsteps    = 5

    iter  = symae.get_data_iterator_with_class_labels(
                data.Dtrain; ntau, batchsize, nsteps)
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
    xb2, cond2 = symae._split_batch_condition(batch)
    @assert xb2  === xb       "xb2 should alias xb"
    @assert cond2 === nothing "class labels should not be treated as condition"

    md"✅ **get_data_iterator_with_class_labels** — all assertions passed"
end

# ╔═╡ a1000010-0000-0000-0000-000000000001
md"## 6 – \_split\_batch\_condition edge cases"

# ╔═╡ a1000011-0000-0000-0000-000000000001
let
    nt, ntau, bs = 100, 10, 8
    x = rand(Float32, nt, ntau, bs)

    # 1. plain array
    xb, c = symae._split_batch_condition(x)
    @assert xb === x && c === nothing  "plain array: expected (x, nothing)"

    # 2. (x, nothing) — was crashing before the ndims(::Nothing) fix
    xb2, c2 = symae._split_batch_condition((x, nothing))
    @assert xb2 === x && c2 === nothing  "(x, nothing): should not error"

    # 3. (x, 2D condition) → condition is returned
    cond2d = rand(Float32, 6, bs)
    xb3, c3 = symae._split_batch_condition((x, cond2d))
    @assert xb3 === x && c3 === cond2d  "(x, 2D cond): cond should pass through"

    # 4. (x, 3D labels) → not a condition, return nothing
    labels = rand(Float32, 1, ntau, bs)
    xb4, c4 = symae._split_batch_condition((x, labels))
    @assert xb4 === x && c4 === nothing  "(x, 3D labels): should return nothing"

    # 5. (x, labels, cond) — 3-element tuple
    cond2d_b = rand(Float32, 6, bs)
    xb5, c5 = symae._split_batch_condition((x, labels, cond2d_b))
    @assert xb5 === x && c5 === cond2d_b  "3-tuple: cond should be batch[3]"

    md"✅ **_split_batch_condition edge cases** — all assertions passed"
end

# ╔═╡ a1000012-0000-0000-0000-000000000001
md"""
---
## Summary
Run cells top-to-bottom. Each section prints ✅ if all assertions pass.
A failure will raise an `AssertionError` with a descriptive message.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
"""
