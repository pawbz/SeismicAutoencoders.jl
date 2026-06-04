### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 4035e565-793a-4bc3-a701-034b0c02f058
using Random, LinearAlgebra

# ╔═╡ f78ead71-604e-412a-8766-dbb7395e685b
using Lux, Enzyme, Optimisers

# ╔═╡ 1f3d5cfc-1eb3-497b-8b25-e49f4d1561b1
using CUDA

# ╔═╡ b543fb06-9d3a-4cf8-9f78-ce807fa875d0
using EnzymeCore

# ╔═╡ 12a7cffc-f64a-4785-8edb-70612e945f6b
using Statistics

# ╔═╡ 51ccb814-5539-402a-9b68-fdd7068f2c07
using Reactant

# ╔═╡ 00000000-0000-0000-0000-000000000010
md"## Shared EMA helpers (mirrors `VQVAE_architecture_v7`)"

# ╔═╡ 00000000-0000-0000-0000-000000000011
begin
	function assignment_matrix(indices::AbstractVector{Int}, K::Int)
	    N = length(indices)
	    enc = zeros(Float32, K, N)
	    for (j, i) in enumerate(indices)
	        enc[i, j] = 1f0
	    end
	    return enc
	end

	function vq_lookup_idx(embedding::Matrix{Float32}, z::AbstractVector{Float32})
	    dists = vec(sum((embedding .- z) .^ 2; dims=1))
	    return argmin(dists)
	end

	function update_stage_ema(embedding, ema_cs, ema_dw, z, indices, K, decay, epsilon)
	    enc    = assignment_matrix(indices, K)
	    counts = vec(sum(enc; dims=2))
	    sums   = z * enc'
	    ema_cs2 = decay .* ema_cs .+ (1f0 - decay) .* counts
	    n = sum(ema_cs2)
	    ema_cs2 = (ema_cs2 .+ epsilon) ./ (n + Float32(K) * epsilon) .* n
	    ema_dw2 = decay .* ema_dw .+ (1f0 - decay) .* sums
	    emb2    = ema_dw2 ./ reshape(max.(ema_cs2, epsilon), 1, :)
	    return emb2, ema_cs2, ema_dw2
	end

	function vq_quantize_cpu(z, embedding, ema_cs, ema_dw, K, decay, epsilon)
	    indices = [vq_lookup_idx(embedding, z[:, j]) for j in 1:size(z, 2)]
	    emb2, ema_cs2, ema_dw2 = update_stage_ema(embedding, ema_cs, ema_dw, z, indices, K, decay, epsilon)
	    z_q    = emb2[:, indices]
	    counts = [sum(i .== indices) for i in 1:K]
	    p      = Float64.(counts) ./ max(sum(counts), 1)
	    perp   = exp(-sum(p[p .> 0] .* log.(p[p .> 0])))
	    return z_q, emb2, ema_cs2, ema_dw2, perp
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000020
md"""
## Test 1: Pure CPU EMA loop (baseline)

Codebook **must** change every step. If it doesn't, something is wrong with the
EMA math itself.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000021
let
	Random.seed!(42)
	K, D, N    = 4, 8, 32
	decay, ε   = 0.99f0, 1f-5
	embedding  = randn(Float32, D, K) .* 0.1f0
	ema_cs     = ones(Float32, K)
	ema_dw     = copy(embedding)
	emb_before = copy(embedding)

	rows = []
	for step in 1:5
	    z = randn(Float32, D, N)
	    _, embedding, ema_cs, ema_dw, perp = vq_quantize_cpu(z, embedding, ema_cs, ema_dw, K, decay, ε)
	    diff = norm(embedding - emb_before)
	    push!(rows, (step=step, codebook_change=round(diff; digits=6), perplexity=round(perp; digits=3),
	        ok = diff > 0 ? "✓" : "✗ ZERO — BUG"))
	    emb_before = copy(embedding)
	end

	rows
end

# ╔═╡ 00000000-0000-0000-0000-000000000030
md"""
## Test 2: Lux + Enzyme CPU backend (simulates training loop)

Replicates the `prepare_rvq_payload → single_train_step!` pattern **without** Reactant.

In this test the codebook lives as a plain Julia variable (like in our CPU payload path).
`single_train_step!` has no way to overwrite it. This should work correctly.

The purpose is to confirm the EMA math and STE loss are correct in isolation.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000031
let
	Random.seed!(42)
	K, D_in, D_lat, N = 4, 16, 8, 32
	decay, ε, β = 0.99f0, 1f-5, 0.25f0

	encoder = Dense(D_in, D_lat)
	ps, st  = Lux.setup(Random.default_rng(), encoder)
	opt     = Optimisers.Adam(1f-3)
	ts      = Training.TrainState(encoder, ps, st, opt)

	embedding  = randn(Float32, D_lat, K) .* 0.1f0
	ema_cs     = ones(Float32, K)
	ema_dw     = copy(embedding)
	emb_before = copy(embedding)

	function loss_fn(model, ps, st, data)
	    x, z_q_pre = data
	    z_e, st2 = model(x, ps, st)
	    z_q_ste    = z_e .+ EnzymeCore.ignore_derivatives(z_q_pre .- z_e)
	    recon_loss  = mean((z_q_ste .- x[1:D_lat, :]) .^ 2)
	    commit_loss = β * mean((z_e .- EnzymeCore.ignore_derivatives(z_q_pre)) .^ 2)
	    return recon_loss + commit_loss, st2, (;)
	end

	rows = []
	for step in 1:5
	    x = randn(Float32, D_in, N)
	    z_e_cpu, _ = Lux.apply(encoder, x, ts.parameters, Lux.testmode(ts.states))
	    z_q_pre, embedding, ema_cs, ema_dw, perp = vq_quantize_cpu(
	        Float32.(z_e_cpu), embedding, ema_cs, ema_dw, K, decay, ε)
	    _, _, _, ts = Training.single_train_step!(AutoEnzyme(), loss_fn, (x, z_q_pre), ts)
	    diff = norm(embedding - emb_before)
	    push!(rows, (step=step, codebook_change=round(diff; digits=6), perplexity=round(perp; digits=3),
	        ok = diff > 0 ? "✓" : "✗ ZERO — BUG"))
	    emb_before = copy(embedding)
	end

	rows
end

# ╔═╡ 00000000-0000-0000-0000-000000000040
md"""
## Test 3: Simulate `ts.states.rvq` clobbering (the actual Reactant bug)

The codebook now lives **inside** `ts.states.rvq` (as in the real architecture).
We simulate what happens when `single_train_step!` returns a `ts` with the **old**
frozen state, and verify the re-inject fix restores the EMA-updated codebook.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000041
let
	Random.seed!(42)
	K, D_in, D_lat, N = 4, 16, 8, 32
	decay, ε, β = 0.99f0, 1f-5, 0.25f0

	encoder = Dense(D_in, D_lat)
	ps, st_init = Lux.setup(Random.default_rng(), encoder)
	opt = Optimisers.Adam(1f-3)

	# Put codebook inside ts.states.rvq (mirrors the real model)
	init_emb   = randn(Float32, D_lat, K) .* 0.1f0
	init_ema_cs = ones(Float32, K)
	init_ema_dw = copy(init_emb)
	rvq_state  = (; embedding=init_emb, ema_cs=init_ema_cs, ema_dw=init_ema_dw)
	st_full    = merge(st_init, (; rvq=rvq_state))

	ts = Training.TrainState(encoder, ps, Lux.trainmode(st_full), opt)

	# st_full = merge(st_init, (; rvq=...))  — Dense state fields are at top level alongside :rvq
	# Strip :rvq to get the encoder-only state for the model forward pass
	enc_st_keys = filter(k -> k != :rvq, keys(st_full))
	enc_st(st) = NamedTuple{enc_st_keys}(map(k -> getfield(st, k), enc_st_keys))

	function loss_fn2(model, ps, st, data)
	    x, z_q_pre = data
	    z_e, st2 = model(x, ps, enc_st(st))
	    z_q_ste    = z_e .+ EnzymeCore.ignore_derivatives(z_q_pre .- z_e)
	    recon_loss  = mean((z_q_ste .- x[1:D_lat, :]) .^ 2)
	    commit_loss = β * mean((z_e .- EnzymeCore.ignore_derivatives(z_q_pre)) .^ 2)
	    # rvq not touched in diff graph — carry it through unchanged
	    return recon_loss + commit_loss, merge(st2, (; rvq=st.rvq)), (;)
	end

	rows = []
	for step in 1:5
	    x = randn(Float32, D_in, N)
	    # CPU-side EMA update from current ts.states.rvq
	    cur_rvq = ts.states.rvq
	    z_e_cpu, _ = Lux.apply(encoder, x, ts.parameters, Lux.testmode(enc_st(ts.states)))
	    z_q_pre, new_emb, new_ema_cs, new_ema_dw, perp = vq_quantize_cpu(
	        Float32.(z_e_cpu), cur_rvq.embedding, cur_rvq.ema_cs, cur_rvq.ema_dw, K, decay, ε)
	    st_updated_rvq = (; embedding=new_emb, ema_cs=new_ema_cs, ema_dw=new_ema_dw)

	    emb_before = copy(cur_rvq.embedding)

	    # inject updated rvq before step (mirrors replace_train_state_states)
	    st_with_rvq = merge(ts.states, (; rvq=st_updated_rvq))
	    ts = Training.TrainState(ts.cache, ts.objective_function, ts.allocator_cache,
	        encoder, ts.parameters, st_with_rvq, ts.optimizer, ts.optimizer_state, ts.step)

	    _, _, _, ts = Training.single_train_step!(AutoEnzyme(), loss_fn2, (x, z_q_pre), ts)

	    emb_after_step = ts.states.rvq.embedding

	    diff_from_ema  = norm(emb_after_step - new_emb)     # should be 0 with fix, >0 means clobbered
	    diff_from_prev = norm(new_emb - emb_before)          # EMA moved the codebook

	    push!(rows, (
	        step            = step,
	        ema_changed     = round(diff_from_prev; digits=6),
	        clobbered_after_step = round(diff_from_ema; digits=6),
	        perplexity      = round(perp; digits=3),
	        ok = diff_from_ema < 1f-6 ? "✓ fix works" : "✗ CLOBBERED"
	    ))
	end

	rows
end

# ╔═╡ 00000000-0000-0000-0000-000000000050
md"""
## Test 4: Logic check for `merge` re-inject

Sanity check that `merge(old_st, (; rvq=new_rvq))` correctly replaces the rvq field.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000051
let
	old_emb = ones(Float32, 4, 4)
	new_emb = 2f0 .* ones(Float32, 4, 4)

	st_old = (; rvq=(; embedding=old_emb, x=42), encoder=(; w=3))
	st_new_rvq = (; embedding=new_emb, x=42)

	st_after_fix    = merge(st_old, (; rvq=st_new_rvq))
	st_after_no_fix = st_old

	(
	    without_fix = st_after_no_fix.rvq.embedding[1,1],   # expect 1.0
	    with_fix    = st_after_fix.rvq.embedding[1,1],       # expect 2.0
	    encoder_preserved = st_after_fix.encoder.w,           # expect 3
	    verdict = st_after_fix.rvq.embedding[1,1] == 2f0 ? "✓ merge re-inject is correct" : "✗ BUG"
	)
end

# ╔═╡ 00000000-0000-0000-0000-000000000099
md"---"

# ╔═╡ 00000000-0000-0000-0000-000000000060
md"""
## Test 5: Reactant GPU backend — does `single_train_step!` clobber `ts.states.rvq`?

This is the definitive test. Same structure as Test 3 but using `ReactantDevice` and
`@compile`. After the step, `ts.states.rvq.embedding` should equal `new_emb` (re-inject
survived), not `init_emb` (clobbered by XLA frozen state).

`clobbered_after_step ≈ 0` → fix works on GPU.
`clobbered_after_step > 0` → Reactant still overwrites rvq despite the re-inject.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000061
let
	
	rdev = reactant_device()
	cdev = cpu_device()

	Random.seed!(42)
	K, D_in, D_lat, N = 4, 16, 8, 32
	decay, ε, β = 0.99f0, 1f-5, 0.25f0

	encoder = Dense(D_in, D_lat)
	ps_cpu, st_cpu = Lux.setup(Random.default_rng(), encoder)

	init_emb    = randn(Float32, D_lat, K) .* 0.1f0
	init_ema_cs = ones(Float32, K)
	init_ema_dw = copy(init_emb)
	rvq_state   = (; embedding=init_emb, ema_cs=init_ema_cs, ema_dw=init_ema_dw)
	st_full_cpu = merge(st_cpu, (; rvq=rvq_state))

	# Move to Reactant device
	ps  = rdev(ps_cpu)
	st  = rdev(Lux.trainmode(st_full_cpu))
	opt = Optimisers.Adam(1f-3)
	ts  = Training.TrainState(encoder, ps, st, opt)

	enc_st_keys = filter(k -> k != :rvq, keys(st_full_cpu))
	enc_st(s) = NamedTuple{enc_st_keys}(map(k -> getfield(s, k), enc_st_keys))

	function loss_fn_r(model, ps, st, data)
	    x, z_q_pre = data
	    z_e, st2 = model(x, ps, enc_st(st))
	    z_q_ste    = z_e .+ Reactant.ignore_derivatives(z_q_pre .- z_e)
	    recon_loss  = mean((z_q_ste .- x[1:D_lat, :]) .^ 2)
	    commit_loss = β * mean((z_e .- Reactant.ignore_derivatives(z_q_pre)) .^ 2)
	    return recon_loss + commit_loss, merge(st2, (; rvq=st.rvq)), (;)
	end

	rows = []
	for step in 1:3
	    x_cpu = randn(Float32, D_in, N)

	    # CPU-side EMA update (mirrors prepare_vq_training_batch)
	    cur_rvq = cdev(ts.states.rvq)
	    ps_c    = cdev(ts.parameters)
	    st_c    = Lux.testmode(cdev(ts.states))
	    z_e_cpu = Float32.(cdev(Lux.apply(encoder, x_cpu, ps_c, enc_st(st_c))[1]))
	    _, new_emb, new_ema_cs, new_ema_dw, perp = vq_quantize_cpu(
	        z_e_cpu, cur_rvq.embedding, cur_rvq.ema_cs, cur_rvq.ema_dw, K, decay, ε)
	    st_updated_rvq = rdev((; embedding=new_emb, ema_cs=new_ema_cs, ema_dw=new_ema_dw))

	    emb_before = copy(new_emb)   # what we injected

	    # inject updated rvq (mirrors replace_train_state_states + re-inject fix)
	    st_with_rvq = merge(ts.states, (; rvq=st_updated_rvq))
	    ts = Training.TrainState(ts.cache, ts.objective_function, ts.allocator_cache,
	        encoder, ts.parameters, st_with_rvq, ts.optimizer, ts.optimizer_state, ts.step)

	    x_dev   = rdev(x_cpu)
	    z_q_dev = rdev(Float32.(Lux.apply(encoder, x_cpu, cdev(ts.parameters), enc_st(cdev(ts.states)))[1]))
	    _, _, _, ts = Training.single_train_step!(
	        AutoEnzyme(), loss_fn_r, (x_dev, z_q_dev), ts)

	    # Re-inject fix (mirrors the fix in VQVAE_architecture_v7)
	    ts = Training.TrainState(ts.cache, ts.objective_function, ts.allocator_cache,
	        encoder, ts.parameters,
	        merge(ts.states, (; rvq=st_updated_rvq)),
	        ts.optimizer, ts.optimizer_state, ts.step)

	    emb_after = Float32.(cdev(ts.states.rvq.embedding))
	    diff = norm(emb_after - emb_before)

	    push!(rows, (
	        step                 = step,
	        clobbered_after_step = round(diff; digits=6),
	        perplexity           = round(perp; digits=3),
	        ok = diff < 1f-4 ? "✓ fix works on GPU" : "✗ CLOBBERED on GPU"
	    ))
	end

	rows
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CUDA = "~6.0.0"
Enzyme = "~0.13.140"
EnzymeCore = "~0.8.20"
Lux = "~1.31.4"
Optimisers = "~0.4.7"
Reactant = "~0.2.254"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "817fb62b38d455cafbea65deb9a2207708877428"

[[deps.ADTypes]]
git-tree-sha1 = "bbc22a9a08a0ef6460041086d8a7b27940ed4ffd"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.22.0"
weakdeps = ["ChainRulesCore", "ConstructionBase", "EnzymeCore"]

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0761717147821d696c9470a7a86364b2fbd22fd8"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.2"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "54f895554d05c83e3dd59f6a396671dae8999573"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.24.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

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

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BufferedStreams]]
git-tree-sha1 = "6863c5b7fc997eadcabdbaf6c5f201dc30032643"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.2.2"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.CUDA]]
deps = ["CUDACore", "CUDATools", "Reexport", "cuBLAS", "cuFFT", "cuRAND", "cuSOLVER", "cuSPARSE"]
git-tree-sha1 = "bcbaecc92b4b8b0fb25997f4d84451b198344d4d"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "6.0.0"

[[deps.CUDACore]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "PrecompileTools", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "StaticArrays"]
git-tree-sha1 = "dc5b6ea53fa3b3bedd2fe1c6037687dd4ee85e70"
uuid = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
version = "6.0.0"
weakdeps = ["ChainRulesCore", "EnzymeCore", "SpecialFunctions"]

    [deps.CUDACore.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.CUDATools]]
deps = ["CUDACore", "CUDA_Compiler_jll", "CUPTI", "Crayons", "GPUCompiler", "LLVM", "NVML", "NVTX", "PrecompileTools", "Preferences", "PrettyTables", "Printf", "Statistics", "demumble_jll"]
git-tree-sha1 = "38ee815c0b8b1423035d10f657f9f756e39c5205"
uuid = "9ec180c6-1c07-47c7-9e6e-ebefa4d1f6d0"
version = "6.0.0"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "b977706846cb0a75d3842a1fed810ab2e6ab2f94"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.3+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "3b759ec65ac87ad192c2925114fa5c126657a5bd"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.2.1+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c0314d9fb0ebd00e404feba4c3fbc04c9975abc1"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.21.0+1"

[[deps.CUPTI]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox"]
git-tree-sha1 = "b37790736de8e067a26ade5cbcd6bf240ddd20ec"
uuid = "9e67e8f6-ba02-4b6c-a7db-3b11ae1e7ab7"
version = "6.0.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

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

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

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

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "42cd00edaac86f941815fe557c1d01e11913e07c"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.28"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "78704dd8d84c93a7f2ac5af0bbb95d26763ec9b9"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.140"
weakdeps = ["ADTypes", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "LogExpFunctions", "SpecialFunctions", "StaticArrays"]

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeGPUArraysCoreExt = "GPUArraysCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

[[deps.EnzymeCore]]
git-tree-sha1 = "c6ee69ee502060982d12dbaaf3d8fcb4e835a0d1"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.20"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "d3ad8f5eca369ac8803ff7db660028d47debc75d"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.258+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "5f1c005ed214356bbe41d442cc1ccd416e510b7e"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.4"

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cddeab6487248a39dae1a960fff0ac17b2a28888"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.3"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "34fd745547978beb471f029f447290ef4dbc7bbd"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.5.3"

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
git-tree-sha1 = "fedfe5e7db7035271c3f58359007f971da1dde87"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.9.1"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "a589b6c1a0eff953571f5d8b0474f5020831114d"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.1.1"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

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
git-tree-sha1 = "fe23330af47b8ab4e135b2ff65f7398c3a2bfc65"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.2"

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

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "f2e76d3ced51a2a9e185abc0b97494c7273f649f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.41"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "85592339c4363f40863f0b61f9cba80b885070c3"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.7.1"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "f1d1adfff151fd02b4062d1af82df02052dc4a0c"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.42+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

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

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.Lux]]
deps = ["ADTypes", "Adapt", "ArrayInterface", "ChainRulesCore", "ConcreteStructs", "DiffResults", "DispatchDoctor", "EnzymeCore", "FastClosures", "ForwardDiff", "Functors", "GPUArraysCore", "LinearAlgebra", "LuxCore", "LuxLib", "MLDataDevices", "MacroTools", "Markdown", "NNlib", "Optimisers", "PrecompileTools", "Preferences", "Random", "ReactantCore", "Reexport", "SciMLPublic", "Setfield", "Static", "StaticArraysCore", "Statistics", "UUIDs", "WeightInitializers"]
git-tree-sha1 = "b7654d9b1144792d7fa165add2e07434329e3193"
uuid = "b2108857-7c20-44ae-9111-449ecde12c47"
version = "1.31.4"

    [deps.Lux.extensions]
    ComponentArraysExt = "ComponentArrays"
    EnzymeExt = "Enzyme"
    FluxExt = "Flux"
    GPUArraysExt = "GPUArrays"
    LossFunctionsExt = "LossFunctions"
    MLUtilsExt = "MLUtils"
    MPIExt = "MPI"
    MPINCCLExt = ["CUDA", "MPI", "NCCL"]
    MooncakeExt = "Mooncake"
    ReactantExt = ["Enzyme", "Reactant"]
    ReverseDiffExt = ["FunctionWrappers", "ReverseDiff"]
    SimpleChainsExt = "SimpleChains"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"

    [deps.Lux.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    FunctionWrappers = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SimpleChains = "de6bee2f-e2f4-4ec7-b6ed-219cc6f6e9e5"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LuxCore]]
deps = ["DispatchDoctor", "Random", "SciMLPublic"]
git-tree-sha1 = "9455b1e829d8dacad236143869be70b7fdb826b8"
uuid = "bb33d45b-7691-41d6-9220-0943567d0623"
version = "1.5.3"

    [deps.LuxCore.extensions]
    ArrayInterfaceReverseDiffExt = ["ArrayInterface", "ReverseDiff"]
    ArrayInterfaceTrackerExt = ["ArrayInterface", "Tracker"]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    FluxExt = "Flux"
    FunctorsExt = "Functors"
    MLDataDevicesExt = ["Adapt", "MLDataDevices"]
    ReactantExt = "Reactant"
    SetfieldExt = "Setfield"

    [deps.LuxCore.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Setfield = "efcf1570-3423-57d1-acb7-fd33fddbac46"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.LuxLib]]
deps = ["ArrayInterface", "CPUSummary", "ChainRulesCore", "DispatchDoctor", "EnzymeCore", "FastClosures", "Functors", "KernelAbstractions", "LinearAlgebra", "LuxCore", "MLDataDevices", "Markdown", "NNlib", "Preferences", "Random", "Reexport", "SciMLPublic", "Static", "StaticArraysCore", "Statistics", "UUIDs"]
git-tree-sha1 = "6a6453d556f7bc3870d797657636b1ad5f45fd27"
uuid = "82251201-b29d-42c6-8e01-566dec8acb11"
version = "1.15.9"

    [deps.LuxLib.extensions]
    AppleAccelerateExt = "AppleAccelerate"
    BLISBLASExt = "BLISBLAS"
    CUDAExt = "CUDA"
    CUDAForwardDiffExt = ["CUDA", "ForwardDiff"]
    EnzymeExt = "Enzyme"
    ForwardDiffExt = "ForwardDiff"
    LoopVectorizationExt = ["LoopVectorization", "Polyester"]
    MKLExt = "MKL"
    OctavianExt = ["Octavian", "LoopVectorization"]
    OneHotArraysExt = ["OneHotArrays"]
    ReactantExt = ["Reactant", "ReactantCore"]
    ReverseDiffExt = "ReverseDiff"
    SLEEFPiratesExt = "SLEEFPirates"
    TrackerAMDGPUExt = ["AMDGPU", "Tracker"]
    TrackerExt = "Tracker"
    cuDNNExt = ["CUDA", "cuDNN"]

    [deps.LuxLib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    AppleAccelerate = "13e28ba4-7ad8-5781-acae-3021b1ed3924"
    BLISBLAS = "6f275bd8-fec0-4d39-945b-7e95a765fa1e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890"
    MKL = "33e6dc65-8f57-5167-99aa-e5a354878fb2"
    Octavian = "6fd5a793-0b7e-452c-907f-f8bfe9c57db4"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    ReactantCore = "a3311ec8-5e00-46d5-b541-4f83e724a433"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SLEEFPirates = "476501e8-09a2-5ece-8869-fb82de89a1fa"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "29b00f22be6fd821a214760f0224329f21998a05"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.10"

    [deps.MLDataDevices.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    ChainRulesExt = "ChainRules"
    ComponentArraysExt = "ComponentArrays"
    FillArraysExt = "FillArrays"
    GPUArraysSparseArraysExt = ["GPUArrays", "SparseArrays"]
    MLUtilsExt = "MLUtils"
    MetalExt = ["GPUArrays", "Metal"]
    OneHotArraysExt = "OneHotArrays"
    OpenCLExt = ["GPUArrays", "OpenCL"]
    ReactantExt = "Reactant"
    RecursiveArrayToolsExt = "RecursiveArrayTools"
    ReverseDiffExt = "ReverseDiff"
    SparseArraysExt = "SparseArrays"
    TrackerExt = "Tracker"
    ZygoteExt = "Zygote"
    cuDNNExt = ["CUDA", "cuDNN"]
    oneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "78cd28dbd5f03f99ccaba45c987107adcb61c115"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.34"

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

[[deps.NVML]]
deps = ["CEnum", "CUDACore", "GPUToolbox", "Libdl"]
git-tree-sha1 = "d041854ab4c16d1b1b6d8ba1092183745a7fe26a"
uuid = "611af6d1-644e-4c5d-bd58-854d7d1254b9"
version = "6.0.0"

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

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "22faba70c22d2f03e60fbc61da99c4ebfc3eb9ba"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.5.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "36b5d2b9dd06290cd65fcf5bdbc3a551ed133af5"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.7"
weakdeps = ["Adapt", "EnzymeCore", "Reactant"]

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "5d5e0a78e971354b1c7bff0655d11fdc1b0e12c8"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.4"

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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "624de6279ab7d94fc9f672f0068107eb6619732c"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.3.2"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProtoBuf]]
deps = ["BufferedStreams", "EnumX", "TOML"]
git-tree-sha1 = "da18083a52d9d57bbe6dadaacad39731e5f7be39"
uuid = "3349acd9-ac6a-5e09-bcdb-63829b23a429"
version = "1.3.0"

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

[[deps.Reactant]]
deps = ["Adapt", "BFloat16s", "CEnum", "Crayons", "Downloads", "EnumX", "Enzyme", "EnzymeCore", "FileWatching", "Functors", "GPUArraysCore", "GPUCompiler", "HTTP", "JSON", "LLVM", "LLVMOpenMP_jll", "Libdl", "LinearAlgebra", "OrderedCollections", "PrecompileTools", "Preferences", "PrettyTables", "ProtoBuf", "Random", "ReactantCore", "Reactant_jll", "ScopedValues", "Scratch", "Serialization", "Setfield", "Sockets", "StableRNGs", "StructUtils", "StyledStrings", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "e02293894a505abfc68ef5e0743d6035d411c64f"
uuid = "3c362404-f566-11ee-1572-e11a4b42c853"
version = "0.2.254"

    [deps.Reactant.extensions]
    ReactantAbstractFFTsExt = "AbstractFFTs"
    ReactantArrayInterfaceExt = "ArrayInterface"
    ReactantCUDAExt = ["CUDA", "Enzyme", "GPUCompiler", "KernelAbstractions", "LLVM", "Printf"]
    ReactantDLFP8TypesExt = "DLFP8Types"
    ReactantDatesExt = "Dates"
    ReactantFFTWExt = ["FFTW", "AbstractFFTs", "LinearAlgebra"]
    ReactantFillArraysExt = "FillArrays"
    ReactantFloat8sExt = "Float8s"
    ReactantKernelAbstractionsExt = "KernelAbstractions"
    ReactantLogExpFunctionsExt = ["IrrationalConstants", "LogExpFunctions"]
    ReactantMCMCDiagnosticToolsExt = ["MCMCDiagnosticTools", "Statistics"]
    ReactantMPIExt = "MPI"
    ReactantNNlibExt = ["NNlib", "Statistics"]
    ReactantNPZExt = "NPZ"
    ReactantOffsetArraysExt = "OffsetArrays"
    ReactantOneHotArraysExt = "OneHotArrays"
    ReactantPythonCallExt = "PythonCall"
    ReactantRandom123Ext = "Random123"
    ReactantSparseArraysExt = "SparseArrays"
    ReactantSpecialFunctionsExt = "SpecialFunctions"
    ReactantStaticArraysExt = "StaticArrays"
    ReactantStatisticsExt = "Statistics"
    ReactantStructArraysExt = "StructArrays"
    ReactantYaoBlocksExt = "YaoBlocks"
    ReactantZygoteExt = "Zygote"

    [deps.Reactant.weakdeps]
    AbstractFFTs = "621f4979-c628-5d54-868e-fcf4e3e8185c"
    ArrayInterface = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    DLFP8Types = "f4c16678-4a16-415b-82ef-ed337c5d6c7c"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    Float8s = "81dfefd7-55b0-40c6-a251-db853704e186"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    MCMCDiagnosticTools = "be115224-59cd-429b-ad48-344e309966f0"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    NPZ = "15e1cf62-19b3-5cfa-8e77-841668bca605"
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    Random123 = "74087812-796a-5b5d-8853-05524746bad3"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    YaoBlocks = "418bc28f-b43b-5e0b-a6e7-61bbc1a2c1df"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.ReactantCore]]
deps = ["ExpressionExplorer", "MacroTools"]
git-tree-sha1 = "5b9e0fe7fb2cf3794fd96ac32bf2732aa4bb9776"
uuid = "a3311ec8-5e00-46d5-b541-4f83e724a433"
version = "0.1.19"

[[deps.Reactant_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2749c35cb1bcc588ad71a50acf19108b9c6e47ed"
uuid = "0192cb87-2b54-54ad-80e0-3be72ad8a3c0"
version = "0.0.371+0"

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

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

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

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "bb072715f158b59ad8819ff80da5ffa90cce6ceb"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.4.0"

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

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "dd974aefe288ef2898733aecf40858dc86742d74"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.1"

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

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

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

[[deps.WeightInitializers]]
deps = ["ConcreteStructs", "GPUArraysCore", "LinearAlgebra", "Random", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "2af44c69f5c37b7b1d14e262347a24ba349052d6"
uuid = "d49dbf32-c5c2-4618-8acc-27bb2598ef2d"
version = "1.3.3"

    [deps.WeightInitializers.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"
    ChainRulesCoreExt = "ChainRulesCore"
    GPUArraysExt = "GPUArrays"
    ReactantExt = "Reactant"

    [deps.WeightInitializers.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.cuBLAS]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "LLVM", "LinearAlgebra"]
git-tree-sha1 = "5df9edbdfff9fed8b818535e7b86e92a85fc7709"
uuid = "182d3088-87b7-4494-8cad-fc6afaa545bc"
version = "6.0.0"
weakdeps = ["EnzymeCore"]

    [deps.cuBLAS.extensions]
    EnzymeCoreExt = "EnzymeCore"

[[deps.cuFFT]]
deps = ["AbstractFFTs", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "c5de5ab272aae86658d3b05999b9ea7bc60503d0"
uuid = "533571aa-0936-420e-b4be-9c66f5f626ca"
version = "6.0.0"

[[deps.cuRAND]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "Random", "Random123", "RandomNumbers"]
git-tree-sha1 = "43d84e8d12e75c401d69d88475d304ca7a038afd"
uuid = "20fd9a0b-12d5-4c2f-a8af-7c34e9e60431"
version = "6.0.0"

[[deps.cuSOLVER]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "SparseArrays", "cuBLAS", "cuSPARSE"]
git-tree-sha1 = "4b15758b0667ba4b715252fe0dfae9dafae1b739"
uuid = "887afef0-6a32-4de5-add4-7827692ba8fc"
version = "6.0.0"

[[deps.cuSPARSE]]
deps = ["Adapt", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "KernelAbstractions", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f5d1fdae1053286374c80e5f6608a913aedad7ef"
uuid = "b26da814-b3bc-49ef-b0ee-c816305aa060"
version = "6.0.0"

    [deps.cuSPARSE.extensions]
    SparseMatricesCSRExt = "SparseMatricesCSR"

    [deps.cuSPARSE.weakdeps]
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"

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
# ╠═00000000-0000-0000-0000-000000000010
# ╠═00000000-0000-0000-0000-000000000011
# ╠═00000000-0000-0000-0000-000000000020
# ╠═4035e565-793a-4bc3-a701-034b0c02f058
# ╠═00000000-0000-0000-0000-000000000021
# ╠═00000000-0000-0000-0000-000000000030
# ╠═00000000-0000-0000-0000-000000000031
# ╠═00000000-0000-0000-0000-000000000040
# ╠═00000000-0000-0000-0000-000000000041
# ╠═00000000-0000-0000-0000-000000000050
# ╠═00000000-0000-0000-0000-000000000051
# ╠═00000000-0000-0000-0000-000000000099
# ╠═f78ead71-604e-412a-8766-dbb7395e685b
# ╠═1f3d5cfc-1eb3-497b-8b25-e49f4d1561b1
# ╠═b543fb06-9d3a-4cf8-9f78-ce807fa875d0
# ╠═12a7cffc-f64a-4785-8edb-70612e945f6b
# ╠═00000000-0000-0000-0000-000000000060
# ╠═51ccb814-5539-402a-9b68-fdd7068f2c07
# ╠═00000000-0000-0000-0000-000000000061
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
