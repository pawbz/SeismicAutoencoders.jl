### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 10000001-0000-0000-0000-000000000001
begin
    using CUDA, ConcreteStructs,
        Dates,
        DSP,
        FFTW,
        Enzyme,
        EnzymeCore,
        Flux,
        JLD2,
        LinearAlgebra,
        MLUtils,
        NNlib,
        Optimisers,
        ProgressLogging,
        PlutoPlotly,
        Random,
        Statistics,
        StatsBase,
        InlineStrings,
        Zygote
end

# ╔═╡ 330652f1-0754-48a4-9a0f-4fb9d6824222
using Distances

# ╔═╡ 10000002-0000-0000-0000-000000000001
md"""
# VQ-VAE v9 Architecture (Flux) — Split-Decoder Interferometric Mixture VQ

Flux single-pair Interferometric Split-Decoder VQ-VAE with SEANet-style encoder/decoder.
Each station pair trains an independent model. Two separate encoder heads map shared features
to z_e1 and z_e2 (each d÷2), quantized independently. Two separate decoders produce x1_hat
and x2_hat; reconstruction is additive: x_hat = x1_hat + x2_hat, forcing specialization.

This is the Flux version of VQVAE_architecture_v9.jl — no Reactant, no XLA compilation.
"""

# ╔═╡ 10000003-0000-0000-0000-000000000001
begin
    const activation = x -> NNlib.leakyrelu(x, 0.1f0)

    default_xdev(; force::Bool=true) = gpu_device()
    default_cdev() = cpu_device()

    # Straight-through / stop-gradient helper for Zygote.
    # EnzymeCore.ignore_derivatives(x) is replaced by stop_grad(x) throughout.
    stop_grad(x) = x
    Zygote.@adjoint stop_grad(x) = x, _ -> (nothing,)
end

# ╔═╡ 10000004-0000-0000-0000-000000000001
md"## Parameters"

# ╔═╡ 10000005-0000-0000-0000-000000000001
Base.@kwdef struct VQVAE_Para
    nt::Int
    d::Int = 64
    beta_commit::Float32 = 0.25f0
    n_filters::Int = 32
    ratios::Vector{Int} = [4, 2]
    n_residual_layers::Int = 1
    dilation_base::Int = 2
    residual_kernel_size::Int = 3
    enc_kernel_size::Int = 7
    dec_kernel_size::Int = 7
    use_bn::Bool = false
    K::Vector{Int} = [5, 5]
    ema_decay::Float32 = 0.99f0
    epsilon::Float32 = 1f-5
    dead_threshold::Int = 50
    entropy_weight::Float32 = 0.01f0
    reconstruction_loss::Symbol = :l2
    seed::Int = 1234
end

# ╔═╡ 10000006-0000-0000-0000-000000000001
Base.@kwdef struct VQVAE_Training_Para
    batchsize::Int = 256
    nepoch::Int = 30
    nprint::Int = 1
    initial_learning_rate::Float64 = 0.001
    weight_decay::Float64 = 0.0
    stop_on_recon_loss::Union{Nothing,Float64} = nothing
    Mnn::Union{Nothing,Int} = nothing
    Mnn_schedule::Vector{Tuple{Int,Int}} = [(1, 5), (6, 10), (26, 25)]
    warmup_epochs::Int = 5
    index_refresh_every::Int = 1
    autodiff_backend::Symbol = :zygote
    normalize_target::Bool = true
    verbose::Bool = false
    knn_search_chunk_size_fraction::Float64 = 0.5
end

# ╔═╡ 10000007-0000-0000-0000-000000000001
md"## Array and Geometry Helpers"

# ╔═╡ 10000008-0000-0000-0000-000000000001
begin
    flatten_batch(x::AbstractVector) = reshape(x, :, 1)
    flatten_batch(x::AbstractMatrix) = x
    flatten_batch(x::AbstractArray) = reshape(x, size(x, 1), :)

    function waveform_to_conv3(x)
        xf = flatten_batch(x)
        return reshape(xf, size(xf, 1), 1, size(xf, 2))
    end

    function conv3_to_waveform(y)
        if ndims(y) == 3 && size(y, 2) == 1
            return reshape(y, size(y, 1), size(y, 3))
        end
        return reshape(y, size(y, 1), :)
    end

    mse_loss(x, y) = mean(abs2, x .- y)

    function replace_tuple(t::Tuple, i::Int, x)
        return ntuple(j -> j == i ? x : t[j], length(t))
    end

end

# ╔═╡ 1000000a-0000-0000-0000-000000000001
md"## Flux Layer Helpers (ResidualWrap, FluxDecoder, FluxEncoderHead)"

# ╔═╡ flux-residual-wrap
begin
    # Drop-in replacement for Lux @compact residual blocks
    struct ResidualWrap
        block
    end
    Flux.@functor ResidualWrap
    (r::ResidualWrap)(x) = x + r.block(x)

    # Drop-in replacement for Lux @compact decoder
    struct FluxDecoder
        linear
        upchain
        bottleneck_len::Int
        bottleneck_channels::Int
    end
    Flux.@functor FluxDecoder (linear, upchain)
    function (d::FluxDecoder)(z)
        y = d.linear(z)
        y3 = reshape(y, d.bottleneck_len, d.bottleneck_channels, size(y, 2))
        out = d.upchain(y3)
        return conv3_to_waveform(out)
    end

    # Drop-in replacement for Lux @compact encoder head
    struct FluxEncoderHead
        conv
        dense
        latent_len::Int
        enc_channels::Int
    end
    Flux.@functor FluxEncoderHead (conv, dense)
    function (h::FluxEncoderHead)(feat)
        y = h.conv(feat)
        B = size(feat, 3)
        y_flat = reshape(permutedims(y, (2, 1, 3)), h.enc_channels * h.latent_len, B)
        return h.dense(y_flat)
    end
end

# ╔═╡ 1000000b-0000-0000-0000-000000000001
md"## Flux Encoder and Decoder"

# ╔═╡ ef72f750-3026-4c6b-8cf6-9fc8964658f8
function make_seanet_residual_block(dim, n_residual_layers, dilation_base, residual_kernel_size)
    layers = Any[]
    hidden = max(1, dim ÷ 2)
    for j in 0:n_residual_layers-1
        d = dilation_base ^ j
        block = Flux.Chain(
            activation,
            Flux.Conv((residual_kernel_size,), dim => hidden, activation; pad=SamePad(), dilation=d),
            Flux.Conv((1,), hidden => dim; pad=SamePad()),
        )
        push!(layers, ResidualWrap(block))
    end
    return Flux.Chain(layers...)
end

# ╔═╡ 6ce85f19-73ba-4785-ab52-1767cf79efa8
function make_encoder(para)
    ratios = reverse(para.ratios)
    mult = 1
    layers = Any[]
    push!(layers, Flux.Conv((para.enc_kernel_size,), 1 => mult * para.n_filters; pad=SamePad()))
    for ratio in ratios
        push!(layers, make_seanet_residual_block(
            mult * para.n_filters, para.n_residual_layers, para.dilation_base, para.residual_kernel_size))
        push!(layers, activation)
        push!(layers, Flux.Conv((ratio * 2,), mult * para.n_filters => mult * para.n_filters * 2;
            stride=ratio, pad=SamePad()))
        mult *= 2
    end
    push!(layers, activation)
    push!(layers, Flux.Conv((para.enc_kernel_size,), mult * para.n_filters => mult * para.n_filters; pad=SamePad()))
    return Flux.Chain(layers...)
end

# ╔═╡ a0416f71-242a-44a7-b8f6-186726318611
function make_decoder(para, latent_len::Int; latent_dim::Int=para.d)
    ratios = para.ratios
    mult = 2 ^ length(ratios)

    bottleneck_len = para.nt
    for r in ratios; bottleneck_len = cld(bottleneck_len, r); end
    bottleneck_channels = para.n_filters * mult

    layers = Any[]
    push!(layers, Flux.Conv((para.dec_kernel_size,), bottleneck_channels => mult * para.n_filters; pad=SamePad()))
    for ratio in ratios
        push!(layers, activation)
        push!(layers, Flux.ConvTranspose((ratio * 2,), mult * para.n_filters => mult * para.n_filters ÷ 2;
            stride=ratio, pad=SamePad()))
        mult ÷= 2
        push!(layers, make_seanet_residual_block(
            mult * para.n_filters, para.n_residual_layers, para.dilation_base, para.residual_kernel_size))
    end
    push!(layers, activation)
    push!(layers, Flux.Conv((para.dec_kernel_size,), para.n_filters => 1; pad=SamePad()))
    upchain = Flux.Chain(layers...)

    linear = Flux.Dense(latent_dim, bottleneck_len * bottleneck_channels, activation)
    return FluxDecoder(linear, upchain, bottleneck_len, bottleneck_channels)
end

# ╔═╡ 1000000c-0000-0000-0000-000000000001
md"## RVQ State and Lookup"

# ╔═╡ e75270cd-5842-4132-9028-559135b9401f
function init_rvq_stage(rng::AbstractRNG, d::Int, K::Int)
    embedding = randn(rng, Float32, d, K) .* (1f0 / max(K, 1))
    stage_rng = MersenneTwister(rand(rng, UInt))
    return (;
        embedding=embedding,
        ema_cluster_size=ones(Float32, K),
        ema_dw=copy(embedding),
        dead_count=zeros(Int32, K),
        rng=copy(stage_rng),
    )
end

# ╔═╡ f31b1233-fdc3-4cef-8836-1122a1b3e7d4
function init_rvq_state(rng::AbstractRNG, d::Int, K::Tuple)
    return (; stages=ntuple(i -> init_rvq_stage(rng, d, K[i]), length(K)))
end

# ╔═╡ b1a7fdb4-7263-496e-8f9f-4d8a4429f351
begin
	function vq_distances(embedding, z)
	    z_sq = sum(abs2, z; dims=1)
	    e_sq = sum(abs2, embedding; dims=1)
	    return e_sq' .+ z_sq .- 2f0 .* (embedding' * z)
	end

	_argmin_col_indices(idx::AbstractArray{<:CartesianIndex}) = getindex.(idx, 1)
	_argmin_col_indices(idx::AbstractArray) = idx
end

# ╔═╡ f8da318e-6ff2-4442-a737-28ff9d8f340a
function assignment_matrix(indices, K::Int)
    codes = reshape(collect(1:K), K, 1)
    return Float32.(codes .== reshape(indices, 1, :))
end

# ╔═╡ fe5ee63d-6944-4557-834e-8e21231485f0
function vq_lookup(embedding, z)
    dist = vq_distances(embedding, z)
    idx_raw = vec(argmin(dist; dims=1))
    indices = vec(_argmin_col_indices(idx_raw))
    enc = assignment_matrix(indices, size(embedding, 2))
    z_q = embedding * enc
    return z_q, indices
end

# ╔═╡ e635aa3f-19de-4ce9-a478-c1252ddb7979
function counts_and_sums(z, indices, K::Int)
    enc = assignment_matrix(indices, K)
    counts = vec(sum(enc; dims=2))
    sums = z * enc'
    return counts, sums
end

# ╔═╡ 5e71db90-a0d1-47a4-bb32-dba21163e029
function update_rvq_stage_state(stage, z, indices, decay::Float32, epsilon::Float32, dead_threshold::Int)
    K = size(stage.embedding, 2)
    z_detached = stop_grad(z)
    counts, sums = counts_and_sums(z_detached, indices, K)
    ema_cluster_size = decay .* stage.ema_cluster_size .+ (1f0 - decay) .* counts
    n = sum(ema_cluster_size)
    ema_cluster_size = (ema_cluster_size .+ epsilon) ./ (n + Float32(K) * epsilon) .* n
    ema_dw = decay .* stage.ema_dw .+ (1f0 - decay) .* sums
    embedding = ema_dw ./ reshape(max.(ema_cluster_size, epsilon), 1, :)

    dead = counts .< 0.5f0
    dead_count = ifelse.(dead, stage.dead_count .+ Int32(1), Int32(0))
    reset = dead_count .>= Int32(dead_threshold)
    N = max(size(z_detached, 2), 1)
    rng = copy(stage.rng)
    donor_idx = [1 + mod(k - 1, N) for k in 1:K]
    donors = z_detached[:, donor_idx]
    donors = donors .+ randn_like(rng, donors) .* 0.01f0
    reset_mask = reshape(Float32.(reset), 1, K)
    keep_mask = 1f0 .- reset_mask
    embedding = embedding .* keep_mask .+ donors .* reset_mask
    ema_dw = ema_dw .* keep_mask .+ donors .* reset_mask
    ema_cluster_size = ifelse.(reset, 1f0, ema_cluster_size)
    dead_count = ifelse.(reset, Int32(0), dead_count)
    return (; embedding=Float32.(embedding), ema_cluster_size=Float32.(ema_cluster_size),
        ema_dw=Float32.(ema_dw), dead_count=Int32.(dead_count), rng), counts
end

# ╔═╡ 0652338c-5ab5-463d-8a4f-1f2314a01ba8
function probs_entropy(counts)
    total = max(sum(counts), 1f-8)
    p = counts ./ total
    psafe = clamp.(p, 1f-10, 1f0)
    entropy = -sum(psafe .* log.(psafe))
    entropy_loss = sum(psafe .* log.(psafe))
    perplexity = exp(entropy)
    return perplexity, entropy_loss, entropy
end

# ╔═╡ 66ddf04d-ee60-4a0b-9444-d09af23685ea
begin
	function rvq_quantize(z_e, rvq_state, K::Tuple; beta_commit::Float32, ema_decay::Float32,
	    epsilon::Float32, dead_threshold::Int, p_stage2_shuffle::Float32=0f0, training::Bool)
	    residual = z_e
	    z_q_total = zero(z_e)
	    stages = rvq_state.stages
	    new_stages = stages
	    all_indices = ()
	    commit_loss = 0f0
	    entropy_loss = 0f0
	    perplexity_total = 0f0
	    stage_perplexities = ()

	    for s in eachindex(K)
	        stage = stages[s]
	        _, indices = stop_grad(vq_lookup(stage.embedding, residual))
	        # randomly permute stage-2+ assignments during training to discourage inter-stage correlation
	        indices = if training && s > 1 && p_stage2_shuffle > 0f0
	            rng_local = copy(stage.rng)
	            rand(rng_local, Float32) < p_stage2_shuffle ?
	                indices[randperm(rng_local, length(indices))] : indices
	        else
	            indices
	        end
	        z_q_detached = stop_grad(stage.embedding[:, indices])
	        if !training
	            all_indices = (all_indices..., reshape(indices, 1, :))
	        end

	        counts = if training
	            new_stage, counts_local = update_rvq_stage_state(stage, residual, indices,
	                ema_decay, epsilon, dead_threshold)
	            new_stages = replace_tuple(new_stages, s, new_stage)
	            counts_local
	        else
	            counts_and_sums(stop_grad(residual), indices, K[s])[1]
	        end

	        z_q = residual .+ stop_grad(z_q_detached .- residual)
	        z_q_total = z_q_total .+ z_q
	        residual = residual .- stop_grad(z_q_detached)
	        commit_loss += beta_commit * mse_loss(residual .+ stop_grad(z_q_detached),
	            stop_grad(z_q_detached))
	        perplexity, stage_entropy_loss, _ = probs_entropy(counts)
	        perplexity_total += perplexity
	        if !training
	            stage_perplexities = (stage_perplexities..., perplexity)
	        end
	        entropy_loss += stage_entropy_loss
	    end

	    nstage = Float32(length(K))
	    stage_indices = training ? nothing : vcat(all_indices...)
	    coarse_indices = training ? nothing : all_indices[1]
	    return (; z_q=z_q_total,
	        stage_indices,
	        coarse_indices=coarse_indices,
	        commit_loss=commit_loss / nstage,
	        entropy_loss=entropy_loss / nstage,
	        perplexity=perplexity_total / nstage,
	        stage_perplexities=(training ? nothing : stage_perplexities)), (; stages=new_stages)
	end

	function prepare_rvq_payload(z_e, rvq_state, K::Tuple; ema_decay::Float32,
	    epsilon::Float32, dead_threshold::Int, p_stage2_shuffle::Float32=0f0, training::Bool)
	    residual = z_e
	    stages = rvq_state.stages
	    new_stages = stages
	    z_q_stages = ()
	    counts_stages = ()
	    entropy_loss = 0f0
	    perplexity_total = 0f0

	    for s in eachindex(K)
	        stage = stages[s]
	        _, indices = vq_lookup(stage.embedding, residual)
	        # randomly permute stage-2+ assignments during training to discourage inter-stage correlation
	        indices = if training && s > 1 && p_stage2_shuffle > 0f0
	            rng_local = copy(stage.rng)
	            rand(rng_local, Float32) < p_stage2_shuffle ?
	                indices[randperm(rng_local, length(indices))] : indices
	        else
	            indices
	        end
	        z_q_detached = stage.embedding[:, indices]
	        z_q_stages = (z_q_stages..., Float32.(z_q_detached))

	        counts = if training
	            new_stage, counts_local = update_rvq_stage_state(stage, residual, indices,
	                ema_decay, epsilon, dead_threshold)
	            new_stages = replace_tuple(new_stages, s, new_stage)
	            counts_local
	        else
	            counts_and_sums(residual, indices, K[s])[1]
	        end

	        counts_stages = (counts_stages..., Float32.(counts))
	        residual = residual .- z_q_detached
	        perplexity, stage_entropy_loss, _ = probs_entropy(counts)
	        perplexity_total += perplexity
	        entropy_loss += stage_entropy_loss
	    end

	    nstage = Float32(length(K))
	    payload = (;
	        z_q_stages,
	        counts_stages,
	        entropy_loss=Float32(entropy_loss / nstage),
	        perplexity=Float32(perplexity_total / nstage),
	    )
	    return payload, (; stages=new_stages)
	end

	function rvq_quantize_precomputed(z_e, payload, K::Tuple; beta_commit::Float32)
	    residual = z_e
	    z_q_total = zero(z_e)
	    commit_loss = 0f0

	    for s in eachindex(K)
	        z_q_detached = stop_grad(payload.z_q_stages[s])
	        z_q = residual .+ stop_grad(z_q_detached .- residual)
	        z_q_total = z_q_total .+ z_q
	        commit_loss += beta_commit * mse_loss(residual, z_q_detached)
	        residual = residual .- z_q_detached
	    end

	    nstage = Float32(length(K))
	    return (; z_q=z_q_total,
	        stage_indices=nothing,
	        coarse_indices=nothing,
	        commit_loss=commit_loss / nstage,
	        entropy_loss=stop_grad(payload.entropy_loss),
	        perplexity=stop_grad(payload.perplexity),
	        stage_perplexities=nothing)
	end
end

# ╔═╡ 10000010-0000-0000-0000-000000000002
md"## Split VQ"

# ╔═╡ 10000010-0000-0000-0000-000000000003
begin
	# Independent quantization of two separate z_e halves — no amplitude mixing.
	function split_vq_quantize(z_e1, z_e2, rvq_state, K::Tuple;
	    beta_commit::Float32, ema_decay::Float32, epsilon::Float32,
	    dead_threshold::Int, training::Bool)

	    stage1 = rvq_state.stages[1]
	    stage2 = rvq_state.stages[2]

	    _, idx1 = stop_grad(vq_lookup(stage1.embedding, z_e1))
	    _, idx2 = stop_grad(vq_lookup(stage2.embedding, z_e2))
	    z_q1_det = stop_grad(stage1.embedding[:, idx1])
	    z_q2_det = stop_grad(stage2.embedding[:, idx2])

	    # STE per sub-vector
	    z_q1 = z_e1 .+ stop_grad(z_q1_det .- z_e1)
	    z_q2 = z_e2 .+ stop_grad(z_q2_det .- z_e2)

	    commit_loss = (beta_commit * mse_loss(z_e1, z_q1_det)
	                 + beta_commit * mse_loss(z_e2, z_q2_det)) / 2f0

	    new_stages = rvq_state.stages
	    counts1 = if training
	        new_s1, c1_loc = update_rvq_stage_state(stage1, z_e1, idx1,
	            ema_decay, epsilon, dead_threshold)
	        new_stages = replace_tuple(new_stages, 1, new_s1)
	        c1_loc
	    else
	        counts_and_sums(stop_grad(z_e1), idx1, K[1])[1]
	    end
	    counts2 = if training
	        new_s2, c2_loc = update_rvq_stage_state(stage2, z_e2, idx2,
	            ema_decay, epsilon, dead_threshold)
	        new_stages = replace_tuple(new_stages, 2, new_s2)
	        c2_loc
	    else
	        counts_and_sums(stop_grad(z_e2), idx2, K[2])[1]
	    end

	    perp1, ent1, _ = probs_entropy(counts1)
	    perp2, ent2, _ = probs_entropy(counts2)
	    stage_indices  = training ? nothing : vcat(reshape(idx1, 1, :), reshape(idx2, 1, :))
	    coarse_indices = training ? nothing : reshape(idx1, 1, :)

	    return (;
	        z_q1, z_q2,
	        stage_indices,
	        coarse_indices,
	        commit_loss,
	        entropy_loss=(ent1 + ent2) / 2f0,
	        perplexity=(perp1 + perp2) / 2f0,
	        stage_perplexities=(training ? nothing : (perp1, perp2)),
	    ), (; stages=new_stages)
	end

	# CPU-side precomputation: independent lookups + EMA, stores detached z_q halves.
	function prepare_split_payload(z_e1, z_e2, rvq_state, K::Tuple;
	    ema_decay::Float32, epsilon::Float32, dead_threshold::Int, training::Bool)

	    stages = rvq_state.stages
	    new_stages = stages

	    _, idx1 = vq_lookup(stages[1].embedding, z_e1)
	    _, idx2 = vq_lookup(stages[2].embedding, z_e2)

	    z_q1_det = Float32.(stages[1].embedding[:, idx1])
	    z_q2_det = Float32.(stages[2].embedding[:, idx2])

	    counts1 = if training
	        new_s1, c1_loc = update_rvq_stage_state(stages[1], z_e1, idx1,
	            ema_decay, epsilon, dead_threshold)
	        new_stages = replace_tuple(new_stages, 1, new_s1)
	        c1_loc
	    else
	        counts_and_sums(z_e1, idx1, K[1])[1]
	    end
	    counts2 = if training
	        new_s2, c2_loc = update_rvq_stage_state(stages[2], z_e2, idx2,
	            ema_decay, epsilon, dead_threshold)
	        new_stages = replace_tuple(new_stages, 2, new_s2)
	        c2_loc
	    else
	        counts_and_sums(z_e2, idx2, K[2])[1]
	    end

	    perp1, ent1, _ = probs_entropy(Float32.(counts1))
	    perp2, ent2, _ = probs_entropy(Float32.(counts2))

	    payload = (;
	        z_q_stages=(z_q1_det, z_q2_det),
	        counts_stages=(Float32.(counts1), Float32.(counts2)),
	        entropy_loss=Float32((ent1 + ent2) / 2f0),
	        perplexity=Float32((perp1 + perp2) / 2f0),
	    )
	    return payload, (; stages=new_stages)
	end

	# Training step: frozen codebook lookups from payload, STE only.
	function split_vq_quantize_precomputed(z_e1, z_e2, payload, K::Tuple; beta_commit::Float32)
	    z_q1_det = stop_grad(payload.z_q_stages[1])
	    z_q2_det = stop_grad(payload.z_q_stages[2])
	    z_q1 = z_e1 .+ stop_grad(z_q1_det .- z_e1)
	    z_q2 = z_e2 .+ stop_grad(z_q2_det .- z_e2)
	    commit_loss = (beta_commit * mse_loss(z_e1, z_q1_det)
	                 + beta_commit * mse_loss(z_e2, z_q2_det)) / 2f0
	    return (;
	        z_q1, z_q2,
	        stage_indices=nothing,
	        coarse_indices=nothing,
	        commit_loss,
	        entropy_loss=stop_grad(payload.entropy_loss),
	        perplexity=stop_grad(payload.perplexity),
	        stage_perplexities=nothing,
	    )
	end
end

# ╔═╡ 1000000e-0000-0000-0000-000000000001
md"## VQ-VAE Model"

# ╔═╡ 1000000f-0000-0000-0000-000000000001
begin
	struct VQVAE
	    encoder
	    head1       # FluxEncoderHead → z_e1 (d÷2)
	    head2       # FluxEncoderHead → z_e2 (d÷2)
	    decoder1    # FluxDecoder: z_q1 → x1_hat
	    decoder2    # FluxDecoder: z_q2 → x2_hat
	    K::Tuple
	    d::Int
	    latent_len::Int
	    beta_commit::Float32
	    ema_decay::Float32
	    epsilon::Float32
	    dead_threshold::Int
	end
	# Only the neural-network fields hold trainable parameters.
	Flux.@functor VQVAE (encoder, head1, head2, decoder1, decoder2)

	function encoder_latents(m, x)
	    feat = m.encoder(waveform_to_conv3(x))
	    z_e1 = m.head1(feat)   # (d÷2, B)
	    z_e2 = m.head2(feat)   # (d÷2, B)
	    z_e = vcat(z_e1, z_e2) # (d, B) — for kNN
	    return (; feat, z_e, z_e1, z_e2)
	end

	function encode(m, rvq_st, x; beta_commit::Float32=m.beta_commit, training::Bool=false)
	    lat = encoder_latents(m, x)
	    q, new_rvq = split_vq_quantize(lat.z_e1, lat.z_e2, rvq_st, m.K;
	        beta_commit, ema_decay=m.ema_decay, epsilon=m.epsilon,
	        dead_threshold=m.dead_threshold, training)
	    return merge(lat, q), new_rvq
	end

	function encode_z_e_inference(m, x)
	    lat = encoder_latents(m, x)
	    return lat.z_e
	end

	function encode_z_e_training(m, x)
	    lat = encoder_latents(m, x)
	    return lat.z_e
	end

	function decode_from_latents(m, result)
	    x1hat = m.decoder1(result.z_q1)
	    x2hat = m.decoder2(result.z_q2)
	    xhat = x1hat .+ x2hat
	    return merge(result, (; xhat, x1hat, x2hat))
	end

	function (m::VQVAE)(x, rvq_st; beta_commit::Float32=m.beta_commit, training::Bool=true)
	    enc, new_rvq = encode(m, rvq_st, x; beta_commit, training)
	    result = decode_from_latents(m, enc)
	    return result, new_rvq
	end

	function forward_with_precomputed_vq(m, x, rvq_st, payload;
	    beta_commit::Float32=m.beta_commit)
	    lat = encoder_latents(m, x)
	    q = split_vq_quantize_precomputed(lat.z_e1, lat.z_e2, payload, m.K; beta_commit)
	    return decode_from_latents(m, merge(lat, q))
	end

	codebook_size(m) = m.K[1]
	num_rvq_stages(m) = length(m.K)
	get_codebook(rvq_st, stage::Int=1) = Array(rvq_st.stages[stage].embedding)
	get_codebooks(rvq_st) = [Array(stage.embedding) for stage in rvq_st.stages]
end

# ╔═╡ 10000011-0000-0000-0000-000000000001
md"## Losses and kNN Targets"

# ╔═╡ 13634a6c-abda-4084-9b5b-f6761fd728ad
begin
	function vqvae_loss(model, rvq_st, x, target, para; training::Bool)
	    result, new_rvq = model(x, rvq_st; beta_commit=para.beta_commit, training)
	    recon_loss = mse_loss(result.xhat, target)
	    total = recon_loss + result.commit_loss + para.entropy_weight * result.entropy_loss
	    return total, new_rvq, (; result, recon_loss,
	        commit_loss=result.commit_loss, entropy_loss=result.entropy_loss,
	        perplexity=result.perplexity, stage_perplexities=result.stage_perplexities)
	end

	function vqvae_precomputed_loss(model, x, rvq_st, payload, target, para)
	    result = forward_with_precomputed_vq(model, x, rvq_st, payload; beta_commit=para.beta_commit)
	    recon_loss = mse_loss(result.xhat, target)
	    total = recon_loss + result.commit_loss + para.entropy_weight * result.entropy_loss
	    return total, (; result, recon_loss,
	        commit_loss=result.commit_loss, entropy_loss=result.entropy_loss,
	        perplexity=result.perplexity, stage_perplexities=result.stage_perplexities)
	end
end

# ╔═╡ 86dfe031-7e05-402b-8916-cc2d2758e6b4
begin
	mutable struct LatentIndex
	    embeddings::Matrix{Float32}
	    neighbor_ids::Matrix{Int}
	    dist_matrix::Matrix{Float32}
	    perm_scratch::Vector{Vector{Int}}
	    Mnn::Int
	end
	LatentIndex(Mnn::Int) = LatentIndex(zeros(Float32, 0, 0), zeros(Int, Mnn, 0), zeros(Float32, 0, 0), Vector{Int}[], Mnn)
end

# ╔═╡ 092511ab-104e-4577-8cf5-dc6deeb73ac7
function _l2_normalize_columns!(X::AbstractMatrix{Float32})
    for j in axes(X, 2)
        nrm = sqrt(sum(abs2, view(X, :, j)))
        X[:, j] ./= Float32(nrm + 1f-8)
    end
    return X
end

# ╔═╡ b1e8b57a-0fa3-492a-a3a1-036423f41373
function rebuild_latent_index!(idx::LatentIndex, model, rvq_st, X;
    Mnn::Int=idx.Mnn, device=identity, cdev=default_cdev(),
    knn_search_chunk_size_fraction::Float64=1.0, n_compiled::Union{Nothing,Int}=nothing)
    X_cpu = Float32.(cdev(flatten_batch(X)))
    _, N_full = size(X_cpu)
    # truncate to compiled size if needed
    N = if !isnothing(n_compiled) && N_full > n_compiled
        @info "Latent index: truncating pair to compiled size" N_full n_compiled ignored=N_full - n_compiled
        n_compiled
    else
        N_full
    end
    X_cpu = X_cpu[:, 1:N]
    Mnn >= 1 || error("Mnn must be >= 1.")
    N >= Mnn + 1 || error("Need at least Mnn + 1 samples; got N=$N and Mnn=$Mnn.")
    D = model.d
    chunk_size = max(Mnn + 1, round(Int, knn_search_chunk_size_fraction * N))
    approximate = chunk_size < N
    embeddings = Matrix{Float32}(undef, D, N)
    # Run encoder on GPU, move result to CPU
    model_gpu = device(model)
    z_e = Float32.(cdev(encode_z_e_inference(model_gpu, device(X_cpu))))
    embeddings .= z_e
    normalize_start = time()
    _l2_normalize_columns!(embeddings)
    normalize_time = time() - normalize_start
    knn_start = time()
    nthreads = Threads.nthreads() + 1
    neighbor_ids = Matrix{Int}(undef, Mnn, N)
    if approximate
        perm = MLUtils.randperm(N)
        candidate_ids = Matrix{Int}(undef, chunk_size, N)
        if size(idx.dist_matrix) != (chunk_size, N)
            idx.dist_matrix = Matrix{Float32}(undef, chunk_size, N)
        end
        chunk_emb    = Matrix{Float32}(undef, D, chunk_size)
        chunk_scores = Matrix{Float32}(undef, chunk_size, chunk_size)
        num_chunks = cld(N, chunk_size)
        for ci in 1:num_chunks
            chunk_start = (ci - 1) * chunk_size + 1
            chunk_end   = min(ci * chunk_size, N)
            chunk       = view(perm, chunk_start:chunk_end)
            nc          = length(chunk)
            chunk_emb_view = view(chunk_emb, :, 1:nc)
            chunk_emb_view .= view(embeddings, :, chunk)
            chunk_scores_view = view(chunk_scores, 1:nc, 1:nc)
            Distances.pairwise!(CosineDist(), chunk_scores_view, chunk_emb_view; dims=2)
            for (li, j) in enumerate(chunk)
                cands = view(candidate_ids, :, j)
                cands[1:nc] .= chunk
                for fi in nc+1:chunk_size
                    cands[fi] = chunk[mod1(fi, nc)]
                end
                scores = view(idx.dist_matrix, :, j)
                scores[1:nc] .= view(chunk_scores_view, :, li)
                scores[li] = Inf32
                scores[nc+1:chunk_size] .= Inf32
            end
        end
        if length(idx.perm_scratch) != nthreads || (!isempty(idx.perm_scratch) && length(idx.perm_scratch[1]) != chunk_size)
            idx.perm_scratch = [collect(1:chunk_size) for _ in 1:nthreads]
        end
        Threads.@threads for j in 1:N
            top_k = partialsortperm!(idx.perm_scratch[Threads.threadid()],
                view(idx.dist_matrix, 1:chunk_size, j), 1:Mnn; rev=false)
            neighbor_ids[:, j] .= view(candidate_ids, top_k, j)
        end
    else
        if size(idx.dist_matrix) != (N, N)
            idx.dist_matrix = Matrix{Float32}(undef, N, N)
        end
        if length(idx.perm_scratch) != nthreads || (!isempty(idx.perm_scratch) && length(idx.perm_scratch[1]) != N)
            idx.perm_scratch = [collect(1:N) for _ in 1:nthreads]
        end
        mul!(idx.dist_matrix, embeddings', embeddings)
        for j in 1:N; idx.dist_matrix[j, j] = -Inf32; end
        Threads.@threads for j in 1:N
            top_k = partialsortperm!(idx.perm_scratch[Threads.threadid()],
                view(idx.dist_matrix, :, j), 1:Mnn; rev=true)
            neighbor_ids[:, j] .= top_k
        end
    end
    knn_time = time() - knn_start
    @debug "Latent index rebuild" N D Mnn chunk_size approximate normalize_time_s=round(normalize_time; digits=3) knn_time_s=round(knn_time; digits=3)
    idx.embeddings = embeddings
    idx.neighbor_ids = neighbor_ids
    idx.Mnn = Mnn
    return idx
end

# ╔═╡ 2f88be66-ffea-4d0e-8ea5-65a39b7d10db
function build_ensemble_targets(X, idx::LatentIndex, batch_indices::AbstractVector{<:Integer};
    Mnn::Int=idx.Mnn)
    X_cpu = Float32.(flatten_batch(X))
    T = size(X_cpu, 1)
    targets = Matrix{Float32}(undef, T, length(batch_indices))
    @inbounds for (b, i_raw) in enumerate(batch_indices)
        nbrs = idx.neighbor_ids[1:Mnn, Int(i_raw)]
        col = view(targets, :, b)
        fill!(col, 0f0)
        for j in nbrs
            col .+= view(X_cpu, :, j)
        end
        col ./= Float32(Mnn)
    end
    return targets
end

# ╔═╡ 2ee07196-8b28-418c-a0d4-40866584bc6f
function ensemble_phase(epoch::Int, training_para::VQVAE_Training_Para)
    post_epoch = epoch - training_para.warmup_epochs
    post_epoch <= 0 && return (; post_epoch=0, Mnn=0)
    phase = training_para.Mnn_schedule[1]
    for candidate in training_para.Mnn_schedule
        candidate[1] <= post_epoch || break
        phase = candidate
    end
    return (; post_epoch, Mnn=phase[2])
end

# ╔═╡ 5c9d71d1-c6a6-4968-814d-66506a78b516
max_Mnn(training_para::VQVAE_Training_Para) =
    isnothing(training_para.Mnn) ? maximum(p[2] for p in training_para.Mnn_schedule) : training_para.Mnn

# ╔═╡ 10000013-0000-0000-0000-000000000001
md"## Training"

# ╔═╡ 5e716b73-ab88-4b84-a8bf-dd064dc82fd8
function make_batches(X_cpu::AbstractMatrix{Float32}, batchsize::Int; shuffle::Bool=true)
    N = size(X_cpu, 2)
    ids = collect(1:N)
    shuffle && Random.shuffle!(ids)
    batches = NamedTuple[]
    for start_idx in 1:batchsize:(N - batchsize + 1)
        batch_ids = ids[start_idx:start_idx + batchsize - 1]
        push!(batches, (; indices=batch_ids, x=X_cpu[:, batch_ids]))
    end
    return batches
end

# ╔═╡ 2d6639d1-ae40-46d0-a811-e1fd34a23613
begin
	function batch_with_target(batch, X_cpu, idx::Union{Nothing,LatentIndex},
	    epoch::Int, training_para::VQVAE_Training_Para; device=identity)
	    phase = ensemble_phase(epoch, training_para)
	    target = if phase.post_epoch == 0 || isnothing(idx)
	        batch.x
	    else
	        build_ensemble_targets(X_cpu, idx, batch.indices; Mnn=phase.Mnn)
	    end
	    return (; x=device(batch.x), target=device(Float32.(target)))
	end

	function make_batch_target(batch, X_cpu, idx::Union{Nothing,LatentIndex},
	    epoch::Int, training_para::VQVAE_Training_Para, ensemble_targets_cpu::Union{Nothing,AbstractMatrix}=nothing)
	    phase = ensemble_phase(epoch, training_para)
	    if phase.post_epoch == 0 || isnothing(idx)
	        return batch.x
	    end
	    if !isnothing(ensemble_targets_cpu)
	        return ensemble_targets_cpu[:, batch.indices]
	    end
	    return build_ensemble_targets(X_cpu, idx, batch.indices; Mnn=phase.Mnn)
	end
end

# ╔═╡ cf13347d-e1fa-4ec1-86dc-38299825f65b
begin
    function fresh_loss_history()
        return (;
            train_objective=Float32[],
            train_target_mse=Float32[],
            train_commit=Float32[],
            train_entropy=Float32[],
            train_perplexity=Float32[],
            test_recon_mse=Float32[],
            epoch_time_s=Float32[],
            throughput=Float32[],
        )
    end

    function reset_vqvae(model::VQVAE, rvq_st; seed::Integer, device=identity)
        # Re-initialise only the RVQ state (codebook + EMA); re-randomise network weights.
        rng = Xoshiro(seed)
        para_d  = model.d
        para_K  = model.K
        new_rvq = init_rvq_state(rng, para_d ÷ 2, para_K)
        # Re-init network weights by building a fresh model with the same architecture and
        # reinitialising via Flux.  We do this by calling Flux.fmap to reset all Dense/Conv.
        reset_layer(l::Flux.Dense)       = Flux.Dense(size(l.weight, 2) => size(l.weight, 1), l.σ)
        reset_layer(l::Flux.Conv)        = Flux.Conv(size(l.weight)[1:ndims(l.weight)-2],
                                                     size(l.weight, ndims(l.weight)-1) => size(l.weight, ndims(l.weight)),
                                                     l.σ; pad=l.pad, stride=l.stride, dilation=l.dilation)
        reset_layer(l::Flux.ConvTranspose) = Flux.ConvTranspose(size(l.weight)[1:ndims(l.weight)-2],
                                                     size(l.weight, ndims(l.weight)-1) => size(l.weight, ndims(l.weight));
                                                     pad=l.pad, stride=l.stride)
        reset_layer(l) = l
        new_model = Flux.fmap(reset_layer, model)
        return device(new_model), device(new_rvq)
    end

	function recon_mse_inference(model, rvq_st, x; normalize_target::Bool=false, cdev=default_cdev())
	    x_cpu = Float32.(cdev(flatten_batch(x)))
	    model_cpu = cdev(model)
	    rvq_cpu = cdev(rvq_st)
	    target = normalize_target ? Float32.(MLUtils.normalise(x_cpu; dims=1)) : x_cpu
	    result, _ = model_cpu(x_cpu, rvq_cpu; training=false)
	    return mse_loss(result.xhat, target)
	end

	function record_train_metrics!(loss_history, train_m, test_recon_mse::Real,
	    epoch_time::Real, throughput::Real)
	    push!(loss_history.train_objective, train_m.total)
	    push!(loss_history.train_target_mse, train_m.recon_loss)
	    push!(loss_history.train_commit, train_m.commit_loss)
	    push!(loss_history.train_entropy, train_m.entropy_loss)
	    push!(loss_history.train_perplexity, train_m.perplexity)
	    push!(loss_history.test_recon_mse, Float32(test_recon_mse))
	    push!(loss_history.epoch_time_s, Float32(epoch_time))
	    push!(loss_history.throughput, Float32(throughput))
	    return loss_history
	end
end

# ╔═╡ prepare-vq-batch
function prepare_vq_training_batch(model, rvq_st, batch, X_cpu, idx,
    epoch::Int, para, training_para;
    device=identity, cdev=default_cdev(),
    ensemble_targets_cpu=nothing)
    target_start = time()
    target_cpu = make_batch_target(batch, X_cpu, idx, epoch, training_para, ensemble_targets_cpu)
    target_time = time() - target_start
    pack_start = time()
    if training_para.normalize_target
        target_cpu = Float32.(MLUtils.normalise(target_cpu; dims=1))
    end
    bdev = (; x=device(batch.x), target=device(Float32.(target_cpu)))
    pack_time = time() - pack_start
    payload_start = time()
    model_cpu = cdev(model)
    rvq_cpu = cdev(rvq_st)
    lat = encoder_latents(model_cpu, batch.x)
    z_e1_cpu = lat.z_e1
    z_e2_cpu = lat.z_e2
    payload_cpu, rvq_cpu_new = prepare_split_payload(z_e1_cpu, z_e2_cpu, rvq_cpu, model.K;
        ema_decay=para.ema_decay,
        epsilon=para.epsilon,
        dead_threshold=para.dead_threshold,
        training=true)
    new_rvq_dev = device(rvq_cpu_new)
    payload_time = time() - payload_start
    return merge(bdev, (; vq_payload=device(payload_cpu))), new_rvq_dev, (; target_time, pack_time, payload_time)
end

# ╔═╡ 10000015-0000-0000-0000-000000000001
md"## Main Training Loop"

# ╔═╡ 11834c5a-4618-11f1-a096-01b9cbdd6fab
function update(model, rvq_st, loss_history, train_data, test_data,
    para, training_para;
    device=identity, cdev=default_cdev())

    setup_start = time()
    train_x_cpu = Float32.(cdev(flatten_batch(train_data)))
    test_x_cpu = Float32.(cdev(flatten_batch(test_data)))
    opt = Optimisers.AdamW(; eta=Float64(training_para.initial_learning_rate),
        lambda=Float64(training_para.weight_decay))
    opt_state = Flux.setup(opt, model)
    idx = LatentIndex(max_Mnn(training_para))
    last_index_Mnn = 0
    ensemble_targets_cpu = nothing
    size(train_x_cpu, 2) >= training_para.batchsize ||
        error("Training set N=$(size(train_x_cpu, 2)) is smaller than batchsize=$(training_para.batchsize).")
    test_eval_x_cpu = test_x_cpu[:, 1:min(512, size(test_x_cpu, 2))]
    training_para.verbose && @info "Prepared v9_flux update loop" setup_time_s=round(time() - setup_start; digits=3) N=size(train_x_cpu, 2) batchsize=training_para.batchsize

    @progress name = "VQ-VAE training" for epoch in 1:training_para.nepoch
        phase = ensemble_phase(epoch, training_para)
        if phase.post_epoch > 0 &&
           (phase.Mnn != last_index_Mnn || mod(phase.post_epoch - 1, training_para.index_refresh_every) == 0)
            index_start = time()
            training_para.verbose && @info "Rebuilding latent index before epoch batches" epoch post_warmup_epoch=phase.post_epoch Mnn=phase.Mnn N=size(train_x_cpu, 2)
            rebuild_latent_index!(idx, model, rvq_st, train_x_cpu;
                Mnn=phase.Mnn, device, cdev,
                knn_search_chunk_size_fraction=training_para.knn_search_chunk_size_fraction)
            latent_index_time = time() - index_start
            target_cache_start = time()
            ensemble_targets_cpu = build_ensemble_targets(train_x_cpu, idx, 1:size(train_x_cpu, 2); Mnn=phase.Mnn)
            target_cache_time = time() - target_cache_start
            last_index_Mnn = phase.Mnn
            training_para.verbose && @info "Rebuilt latent index" epoch post_warmup_epoch=phase.post_epoch Mnn=phase.Mnn latent_index_time_s=round(latent_index_time; digits=3) ensemble_target_cache_time_s=round(target_cache_time; digits=3)
        end

        batches = make_batches(train_x_cpu, training_para.batchsize)
        start = time()
        total_seen = 0
        last_loss = NaN32
        last_recon = NaN32
        last_commit = NaN32
        last_entropy = NaN32
        last_perplexity = NaN32
        epoch_counts = zeros(Float32, sum(para.K))
        prep_time = 0.0
        step_time = 0.0

        for (batch_idx, batch) in enumerate(batches)
            prep_start = time()
            bdev, rvq_st, prep_stats = prepare_vq_training_batch(
                model, rvq_st, batch, train_x_cpu,
                phase.post_epoch == 0 ? nothing : idx,
                epoch, para, training_para; device, cdev,
                ensemble_targets_cpu=ensemble_targets_cpu,
            )
            prep_time += time() - prep_start

            step_start = time()
            loss_val, grads = Flux.withgradient(model) do m
                result = forward_with_precomputed_vq(m, bdev.x, rvq_st, bdev.vq_payload;
                    beta_commit=para.beta_commit)
                recon_loss = mse_loss(result.xhat, bdev.target)
                recon_loss + result.commit_loss + para.entropy_weight * result.entropy_loss
            end
            Flux.update!(opt_state, model, grads[1])
            step_time += time() - step_start

            total_seen += size(batch.x, 2)

            # accumulate per-stage EMA cluster sizes for epoch-level perplexity
            offset = 0
            for stage in rvq_st.stages
                cs = Float32.(cdev(stage.ema_cluster_size))
                epoch_counts[offset+1:offset+length(cs)] .+= cs
                offset += length(cs)
            end

            if batch_idx == length(batches)
                # compute metrics for the last batch
                with_metrics = forward_with_precomputed_vq(cdev(model), cdev(bdev.x), cdev(rvq_st), cdev(bdev.vq_payload);
                    beta_commit=para.beta_commit)
                last_loss = Float32(loss_val)
                isnan(last_loss) && error("NaN loss encountered.")
                last_recon = Float32(mse_loss(with_metrics.xhat, cdev(bdev.target)))
                last_commit = Float32(with_metrics.commit_loss)
                last_entropy = Float32(with_metrics.entropy_loss)
            end
        end

        epoch_time = time() - start
        throughput = total_seen / max(epoch_time, 1e-8)

        # epoch perplexity from accumulated counts
        nstages = length(para.K)
        epoch_perplexity = 0f0
        offset = 0
        for k in para.K
            stage_counts = epoch_counts[offset+1:offset+k]
            p = stage_counts ./ max(sum(stage_counts), 1f-8)
            psafe = clamp.(p, 1f-10, 1f0)
            epoch_perplexity += exp(-sum(psafe .* log.(psafe)))
            offset += k
        end
        last_perplexity = epoch_perplexity / nstages

        epoch_entropy = 0f0
        offset2 = 0
        for k in para.K
            stage_counts = epoch_counts[offset2+1:offset2+k]
            p = stage_counts ./ max(sum(stage_counts), 1f-8)
            psafe = clamp.(p, 1f-10, 1f0)
            epoch_entropy += sum(psafe .* log.(psafe))
            offset2 += k
        end
        last_entropy = epoch_entropy / nstages

        train_m = (;
            total=last_loss,
            recon_loss=last_recon,
            commit_loss=last_commit,
            entropy_loss=last_entropy,
            perplexity=last_perplexity,
        )
        test_recon_mse = recon_mse_inference(
            model, rvq_st, test_eval_x_cpu;
            normalize_target=training_para.normalize_target, cdev)
        record_train_metrics!(loss_history, train_m, test_recon_mse, epoch_time, throughput)

        if training_para.verbose && mod(epoch, training_para.nprint) == 0
            r(x) = round(x; digits=4)
            weighted_entropy = para.entropy_weight * train_m.entropy_loss
            objective_str = "$(r(train_m.recon_loss)) + $(r(train_m.commit_loss)) + $(r(weighted_entropy)) [raw_entropy=$(r(train_m.entropy_loss))] = $(r(train_m.total))"
            @info "Epoch $epoch" objective="mse+commit+w_entropy = $objective_str" test_recon_mse=r(test_recon_mse) perplexity=r(train_m.perplexity) post_warmup_epoch=phase.post_epoch Mnn=phase.Mnn throughput=round(throughput; digits=1) epoch_time_s=round(epoch_time; digits=3)
            @info "Epoch timing" epoch prep_time_s=round(prep_time; digits=3) step_time_s=round(step_time; digits=3)
        end
        if !isnothing(training_para.stop_on_recon_loss) && train_m.recon_loss < training_para.stop_on_recon_loss
            training_para.verbose && @info "Early stopping" epoch train_target_mse=train_m.recon_loss threshold=training_para.stop_on_recon_loss
            break
        end
    end
    return model, rvq_st, loss_history
end

# ╔═╡ 10000015-0000-0000-0000-000000000002
md"## Data Loading and Pair Loop"

# ╔═╡ 566e6a4c-1153-4c6c-bf2b-385478f684c4
function taper(x)
    w = cat(DSP.tukey(size(x, 1), 0.1), dims=ndims(x))
    return w .* x
end

# ╔═╡ a1e5a8cb-0bd1-44b8-8cd4-c95a667d830d
function get_acausal_causal(pair::String, filepath::String)
    matches = filter(x -> occursin(pair, basename(x)), readdir(filepath, join=true))
    isempty(matches) && error("No JLD2 file matching pair $(pair) found in $(filepath).")
    jldfile = load(matches[1])
    correlations = haskey(jldfile, "correlations") ? jldfile["correlations"] : jldfile["D"][1]
    headers = haskey(jldfile, "headers") ? jldfile["headers"] : nothing
    latitudes = haskey(jldfile, "latitudes") ? Float64.(jldfile["latitudes"]) : nothing
    longitudes = haskey(jldfile, "longitudes") ? Float64.(jldfile["longitudes"]) : nothing
    distance = haskey(jldfile, "dist") ? Float64(jldfile["dist"]) :
        (haskey(jldfile, "Distances") ? Float64(jldfile["Distances"][1]) : nothing)
    return (; correlations, headers, distance, latitudes, longitudes)
end

# ╔═╡ 0b79d043-0805-43b3-80d7-f64d2018525f
function split_causal_acausal(X::AbstractMatrix, zero_lag::Bool, max_lag=nothing)
    nt, ntr = size(X)
    !isodd(nt) && error("nt should be odd.")
    center = div(nt + 1, 2)
    half = div(nt - 1, 2)
    N = isnothing(max_lag) ? half : max(0, min(half, max_lag))
    X_acausal = reverse(X[center-N:center-1, :], dims=1)
    X_causal = X[center+1:center+N, :]
    if zero_lag
        return vcat(zeros(1, ntr), Array(X_acausal)), vcat(zeros(1, ntr), Array(X_causal))
    end
    return Array(X_acausal), Array(X_causal)
end

# ╔═╡ 4b8ffb0f-23b0-443d-b4c7-12a3ed4ac76d
function build_training_bundle(pair::Tuple{String,String}; filepath::String, dt::Real=1.0,
    period_min::Real=10, period_max::Real=50)
    pair_name = join(pair, "_")
    raw = get_acausal_causal(pair_name, filepath)
    D1 = MLUtils.normalise(raw.correlations, dims=1)
    D1ac, D1c = split_causal_acausal(D1, true)
    D1fac = Float32.(MLUtils.normalise(taper(D1ac)[2:end, :], dims=1))
    D1fc  = Float32.(MLUtils.normalise(taper(D1c)[2:end,  :], dims=1))
    return (; pair, D1=Float32.(D1), D1fac, D1fc, headers=raw.headers, distance=raw.distance,
            latitudes=raw.latitudes, longitudes=raw.longitudes)
end

function trim_training_bundle(bundle; n_max::Union{Nothing,Integer}=nothing,
    rng=Random.default_rng())
    isnothing(n_max) && return bundle
    n_max >= 2 || error("n_max must be at least 2 so both acausal and causal branches are represented.")
    n_windows_max = n_max ÷ 2
    n_windows = min(size(bundle.D1fac, 2), size(bundle.D1fc, 2))
    n_windows == n_windows_max && return bundle
    if n_windows > n_windows_max
        @info "Trimming pair waveforms to n_max" pair=bundle.pair n_original=2n_windows n_kept=2n_windows_max n_max
        trim_headers(headers) = headers === nothing ? nothing :
            (length(headers) == n_windows ? headers[1:n_windows_max] : headers)
        return merge(bundle, (;
            D1fac=bundle.D1fac[:, 1:n_windows_max],
            D1fc=bundle.D1fc[:, 1:n_windows_max],
            headers=trim_headers(bundle.headers),
        ))
    else
        @info "Upsampling pair waveforms to n_max" pair=bundle.pair n_original=2n_windows n_target=2n_windows_max n_max
        idx_ac = rand(rng, 1:n_windows, n_windows_max)
        idx_c  = rand(rng, 1:n_windows, n_windows_max)
        return merge(bundle, (;
            D1fac=bundle.D1fac[:, idx_ac],
            D1fc=bundle.D1fc[:, idx_c],
            headers=nothing,
        ))
    end
end

# ╔═╡ 8dd1c50c-587c-471d-bc80-cd77012302a9
function make_pooled_split(D1fac, D1fc; at=0.9, shuffle=true, rng=Random.default_rng())
    D_all = Float32.(hcat(D1fac, D1fc))
    nw = size(D_all, 2)
    idx = collect(1:nw)
    shuffle && Random.shuffle!(rng, idx)
    ntrain = round(Int, at * nw)
    train_idx = idx[1:ntrain]
    test_idx = idx[ntrain+1:end]
    return (; D_train=D_all[:, train_idx], D_test=D_all[:, test_idx],
        D_all, D_ac_all=Float32.(D1fac), D_c_all=Float32.(D1fc))
end

# ╔═╡ 7e26f064-6a32-41ce-b416-90a04adfbcc9
function pair_run_dir(save_root::String, pair, timestamp=Dates.now())
    pair_str = join(pair, "_")
    run_tag = Dates.format(timestamp, "yyyymmdd_HHMMSS")
    return joinpath(save_root, pair_str, run_tag)
end

# ╔═╡ f3583928-80f5-4e89-8d86-463eda8b97bd
function load_pairs_data(selected_pairs; filepath::String,
    seed::Int=1234, dt::Real=1.0, period_min::Real=10, period_max::Real=50,
    n_max::Union{Nothing,Integer}=nothing)
    rng = Xoshiro(seed)
    pairs_data = Any[]
    for pair_raw in selected_pairs
        pair = (String(pair_raw[1]), String(pair_raw[2]))
        @info "Loading pair data" pair
        bundle = build_training_bundle(pair; filepath, dt, period_min, period_max)
        bundle = trim_training_bundle(bundle; n_max, rng)
        @info "Loaded pair bundle" pair distance=bundle.distance D1fac_size=size(bundle.D1fac) D1fc_size=size(bundle.D1fc)
        data = make_pooled_split(bundle.D1fac, bundle.D1fc; rng)
        @info "Built train/test split" pair train_size=size(data.D_train) test_size=size(data.D_test)
        push!(pairs_data, (; pair, data, data_bundle=bundle))
    end
    return pairs_data
end

# ╔═╡ a4957762-faf0-40d2-a71e-1d65fe873065
function run_dir_for_seed(save_root::String, pair, seed::Integer, timestamp=Dates.now())
    pair_str = join(pair, "_")
    run_tag = Dates.format(timestamp, "yyyymmdd_HHMMSS")
    return joinpath(save_root, pair_str, "seed$(seed)_$(run_tag)")
end

# ╔═╡ 10000017-0000-0000-0000-000000000001
md"## Analysis and Plotting"

# ╔═╡ 2f151b20-b956-404d-8fee-1e9cddfd6b62
function get_cluster_percentages(model, rvq_st, x; stage::Int=1, return_labels::Bool=false,
    device=identity, cdev=default_cdev())
    x_cpu = Float32.(cdev(flatten_batch(x)))
    res, _ = encode(cdev(model), cdev(rvq_st), x_cpu; training=false)
    idx = vec(Array(cdev(res.stage_indices[stage, :])))
    K = model.K[stage]
    counts = zeros(Float32, K)
    for k in idx
        counts[Int(k)] += 1f0
    end
    pct = counts ./ max(sum(counts), 1f-8) .* 100f0
    labels = string.(1:K)
    return return_labels ? (; percentages=pct, labels) : pct
end

# ╔═╡ b478bec8-14f6-4d78-89e5-2e76414c4d46
function select_state_indices_from_codes(ci::AbstractMatrix{<:Integer}, state_tuple::Tuple)
    mask = trues(size(ci, 2))
    Tlocal = min(length(state_tuple), size(ci, 1))
    for t in 1:Tlocal
        mask .&= vec(ci[t, :]) .== state_tuple[t]
    end
    return findall(mask)
end

# ╔═╡ b387cc9c-c64f-4ca6-9e44-6df5551f6d7a
function cluster_averages_from_codes(x_cpu, ci; K::Int, stage::Int=1)
    nt = size(x_cpu, 1)
    out = zeros(Float32, nt, K)
    counts = zeros(Int, K)
    labels = vec(ci[stage, :])
    for j in eachindex(labels)
        k = Int(labels[j])
        out[:, k] .+= x_cpu[:, j]
        counts[k] += 1
    end
    for k in 1:K
        counts[k] > 0 && (out[:, k] ./= counts[k])
    end
    return (; averages=out, counts)
end

# ╔═╡ 987c9325-5908-4047-aaf3-39e5b251309d
function joint_cluster_averages(x_cpu, ci; K1::Int, K2::Int)
    nt = size(x_cpu, 1)
    n_combos = K1 * K2
    out = zeros(Float32, nt, n_combos)
    counts = zeros(Int, n_combos)
    codes1 = vec(ci[1, :])
    codes2 = vec(ci[2, :])
    for j in eachindex(codes1)
        k1 = Int(codes1[j])
        k2 = Int(codes2[j])
        col = (k2 - 1) * K1 + k1
        out[:, col] .+= x_cpu[:, j]
        counts[col] += 1
    end
    for col in 1:n_combos
        counts[col] > 0 && (out[:, col] ./= counts[col])
    end
    labels = ["($k1,$k2)" for k2 in 1:K2 for k1 in 1:K1]
    return (; averages=out, counts, labels)
end

# ╔═╡ db53da0e-96ce-4a75-bcfc-32fdc4ffe064
function encoded_cache(model, rvq_st, data; device=identity, cdev=default_cdev())
    model_cpu = cdev(model)
    rvq_cpu = cdev(rvq_st)
    res_ac, _ = encode(model_cpu, rvq_cpu, Float32.(cdev(data.D_ac_all)); training=false)
    res_c, _ = encode(model_cpu, rvq_cpu, Float32.(cdev(data.D_c_all)); training=false)
    return (;
        stage_ac=Array(cdev(res_ac.stage_indices)),
        stage_c=Array(cdev(res_c.stage_indices)),
        coarse_ac=Array(cdev(res_ac.coarse_indices)),
        coarse_c=Array(cdev(res_c.coarse_indices)),
    )
end

# ╔═╡ 4f2b1382-c158-417b-9fc0-1b8d04d90ed2
function codebook_cross_analysis(model, rvq_st, D_ac, D_c; device=identity, cdev=default_cdev())
    cache = encoded_cache(model, rvq_st, (; D_ac_all=D_ac, D_c_all=D_c); device, cdev)
    K = model.K[1]
    idx_ac = vec(cache.coarse_ac)
    idx_c = vec(cache.coarse_c)
    nw = min(length(idx_ac), length(idx_c))
    confusion = zeros(Float32, K, K)
    for i in 1:nw
        confusion[idx_ac[i], idx_c[i]] += 1f0
    end
    confusion ./= max(sum(confusion), 1f-8)
    agreement = mean(idx_ac[1:nw] .== idx_c[1:nw])
    pct_ac = get_cluster_percentages(model, rvq_st, D_ac; device, cdev)
    pct_c = get_cluster_percentages(model, rvq_st, D_c; device, cdev)
    return (; pct_ac, pct_c, confusion, agreement, labels=string.(1:K), cache)
end

# ╔═╡ c974342d-4bcc-4175-9d71-8f9cfbb7105a
function source_state_averages(model, rvq_st, data; device=identity, cdev=default_cdev())
    cache = encoded_cache(model, rvq_st, data; device, cdev)
    if length(model.K) >= 2
        K1, K2 = model.K[1], model.K[2]
        ac = joint_cluster_averages(Float32.(cdev(data.D_ac_all)), cache.stage_ac; K1, K2)
        c  = joint_cluster_averages(Float32.(cdev(data.D_c_all)),  cache.stage_c;  K1, K2)
        return (; acausal=ac.averages, causal=c.averages,
            counts_ac=ac.counts, counts_c=c.counts, combo_labels=ac.labels, cache)
    else
        K = model.K[1]
        ac = cluster_averages_from_codes(Float32.(cdev(data.D_ac_all)), cache.coarse_ac; K)
        c  = cluster_averages_from_codes(Float32.(cdev(data.D_c_all)),  cache.coarse_c;  K)
        labels = string.(1:K)
        return (; acausal=ac.averages, causal=c.averages,
            counts_ac=ac.counts, counts_c=c.counts, combo_labels=labels, cache)
    end
end

# ╔═╡ a1b2c3d4-0000-0000-0000-000000000001
function marginal_decomposition(averages::AbstractMatrix{Float32}, counts::AbstractVector{Int};
        K1::Int, K2::Int)
    nt = size(averages, 1)
    W = reshape(Float32.(counts), K1, K2)
    A = reshape(averages, nt, K1, K2)

    w_row = sum(W; dims=2)
    w_col = sum(W; dims=1)
    w_tot = sum(W)

    grand_mean = dropdims(
        sum(A .* reshape(W, 1, K1, K2); dims=(2,3)); dims=(2,3)) ./ max(w_tot, 1f0)

    stage1_marginal = dropdims(
        sum(A .* reshape(W, 1, K1, K2); dims=3); dims=3) ./
        reshape(max.(vec(w_row), 1f0), 1, K1)

    stage2_marginal = dropdims(
        sum(A .* reshape(W, 1, K1, K2); dims=2); dims=2) ./
        reshape(max.(vec(w_col), 1f0), 1, K2)

    stage1_effects = stage1_marginal .- grand_mean
    stage2_effects = stage2_marginal .- grand_mean

    return (;
        grand_mean,
        stage1_waves  = stage1_marginal,
        stage2_waves  = stage2_marginal,
        stage1_effects,
        stage2_effects,
        stage1_labels = ["s1=$k" for k in 1:K1],
        stage2_labels = ["s2=$k" for k in 1:K2],
    )
end

# ╔═╡ 70a460bf-b3e4-4e7c-aa4d-2674a450379a
function plot_training_dashboard(loss_history; title="VQ-VAE v9 Flux Training")
    epochs = collect(1:length(loss_history.train_target_mse))
    traces = [
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_objective, mode="lines", name="train_objective"),
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_target_mse, mode="lines", name="train_target_mse"),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_recon_mse, mode="lines", name="test_recon_mse"),
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_perplexity, mode="lines", name="Train perplexity", yaxis="y2"),
    ]
    layout = PlutoPlotly.Layout(title=title, xaxis_title="Epoch",
        yaxis=PlutoPlotly.attr(title="Loss", type="log"),
        yaxis2=PlutoPlotly.attr(title="Perplexity", overlaying="y", side="right"),
        width=900, height=500, plot_bgcolor="white", paper_bgcolor="white")
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 2168a07c-e17a-4e94-bec7-5f881a5b5f09
function plot_codebook_heatmap(rvq_st; stage::Int=1, kmax::Int=20,
    title="RVQ stage $(stage) codebook")
    E = get_codebook(rvq_st, stage)
    ksel = 1:min(kmax, size(E, 2))
    trace = PlutoPlotly.heatmap(z=E[:, ksel], x=string.(ksel), y=string.(1:size(E, 1)), zmid=0)
    return PlutoPlotly.plot([trace], PlutoPlotly.Layout(title=title,
        xaxis_title="Code", yaxis_title="Embedding dim", width=850, height=550))
end

# ╔═╡ d7d22b36-79b6-41d6-b9ce-403d34d4165b
function plot_codebook_confusion(confusion; title="Codebook Confusion", labels=nothing)
    K = size(confusion, 1)
    labels = isnothing(labels) ? string.(1:K) : labels
    trace = PlutoPlotly.heatmap(z=confusion, x=labels, y=labels, colorscale="Blues")
    return PlutoPlotly.plot([trace], PlutoPlotly.Layout(title=title, xaxis_title="Causal code",
        yaxis_title="Acausal code", width=750, height=700))
end

# ╔═╡ 16d60aa8-ecc0-47f7-b454-0ca1e1d2e3d0
function plot_state_average_matrix(avg; title::String, dt::Real=1.0, reverse_time::Bool=false)
    nt, nstates = size(avg)
    t = collect(1:nt) .* dt
    traces = PlutoPlotly.AbstractTrace[]
    for k in 1:nstates
        y = reverse_time ? reverse(avg[:, k]) : avg[:, k]
        push!(traces, PlutoPlotly.scatter(x=t, y=y .+ (k - 1) * 2.5,
            mode="lines", name="state $k"))
    end
    return PlutoPlotly.plot(traces, PlutoPlotly.Layout(title=title, xaxis_title="Time (s)",
        yaxis_title="State + offset", width=900, height=max(400, 70 * nstates)))
end

# ╔═╡ a6751ceb-46af-4ec7-840b-d42ea46c93a5
function plot_cluster_histogram(counts_ac, counts_c; title="Cluster Usage", labels=nothing)
    K = length(counts_ac)
    total_ac = max(sum(counts_ac), 1)
    total_c  = max(sum(counts_c),  1)
    pct_ac = 100f0 .* counts_ac ./ total_ac
    pct_c  = 100f0 .* counts_c  ./ total_c
    xlabels = isnothing(labels) ? string.(1:K) : labels
    traces = [
        PlutoPlotly.bar(x=xlabels, y=pct_ac, name="Acausal",
            marker=PlutoPlotly.attr(color="rgba(31,119,180,0.7)")),
        PlutoPlotly.bar(x=xlabels, y=pct_c,  name="Causal",
            marker=PlutoPlotly.attr(color="rgba(214,39,40,0.7)")),
    ]
    layout = PlutoPlotly.Layout(
        title=PlutoPlotly.attr(text=title, font=PlutoPlotly.attr(size=18)),
        barmode="group", height=400, width=700,
        xaxis=PlutoPlotly.attr(title="Source state"),
        yaxis=PlutoPlotly.attr(title="Usage (%)"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ f140b608-0c12-4c5e-8dad-1ac81f6e2d99
function plot_reconstruction_examples(model, rvq_st, X; nsamples::Int=8, dt::Real=1.0,
    device=identity, cdev=default_cdev(), title="Reconstruction examples")
    X_cpu = Float32.(cdev(flatten_batch(X)))
    ids = sort(randperm(size(X_cpu, 2))[1:min(nsamples, size(X_cpu, 2))])
    model_cpu = cdev(model)
    rvq_cpu = cdev(rvq_st)
    res, _ = model_cpu(X_cpu[:, ids], rvq_cpu; training=false)
    recon = Float32.(res.xhat)
    t = collect(1:size(X_cpu, 1)) .* dt
    traces = PlutoPlotly.AbstractTrace[]
    for (j, id) in enumerate(ids)
        offset = (j - 1) * 4
        push!(traces, PlutoPlotly.scatter(x=t, y=X_cpu[:, id] .+ offset, mode="lines",
            name="raw", showlegend=j == 1, line=PlutoPlotly.attr(color="black", width=1)))
        push!(traces, PlutoPlotly.scatter(x=t, y=recon[:, j] .+ offset, mode="lines",
            name="recon", showlegend=j == 1, line=PlutoPlotly.attr(color="red", width=2)))
    end
    return PlutoPlotly.plot(traces, PlutoPlotly.Layout(title=title, xaxis_title="Time (s)",
        yaxis_title="Trace + offset", width=900, height=650))
end

# ╔═╡ whitening-helpers
begin
    function compute_whitening_fir(X_cpu::AbstractMatrix{Float32}, dt::Float64;
        kernel_length::Int=64, min_power_fraction::Float64=0.05)
        nt = size(X_cpu, 1)
        P = vec(mean(abs2.(fft(Float64.(X_cpu), 1)); dims=2))
        P[1] = 0.0
        P .= max.(P, min_power_fraction * maximum(P))
        W = P .^ (-0.25)
        W[1] = 0.0
        w_full = real.(ifft(W))
        w_shift = fftshift(w_full)
        center = div(nt, 2) + 1
        lo = center - div(kernel_length, 2)
        taps = w_shift[lo : lo + kernel_length - 1]
        taps .*= DSP.hann(kernel_length)
        taps ./= sum(abs.(taps))
        return Float32.(taps)
    end

    function apply_whitening_fir(X_cpu::AbstractMatrix{Float32}, fir::AbstractVector{Float32})
        return Float32.(DSP.filtfilt(Float64.(fir), [1.0], Float64.(X_cpu)))
    end

    function detect_spectral_spikes(X_cpu::AbstractMatrix{Float32};
            spike_threshold::Float64=10.0)
        nt = size(X_cpu, 1)
        P = vec(mean(abs2.(fft(Float64.(X_cpu), 1)); dims=2))
        half = nt ÷ 2 + 1
        P_half = P[1:half]
        med = median(P_half[2:end])
        spike_bins = findall(P_half .> spike_threshold * med)
        return spike_bins, P_half, med
    end

    function apply_spike_suppression(X_cpu::AbstractMatrix{Float32}, spike_bins::Vector{Int};
            suppression_factor::Float64=0.05, bandwidth_bins::Int=2)
        nt = size(X_cpu, 1)
        half = nt ÷ 2 + 1
        mask = ones(Float64, nt)
        for b in spike_bins
            for k in max(1, b - 2*bandwidth_bins):min(half, b + 2*bandwidth_bins)
                g = exp(-0.5 * ((k - b) / bandwidth_bins)^2)
                mask[k] = min(mask[k], 1.0 - (1.0 - suppression_factor) * g)
            end
        end
        for b in 2:half-1
            mirror = nt - b + 2
            mask[mirror] = mask[b]
        end
        X_f = fft(Float64.(X_cpu), 1)
        X_f .*= mask
        return Float32.(real.(ifft(X_f, 1)))
    end

    function whiten_amplitude_spectrum(X_cpu::AbstractMatrix{Float32}; smooth_bins::Int=15)
        nt = size(X_cpu, 1)
        X_f = fft(Float64.(X_cpu), 1)
        A_mean = vec(mean(abs.(X_f); dims=2))
        half_w = smooth_bins ÷ 2
        kernel = ones(smooth_bins) / smooth_bins
        c = DSP.conv(A_mean, kernel)
        A_smooth = max.(c[half_w+1:half_w+nt], 1e-10)
        X_f ./= A_smooth
        return Float32.(real.(ifft(X_f, 1)))
    end

    function per_waveform_whiten(X_cpu::AbstractMatrix{Float32};
        kernel_length::Int=128, min_power_fraction::Float64=0.05)
        nt, nw = size(X_cpu)
        X64 = Float64.(X_cpu)
        P = abs2.(fft(X64, 1))
        P[1, :] .= 0.0
        col_max = maximum(P; dims=1)
        P .= max.(P, min_power_fraction .* col_max)
        W = P .^ (-0.25)
        W[1, :] .= 0.0
        w_full = real.(ifft(W, 1))
        w_shift = fftshift(w_full, 1)
        center = div(nt, 2) + 1
        lo = center - div(kernel_length, 2)
        taps = w_shift[lo:lo+kernel_length-1, :]
        taps .*= DSP.hann(kernel_length)
        taps ./= sum(abs.(taps); dims=1)
        X_whitened = similar(X64)
        for i in 1:nw
            X_whitened[:, i] = DSP.filtfilt(taps[:, i], [1.0], X64[:, i])
        end
        return Float32.(X_whitened)
    end
end

# ╔═╡ f5000001-0000-0000-0000-000000000001
function whiten_pair_entry(pd; bp_filter, per_waveform_whitening_kernel_length::Int)
    process(X) = Float32.(MLUtils.normalise(
        DSP.filtfilt(bp_filter, Float64.(per_waveform_whiten(X;
            kernel_length=per_waveform_whitening_kernel_length,
            min_power_fraction=0.05))); dims=1))
    D_ac_w    = process(pd.data.D_ac_all)
    D_c_w     = process(pd.data.D_c_all)
    D_train_w = process(pd.data.D_train)
    D_test_w  = process(pd.data.D_test)
    D_all_w   = Float32.(hcat(D_ac_w, D_c_w))
    return merge(pd, (;
        data=merge(pd.data, (; D_train=D_train_w, D_test=D_test_w,
            D_all=D_all_w, D_ac_all=D_ac_w, D_c_all=D_c_w)),
        data_bundle=merge(pd.data_bundle, (;
            D1fac_raw=pd.data_bundle.D1fac,
            D1fc_raw=pd.data_bundle.D1fc,
            D1fac=D_ac_w,
            D1fc=D_c_w,
            whitening_fir=nothing,
            spike_bins=Int[],
        )),
        whitening_fir=nothing,
        spike_bins=Int[],
    ))
end

# ╔═╡ a1b2c3d4-0000-0000-0000-000000000002
function decode_codebook_waveforms(model, rvq_st; cdev=default_cdev())
    model_cpu = cdev(model)
    rvq_cpu = cdev(rvq_st)
    embeddings = [Array(stage.embedding) for stage in rvq_cpu.stages]

    K1, K2 = model.K[1], model.K[2]
    e1 = embeddings[1]
    e2 = embeddings[2]

    result1 = (; z_q1=Float32.(e1), z_q2=zeros(Float32, model.d÷2, K1))
    out1 = decode_from_latents(model_cpu, result1)
    stage1_waves = Array(out1.x1hat)

    result2 = (; z_q1=zeros(Float32, model.d÷2, K2), z_q2=Float32.(e2))
    out2 = decode_from_latents(model_cpu, result2)
    stage2_waves = Array(out2.x2hat)

    n_joint = K1 * K2
    Z1_rep = hcat([e1[:, k1] for k2 in 1:K2 for k1 in 1:K1]...)
    Z2_rep = hcat([e2[:, k2] for k2 in 1:K2 for k1 in 1:K1]...)
    result_joint = (; z_q1=Float32.(Z1_rep), z_q2=Float32.(Z2_rep))
    out_joint = decode_from_latents(model_cpu, result_joint)
    joint_waves = Array(out_joint.xhat)

    return (;
        joint=joint_waves,
        stage1=stage1_waves,
        stage2=stage2_waves,
        joint_labels=["($k1,$k2)" for k2 in 1:K2 for k1 in 1:K1],
        stage1_labels=["s1=$k" for k in 1:K1],
        stage2_labels=["s2=$k" for k in 1:K2],
    )
end

# ╔═╡ helper-window-fns
begin
    function _window_headers(headers, n::Integer)
        if isnothing(headers)
            return ["window_$(i)" for i in 1:n]
        end
        out = string.(collect(headers))
        if length(out) < n
            append!(out, ["window_$(i)" for i in (length(out)+1):n])
        end
        return out[1:n]
    end

    function _header_time_label(header::AbstractString)
        m = match(r"^(\d{4})\.(\d{3})\.(\d{4})\.(\d{4})", header)
        return m === nothing ? header : join(m.captures, ".")
    end

    function _source_state_indices(cache, K)
        if length(K) >= 2 && size(cache.stage_ac, 1) >= 2 && size(cache.stage_c, 1) >= 2
            K1 = Int(K[1])
            source_state_ac = Int.(vec(cache.stage_ac[1, :]) .+ (vec(cache.stage_ac[2, :]) .- 1) .* K1)
            source_state_c = Int.(vec(cache.stage_c[1, :]) .+ (vec(cache.stage_c[2, :]) .- 1) .* K1)
        else
            source_state_ac = Int.(vec(cache.coarse_ac))
            source_state_c = Int.(vec(cache.coarse_c))
        end
        return (; source_state_ac, source_state_c)
    end

    function _assignment_table(headers, source_state::AbstractVector{<:Integer},
            stage_assignments::AbstractMatrix{<:Integer})
        n = length(source_state)
        time_labels = _header_time_label.(headers)
        columns = ["header", "time_label", "source_state"]
        table = hcat(headers, time_labels, string.(source_state))
        for stage in 1:size(stage_assignments, 1)
            columns = vcat(columns, "stage_$(stage)")
            table = hcat(table, string.(vec(stage_assignments[stage, 1:n])))
        end
        return (; table=Matrix{String}(table), columns)
    end

    function _combined_assignment_table(headers, source_state_ac::AbstractVector{<:Integer},
            source_state_c::AbstractVector{<:Integer},
            stage_assignments_ac::AbstractMatrix{<:Integer},
            stage_assignments_c::AbstractMatrix{<:Integer})
        n = length(source_state_ac)
        time_labels = _header_time_label.(headers)
        columns = ["header", "time_label", "source_state_ac", "source_state_c"]
        table = hcat(headers, time_labels, string.(source_state_ac), string.(source_state_c))
        for stage in 1:size(stage_assignments_ac, 1)
            columns = vcat(columns, "ac_stage_$(stage)")
            table = hcat(table, string.(vec(stage_assignments_ac[stage, 1:n])))
        end
        for stage in 1:size(stage_assignments_c, 1)
            columns = vcat(columns, "c_stage_$(stage)")
            table = hcat(table, string.(vec(stage_assignments_c[stage, 1:n])))
        end
        return (; table=Matrix{String}(table), columns)
    end

    function _positive_period_psd(X::AbstractVecOrMat, dt::Real)
        Xmat = flatten_batch(X)
        nt = size(Xmat, 1)
        freqs = FFTW.fftfreq(nt, inv(Float64(dt)))
        pos = freqs .> 0
        periods = 1.0 ./ freqs[pos]
        order = sortperm(periods)
        psd = abs2.(fft(Float64.(Xmat), 1))
        return (;
            periods=Float64.(periods[order]),
            frequencies=Float64.(freqs[pos][order]),
            psd=Float32.(psd[pos, :][order, :]),
        )
    end

    function _bundle_stage_waveforms(data_bundle, stage::Symbol)
        if stage == :raw
            ac = hasproperty(data_bundle, :D1fac_raw) ? data_bundle.D1fac_raw : data_bundle.D1fac
            c = hasproperty(data_bundle, :D1fc_raw) ? data_bundle.D1fc_raw : data_bundle.D1fc
        elseif stage == :whitened
            ac = hasproperty(data_bundle, :D1fac_whitened) ? data_bundle.D1fac_whitened : data_bundle.D1fac
            c = hasproperty(data_bundle, :D1fc_whitened) ? data_bundle.D1fc_whitened : data_bundle.D1fc
        elseif stage == :despiked
            ac = hasproperty(data_bundle, :D1fac_despiked) ? data_bundle.D1fac_despiked : data_bundle.D1fac
            c = hasproperty(data_bundle, :D1fc_despiked) ? data_bundle.D1fc_despiked : data_bundle.D1fc
        else
            error("Unknown preprocessing stage $(stage).")
        end
        return (; ac=Float32.(ac), c=Float32.(c))
    end
end

# ╔═╡ a6066ca6-1350-4c54-9857-99d195873e6c
function save_vqvae_run(run_dir; model, rvq_st, para, training_para, loss_history, pair, data_bundle,
        analysis_settings=(;))
    mkpath(run_dir)
    averages = source_state_averages(model, rvq_st,
        (; D_ac_all=data_bundle.D1fac, D_c_all=data_bundle.D1fc);
        device=identity, cdev=default_cdev())
    labels = hasproperty(averages, :combo_labels) ? string.(averages.combo_labels) :
        string.(1:size(averages.acausal, 2))
    n_windows = min(size(data_bundle.D1fac, 2), size(data_bundle.D1fc, 2))
    window_headers = _window_headers(
        hasproperty(data_bundle, :headers) ? data_bundle.headers : nothing,
        n_windows,
    )
    window_time_labels = _header_time_label.(window_headers)
    source_states = _source_state_indices(averages.cache, model.K)
    stage_assignments_ac = Int.(averages.cache.stage_ac[:, 1:n_windows])
    stage_assignments_c = Int.(averages.cache.stage_c[:, 1:n_windows])
    source_state_ac = Int.(source_states.source_state_ac[1:n_windows])
    source_state_c = Int.(source_states.source_state_c[1:n_windows])
    assignment_ac = _assignment_table(window_headers, source_state_ac, stage_assignments_ac)
    assignment_c = _assignment_table(window_headers, source_state_c, stage_assignments_c)
    assignment = _combined_assignment_table(
        window_headers, source_state_ac, source_state_c,
        stage_assignments_ac, stage_assignments_c,
    )
    global_avg_ac = vec(mean(data_bundle.D1fac; dims=2))
    global_avg_c = vec(mean(data_bundle.D1fc; dims=2))
    raw_stage = _bundle_stage_waveforms(data_bundle, :raw)
    whitened_stage = _bundle_stage_waveforms(data_bundle, :whitened)
    global_avg_raw_ac = vec(mean(raw_stage.ac; dims=2))
    global_avg_raw_c = vec(mean(raw_stage.c; dims=2))
    global_avg_whitened_ac = vec(mean(whitened_stage.ac; dims=2))
    global_avg_whitened_c = vec(mean(whitened_stage.c; dims=2))
    codebook_waves = decode_codebook_waveforms(model, rvq_st)
    marginals_ac = length(model.K) >= 2 ?
        marginal_decomposition(Float32.(averages.acausal), Int.(averages.counts_ac);
            K1=model.K[1], K2=model.K[2]) : nothing
    marginals_c = length(model.K) >= 2 ?
        marginal_decomposition(Float32.(averages.causal), Int.(averages.counts_c);
            K1=model.K[1], K2=model.K[2]) : nothing

    jldsave(joinpath(run_dir, "source_state_averages.jld2");
        acausal=Float32.(averages.acausal),
        causal=Float32.(averages.causal),
        counts_ac=Float32.(averages.counts_ac),
        counts_c=Float32.(averages.counts_c),
        combo_labels=labels,
        marginal_stage1_ac=isnothing(marginals_ac) ? Float32[;;] : Float32.(marginals_ac.stage1_waves),
        marginal_stage1_c=isnothing(marginals_c) ? Float32[;;] : Float32.(marginals_c.stage1_waves),
        marginal_stage2_ac=isnothing(marginals_ac) ? Float32[;;] : Float32.(marginals_ac.stage2_waves),
        marginal_stage2_c=isnothing(marginals_c) ? Float32[;;] : Float32.(marginals_c.stage2_waves),
        marginal_stage1_labels=isnothing(marginals_ac) ? String[] : marginals_ac.stage1_labels,
        marginal_stage2_labels=isnothing(marginals_ac) ? String[] : marginals_ac.stage2_labels,
        marginal_stage1_counts_ac=isnothing(marginals_ac) ? Int[] : Int.(round.(vec(sum(reshape(Int.(averages.counts_ac), model.K[1], model.K[2]); dims=2)))),
        marginal_stage2_counts_ac=isnothing(marginals_ac) ? Int[] : Int.(round.(vec(sum(reshape(Int.(averages.counts_ac), model.K[1], model.K[2]); dims=1)))),
        marginal_stage1_counts_c=isnothing(marginals_c) ? Int[] : Int.(round.(vec(sum(reshape(Int.(averages.counts_c), model.K[1], model.K[2]); dims=2)))),
        marginal_stage2_counts_c=isnothing(marginals_c) ? Int[] : Int.(round.(vec(sum(reshape(Int.(averages.counts_c), model.K[1], model.K[2]); dims=1)))),
        global_avg_ac=Float32.(global_avg_ac),
        global_avg_c=Float32.(global_avg_c),
        window_headers=window_headers,
        window_time_labels=window_time_labels,
        source_state_ac=source_state_ac,
        source_state_c=source_state_c,
        stage_assignments_ac=stage_assignments_ac,
        stage_assignments_c=stage_assignments_c,
        assignment_table=assignment.table,
        assignment_table_columns=assignment.columns,
        assignment_table_ac=assignment_ac.table,
        assignment_table_c=assignment_c.table,
        assignment_table_ac_columns=assignment_ac.columns,
        assignment_table_c_columns=assignment_c.columns,
        analysis_settings=analysis_settings,
        global_avg_raw_ac=Float32.(global_avg_raw_ac),
        global_avg_raw_c=Float32.(global_avg_raw_c),
        global_avg_whitened_ac=Float32.(global_avg_whitened_ac),
        global_avg_whitened_c=Float32.(global_avg_whitened_c),
        codebook_stage1_waves=Float32.(codebook_waves.stage1),
        codebook_stage1_labels=codebook_waves.stage1_labels,
        codebook_stage2_waves=hasproperty(codebook_waves, :stage2) ? Float32.(codebook_waves.stage2) : Float32[;;],
        codebook_stage2_labels=hasproperty(codebook_waves, :stage2) ? codebook_waves.stage2_labels : String[],
        codebook_joint_waves=hasproperty(codebook_waves, :joint) ? Float32.(codebook_waves.joint) : Float32[;;],
        codebook_joint_labels=hasproperty(codebook_waves, :joint) ? codebook_waves.joint_labels : String[],
        whitening_fir=hasproperty(data_bundle, :whitening_fir) && !isnothing(data_bundle.whitening_fir) ? Float32.(data_bundle.whitening_fir) : Float32[],
        spike_bins=hasproperty(data_bundle, :spike_bins) ? Int.(data_bundle.spike_bins) : Int[],
        distance=data_bundle.distance,
        latitudes=data_bundle.latitudes,
        longitudes=data_bundle.longitudes,
        pair=pair)
    jldsave(joinpath(run_dir, "run_summary.jld2");
        vqvae_para=para, training_para=training_para, pair=pair,
        analysis_settings=analysis_settings,
        distance=data_bundle.distance,
        latitudes=data_bundle.latitudes,
        longitudes=data_bundle.longitudes,
        loss_history=loss_history)
    jldsave(joinpath(run_dir, "loss_history.jld2"); loss_history)
    @info "Saved v9_flux source-state analysis artifact" run_dir
    return run_dir
end

# ╔═╡ a1b2c3d5-0000-0000-0000-000000000001
function make_encoder_head(para, latent_len::Int, enc_channels::Int)
    conv = Flux.Conv((3,), enc_channels => enc_channels, activation; pad=SamePad())
    dense = Flux.Dense(latent_len * enc_channels, para.d ÷ 2)
    return FluxEncoderHead(conv, dense, latent_len, enc_channels)
end

# ╔═╡ 10000010-0000-0000-0000-000000000001
function get_vqvae(para; rng=Random.default_rng(), device=identity)
    length(para.K) == 2 || error("v9_flux requires exactly 2 codebook stages (length(K) must be 2). Got K=$(para.K).")
    all(>(1), para.K) || error("All K entries must be > 1.")
    iseven(para.d) || error("d must be even for v9_flux (required for split heads). Got d=$(para.d).")
    Random.seed!(rng, para.seed)

    encoder = make_encoder(para)
    # Probe encoder geometry with a dummy batch
    dummy = randn(rng, Float32, para.nt, 2)
    enc_dummy = encoder(waveform_to_conv3(dummy))
    latent_len, enc_channels, _ = size(enc_dummy)

    head1 = make_encoder_head(para, latent_len, enc_channels)
    head2 = make_encoder_head(para, latent_len, enc_channels)
    decoder1 = make_decoder(para, latent_len; latent_dim=para.d ÷ 2)
    decoder2 = make_decoder(para, latent_len; latent_dim=para.d ÷ 2)

    # Verify decoder geometry
    dec_dummy = decoder1(randn(rng, Float32, para.d ÷ 2, 2))
    size(dec_dummy, 1) == para.nt ||
        error("Decoder geometry mismatch: output length $(size(dec_dummy, 1)) != nt $(para.nt). Adjust decoder strides/kernels.")

    model = VQVAE(encoder, head1, head2, decoder1, decoder2, Tuple(Int.(para.K)),
        para.d, latent_len,
        para.beta_commit, para.ema_decay,
        para.epsilon, para.dead_threshold)
    rvq_st = init_rvq_state(rng, para.d ÷ 2, model.K)
    model = device(model)
    rvq_st = device(rvq_st)

    loss_history = fresh_loss_history()
    @info "VQ-VAE v9_flux geometry" nt=para.nt d=para.d half=para.d÷2 latent_len K=para.K enc_channels
    return model, rvq_st, loss_history
end

# ╔═╡ train-pairs-lazy
function train_selected_pairs_lazy(selected_pairs;
    seeds, training_para::VQVAE_Training_Para,
    vqvae_parameters::NamedTuple,
    save_root::String,
    filepath::String, dt::Real=1.0, period_min::Real=10, period_max::Real=50,
    n_max::Union{Nothing,Integer}=nothing,
    bp_filter, per_waveform_whitening_kernel_length::Int,
    device=nothing, analysis_settings=(;))
    isempty(selected_pairs) && return Any[]
    isempty(seeds) && error("Provide at least one seed.")
    xdev = isnothing(device) ? default_xdev() : device
    cdev = default_cdev()
    rng = Xoshiro(1234)
    results = Any[]
    for pair_raw in selected_pairs
        pair = (String(pair_raw[1]), String(pair_raw[2]))
        @info "Loading pair" pair
        bundle = build_training_bundle(pair; filepath, dt, period_min, period_max)
        bundle = trim_training_bundle(bundle; n_max, rng)
        data = make_pooled_split(bundle.D1fac, bundle.D1fc; rng)
        pd_raw = (; pair, data, data_bundle=bundle)
        @info "Whitening pair" pair
        pd = whiten_pair_entry(pd_raw; bp_filter, per_waveform_whitening_kernel_length)
        nt = size(pd.data.D_train, 1)
        for (run_index, seed) in enumerate(seeds)
            @info "Training pair" pair run_index seed
            rng_seed = Xoshiro(seed)
            para = VQVAE_Para(; merge(vqvae_parameters, (; nt, seed))...)
            model, rvq_st, loss_history = get_vqvae(para; rng=rng_seed, device=xdev)
            train_start = time()
            model, rvq_st, loss_history = update(
                model, rvq_st, loss_history,
                pd.data.D_train, pd.data.D_test,
                para, training_para;
                device=xdev, cdev,
            )
            @info "Finished pair run" pair run_index seed time_s=round(time() - train_start; digits=3)
            run_dir = run_dir_for_seed(save_root, pair, seed)
            save_vqvae_run(run_dir; model, rvq_st,
                para, training_para, loss_history, pair,
                data_bundle=pd.data_bundle, analysis_settings)
            push!(results, (; pair, run_index, seed, run_dir,
                model, rvq_st, para,
                training_para, loss_history, data=pd.data,
                data_bundle=pd.data_bundle))
        end
        pd = nothing
        pd_raw = nothing
        GC.gc()
    end
    return results
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
ConcreteStructs = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
InlineStrings = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
CUDA = "~5"
ConcreteStructs = "~0.2"
DSP = "~0.8"
Distances = "~0.10"
Enzyme = "~0.13"
EnzymeCore = "~0.8"
FFTW = "~1"
Flux = "~0.14"
InlineStrings = "~1.4"
JLD2 = "~0.4"
MLUtils = "~0.4"
NNlib = "~0.9"
Optimisers = "~0.3"
PlutoPlotly = "~0.6"
ProgressLogging = "~0.1"
StatsBase = "~0.34"
Zygote = "~0.6"
"""
