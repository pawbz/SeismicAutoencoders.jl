### A Pluto.jl notebook ###
# v0.20.27

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
        JLD2,
        LinearAlgebra,
        Lux,
        MLUtils,
        NNlib,
        Optimisers,
        ProgressLogging,
        ProgressMeter,
        PlutoPlotly,
        Random,
        Reactant,
        Statistics,
        StatsBase,
        InlineStrings,
        UnicodePlots,
        Zygote
end

# ╔═╡ 330652f1-0754-48a4-9a0f-4fb9d6824222
using Distances

# ╔═╡ 10000002-0000-0000-0000-000000000001
md"""
# VQ-VAE v10 Architecture — Split-Decoder Interferometric Mixture VQ

Lux/Reactant single-pair Interferometric Split-Decoder VQ-VAE with SEANet-style encoder/decoder.
Each station pair trains an independent model. Two separate encoder heads map shared features
to z_e1 and z_e2 (each d÷2), quantized independently. Two separate decoders produce x1_hat
and x2_hat; reconstruction is additive: x_hat = x1_hat + x2_hat, forcing specialization.
"""

# ╔═╡ 10000003-0000-0000-0000-000000000001
begin
    const activation = x -> NNlib.leakyrelu(x, 0.1f0)

    function ensure_reactant_xla_flags!()
        xla_flags = get(ENV, "XLA_FLAGS", "")
        flag = "--xla_gpu_enable_cublaslt=true"
        if !occursin(flag, xla_flags)
            ENV["XLA_FLAGS"] = isempty(xla_flags) ? flag : "$(xla_flags) $(flag)"
        end
        return ENV["XLA_FLAGS"]
    end

    default_xdev(; force::Bool=true) = reactant_device(; force)
    default_cdev() = cpu_device()
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
    codebook_exclusivity_weight::Float32 = 0.1f0
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
    autodiff_backend::Symbol = :auto
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
md"## Lux Encoder and Decoder"

# ╔═╡ ef72f750-3026-4c6b-8cf6-9fc8964658f8
function make_seanet_residual_block(dim, n_residual_layers, dilation_base, residual_kernel_size)
    layers = Any[]
    hidden = max(1, dim ÷ 2)
    for j in 0:n_residual_layers-1
        d = dilation_base ^ j
        block = Chain(
            activation,
            Conv((residual_kernel_size,), dim => hidden, activation; pad=SamePad(), dilation=d),
            Conv((1,), hidden => dim; pad=SamePad()),
        )
        push!(layers, @compact(; block) do x
            @return x + block(x)
        end)
    end
    return Chain(layers...)
end

# ╔═╡ 6ce85f19-73ba-4785-ab52-1767cf79efa8
function make_encoder(para)
    ratios = reverse(para.ratios)
    mult = 1
    layers = Any[]
    push!(layers, Conv((para.enc_kernel_size,), 1 => mult * para.n_filters; pad=SamePad()))
    for ratio in ratios
        push!(layers, make_seanet_residual_block(
            mult * para.n_filters, para.n_residual_layers, para.dilation_base, para.residual_kernel_size))
        push!(layers, activation)
        push!(layers, Conv((ratio * 2,), mult * para.n_filters => mult * para.n_filters * 2;
            stride=ratio, pad=SamePad()))
        mult *= 2
    end
    push!(layers, activation)
    push!(layers, Conv((para.enc_kernel_size,), mult * para.n_filters => mult * para.n_filters; pad=SamePad()))
    return Chain(layers...)
end

# ╔═╡ a0416f71-242a-44a7-b8f6-186726318611
function make_decoder(para, latent_len::Int; latent_dim::Int=para.d)
    ratios = para.ratios
    mult = 2 ^ length(ratios)

    bottleneck_len = para.nt
    for r in ratios; bottleneck_len = cld(bottleneck_len, r); end
    bottleneck_channels = para.n_filters * mult

    layers = Any[]
    push!(layers, Conv((para.dec_kernel_size,), bottleneck_channels => mult * para.n_filters; pad=SamePad()))
    for ratio in ratios
        push!(layers, activation)
        push!(layers, ConvTranspose((ratio * 2,), mult * para.n_filters => mult * para.n_filters ÷ 2;
            stride=ratio, pad=SamePad()))
        mult ÷= 2
        push!(layers, make_seanet_residual_block(
            mult * para.n_filters, para.n_residual_layers, para.dilation_base, para.residual_kernel_size))
    end
    push!(layers, activation)
    push!(layers, Conv((para.dec_kernel_size,), para.n_filters => 1; pad=SamePad()))
    upchain = Chain(layers...)

    linear = Dense(latent_dim, bottleneck_len * bottleneck_channels, activation)
    return @compact(; linear, upchain, bottleneck_len, bottleneck_channels) do z
        y = linear(z)
        y3 = reshape(y, bottleneck_len, bottleneck_channels, size(y, 2))
        out = upchain(y3)
        wave = conv3_to_waveform(out)
        @return wave
    end
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
        rng=Lux.replicate(stage_rng),
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
    z_detached = EnzymeCore.ignore_derivatives(z)
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
    rng = Lux.replicate(stage.rng)
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
	        _, indices = EnzymeCore.ignore_derivatives(vq_lookup(stage.embedding, residual))
	        # randomly permute stage-2+ assignments during training to discourage inter-stage correlation
	        indices = if training && s > 1 && p_stage2_shuffle > 0f0
	            rng_local = Lux.replicate(stage.rng)
	            rand(rng_local, Float32) < p_stage2_shuffle ?
	                indices[randperm(rng_local, length(indices))] : indices
	        else
	            indices
	        end
	        z_q_detached = EnzymeCore.ignore_derivatives(stage.embedding[:, indices])
	        if !training
	            all_indices = (all_indices..., reshape(indices, 1, :))
	        end

	        counts = if training
	            new_stage, counts_local = update_rvq_stage_state(stage, residual, indices,
	                ema_decay, epsilon, dead_threshold)
	            new_stages = replace_tuple(new_stages, s, new_stage)
	            counts_local
	        else
	            counts_and_sums(EnzymeCore.ignore_derivatives(residual), indices, K[s])[1]
	        end

	        z_q = residual .+ EnzymeCore.ignore_derivatives(z_q_detached .- residual)
	        z_q_total = z_q_total .+ z_q
	        residual = residual .- EnzymeCore.ignore_derivatives(z_q_detached)
	        commit_loss += beta_commit * mse_loss(residual .+ EnzymeCore.ignore_derivatives(z_q_detached),
	            EnzymeCore.ignore_derivatives(z_q_detached))
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
	            rng_local = Lux.replicate(stage.rng)
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
	        z_q_detached = EnzymeCore.ignore_derivatives(payload.z_q_stages[s])
	        z_q = residual .+ EnzymeCore.ignore_derivatives(z_q_detached .- residual)
	        z_q_total = z_q_total .+ z_q
	        commit_loss += beta_commit * mse_loss(residual, z_q_detached)
	        residual = residual .- z_q_detached
	    end
	
	    nstage = Float32(length(K))
	    return (; z_q=z_q_total,
	        stage_indices=nothing,
	        coarse_indices=nothing,
	        commit_loss=commit_loss / nstage,
	        entropy_loss=EnzymeCore.ignore_derivatives(payload.entropy_loss),
	        perplexity=EnzymeCore.ignore_derivatives(payload.perplexity),
	        stage_perplexities=nothing)
	end
end

# ╔═╡ 10000010-0000-0000-0000-000000000002
md"## Split VQ (v10)"

# ╔═╡ 10000010-0000-0000-0000-000000000003
begin
	# Independent quantization of two separate z_e halves — no amplitude mixing.
	function split_vq_quantize(z_e1, z_e2, rvq_state, K::Tuple;
	    beta_commit::Float32, ema_decay::Float32, epsilon::Float32,
	    dead_threshold::Int, training::Bool)

	    stage1 = rvq_state.stages[1]
	    stage2 = rvq_state.stages[2]

	    _, idx1 = EnzymeCore.ignore_derivatives(vq_lookup(stage1.embedding, z_e1))
	    _, idx2 = EnzymeCore.ignore_derivatives(vq_lookup(stage2.embedding, z_e2))
	    z_q1_det = EnzymeCore.ignore_derivatives(stage1.embedding[:, idx1])
	    z_q2_det = EnzymeCore.ignore_derivatives(stage2.embedding[:, idx2])

	    # STE per sub-vector
	    z_q1 = z_e1 .+ EnzymeCore.ignore_derivatives(z_q1_det .- z_e1)
	    z_q2 = z_e2 .+ EnzymeCore.ignore_derivatives(z_q2_det .- z_e2)

	    commit_loss = (beta_commit * mse_loss(z_e1, z_q1_det)
	                 + beta_commit * mse_loss(z_e2, z_q2_det)) / 2f0

	    new_stages = rvq_state.stages
	    counts1 = if training
	        new_s1, c1_loc = update_rvq_stage_state(stage1, z_e1, idx1,
	            ema_decay, epsilon, dead_threshold)
	        new_stages = replace_tuple(new_stages, 1, new_s1)
	        c1_loc
	    else
	        counts_and_sums(EnzymeCore.ignore_derivatives(z_e1), idx1, K[1])[1]
	    end
	    counts2 = if training
	        new_s2, c2_loc = update_rvq_stage_state(stage2, z_e2, idx2,
	            ema_decay, epsilon, dead_threshold)
	        new_stages = replace_tuple(new_stages, 2, new_s2)
	        c2_loc
	    else
	        counts_and_sums(EnzymeCore.ignore_derivatives(z_e2), idx2, K[2])[1]
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

	# Compiled training step: frozen codebook lookups from payload, STE only.
	function split_vq_quantize_precomputed(z_e1, z_e2, payload, K::Tuple; beta_commit::Float32)
	    z_q1_det = EnzymeCore.ignore_derivatives(payload.z_q_stages[1])
	    z_q2_det = EnzymeCore.ignore_derivatives(payload.z_q_stages[2])
	    z_q1 = z_e1 .+ EnzymeCore.ignore_derivatives(z_q1_det .- z_e1)
	    z_q2 = z_e2 .+ EnzymeCore.ignore_derivatives(z_q2_det .- z_e2)
	    commit_loss = (beta_commit * mse_loss(z_e1, z_q1_det)
	                 + beta_commit * mse_loss(z_e2, z_q2_det)) / 2f0
	    return (;
	        z_q1, z_q2,
	        stage_indices=nothing,
	        coarse_indices=nothing,
	        commit_loss,
	        entropy_loss=EnzymeCore.ignore_derivatives(payload.entropy_loss),
	        perplexity=EnzymeCore.ignore_derivatives(payload.perplexity),
	        stage_perplexities=nothing,
	    )
	end
end

# ╔═╡ 1000000e-0000-0000-0000-000000000001
md"## VQ-VAE Model"

# ╔═╡ 1000000f-0000-0000-0000-000000000001
begin
	@concrete struct VQVAE <: AbstractLuxContainerLayer{(:encoder, :head1, :head2, :decoder1, :decoder2)}
	    encoder  <: AbstractLuxLayer
	    head1    <: AbstractLuxLayer   # Conv+Dense → z_e1 (d÷2)
	    head2    <: AbstractLuxLayer   # Conv+Dense → z_e2 (d÷2)
	    decoder1 <: AbstractLuxLayer   # z_q1 → x1_hat
	    decoder2 <: AbstractLuxLayer   # z_q2 → x2_hat
	    K::Tuple
	    d::Int
	    latent_len::Int
	    beta_commit::Float32
	    ema_decay::Float32
	    epsilon::Float32
	    dead_threshold::Int
	end

	function Lux.initialparameters(rng::AbstractRNG, m::VQVAE)
	    return (;
	        encoder=Lux.initialparameters(rng, m.encoder),
	        head1=Lux.initialparameters(rng, m.head1),
	        head2=Lux.initialparameters(rng, m.head2),
	        decoder1=Lux.initialparameters(rng, m.decoder1),
	        decoder2=Lux.initialparameters(rng, m.decoder2),
	    )
	end

	function Lux.initialstates(rng::AbstractRNG, m::VQVAE)
	    return (;
	        encoder=Lux.initialstates(rng, m.encoder),
	        head1=Lux.initialstates(rng, m.head1),
	        head2=Lux.initialstates(rng, m.head2),
	        decoder1=Lux.initialstates(rng, m.decoder1),
	        decoder2=Lux.initialstates(rng, m.decoder2),
	        rvq=init_rvq_state(rng, m.d ÷ 2, m.K),   # each codebook is (d÷2, K[s])
	    )
	end

	function encoder_latents(m, x, ps, st)
	    feat, st_enc = m.encoder(waveform_to_conv3(x), ps.encoder, st.encoder)
	    z_e1, st_h1 = m.head1(feat, ps.head1, st.head1)   # (d÷2, B)
	    z_e2, st_h2 = m.head2(feat, ps.head2, st.head2)   # (d÷2, B)
	    z_e = vcat(z_e1, z_e2)                             # (d, B) — for kNN
	    return (; feat, z_e, z_e1, z_e2),
	           (; encoder=st_enc, head1=st_h1, head2=st_h2)
	end

	function encode(m, ps, st, x; beta_commit::Float32=m.beta_commit, training::Bool=false)
	    lat, st_lat = encoder_latents(m, x, ps, st)
	    q, st_rvq = split_vq_quantize(lat.z_e1, lat.z_e2, st.rvq, m.K;
	        beta_commit, ema_decay=m.ema_decay, epsilon=m.epsilon,
	        dead_threshold=m.dead_threshold, training)
	    st_new = (; encoder=st_lat.encoder, head1=st_lat.head1, head2=st_lat.head2,
	        decoder1=st.decoder1, decoder2=st.decoder2, rvq=st_rvq)
	    return merge(lat, q), st_new
	end

	function encode_z_e_inference(m, ps, st, x)
	    st = Lux.testmode(st)
	    enc, _ = encode(m, ps, st, x; training=false)
	    return enc.z_e
	end

	function encode_z_e_training(m, ps, st, x)
	    lat, _ = encoder_latents(m, x, ps, st)
	    return lat.z_e
	end

	function decode_from_latents(m, ps, st, result)
	    x1hat, st_dec1 = m.decoder1(result.z_q1, ps.decoder1, st.decoder1)
	    x2hat, st_dec2 = m.decoder2(result.z_q2, ps.decoder2, st.decoder2)
	    xhat = x1hat .+ x2hat
	    st_new = merge(st, (; decoder1=st_dec1, decoder2=st_dec2))
	    return merge(result, (; xhat, x1hat, x2hat)), st_new
	end

	function (m::VQVAE)(x, ps, st; beta_commit::Float32=m.beta_commit, training::Bool=true)
	    enc, st_enc = encode(m, ps, st, x; beta_commit, training)
	    return decode_from_latents(m, ps, st_enc, enc)
	end

	function forward_with_precomputed_vq(m, x, ps, st, payload;
	    beta_commit::Float32=m.beta_commit)
	    lat, st_lat = encoder_latents(m, x, ps, st)
	    q = split_vq_quantize_precomputed(lat.z_e1, lat.z_e2, payload, m.K; beta_commit)
	    st_mid = (; encoder=st_lat.encoder, head1=st_lat.head1, head2=st_lat.head2,
	        decoder1=st.decoder1, decoder2=st.decoder2, rvq=st.rvq)
	    return decode_from_latents(m, ps, st_mid, merge(lat, q))
	end

	codebook_size(m) = m.K[1]
	num_rvq_stages(m) = length(m.K)
	get_codebook(st, stage::Int=1) = Array(st.rvq.stages[stage].embedding)
	get_codebooks(st) = [Array(stage.embedding) for stage in st.rvq.stages]
end

# ╔═╡ 10000011-0000-0000-0000-000000000001
md"## Losses and kNN Targets"

# ╔═╡ 13634a6c-abda-4084-9b5b-f6761fd728ad
begin
    function _normalize_codebook_waveforms(waves)
        x = waves .- mean(waves; dims=1)
        den = sqrt.(sum(abs2, x; dims=1) .+ 1f-8)
        x ./ den
    end

    function codebook_component_exclusivity_loss(model, ps, st)
        z1 = st.rvq.stages[1].embedding
        z2 = st.rvq.stages[2].embedding
        x1hat, _ = model.decoder1(z1, ps.decoder1, st.decoder1)
        x2hat, _ = model.decoder2(z2, ps.decoder2, st.decoder2)
        x1 = _normalize_codebook_waveforms(x1hat)
        x2 = _normalize_codebook_waveforms(x2hat)
        corr = transpose(x1) * x2
        return mean(abs2, corr)
    end

	function vqvae_loss(model, ps, st, x, target, para; training::Bool)
	    result, st_new = model(x, ps, st; beta_commit=para.beta_commit, training)
	    recon_loss = mse_loss(result.xhat, target)
        w_excl = para.codebook_exclusivity_weight
        codebook_exclusivity_loss = iszero(w_excl) ?
            zero(recon_loss) : codebook_component_exclusivity_loss(model, ps, st_new)
	    total = recon_loss + result.commit_loss +
            para.entropy_weight * result.entropy_loss +
            w_excl * codebook_exclusivity_loss
	    return total, st_new, (; result, recon_loss,
	        commit_loss=result.commit_loss, entropy_loss=result.entropy_loss,
            codebook_exclusivity_loss,
	        perplexity=result.perplexity, stage_perplexities=result.stage_perplexities)
	end

	function vqvae_precomputed_loss(model, ps, st, batch, para)
	    result, st_new = forward_with_precomputed_vq(
	        model, batch.x, ps, st, batch.vq_payload; beta_commit=para.beta_commit
	    )
	    recon_loss = mse_loss(result.xhat, batch.target)
        w_excl = para.codebook_exclusivity_weight
        codebook_exclusivity_loss = iszero(w_excl) ?
            zero(recon_loss) : codebook_component_exclusivity_loss(model, ps, st_new)
	    total = recon_loss + result.commit_loss +
            para.entropy_weight * result.entropy_loss +
            w_excl * codebook_exclusivity_loss
	    return total, st_new, (; result, recon_loss,
	        commit_loss=result.commit_loss, entropy_loss=result.entropy_loss,
            codebook_exclusivity_loss,
	        perplexity=result.perplexity, stage_perplexities=result.stage_perplexities)
	end
end

# ╔═╡ 8950cf6d-f5d2-4bcc-90ab-12ecf79f7c35


# ╔═╡ 91a4fb9f-f5d4-406f-b384-282d8a48257f
begin
	@concrete struct VQVAELoss
	    para
	end
	function (l::VQVAELoss)(model, ps, st, batch)
	    return vqvae_precomputed_loss(model, ps, st, batch, l.para)
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
function rebuild_latent_index!(idx::LatentIndex, model, ps, st, X;
    Mnn::Int=idx.Mnn, device=identity, cdev=default_cdev(), encode_compiled=nothing,
    knn_search_chunk_size_fraction::Float64=1.0, n_compiled::Union{Nothing,Int}=nothing)
    X_cpu = Float32.(cdev(flatten_batch(X)))
    _, N = size(X_cpu)
    # Use all waveforms - batching with padding handles variable-N inference
    Mnn >= 1 || error("Mnn must be >= 1.")
    N >= Mnn + 1 || error("Need at least Mnn + 1 samples; got N=$N and Mnn=$Mnn.")
    D = model.d
    chunk_size = max(Mnn + 1, round(Int, knn_search_chunk_size_fraction * N))
    approximate = chunk_size < N
    if isnothing(encode_compiled) && device isa ReactantDevice
        error("A compiled encoder is required to rebuild the latent index on a Reactant device. Ensure encode_z_e_inference compiles successfully.")
    end
    embeddings = Matrix{Float32}(undef, D, N)
    ps_dev = device(ps)
    st_eval = Lux.testmode(device(st))
    embedding_start = time()
    if isnothing(encode_compiled)
        enc, _ = encode(model, ps_dev, st_eval, device(X_cpu); training=false)
        embeddings .= Float32.(cdev(enc.z_e))
    else
        # Use batch_inference_with_padding to handle variable-N with fixed-size compiled executor
        batch_size = n_compiled  # Compiled at batchsize
        z_list = Vector{Matrix{Float32}}()
        for start_idx in 1:batch_size:N
            end_idx = min(start_idx + batch_size - 1, N)
            n_batch = end_idx - start_idx + 1

            # Extract batch and pad if needed
            X_batch = X_cpu[:, start_idx:end_idx]
            if n_batch < batch_size
                pad_size = batch_size - n_batch
                X_batch = hcat(X_batch, randn(Float32, size(X_cpu, 1), pad_size))
            end

            # Run compiled inference
            z_batch = Float32.(cdev(encode_compiled(model, ps_dev, st_eval, device(X_batch))))
            z_valid = z_batch[:, 1:n_batch]
            push!(z_list, z_valid)
        end
        z_cpu = hcat(z_list...)
        embeddings .= z_cpu
    end
    embedding_time = time() - embedding_start
    normalize_start = time()
    _l2_normalize_columns!(embeddings)
    normalize_time = time() - normalize_start
    knn_start = time()
    nthreads = Threads.nthreads() + 1
    neighbor_ids = Matrix{Int}(undef, Mnn, N)
    if approximate
        # Partition a random permutation of 1:N into chunks of chunk_size.
        # Every query in the same chunk shares the same candidate set → one
        # Distances.pairwise! call per chunk (chunk_size×chunk_size dot products)
        # instead of N individual BLAS calls.
        perm = MLUtils.randperm(N)
        # candidate_ids[:,j] = chunk_size candidate indices for query j (self at end)
        candidate_ids = Matrix{Int}(undef, chunk_size, N)
        # score buffer: chunk_size × N; each column holds intra-chunk cosine sims for j
        if size(idx.dist_matrix) != (chunk_size, N)
            idx.dist_matrix = Matrix{Float32}(undef, chunk_size, N)
        end
        chunk_emb    = Matrix{Float32}(undef, D, chunk_size)           # preallocated, reused per chunk
        chunk_scores = Matrix{Float32}(undef, chunk_size, chunk_size)   # preallocated, reused per chunk
        num_chunks = cld(N, chunk_size)
        for ci in 1:num_chunks
            chunk_start = (ci - 1) * chunk_size + 1
            chunk_end   = min(ci * chunk_size, N)
            chunk       = view(perm, chunk_start:chunk_end)
            nc          = length(chunk)   # may be < chunk_size for the last chunk
            # gather embeddings for this chunk's members
            chunk_emb_view = view(chunk_emb, :, 1:nc)
            chunk_emb_view .= view(embeddings, :, chunk)
            # pairwise cosine similarities within chunk: nc × nc (L2-normalised → dot product)
            chunk_scores_view = view(chunk_scores, 1:nc, 1:nc)
            Distances.pairwise!(CosineDist(), chunk_scores_view, chunk_emb_view; dims=2)
            # fill candidate_ids and dist_matrix columns for each member of this chunk
            for (li, j) in enumerate(chunk)
                cands = view(candidate_ids, :, j)
                cands[1:nc] .= chunk
                # pad short last chunk by wrapping within the same chunk
                for fi in nc+1:chunk_size
                    cands[fi] = chunk[mod1(fi, nc)]
                end
                # copy distances from the pairwise matrix; self-distance (0.0) is at position li
                scores = view(idx.dist_matrix, :, j)
                scores[1:nc] .= view(chunk_scores_view, :, li)
                scores[li] = Inf32              # exclude self (CosineDist(x,x) == 0)
                scores[nc+1:chunk_size] .= Inf32   # padded slots never selected
            end
        end
        if length(idx.perm_scratch) != nthreads || (!isempty(idx.perm_scratch) && length(idx.perm_scratch[1]) != chunk_size)
            idx.perm_scratch = [collect(1:chunk_size) for _ in 1:nthreads]
        end
        Threads.@threads for j in 1:N
            # CosineDist: smaller = more similar; exclude self (+Inf) and padding (+Inf)
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
        for j in 1:N; idx.dist_matrix[j, j] = -Inf32; end   # zero diagonal before threaded read
        Threads.@threads for j in 1:N
            top_k = partialsortperm!(idx.perm_scratch[Threads.threadid()],
                view(idx.dist_matrix, :, j), 1:Mnn; rev=true)
            neighbor_ids[:, j] .= top_k
        end
    end
    knn_time = time() - knn_start
    @debug "Latent index rebuild breakdown" N D Mnn chunk_size approximate julia_threads=Threads.nthreads() embedding_time_s=round(embedding_time; digits=3) normalize_time_s=round(normalize_time; digits=3) knn_time_s=round(knn_time; digits=3)
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
	
	function replace_train_state_states(train_state, states)
	    return Training.TrainState(
	        train_state.cache,
	        train_state.objective_function,
	        train_state.allocator_cache,
	        train_state.model,
	        train_state.parameters,
	        states,
	        train_state.optimizer,
	        train_state.optimizer_state,
	        train_state.step,
	    )
	end

	function replace_train_state_cache(train_state, cache)
	    return Training.TrainState(
	        cache,
	        train_state.objective_function,
	        train_state.allocator_cache,
	        train_state.model,
	        train_state.parameters,
	        train_state.states,
	        train_state.optimizer,
	        train_state.optimizer_state,
	        train_state.step,
	    )
	end
	
	function prepare_vq_training_batch(model, ps, st, batch, X_cpu, idx,
	    epoch::Int, para, training_para;
	    device=identity, cdev=default_cdev(), encode_compiled=nothing,
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
	    if isnothing(encode_compiled)
	        ps_cpu = cdev(ps)
	        st_cpu = Lux.testmode(cdev(st))
	        lat, _ = encoder_latents(model, batch.x, ps_cpu, st_cpu)
	        z_e1_cpu = lat.z_e1
	        z_e2_cpu = lat.z_e2
	        rvq_cpu = st_cpu.rvq
	    else
	        z_e_cpu = Float32.(cdev(encode_compiled(model, ps, Lux.testmode(st), device(batch.x))))
	        half = model.d ÷ 2
	        z_e1_cpu = z_e_cpu[1:half, :]
	        z_e2_cpu = z_e_cpu[half+1:end, :]
	        rvq_cpu = cdev(st.rvq)
	    end
	    payload_cpu, rvq_cpu = prepare_split_payload(z_e1_cpu, z_e2_cpu, rvq_cpu, model.K;
	        ema_decay=para.ema_decay,
	        epsilon=para.epsilon,
	        dead_threshold=para.dead_threshold,
	        training=true)
	    st_dev = merge(st, (; rvq=device(rvq_cpu)))
	    payload_time = time() - payload_start
	    return merge(bdev, (; vq_payload=device(payload_cpu))), st_dev, (; target_time, pack_time, payload_time)
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
            train_codebook_exclusivity=Float32[],
            train_perplexity=Float32[],
            test_recon_mse=Float32[],
            epoch_time_s=Float32[],
            throughput=Float32[],
        )
    end

    function reset_vqvae(model; seed::Integer, device=identity)
        rng = Xoshiro(seed)
        ps, st = Lux.setup(rng, model)
        return (ps, st) |> device
    end

	function recon_mse_inference(model, ps, st, x; normalize_target::Bool=false)
	    target = normalize_target ? Float32.(MLUtils.normalise(x; dims=1)) : x
	    result, _ = model(x, ps, Lux.testmode(st); training=false)
	    return mse_loss(result.xhat, target)
	end
	
	function record_train_metrics!(loss_history, train_m, test_recon_mse::Real,
	    epoch_time::Real, throughput::Real)
	    push!(loss_history.train_objective, train_m.total)
	    push!(loss_history.train_target_mse, train_m.recon_loss)
	    push!(loss_history.train_commit, train_m.commit_loss)
	    push!(loss_history.train_entropy, train_m.entropy_loss)
	    push!(loss_history.train_codebook_exclusivity, train_m.codebook_exclusivity_loss)
	    push!(loss_history.train_perplexity, train_m.perplexity)
	    push!(loss_history.test_recon_mse, Float32(test_recon_mse))
	    push!(loss_history.epoch_time_s, Float32(epoch_time))
	    push!(loss_history.throughput, Float32(throughput))
	    return loss_history
	end
end

# ╔═╡ 2ed0a636-4fc5-4bbe-9886-1aa351510c67
function maybe_compile_inference(model, ps, st, sample_x, training_para::VQVAE_Training_Para;
    device=identity, sample_batch_x=nothing, compile_latent_index::Bool=true,
    compile_train_encoder::Bool=true)
    start = time()
    st_eval = Lux.testmode(st)
    encode_z_e_compiled = if compile_latent_index
        try
            @compile encode_z_e_inference(model, ps, st_eval, sample_x)
        catch err
            error("Reactant @compile failed for latent-index encoder. Cannot rebuild latent index without compiled encoder path: $(err)")
        end
    else
        nothing
    end
    st_payload = Lux.testmode(st)
    train_sample = isnothing(sample_batch_x) ? sample_x : sample_batch_x
    encode_z_e_train_compiled = if compile_train_encoder
        try
            @compile encode_z_e_training(model, ps, st_payload, train_sample)
        catch err
            error("Reactant @compile failed for training-batch encoder. Cannot prepare VQ payload efficiently: $(err)")
        end
    else
        nothing
    end
    @info "Reactant inference compile complete" N=size(sample_x, 2) batch_N=size(train_sample, 2) compile_time_s=round(time() - start; digits=3) encode_compiled=!isnothing(encode_z_e_compiled) train_encode_compiled=!isnothing(encode_z_e_train_compiled)
    return (; encode_z_e=encode_z_e_compiled, encode_z_e_train=encode_z_e_train_compiled)
end

# ╔═╡ batch_inference_with_padding helper
"""
    batch_inference_with_padding(X_cpu::AbstractMatrix{Float32}, compiled_encode::Function,
                                  batch_size::Int)

Run inference on variable-N data using a fixed batch-size compiled executable.
Splits X into batch_size chunks, pads last chunk with randn if needed, runs compiled inference,
and extracts only the valid outputs (discarding padding).

Function signature: compiled_encode(X_batch_on_device) -> z_batch_on_device

Returns latent codes with shape (latent_dim, N_original) on CPU
"""
function batch_inference_with_padding(X_cpu::AbstractMatrix{Float32}, compiled_encode::Function,
                                       batch_size::Int)
    nt, N = size(X_cpu)
    if isnothing(compiled_encode)
        error("compiled_encode cannot be nothing")
    end

    latent_codes = Vector{Float32}[]

    for start_idx in 1:batch_size:N
        end_idx = min(start_idx + batch_size - 1, N)
        n_batch = end_idx - start_idx + 1

        # Extract batch and pad if needed
        X_batch = X_cpu[:, start_idx:end_idx]
        if n_batch < batch_size
            pad_size = batch_size - n_batch
            X_batch = hcat(X_batch, randn(Float32, nt, pad_size))  # Pad with noise
        end

        # Run compiled inference (function handles device transfer)
        z_batch = compiled_encode(X_batch)  # shape: (latent_dim, batch_size) on device
        z_batch_cpu = Float32.(vec(z_batch))  # Move to CPU and flatten

        # Extract only valid outputs (discard padding)
        latent_dim = size(z_batch, 1)
        z_valid = reshape(z_batch_cpu[1:latent_dim * n_batch], latent_dim, n_batch)
        push!(latent_codes, vec(z_valid))
    end

    # Concatenate all valid latent codes
    return hcat(latent_codes...)
end

# ╔═╡ 81ffa3b4-8a3c-4571-a494-6108378fff4a
function compute_whitening_fir(X_cpu::AbstractMatrix{Float32}, dt::Float64;
    kernel_length::Int=64, min_power_fraction::Float64=0.05)
    nt = size(X_cpu, 1)
    P = vec(mean(abs2.(fft(Float64.(X_cpu), 1)); dims=2))
    P[1] = 0.0
    P .= max.(P, min_power_fraction * maximum(P))
    W = P .^ (-0.25)   # ¼ power: filtfilt squares the magnitude response → net P^(-0.5) whitening
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

# ╔═╡ 6322b22f-ce89-475f-9073-fdca44f212b1
function apply_whitening_fir(X_cpu::AbstractMatrix{Float32}, fir::AbstractVector{Float32})
    return Float32.(DSP.filtfilt(Float64.(fir), [1.0], Float64.(X_cpu)))
end

# ╔═╡ 990c86a6-6267-455f-b62f-9295de5ecb75
function detect_spectral_spikes(X_cpu::AbstractMatrix{Float32};
        spike_threshold::Float64=10.0)
    nt = size(X_cpu, 1)
    P = vec(mean(abs2.(fft(Float64.(X_cpu), 1)); dims=2))
    half = nt ÷ 2 + 1
    P_half = P[1:half]
    med = median(P_half[2:end])   # skip DC bin for median
    spike_bins = findall(P_half .> spike_threshold * med)
    return spike_bins, P_half, med
end

# ╔═╡ 5492fa27-20fc-4526-aaab-9bee3f866c51
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

# ╔═╡ 89a363dd-b142-4dce-8783-307136602c8e
function compile_vqvae_helpers(model, ps, st, train_x_cpu, training_para::VQVAE_Training_Para;
    device=identity, sample_batch_x=nothing)
    train_x_cpu = Float32.(flatten_batch(train_x_cpu))
    size(train_x_cpu, 2) >= training_para.batchsize ||
        error("Training set N=$(size(train_x_cpu, 2)) is smaller than batchsize=$(training_para.batchsize).")
    sample_batch_x = isnothing(sample_batch_x) ?
        device(train_x_cpu[:, 1:training_para.batchsize]) : sample_batch_x
    @info "Compiling v10 pair helpers" N=size(train_x_cpu, 2) batchsize=training_para.batchsize
    return maybe_compile_inference(
        model, ps, st, device(train_x_cpu), training_para;
        device,
        sample_batch_x,
        compile_latent_index=true,
        compile_train_encoder=true,
    )
end

# ╔═╡ 3442ef19-bf2f-4ebf-94fd-bce4dd378745
begin
	enzyme_training_backend() = AutoEnzyme(mode=EnzymeCore.set_runtime_activity(EnzymeCore.Reverse))
	
	function training_backend(training_para::VQVAE_Training_Para, device)
	    backend = training_para.autodiff_backend
	    if backend === :auto
	        backend = device isa ReactantDevice ? :enzyme : :zygote
	    end
	    backend === :zygote && return AutoZygote()
	    backend === :enzyme && return enzyme_training_backend()
	    error("Unsupported autodiff_backend=$(training_para.autodiff_backend). Use :auto, :zygote, or :enzyme.")
	end
end

# ╔═╡ 7f8c942b-1957-4721-8c21-37f9278e342d
function compile_train_step(model, ps, st, train_x_cpu, para, training_para::VQVAE_Training_Para;
    device=identity, cdev=default_cdev())
    @info "Compiling v10 training step (forward+backward+optimizer)"
    start = time()
    opt = Optimisers.AdamW(; eta=Float64(training_para.initial_learning_rate),
        lambda=Float64(training_para.weight_decay))
    loss_fn = VQVAELoss(para)
    ad_backend = training_backend(training_para, device)
    dummy_batch_cpu = train_x_cpu[:, 1:training_para.batchsize]
    dummy_target_cpu = Float32.(MLUtils.normalise(dummy_batch_cpu; dims=1))
    ps_cpu = cdev(ps); st_cpu = Lux.testmode(cdev(st))
    lat, _ = encoder_latents(model, dummy_batch_cpu, ps_cpu, st_cpu)
    payload_cpu, rvq_cpu = prepare_split_payload(lat.z_e1, lat.z_e2, st_cpu.rvq, model.K;
        ema_decay=para.ema_decay, epsilon=para.epsilon,
        dead_threshold=para.dead_threshold, training=true)
    st_dev = merge(st, (; rvq=device(rvq_cpu)))
    dummy_bdev = (;
        x=device(Float32.(dummy_batch_cpu)),
        target=device(dummy_target_cpu),
        vq_payload=device(payload_cpu),
    )
    ts = Training.TrainState(model, ps, Lux.trainmode(st_dev), opt)
    (_, _, _, ts_warmed) = Training.single_train_step!(
        ad_backend, loss_fn, dummy_bdev, ts; return_gradients=Val(false))
    @info "Training step compile complete" compile_time_s=round(time() - start; digits=3)
    return ts_warmed.cache
end

# ╔═╡ 10000015-0000-0000-0000-000000000001
md"## Data Loading and Pair Loop"

# ╔═╡ 566e6a4c-1153-4c6c-bf2b-385478f684c4
function taper(x)
    w = cat(tukey(size(x, 1), 0.1), dims=ndims(x))
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
    D1 = normalise(raw.correlations, dims=1)
    D1ac, D1c = split_causal_acausal(D1, true)
    D1fac = Float32.(normalise(taper(D1ac)[2:end, :], dims=1))
    D1fc  = Float32.(normalise(taper(D1c)[2:end,  :], dims=1))
    return (; pair, D1=Float32.(D1), D1fac, D1fc, headers=raw.headers, distance=raw.distance,
            latitudes=raw.latitudes, longitudes=raw.longitudes)
end

function trim_training_bundle(bundle; n_max::Union{Nothing,Integer}=nothing,
    rng=Random.default_rng())
    # n_max parameter DEPRECATED: XLA compilation now uses batchsize (not n_train)
    # All pairs train on their full waveform count, no trimming needed
    if !isnothing(n_max)
        @warn "n_max parameter deprecated: XLA compiles at batchsize now. Training on all available waveforms."
    end
    return bundle
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
function pair_run_dir(save_root::String, pair, timestamp=now())
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
        @info "Loading v10 pair data" pair
        bundle = build_training_bundle(pair; filepath, dt, period_min, period_max)
        bundle = trim_training_bundle(bundle; n_max, rng)
        @info "Loaded v10 pair bundle" pair distance=bundle.distance D1fac_size=size(bundle.D1fac) D1fc_size=size(bundle.D1fc)
        data = make_pooled_split(bundle.D1fac, bundle.D1fc; rng)
        @info "Built v10 train/test split" pair train_size=size(data.D_train) test_size=size(data.D_test)
        push!(pairs_data, (; pair, data, data_bundle=bundle))
    end
    return pairs_data
end

# ╔═╡ a4957762-faf0-40d2-a71e-1d65fe873065
function run_dir_for_seed(save_root::String, pair, seed::Integer, timestamp=now())
    pair_str = join(pair, "_")
    run_tag = Dates.format(timestamp, "yyyymmdd_HHMMSS")
    return joinpath(save_root, pair_str, "seed$(seed)_$(run_tag)")
end

# ╔═╡ 10000017-0000-0000-0000-000000000001
md"## Analysis and Plotting"

# ╔═╡ 2f151b20-b956-404d-8fee-1e9cddfd6b62
function get_cluster_percentages(model, ps, st, x; stage::Int=1, return_labels::Bool=false,
    device=identity, cdev=default_cdev())
    x_cpu = Float32.(cdev(flatten_batch(x)))
    res, _ = encode(model, cdev(ps), Lux.testmode(cdev(st)), x_cpu; training=false)
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
# Joint 2-stage cluster averages: groups by (code_1, code_2) pairs → K1*K2 columns
# Column ordering: combo index = (k2-1)*K1 + k1  (k1 varies fastest)
# Labels returned as "(k1,k2)" strings in the same column order
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
function encoded_cache(model, ps, st, data; device=identity, cdev=default_cdev())
    ps_cpu = cdev(ps)
    st_eval = Lux.testmode(cdev(st))
    res_ac, _ = encode(model, ps_cpu, st_eval, Float32.(cdev(data.D_ac_all)); training=false)
    res_c, _ = encode(model, ps_cpu, st_eval, Float32.(cdev(data.D_c_all)); training=false)
    return (;
        stage_ac=Array(cdev(res_ac.stage_indices)),
        stage_c=Array(cdev(res_c.stage_indices)),
        coarse_ac=Array(cdev(res_ac.coarse_indices)),
        coarse_c=Array(cdev(res_c.coarse_indices)),
    )
end

# ╔═╡ 4f2b1382-c158-417b-9fc0-1b8d04d90ed2
function codebook_cross_analysis(model, ps, st, D_ac, D_c; device=identity, cdev=default_cdev())
    cache = encoded_cache(model, ps, st, (; D_ac_all=D_ac, D_c_all=D_c); device, cdev)
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
    pct_ac = get_cluster_percentages(model, ps, st, D_ac; device, cdev)
    pct_c = get_cluster_percentages(model, ps, st, D_c; device, cdev)
    return (; pct_ac, pct_c, confusion, agreement, labels=string.(1:K), cache)
end

# ╔═╡ c974342d-4bcc-4175-9d71-8f9cfbb7105a
function source_state_averages(model, ps, st, data; device=identity, cdev=default_cdev())
    cache = encoded_cache(model, ps, st, data; device, cdev)
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
# Given joint_averages of shape (nt, K1*K2) with column ordering (k2-1)*K1+k1,
# returns marginal waveforms for each stage via 2-way ANOVA decomposition:
#   grand_mean     : (nt,)
#   stage1_effects : (nt, K1)   — α(k1) = row-mean minus grand mean
#   stage2_effects : (nt, K2)   — β(k2) = col-mean minus grand mean
#   stage1_waves   : (nt, K1)   — μ + α(k1)  (ready to run MFT on)
#   stage2_waves   : (nt, K2)   — μ + β(k2)
#   stage1_labels  : K1 strings
#   stage2_labels  : K2 strings
function marginal_decomposition(averages::AbstractMatrix{Float32}, counts::AbstractVector{Int};
        K1::Int, K2::Int)
    nt = size(averages, 1)
    W = reshape(Float32.(counts), K1, K2)          # weight matrix
    A = reshape(averages, nt, K1, K2)              # (nt, K1, K2)

    # weighted row/col means (weight by count so empty cells don't distort)
    w_row = sum(W; dims=2)                         # (K1,)
    w_col = sum(W; dims=1)                         # (K2,)
    w_tot = sum(W)

    # grand mean weighted by count
    grand_mean = dropdims(
        sum(A .* reshape(W, 1, K1, K2); dims=(2,3)); dims=(2,3)) ./ max(w_tot, 1f0)

    # stage-1 marginal: mean over k2 for each k1
    stage1_marginal = dropdims(
        sum(A .* reshape(W, 1, K1, K2); dims=3); dims=3) ./
        reshape(max.(vec(w_row), 1f0), 1, K1)      # (nt, K1)

    # stage-2 marginal: mean over k1 for each k2
    stage2_marginal = dropdims(
        sum(A .* reshape(W, 1, K1, K2); dims=2); dims=2) ./
        reshape(max.(vec(w_col), 1f0), 1, K2)      # (nt, K2)

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
function plot_training_dashboard(loss_history; title="VQ-VAE v10 Training")
    epochs = collect(1:length(loss_history.train_target_mse))
    traces = [
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_objective, mode="lines", name="train_objective"),
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_target_mse, mode="lines", name="train_target_mse"),
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_codebook_exclusivity, mode="lines", name="train_codebook_exclusivity"),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_recon_mse, mode="lines", name="test_recon_mse"),
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_perplexity, mode="lines", name="Train perplexity", yaxis="y2"),
    ]
    layout = Layout(title=title, xaxis_title="Epoch",
        yaxis=attr(title="Loss", type="log"),
        yaxis2=attr(title="Perplexity", overlaying="y", side="right"),
        width=900, height=500, plot_bgcolor="white", paper_bgcolor="white")
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 2168a07c-e17a-4e94-bec7-5f881a5b5f09
function plot_codebook_heatmap(st; stage::Int=1, kmax::Int=20,
    title="RVQ stage $(stage) codebook")
    E = get_codebook(st, stage)
    ksel = 1:min(kmax, size(E, 2))
    trace = PlutoPlotly.heatmap(z=E[:, ksel], x=string.(ksel), y=string.(1:size(E, 1)), zmid=0)
    return PlutoPlotly.plot([trace], Layout(title=title,
        xaxis_title="Code", yaxis_title="Embedding dim", width=850, height=550))
end

# ╔═╡ d7d22b36-79b6-41d6-b9ce-403d34d4165b
function plot_codebook_confusion(confusion; title="Codebook Confusion", labels=nothing)
    K = size(confusion, 1)
    labels = isnothing(labels) ? string.(1:K) : labels
    trace = PlutoPlotly.heatmap(z=confusion, x=labels, y=labels, colorscale="Blues")
    return PlutoPlotly.plot([trace], Layout(title=title, xaxis_title="Causal code",
        yaxis_title="Acausal code", width=750, height=700))
end

# ╔═╡ 16d60aa8-ecc0-47f7-b454-0ca1e1d2e3d0
function plot_state_average_matrix(avg; title::String, dt::Real=1.0, reverse_time::Bool=false)
    nt, nstates = size(avg)
    t = collect(1:nt) .* dt
    traces = AbstractTrace[]
    for k in 1:nstates
        y = reverse_time ? reverse(avg[:, k]) : avg[:, k]
        push!(traces, PlutoPlotly.scatter(x=t, y=y .+ (k - 1) * 2.5,
            mode="lines", name="state $k"))
    end
    return PlutoPlotly.plot(traces, Layout(title=title, xaxis_title="Time (s)",
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
            marker=attr(color="rgba(31,119,180,0.7)")),
        PlutoPlotly.bar(x=xlabels, y=pct_c,  name="Causal",
            marker=attr(color="rgba(214,39,40,0.7)")),
    ]
    layout = Layout(
        title=attr(text=title, font=attr(size=18)),
        barmode="group", height=400, width=700,
        xaxis=attr(title="Source state"),
        yaxis=attr(title="Usage (%)"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ f140b608-0c12-4c5e-8dad-1ac81f6e2d99
function plot_reconstruction_examples(model, ps, st, X; nsamples::Int=8, dt::Real=1.0,
    device=identity, cdev=default_cdev(), title="Reconstruction examples")
    X_cpu = Float32.(cdev(flatten_batch(X)))
    ids = sort(randperm(size(X_cpu, 2))[1:min(nsamples, size(X_cpu, 2))])
    res, _ = model(X_cpu[:, ids], cdev(ps), Lux.testmode(cdev(st)); training=false)
    recon = Float32.(cdev(res.xhat))
    t = collect(1:size(X_cpu, 1)) .* dt
    traces = AbstractTrace[]
    for (j, id) in enumerate(ids)
        offset = (j - 1) * 4
        push!(traces, PlutoPlotly.scatter(x=t, y=X_cpu[:, id] .+ offset, mode="lines",
            name="raw", showlegend=j == 1, line=attr(color="black", width=1)))
        push!(traces, PlutoPlotly.scatter(x=t, y=recon[:, j] .+ offset, mode="lines",
            name="recon", showlegend=j == 1, line=attr(color="red", width=2)))
    end
    return PlutoPlotly.plot(traces, Layout(title=title, xaxis_title="Time (s)",
        yaxis_title="Trace + offset", width=900, height=650))
end

# ╔═╡ 11834c5a-4618-11f1-a096-01b9cbdd6fab
function update(model, ps, st, loss_history, train_data, test_data,
    para, training_para;
    device=identity, cdev=default_cdev(), compiled=nothing, compile_missing::Bool=false,
    train_step_cache=nothing, n_compiled::Union{Nothing,Int}=nothing,
    pbar_pair=nothing, pbar_seed=nothing)

    setup_start = time()
    train_x_cpu_full = Float32.(cdev(flatten_batch(train_data)))
    # Use all waveforms - batching with padding in rebuild_latent_index handles variable-N
    train_x_cpu = train_x_cpu_full
    test_x_cpu = Float32.(cdev(flatten_batch(test_data)))
    opt = Optimisers.AdamW(; eta=Float64(training_para.initial_learning_rate),
        lambda=Float64(training_para.weight_decay))
    train_state = Training.TrainState(model, ps, Lux.trainmode(st), opt)
    if !isnothing(train_step_cache)
        train_state = replace_train_state_cache(train_state, train_step_cache)
    end
    loss_fn = VQVAELoss(para)
    ad_backend = training_backend(training_para, device)
    idx = LatentIndex(max_Mnn(training_para))
    last_index_Mnn = 0
    ensemble_targets_cpu = nothing
    size(train_x_cpu, 2) >= training_para.batchsize ||
        error("Training set N=$(size(train_x_cpu, 2)) is smaller than batchsize=$(training_para.batchsize). Reactant training uses fixed full batches.")
    test_eval_x_cpu = test_x_cpu[:, 1:min(512, size(test_x_cpu, 2))]
    inference_compiled = compiled
    if isnothing(inference_compiled)
        if !compile_missing && device isa ReactantDevice
            error("Reactant training requires compiled helpers. Run compile_model first, or pass compile_missing=true.")
        end
        if compile_missing
            training_para.verbose && @info "Compiling missing Reactant training helpers" N=size(train_x_cpu, 2) batchsize=training_para.batchsize
            inference_compiled = compile_vqvae_helpers(
                model, ps, st, train_x_cpu, training_para;
                device,
                sample_batch_x=device(train_x_cpu[:, 1:training_para.batchsize]),
            )
        else
            inference_compiled = (; encode_z_e=nothing, encode_z_e_train=nothing)
        end
    end
    training_para.verbose && @info "Prepared v10 update loop" setup_time_s=round(time() - setup_start; digits=3) N=size(train_x_cpu, 2) batchsize=training_para.batchsize compiled_helpers=!isnothing(compiled)

    pbar_epochs = Progress(training_para.nepoch; desc="Epochs"; parent=pbar_seed, showspeed=true, output=stdout)
    for epoch in 1:training_para.nepoch
        phase = ensemble_phase(epoch, training_para)
        if phase.post_epoch > 0 &&
           (phase.Mnn != last_index_Mnn || mod(phase.post_epoch - 1, training_para.index_refresh_every) == 0)
            if isnothing(inference_compiled.encode_z_e)
                if !compile_missing && device isa ReactantDevice
                    error("Latent-index rebuild requires compiled.encode_z_e on Reactant. Recompile with compile_vqvae_helpers or pass compile_missing=true.")
                end
                if compile_missing
                    latent_compiled = maybe_compile_inference(
                        train_state.model, train_state.parameters, train_state.states,
                        device(train_x_cpu[:, 1:training_para.batchsize]),
                        training_para;
                        device,
                        sample_batch_x=device(train_x_cpu[:, 1:training_para.batchsize]),
                        compile_latent_index=true,
                        compile_train_encoder=false,
                    )
                    inference_compiled = (;
                        encode_z_e=latent_compiled.encode_z_e,
                        encode_z_e_train=inference_compiled.encode_z_e_train,
                    )
                end
            end
            index_start = time()
            training_para.verbose && @info "Rebuilding latent index before epoch batches" epoch post_warmup_epoch=phase.post_epoch Mnn=phase.Mnn N=size(train_x_cpu, 2)
            rebuild_latent_index!(idx, train_state.model, train_state.parameters, train_state.states, train_x_cpu;
                Mnn=phase.Mnn, device, cdev,
                encode_compiled=inference_compiled.encode_z_e,
                knn_search_chunk_size_fraction=training_para.knn_search_chunk_size_fraction,
                n_compiled)
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
        last_codebook_exclusivity = NaN32
        last_perplexity = NaN32
        epoch_counts = zeros(Float32, sum(para.K))   # accumulated codebook counts across all batches
        prep_time = 0.0
        target_time = 0.0
        pack_time = 0.0
        payload_time = 0.0
        state_swap_time = 0.0
        step_time = 0.0
        metric_sync_time = 0.0
        pbar = Progress(length(batches); desc="Epoch $epoch", showspeed=true, output=stdout)
        for (batch_idx, batch) in enumerate(batches)
            prep_start = time()
            bdev, st_updated, prep_stats = prepare_vq_training_batch(
                train_state.model, train_state.parameters, train_state.states,
                batch, train_x_cpu, phase.post_epoch == 0 ? nothing : idx,
                epoch, para, training_para; device, cdev,
                encode_compiled=inference_compiled.encode_z_e_train,
                ensemble_targets_cpu=ensemble_targets_cpu
            )
            prep_time += time() - prep_start
            target_time += prep_stats.target_time
            pack_time += prep_stats.pack_time
            payload_time += prep_stats.payload_time
            state_swap_start = time()
            train_state = replace_train_state_states(train_state, st_updated)
            state_swap_time += time() - state_swap_start
            step_start = time()
            (_, loss, stats, train_state) = Training.single_train_step!(
                ad_backend, loss_fn, bdev, train_state; return_gradients=Val(false)
            )
            # Reactant's compiled step overwrites ts.states with the XLA-traced st (which
            # has the old rvq frozen in). Re-inject the CPU-side EMA-updated rvq so it
            # survives into the next batch's prepare_vq_training_batch call.
            train_state = replace_train_state_states(train_state,
                merge(train_state.states, (; rvq=st_updated.rvq)))
            step_time += time() - step_start
            total_seen += size(batch.x, 2)
            # accumulate per-stage EMA cluster sizes for epoch-level perplexity
            offset = 0
            for stage in st_updated.rvq.stages
                cs = Float32.(cdev(stage.ema_cluster_size))
                epoch_counts[offset+1:offset+length(cs)] .+= cs
                offset += length(cs)
            end
            if batch_idx == length(batches)
                metric_sync_start = time()
                batch_metrics = cdev((;
                    loss,
                    recon_loss=stats.recon_loss,
                    commit_loss=stats.commit_loss,
                    entropy_loss=stats.entropy_loss,
                    codebook_exclusivity_loss=stats.codebook_exclusivity_loss,
                ))
                metric_sync_time += time() - metric_sync_start
                last_loss = Float32(batch_metrics.loss)
                isnan(last_loss) && error("NaN loss encountered.")
                last_recon = Float32(batch_metrics.recon_loss)
                last_commit = Float32(batch_metrics.commit_loss)
                last_entropy = Float32(batch_metrics.entropy_loss)
                last_codebook_exclusivity = Float32(batch_metrics.codebook_exclusivity_loss)
            end
            # Update progress bar with metrics
            elapsed = time() - start
            next!(pbar; showvalues=[
                (:loss, round(last_loss; digits=4)),
                (:recon, round(last_recon; digits=4)),
                (:commit, round(last_commit; digits=4)),
            ])
        end
        epoch_time = time() - start
        throughput = total_seen / max(epoch_time, 1e-8)
        # epoch perplexity from accumulated counts across all batches
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
        # recompute entropy from same accumulated counts so it matches perplexity
        epoch_entropy = 0f0
        offset2 = 0
        for k in para.K
            stage_counts = epoch_counts[offset2+1:offset2+k]
            p = stage_counts ./ max(sum(stage_counts), 1f-8)
            psafe = clamp.(p, 1f-10, 1f0)
            epoch_entropy += sum(psafe .* log.(psafe))  # negative value, consistent with entropy_loss convention
            offset2 += k
        end
        last_entropy = epoch_entropy / nstages
        train_m = (;
            total=last_loss,
            recon_loss=last_recon,
            commit_loss=last_commit,
            entropy_loss=last_entropy,
            codebook_exclusivity_loss=last_codebook_exclusivity,
            perplexity=last_perplexity,
        )
        test_recon_mse = recon_mse_inference(
            train_state.model, cdev(train_state.parameters), cdev(train_state.states), test_eval_x_cpu;
            normalize_target=training_para.normalize_target)
        record_train_metrics!(loss_history, train_m, test_recon_mse, epoch_time, throughput)

        if training_para.verbose && mod(epoch, training_para.nprint) == 0
            r(x) = round(x; digits=4)
            weighted_entropy = para.entropy_weight * train_m.entropy_loss
            weighted_codebook_exclusivity = para.codebook_exclusivity_weight * train_m.codebook_exclusivity_loss
            objective_str = "$(r(train_m.recon_loss)) + $(r(train_m.commit_loss)) + $(r(weighted_entropy)) [raw_entropy=$(r(train_m.entropy_loss))] + $(r(weighted_codebook_exclusivity)) [raw_codebook_excl=$(r(train_m.codebook_exclusivity_loss))] = $(r(train_m.total))"
            @info "Epoch $epoch" objective="mse+commit+w_entropy+w_codebook_exclusivity = $objective_str" test_recon_mse=r(test_recon_mse) perplexity=r(train_m.perplexity) post_warmup_epoch=phase.post_epoch Mnn=phase.Mnn throughput=round(throughput; digits=1) epoch_time_s=round(epoch_time; digits=3)
            @info "Epoch timing breakdown" epoch post_warmup_epoch=phase.post_epoch prep_time_s=round(prep_time; digits=3) target_time_s=round(target_time; digits=3) pack_time_s=round(pack_time; digits=3) payload_time_s=round(payload_time; digits=3) state_swap_time_s=round(state_swap_time; digits=3) step_time_s=round(step_time; digits=3) metric_sync_time_s=round(metric_sync_time; digits=3) other_time_s=round(epoch_time - prep_time - state_swap_time - step_time - metric_sync_time; digits=3)
        end
        if !isnothing(training_para.stop_on_recon_loss) && train_m.recon_loss < training_para.stop_on_recon_loss
            training_para.verbose && @info "Early stopping" epoch train_target_mse=train_m.recon_loss threshold=training_para.stop_on_recon_loss
            break
        end
        # Update epoch progress bar
        next!(pbar_epochs; showvalues=[
            (:loss, round(train_m.total; digits=4)),
            (:recon, round(train_m.recon_loss; digits=4)),
            (:test_mse, round(test_recon_mse; digits=4)),
        ])
    end
    return train_state.parameters, train_state.states, loss_history
end

# ╔═╡ b8e82af3-5ebb-4eec-9c7f-4813dcf3d6b6
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

# ╔═╡ b890360d-0a9a-42a4-93a3-71dcb92d1ff9
function _header_time_label(header::AbstractString)
    m = match(r"^(\d{4})\.(\d{3})\.(\d{4})\.(\d{4})", header)
    return m === nothing ? header : join(m.captures, ".")
end

# ╔═╡ ccf43f01-b99d-42f0-941d-913fbc27da83
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

# ╔═╡ e8ef982b-868f-473f-b0b8-e8435ef38899
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

# ╔═╡ e0a7c40a-7b56-45ac-9498-d6ccdb22e9eb
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

# ╔═╡ f79b9080-afd2-4b16-8757-6db0e3db97fa
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

# ╔═╡ c4e50d34-7af6-4c3e-90fb-c375928639d8
function _state_window_mean_psd(window_psd::AbstractMatrix, source_state::AbstractVector{<:Integer},
        nstates::Integer)
    nfreq = size(window_psd, 1)
    out = fill(Float32(NaN), nfreq, nstates)
    counts = zeros(Int, nstates)
    sums = zeros(Float32, nfreq, nstates)
    nw = min(size(window_psd, 2), length(source_state))
    for j in 1:nw
        k = Int(source_state[j])
        1 <= k <= nstates || continue
        sums[:, k] .+= Float32.(window_psd[:, j])
        counts[k] += 1
    end
    for k in 1:nstates
        counts[k] > 0 && (out[:, k] .= sums[:, k] ./ counts[k])
    end
    return (; psd=out, counts)
end

# ╔═╡ bfb64aec-d8cd-43ad-80c0-b8191bb6f5eb
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

# ╔═╡ 5492fa27-20fc-4526-aaab-9bee3f866c52
function whiten_amplitude_spectrum(X_cpu::AbstractMatrix{Float32}; smooth_bins::Int=15)
    nt = size(X_cpu, 1)
    X_f = fft(Float64.(X_cpu), 1)
    # mean amplitude spectrum across all waveforms
    A_mean = vec(mean(abs.(X_f); dims=2))
    # smooth with a centred boxcar
    half_w = smooth_bins ÷ 2
    kernel = ones(smooth_bins) / smooth_bins
    c = DSP.conv(A_mean, kernel)
    A_smooth = max.(c[half_w+1:half_w+nt], 1e-10)
    X_f ./= A_smooth
    return Float32.(real.(ifft(X_f, 1)))
end

# ╔═╡ 6322b22f-ce89-475f-9073-fdca44f212b2
function per_waveform_whiten(X_cpu::AbstractMatrix{Float32};
    kernel_length::Int=128, min_power_fraction::Float64=0.05)
    nt, nw = size(X_cpu)
    X64 = Float64.(X_cpu)
    # per-column power spectrum: (nt, nw)
    P = abs2.(fft(X64, 1))
    P[1, :] .= 0.0
    # floor each column relative to its own max
    col_max = maximum(P; dims=1)
    P .= max.(P, min_power_fraction .* col_max)
    # whitening filter in frequency domain: (nt, nw)
    W = P .^ (-0.25)
    W[1, :] .= 0.0
    # per-column FIR taps via ifft, then window and truncate
    w_full = real.(ifft(W, 1))                          # (nt, nw)
    w_shift = fftshift(w_full, 1)                       # center zero-lag
    center = div(nt, 2) + 1
    lo = center - div(kernel_length, 2)
    taps = w_shift[lo:lo+kernel_length-1, :]            # (kernel_length, nw)
    taps .*= DSP.hann(kernel_length)
    taps ./= sum(abs.(taps); dims=1)
    # apply each column's FIR via filtfilt (zero-phase, forward+backward)
    X_whitened = similar(X64)
    for i in 1:nw
        X_whitened[:, i] = DSP.filtfilt(taps[:, i], [1.0], X64[:, i])
    end
    return Float32.(X_whitened)
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
# Decode every codebook entry through the separate decoders on CPU.
# For v10 (split-decoder) returns:
#   joint    : (nt, K1*K2)  — decoder1(e1[:,k1]) + decoder2(e2[:,k2]), col order (k2-1)*K1+k1
#   stage1   : (nt, K1)     — decoder1(e1[:,k1]) alone
#   stage2   : (nt, K2)     — decoder2(e2[:,k2]) alone
#   joint_labels  : K1*K2 strings "(k1,k2)"
#   stage1_labels : K1 strings
#   stage2_labels : K2 strings
function decode_codebook_waveforms(model, ps, st; cdev=default_cdev())
    ps_cpu = cdev(ps)
    st_cpu = Lux.testmode(cdev(st))
    embeddings = [Array(stage.embedding) for stage in st_cpu.rvq.stages]  # [(d÷2,K1), (d÷2,K2)]

    K1, K2 = model.K[1], model.K[2]
    e1 = embeddings[1]   # (d÷2, K1)
    e2 = embeddings[2]   # (d÷2, K2)

    # stage-1 marginal: decoder1 applied to each K1 codebook vector
    Z1_all = Float32.(e1)   # (d÷2, K1)
    result1 = (; z_q1=Z1_all, z_q2=zeros(Float32, model.d÷2, K1))
    out1, _ = decode_from_latents(model, ps_cpu, st_cpu, result1)
    stage1_waves = Array(out1.x1hat)   # (nt, K1) — decoder1 contribution only

    # stage-2 marginal: decoder2 applied to each K2 codebook vector
    Z2_all = Float32.(e2)   # (d÷2, K2)
    result2 = (; z_q1=zeros(Float32, model.d÷2, K2), z_q2=Z2_all)
    out2, _ = decode_from_latents(model, ps_cpu, st_cpu, result2)
    stage2_waves = Array(out2.x2hat)   # (nt, K2) — decoder2 contribution only

    # joint: additive combination of all K1*K2 pairs
    n_joint = K1 * K2
    Z1_rep = hcat([e1[:, k1] for k2 in 1:K2 for k1 in 1:K1]...)   # (d÷2, n_joint)
    Z2_rep = hcat([e2[:, k2] for k2 in 1:K2 for k1 in 1:K1]...)   # (d÷2, n_joint)
    result_joint = (; z_q1=Float32.(Z1_rep), z_q2=Float32.(Z2_rep))
    out_joint, _ = decode_from_latents(model, ps_cpu, st_cpu, result_joint)
    joint_waves = Array(out_joint.xhat)   # (nt, K1*K2)

    return (;
        joint=joint_waves,
        stage1=stage1_waves,
        stage2=stage2_waves,
        joint_labels=["($k1,$k2)" for k2 in 1:K2 for k1 in 1:K1],
        stage1_labels=["s1=$k" for k in 1:K1],
        stage2_labels=["s2=$k" for k in 1:K2],
    )
end

# ╔═╡ a6066ca6-1350-4c54-9857-99d195873e6c
function save_vqvae_run(run_dir; model, ps, st, para, training_para, loss_history, pair, data_bundle,
        analysis_settings=(;))
    mkpath(run_dir)
    averages = source_state_averages(model, ps, st,
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
    codebook_waves = decode_codebook_waveforms(model, ps, st)
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
    @info "Saved v10 source-state analysis artifact" run_dir
    return run_dir
end

# ╔═╡ 4eac1994-2dae-428c-a70d-5d5de672e4bf
function train_selected_pairs(pairs_data, compiled_model;
    seeds, training_para::VQVAE_Training_Para, save_root::String, device=nothing,
    analysis_settings=(;))
    isempty(pairs_data) && return Any[]
    isempty(seeds) && error("Provide at least one seed.")
    xdev = isnothing(device) ? default_xdev(; force=true) : device
    cdev = default_cdev()
    results = Any[]
    pbar_pairs = Progress(length(pairs_data); desc="Pairs"; showspeed=true, output=stdout)
    for pair_entry in pairs_data
        pbar_seeds = Progress(length(seeds); desc="Seeds"; parent=pbar_pairs, showspeed=true, output=stdout)
        for (run_index, seed) in enumerate(seeds)
            pair = pair_entry.pair
            @info "Training v10 pair run" pair run_index seed
            reset_start = time()
            ps, st = reset_vqvae(compiled_model.model; seed, device=xdev)
            training_para.verbose && @info "Reset v10 model parameters" pair run_index seed reset_time_s=round(time() - reset_start; digits=3)
            loss_history = fresh_loss_history()
            train_start = time()
            ps, st, loss_history = update(
                compiled_model.model, ps, st, loss_history,
                pair_entry.data.D_train, pair_entry.data.D_test,
                compiled_model.para, training_para;
                device=xdev, cdev,
                compiled=compiled_model.compiled,
                train_step_cache=compiled_model.train_step_cache,
                n_compiled=compiled_model.n_train,
                compile_missing=false,
                pbar_pair=pbar_pairs,
                pbar_seed=pbar_seeds,
            )
            @info "Finished v10 pair run" pair run_index seed time_s=round(time() - train_start; digits=3)
            run_dir = run_dir_for_seed(save_root, pair, seed)
            save_vqvae_run(run_dir; model=compiled_model.model, ps, st,
                para=compiled_model.para, training_para, loss_history, pair,
                data_bundle=pair_entry.data_bundle, analysis_settings)
            push!(results, (; pair, run_index, seed, run_dir,
                model=compiled_model.model, ps, st, para=compiled_model.para,
                training_para, loss_history, data=pair_entry.data,
                data_bundle=pair_entry.data_bundle))
            # Update seed progress bar
            final_loss = loss_history.record[:, 1][end]  # Get final epoch loss
            next!(pbar_seeds; showvalues=[(:final_loss, round(final_loss; digits=4))])
        end
        # Update pair progress bar
        next!(pbar_pairs; showvalues=[(:pair, pair_entry.pair)])
    end
    return results
end

# ╔═╡ f5000002-0000-0000-0000-000000000001
function train_selected_pairs_lazy(selected_pairs, compiled_model;
    seeds, training_para::VQVAE_Training_Para, save_root::String,
    filepath::String, dt::Real=1.0, period_min::Real=10, period_max::Real=50,
    n_max::Union{Nothing,Integer}=nothing,
    bp_filter, per_waveform_whitening_kernel_length::Int,
    device=nothing, analysis_settings=(;))
    isempty(selected_pairs) && return Any[]
    isempty(seeds) && error("Provide at least one seed.")
    xdev = isnothing(device) ? default_xdev(; force=true) : device
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
        let D_ac = pd.data.D_ac_all, D_c = pd.data.D_c_all
            nt      = size(D_ac, 1)
            lags_s  = collect((1:nt) .* dt)
            mean_ac = vec(mean(D_ac, dims=2))
            mean_c  = vec(mean(D_c,  dims=2))
            plt_w = UnicodePlots.lineplot(
                lags_s, mean_ac;
                name="acausal", xlabel="lag (s)", ylabel="amplitude",
                title="$(pair[1])-$(pair[2])  whitened waveforms  ($(size(D_ac,2)) ac / $(size(D_c,2)) c)",
                width=80, height=10,
            )
            UnicodePlots.lineplot!(plt_w, lags_s, mean_c; name="causal")
            println(plt_w)
            psd_ac = _positive_period_psd(D_ac, dt)
            psd_c  = _positive_period_psd(D_c,  dt)
            mean_pow_ac = vec(mean(10 .* log10.(max.(psd_ac.psd, 1f-30)), dims=2))
            mean_pow_c  = vec(mean(10 .* log10.(max.(psd_c.psd,  1f-30)), dims=2))
            plt_p = UnicodePlots.lineplot(
                psd_ac.periods, mean_pow_ac;
                name="acausal", xlabel="period (s)", ylabel="PSD (dB)",
                title="$(pair[1])-$(pair[2])  input PSD",
                width=80, height=10,
                xscale=:log10,
            )
            UnicodePlots.lineplot!(plt_p, psd_c.periods, mean_pow_c; name="causal")
            println(plt_p)
        end
        for (run_index, seed) in enumerate(seeds)
            @info "Training pair" pair run_index seed
            reset_start = time()
            ps, st = reset_vqvae(compiled_model.model; seed, device=xdev)
            loss_history = fresh_loss_history()
            ps, st, loss_history = update(
                compiled_model.model, ps, st, loss_history,
                pd.data.D_train, pd.data.D_test,
                compiled_model.para, training_para;
                device=xdev, cdev,
                compiled=compiled_model.compiled,
                train_step_cache=compiled_model.train_step_cache,
                n_compiled=compiled_model.n_train,
                compile_missing=false,
            )
            @info "Finished pair run" pair run_index seed time_s=round(time() - reset_start; digits=3)
            let epochs = collect(1:length(loss_history.train_target_mse))
                if !isempty(epochs)
                    plt = UnicodePlots.lineplot(
                        epochs, loss_history.train_target_mse;
                        name="train", xlabel="epoch", ylabel="recon MSE",
                        title="$(pair[1])-$(pair[2])  seed=$(seed)  loss curve",
                        width=80, height=12,
                    )
                    UnicodePlots.lineplot!(plt, epochs, loss_history.test_recon_mse; name="test")
                    println(plt)
                end
            end
            run_dir = run_dir_for_seed(save_root, pair, seed)
            save_vqvae_run(run_dir; model=compiled_model.model, ps, st,
                para=compiled_model.para, training_para, loss_history, pair,
                data_bundle=pd.data_bundle, analysis_settings)
            push!(results, (; pair, run_index, seed, run_dir,
                model=compiled_model.model, ps, st, para=compiled_model.para,
                training_para, loss_history, data=pd.data,
                data_bundle=pd.data_bundle))
        end
        # discard raw and whitened data for this pair before loading the next
        pd = nothing
        pd_raw = nothing
        GC.gc()
    end
    return results
end

# ╔═╡ a1b2c3d5-0000-0000-0000-000000000001
function make_encoder_head(para, latent_len::Int, enc_channels::Int)
    conv = Conv((3,), enc_channels => enc_channels, activation; pad=SamePad())
    dense = Dense(latent_len * enc_channels, para.d ÷ 2)
    return @compact(; conv, dense, latent_len, enc_channels) do feat
        # feat: (L, C, B)
        y = conv(feat)                                              # (L, C, B)
        B = size(feat, 3)
        y_flat = reshape(permutedims(y, (2, 1, 3)), enc_channels * latent_len, B)
        @return dense(y_flat)                                       # (d÷2, B)
    end
end

# ╔═╡ 10000010-0000-0000-0000-000000000001
function get_vqvae(para; rng=Random.default_rng(), device=identity)
    length(para.K) == 2 || error("v10 requires exactly 2 codebook stages (length(K) must be 2). Got K=$(para.K).")
    all(>(1), para.K) || error("All K entries must be > 1.")
    iseven(para.d) || error("d must be even for v10 (required for split heads). Got d=$(para.d).")
    Random.seed!(rng, para.seed)

    encoder = make_encoder(para)
    ps_enc, st_enc = Lux.setup(rng, encoder)
    dummy = randn(rng, Float32, para.nt, 2)
    enc_dummy, _ = encoder(waveform_to_conv3(dummy), ps_enc, Lux.testmode(st_enc))
    latent_len, enc_channels, _ = size(enc_dummy)

    head1 = make_encoder_head(para, latent_len, enc_channels)
    head2 = make_encoder_head(para, latent_len, enc_channels)
    decoder1 = make_decoder(para, latent_len; latent_dim=para.d ÷ 2)
    decoder2 = make_decoder(para, latent_len; latent_dim=para.d ÷ 2)
    ps_dec1, st_dec1 = Lux.setup(rng, decoder1)
    dec_dummy, _ = decoder1(randn(rng, Float32, para.d ÷ 2, 2), ps_dec1, Lux.testmode(st_dec1))
    size(dec_dummy, 1) == para.nt ||
        error("Decoder geometry mismatch: output length $(size(dec_dummy, 1)) != nt $(para.nt). Adjust decoder strides/kernels.")
    model = VQVAE(encoder, head1, head2, decoder1, decoder2, Tuple(Int.(para.K)),
        para.d, latent_len,
        para.beta_commit, para.ema_decay,
        para.epsilon, para.dead_threshold)
    ps, st = Lux.setup(rng, model)
    ps, st = (ps, st) |> device

    loss_history = fresh_loss_history()
    @info "VQ-VAE v10 geometry" nt=para.nt d=para.d half=para.d÷2 latent_len K=para.K enc_channels
    return model, ps, st, loss_history
end

# ╔═╡ a74f1dab-d658-4f37-9d26-4ddd9097d15e
function compile_model(nt::Int, n_train::Int; vqvae_parameters::NamedTuple,
    training_para::VQVAE_Training_Para, seed::Int=1234, device=nothing)
    ensure_reactant_xla_flags!()
    xdev = isnothing(device) ? default_xdev(; force=true) : device
    cdev = default_cdev()
    rng = Xoshiro(seed)
    para = VQVAE_Para(; merge(vqvae_parameters, (; nt, seed))...)
    model, ps, st, _ = get_vqvae(para; rng, device=xdev)
    # XLA compile at batchsize (not n_train) to fix RuntimeProgramInputMismatch errors
    # Allows inference with any N via batching + padding to batchsize
    dummy_full_cpu = randn(rng, Float32, nt, training_para.batchsize)
    dummy_batch_cpu = dummy_full_cpu[:, 1:training_para.batchsize]
    compiled = compile_vqvae_helpers(model, ps, st, dummy_full_cpu, training_para;
        device=xdev, sample_batch_x=xdev(dummy_batch_cpu))
    train_step_cache = compile_train_step(model, xdev(ps), xdev(st),
        dummy_batch_cpu, para, training_para; device=xdev, cdev)
    @info "compile_model complete (using batchsize for variable-N inference)" nt batchsize=training_para.batchsize
    return (; model, para, compiled, train_step_cache, compile_seed=seed, n_train=training_para.batchsize)
end

# ╔═╡ a848319e-7bda-4844-a916-2bbdef1d5417
function train_one_pair(pair::Tuple{<:AbstractString,<:AbstractString}; filepath::String,
    vqvae_parameters::NamedTuple, training_para::VQVAE_Training_Para,
    save_root::String=joinpath(filepath, "SavedModels", "vqvae_v10"),
    seed::Int=1234, dt::Real=1.0, period_min::Real=10, period_max::Real=50,
    device=nothing, n_max::Union{Nothing,Integer}=nothing)
    pairs_data = load_pairs_data([pair]; filepath, seed, dt, period_min, period_max, n_max=n_max)
    nt = size(pairs_data[1].data.D_train, 1)
    n_train = size(pairs_data[1].data.D_train, 2)
    compiled_model = compile_model(nt, n_train; vqvae_parameters, training_para, seed, device)
    return only(train_selected_pairs(pairs_data, compiled_model; seeds=[seed],
        training_para, save_root, device))
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
InlineStrings = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
LuxCUDA = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
CUDA = "~6.1.0"
ConcreteStructs = "~0.2.4"
DSP = "~0.8.5"
Distances = "~0.10.12"
Enzyme = "~0.13.152"
EnzymeCore = "~0.8.20"
FFTW = "~1.10.0"
InlineStrings = "~1.4.5"
JLD2 = "~0.6.4"
Lux = "~1.31.4"
LuxCUDA = "~0.3.6"
MLUtils = "~0.4.8"
NNlib = "~0.9.34"
Optimisers = "~0.4.7"
PlutoPlotly = "~0.6.5"
ProgressLogging = "~0.1.6"
Reactant = "~0.2.262"
StatsBase = "~0.34.11"
Zygote = "~0.7.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "f52b1997be4c024f9c701380a92b42220258398a"

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

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

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
deps = ["LinearAlgebra"]
git-tree-sha1 = "28e1637322d4019ed2577cbec9268fab9b7da117"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.6.0"
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

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "3d0cabd25fab32390e3bcb82cd67e700aebd9816"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.25.0"

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

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

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
git-tree-sha1 = "6e0ed8fe61c12fced0fcb6912bc73196e323fff2"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "6.1.0"

[[deps.CUDACore]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "PrecompileTools", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "StaticArrays"]
git-tree-sha1 = "dd9b7208970e29c459f254a52393f10cdc4357b8"
uuid = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
version = "6.1.1"
weakdeps = ["CUDA", "ChainRulesCore", "EnzymeCore", "SpecialFunctions"]

    [deps.CUDACore.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.CUDATools]]
deps = ["CUDACore", "CUDA_Compiler_jll", "CUPTI", "Crayons", "GPUCompiler", "LLVM", "NVML", "NVTX", "PrecompileTools", "Preferences", "PrettyTables", "Printf", "Statistics", "demumble_jll"]
git-tree-sha1 = "3de36a048e6abb49f9d34405e8cf7d0b27715ae7"
uuid = "9ec180c6-1c07-47c7-9e6e-ebefa4d1f6d0"
version = "6.1.0"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "6ca8941cec33550807f849ffc566492db6ede90d"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.4.4+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "TOML"]
git-tree-sha1 = "2d6222474d868469a72de5bd47c5c25c0e1fe518"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.3.0+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "79312abe5261a660f94e746e449d2cb2fe3284d9"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "2.1.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "c0314d9fb0ebd00e404feba4c3fbc04c9975abc1"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.21.0+1"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "70dea6a7133d2100a143b515a00d6d887e208500"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "9.20.0+0"

[[deps.CUPTI]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox"]
git-tree-sha1 = "d57a233b32dc0f7f8eeb73dd986a01348dc4bef5"
uuid = "9e67e8f6-ba02-4b6c-a7db-3b11ae1e7ab7"
version = "6.1.0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3c190c570fb3108c09f838607386d10c71701789"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.73.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "1a3ad7e16a321667698a19e77362b35a1e94c544"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.1"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcreteStructs]]
git-tree-sha1 = "ed1da4eac5ba9b3f6d061c90f3ca6ba190dd6595"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.4"

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

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "d335b2929e1b6067951a1250df247cc5fab7d40e"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.5"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6fb53a69613a0b2b68a0d12671717d307ab8b24e"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.5"

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

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "79a2aca180a85c690c58a020d47b426954b590f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.16.0"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "42cd00edaac86f941815fe557c1d01e11913e07c"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.28"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

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

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "21778bf207a725f43048ec60918465d4b2d6ecab"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.152"
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
git-tree-sha1 = "08f3e8d675fb14e361d6e7174bfe229af150d580"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.263+0"

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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

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

[[deps.FastClosures]]
git-tree-sha1 = "acebe244d53ee1b461970f8910c235b259e772ef"
uuid = "9aa1b823-49e4-5ca5-8b0f-3971ec8bab6a"
version = "0.3.2"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "8e9c059d6857607253e837730dbf780b6b151acd"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.19.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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
git-tree-sha1 = "cf230532c7871ce9bfa5812dd699cb68da888364"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.5.4"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "201bc54b04903f12dd014816c8f1ac4624db7221"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.14.0"

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

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

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

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "941f87a0ae1b14d1ac2fa57245425b23a9d7a516"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.4"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "f76f7560267b840e492180f9899b472f30b88450"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.0"

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
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "PrecompileTools", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "0eeef733eebb4bb1464dd9d1d8c8f3ad20a090d1"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.8.2"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "70c96f133c78c3cdc06234157144fab3744c6b38"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.43+1"

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

[[deps.LuxCUDA]]
deps = ["CUDA", "Reexport", "cuDNN"]
git-tree-sha1 = "8f4b360da604b166d34220ee94ab83244da11e6d"
uuid = "d0bbae9a-e099-4d5b-a835-1c6931763bda"
version = "0.3.6"

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

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

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
git-tree-sha1 = "269695e0ac0bfd854feba8179433c57e10dc95a7"
uuid = "611af6d1-644e-4c5d-bd58-854d7d1254b9"
version = "6.1.0"

[[deps.NVTX]]
deps = ["JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "a9083c3e469e63cca454d1fc3b19472d9d92c14a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.3"
weakdeps = ["Colors"]

    [deps.NVTX.extensions]
    NVTXColorsExt = "Colors"

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

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

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
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "2d99b4c8a7845ab1342921733fa29366dae28b24"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.1"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"
    PolynomialsRecipesBaseExt = "RecipesBase"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

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

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.ProtoBuf]]
deps = ["BufferedStreams", "EnumX", "TOML"]
git-tree-sha1 = "da18083a52d9d57bbe6dadaacad39731e5f7be39"
uuid = "3349acd9-ac6a-5e09-bcdb-63829b23a429"
version = "1.3.0"

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

[[deps.Reactant]]
deps = ["Adapt", "BFloat16s", "CEnum", "Crayons", "Downloads", "EnumX", "Enzyme", "EnzymeCore", "FileWatching", "Functors", "GPUArraysCore", "GPUCompiler", "HTTP", "JSON", "LLVM", "LLVMOpenMP_jll", "Libdl", "LinearAlgebra", "OrderedCollections", "PrecompileTools", "Preferences", "PrettyTables", "ProtoBuf", "Random", "ReactantCore", "Reactant_jll", "ScopedValues", "Scratch", "Serialization", "Setfield", "Sockets", "StableRNGs", "StructUtils", "StyledStrings", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "34cde6a3f9e36aa438401b0f1d719151437b39b9"
uuid = "3c362404-f566-11ee-1572-e11a4b42c853"
version = "0.2.262"

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
git-tree-sha1 = "ca0ea2ee6d2942abf0dc5d7c6c9a27245c7f97d8"
uuid = "0192cb87-2b54-54ad-80e0-3be72ad8a3c0"
version = "0.0.383+0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

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

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "6547cbdd8ce32efba0d21c5a40fa96d1a3548f9f"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

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

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "c6f18e5a52a176a383f6f6c635e0f81feed1d6d4"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.11"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

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

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

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

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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
git-tree-sha1 = "d60c958a009b5a25cf8ea4eb3f9a2cf7352a9c6f"
uuid = "d49dbf32-c5c2-4618-8acc-27bb2598ef2d"
version = "1.3.4"

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

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a29cbf3968d36022198bcc6f23fdfd70f7caf737"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.10"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.cuBLAS]]
deps = ["Adapt", "BFloat16s", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "LLVM", "LinearAlgebra"]
git-tree-sha1 = "9d6409c7fbdef7105385105b172bb193897ec7a3"
uuid = "182d3088-87b7-4494-8cad-fc6afaa545bc"
version = "6.1.0"
weakdeps = ["EnzymeCore"]

    [deps.cuBLAS.extensions]
    EnzymeCoreExt = "EnzymeCore"

[[deps.cuDNN]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "b0fc7ed6d9597e796b0ef9d4c47c06617351162e"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "6.1.0"

[[deps.cuFFT]]
deps = ["AbstractFFTs", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "772beeca0de8e7e58ec013b14fbda981b74108f5"
uuid = "533571aa-0936-420e-b4be-9c66f5f626ca"
version = "6.1.0"

[[deps.cuRAND]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "Random", "Random123", "RandomNumbers"]
git-tree-sha1 = "d0772ca43c4500c0cce1ff02a9dfda8faf65c960"
uuid = "20fd9a0b-12d5-4c2f-a8af-7c34e9e60431"
version = "6.1.0"

[[deps.cuSOLVER]]
deps = ["CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUToolbox", "LinearAlgebra", "SparseArrays", "cuBLAS", "cuSPARSE"]
git-tree-sha1 = "dae250870a645b30a534a4176985993e615eb944"
uuid = "887afef0-6a32-4de5-add4-7827692ba8fc"
version = "6.1.0"

[[deps.cuSPARSE]]
deps = ["Adapt", "CEnum", "CUDACore", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "GPUArrays", "GPUToolbox", "KernelAbstractions", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5da59442cc13bb1a8647bbe20eba66917dc3a64b"
uuid = "b26da814-b3bc-49ef-b0ee-c816305aa060"
version = "6.1.0"

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

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "da8c1f6eee04831f14edcfa5dae611d309807e57"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.3.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═10000001-0000-0000-0000-000000000001
# ╠═330652f1-0754-48a4-9a0f-4fb9d6824222
# ╟─10000002-0000-0000-0000-000000000001
# ╠═10000003-0000-0000-0000-000000000001
# ╠═10000004-0000-0000-0000-000000000001
# ╠═10000005-0000-0000-0000-000000000001
# ╠═10000006-0000-0000-0000-000000000001
# ╠═10000007-0000-0000-0000-000000000001
# ╠═10000008-0000-0000-0000-000000000001
# ╟─1000000a-0000-0000-0000-000000000001
# ╠═ef72f750-3026-4c6b-8cf6-9fc8964658f8
# ╠═6ce85f19-73ba-4785-ab52-1767cf79efa8
# ╠═a0416f71-242a-44a7-b8f6-186726318611
# ╠═1000000c-0000-0000-0000-000000000001
# ╠═e75270cd-5842-4132-9028-559135b9401f
# ╠═f31b1233-fdc3-4cef-8836-1122a1b3e7d4
# ╠═b1a7fdb4-7263-496e-8f9f-4d8a4429f351
# ╠═fe5ee63d-6944-4557-834e-8e21231485f0
# ╠═f8da318e-6ff2-4442-a737-28ff9d8f340a
# ╠═e635aa3f-19de-4ce9-a478-c1252ddb7979
# ╠═5e71db90-a0d1-47a4-bb32-dba21163e029
# ╠═0652338c-5ab5-463d-8a4f-1f2314a01ba8
# ╠═66ddf04d-ee60-4a0b-9444-d09af23685ea
# ╠═10000010-0000-0000-0000-000000000002
# ╠═10000010-0000-0000-0000-000000000003
# ╠═1000000e-0000-0000-0000-000000000001
# ╠═1000000f-0000-0000-0000-000000000001
# ╠═10000010-0000-0000-0000-000000000001
# ╠═10000011-0000-0000-0000-000000000001
# ╠═13634a6c-abda-4084-9b5b-f6761fd728ad
# ╠═8950cf6d-f5d2-4bcc-90ab-12ecf79f7c35
# ╠═91a4fb9f-f5d4-406f-b384-282d8a48257f
# ╠═86dfe031-7e05-402b-8916-cc2d2758e6b4
# ╠═092511ab-104e-4577-8cf5-dc6deeb73ac7
# ╠═b1e8b57a-0fa3-492a-a3a1-036423f41373
# ╠═2f88be66-ffea-4d0e-8ea5-65a39b7d10db
# ╠═2ee07196-8b28-418c-a0d4-40866584bc6f
# ╠═5c9d71d1-c6a6-4968-814d-66506a78b516
# ╠═10000013-0000-0000-0000-000000000001
# ╠═5e716b73-ab88-4b84-a8bf-dd064dc82fd8
# ╠═2d6639d1-ae40-46d0-a811-e1fd34a23613
# ╠═cf13347d-e1fa-4ec1-86dc-38299825f65b
# ╠═2ed0a636-4fc5-4bbe-9886-1aa351510c67
# ╠═81ffa3b4-8a3c-4571-a494-6108378fff4a
# ╠═6322b22f-ce89-475f-9073-fdca44f212b1
# ╠═990c86a6-6267-455f-b62f-9295de5ecb75
# ╠═5492fa27-20fc-4526-aaab-9bee3f866c51
# ╠═89a363dd-b142-4dce-8783-307136602c8e
# ╠═7f8c942b-1957-4721-8c21-37f9278e342d
# ╠═3442ef19-bf2f-4ebf-94fd-bce4dd378745
# ╠═10000015-0000-0000-0000-000000000001
# ╠═566e6a4c-1153-4c6c-bf2b-385478f684c4
# ╠═a1e5a8cb-0bd1-44b8-8cd4-c95a667d830d
# ╠═0b79d043-0805-43b3-80d7-f64d2018525f
# ╠═4b8ffb0f-23b0-443d-b4c7-12a3ed4ac76d
# ╠═8dd1c50c-587c-471d-bc80-cd77012302a9
# ╠═7e26f064-6a32-41ce-b416-90a04adfbcc9
# ╠═a6066ca6-1350-4c54-9857-99d195873e6c
# ╠═f3583928-80f5-4e89-8d86-463eda8b97bd
# ╠═a74f1dab-d658-4f37-9d26-4ddd9097d15e
# ╠═a4957762-faf0-40d2-a71e-1d65fe873065
# ╠═4eac1994-2dae-428c-a70d-5d5de672e4bf
# ╠═f5000001-0000-0000-0000-000000000001
# ╠═f5000002-0000-0000-0000-000000000001
# ╠═a848319e-7bda-4844-a916-2bbdef1d5417
# ╠═10000017-0000-0000-0000-000000000001
# ╠═2f151b20-b956-404d-8fee-1e9cddfd6b62
# ╠═b478bec8-14f6-4d78-89e5-2e76414c4d46
# ╠═b387cc9c-c64f-4ca6-9e44-6df5551f6d7a
# ╠═987c9325-5908-4047-aaf3-39e5b251309d
# ╠═db53da0e-96ce-4a75-bcfc-32fdc4ffe064
# ╠═4f2b1382-c158-417b-9fc0-1b8d04d90ed2
# ╠═c974342d-4bcc-4175-9d71-8f9cfbb7105a
# ╠═a1b2c3d4-0000-0000-0000-000000000001
# ╠═70a460bf-b3e4-4e7c-aa4d-2674a450379a
# ╠═2168a07c-e17a-4e94-bec7-5f881a5b5f09
# ╠═d7d22b36-79b6-41d6-b9ce-403d34d4165b
# ╠═16d60aa8-ecc0-47f7-b454-0ca1e1d2e3d0
# ╠═a6751ceb-46af-4ec7-840b-d42ea46c93a5
# ╠═f140b608-0c12-4c5e-8dad-1ac81f6e2d99
# ╠═11834c5a-4618-11f1-a096-01b9cbdd6fab
# ╠═b8e82af3-5ebb-4eec-9c7f-4813dcf3d6b6
# ╠═b890360d-0a9a-42a4-93a3-71dcb92d1ff9
# ╠═ccf43f01-b99d-42f0-941d-913fbc27da83
# ╠═e8ef982b-868f-473f-b0b8-e8435ef38899
# ╠═e0a7c40a-7b56-45ac-9498-d6ccdb22e9eb
# ╠═f79b9080-afd2-4b16-8757-6db0e3db97fa
# ╠═c4e50d34-7af6-4c3e-90fb-c375928639d8
# ╠═bfb64aec-d8cd-43ad-80c0-b8191bb6f5eb
# ╠═5492fa27-20fc-4526-aaab-9bee3f866c52
# ╠═6322b22f-ce89-475f-9073-fdca44f212b2
# ╠═a1b2c3d4-0000-0000-0000-000000000002
# ╠═a1b2c3d5-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
