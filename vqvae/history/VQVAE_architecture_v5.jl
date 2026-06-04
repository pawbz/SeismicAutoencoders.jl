### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ cc11647d-1c56-4ceb-9677-703aca03c9f4
using Functors

# ╔═╡ d73472ff-9e09-45b0-8811-b7dd8d820358
using CUDA,
    cuDNN,
    Enzyme,
    Flux,
    Zygote,
    Distances,
    JLD2,
    Random,
    MLUtils,
    DSP,
    ProgressLogging,
    Statistics,
    LinearAlgebra,
    PlutoUI,
    PlutoHooks,
    FFTW,
    StatsBase,
    Optimisers

# ╔═╡ 76dbf599-a9b3-459f-992b-16ab2f7b74f1
using PlutoLinks

# ╔═╡ 4a95997e-5c12-4658-9b8e-a5065328e1c1
using BenchmarkTools

# ╔═╡ a0000018-0000-0000-0000-000000000001
using ParameterSchedulers

# ╔═╡ a0000021-0000-0000-0000-000000000001
using PlutoPlotly

# ╔═╡ 461f0505-2230-4b84-b6c6-1a9730808437
md"""# VQ-VAE for Seismic Waveform Clustering

## Architecture Overview

**Vector Quantized Variational Autoencoder (VQ-VAE)** for extracting shared coherent features
between causal and acausal cross-correlation branches.

### Key Ideas
- **Shared encoder** processes both causal and acausal branches identically
- **Vector quantization** with finite codebook forces discrete clustering
- **Optional latent-time windowing** lets you focus on a small region around an arrival
- **EMA codebook updates** (no gradient through codebook — more stable)
- **Configurable T** (number of quantized vectors per waveform) controls information bottleneck
- **Unpaired training**: pool all causal + acausal waveforms, train standard VQ-VAE
- **Post-hoc analysis**: examine which codebook entries are shared vs branch-specific
- **Codebook reset** for dead entries prevents codebook collapse

### Architecture
```
waveform (nt,) → ConvEncoder → dense flatten or latent-time window → z_e (d, T) → VQ → z_q (d, T) → ConvDecoder → x̂ (nt,)
                                                  ↓
                                           codebook indices (T,)
```
"""

# ╔═╡ 97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
TableOfContents(include_definitions=true)

# ╔═╡ 26fb86d5-c844-469a-aef5-ed3c2a9ba949
xpu = gpu

# ╔═╡ 6affb3b3-9dc4-4bbc-a582-495fc1783a7a
const activation = x -> leakyrelu(x, 0.1f0)

# ╔═╡ a0000001-0000-0000-0000-000000000001
md"## Utilities"

# ╔═╡ 80f77b52-84e0-4664-8aa0-3d79fded40de
"""
Instead of cat(x, dims=3)
"""
add_dim3_reshape(::Nothing) = nothing

# ╔═╡ 6ba143e2-50df-441a-8f38-3ea8d9edd4d8
begin
    function add_dim3_reshape(x)
        nd = ndims(x)
        if nd == 2
            return reshape(x, size(x, 1), 1, size(x, 2))
        elseif nd == 3
            return x
        else
            return x
        end
    end

    function flatten_batch(x)
        return reshape(x, size(x, 1), :)
    end
end

# ╔═╡ a0000002-0000-0000-0000-000000000001
md"## Parameters"

# ╔═╡ 91a25156-e121-4d53-a5a1-422f1230d235
Base.@kwdef struct VQVAE_Para
    nt::Int                                # waveform length (time samples)
    d::Int = 64                            # codebook embedding dimension
    T::Int = 1                             # quantized vectors per waveform (1 = single-vector VQ). When per_position=true, T is auto-computed as the latent-time window width and this value is ignored.
    beta_commit::Float32 = 0.25f0          # commitment loss weight
    enc_kernels::Vector{Int} = [32, 16, 8, 4]
    enc_filters::Vector{Int} = [8, 16, 32, 64]
    enc_strides::Vector{Int} = [2, 2, 2, 2]
    dec_kernels::Vector{Int} = [4, 8, 16]
    dec_filters::Vector{Int} = [64, 48, 16, 1]
    use_bn::Bool = true
    ema_decay::Float32 = 0.99f0            # EMA decay for codebook updates
    epsilon::Float32 = 1f-5                # EMA Laplace smoothing
    dead_threshold::Int = 2                # reset codebook entry after this many batches unused
    entropy_weight::Float32 = 0.01f0       # entropy bonus to encourage uniform codebook usage
    reconstruction_loss::Symbol = :l1      # :l1 for mean absolute error, :l2 for mean squared error
    interstation_distance::Union{Nothing,Float64} = nothing  # km; enables latent-time window mode when set
    dt::Float64 = 1.0                                        # sampling interval (s)
    velocity_range::Tuple{Float64,Float64} = (2.0, 4.0)      # km/s; (vmin, vmax) group velocity range for window
    per_position::Bool = false             # true: quantize each latent-time position independently via 1x1 conv; T is auto-computed
    per_slot_codebook::Bool = false        # true: use one EMA codebook per latent slot instead of a single shared codebook
    per_position_use_lstm::Bool = false    # true: run an LSTM across latent-time positions before 1x1 pre-VQ projection
    per_position_lstm_hidden::Union{Nothing,Int} = nothing  # LSTM hidden size; nothing => use encoder channel count
    seed::Int = 1234                        # random seed for reproducibility
    arrival_mse_weight::Float32 = 1.0f0    # Deprecated in v4: arrival focus is handled by latent-time window selection, not weighted MSE

    # Multiscale RVQ: coherent codebook (with per-pair self-attention pooling) + nuisance residual codebooks
    use_multiscale_rvq::Bool = true
    Kcoherent::Int = 8                      # coherent/coarse codebook size
    Knuisance::Int = 16                     # nuisance/detail codebook size
    detail_stages::Int = 1                  # number of nuisance residual stages (>=1)
    ema_decay_small::Float32 = 0.999f0      # slower EMA for coarse anchors
    ema_decay_large::Float32 = 0.99f0       # faster EMA for nuisance residuals

    # Self-attention pooling for coarse quantization
    attn_temperature::Union{Float32, Vector{Float32}} = 0.1f0  # scalar or per-pair
    attn_init_scale::Float32 = 1.0f0  # scale for U initialization
    attn_blend::Float32 = 0.25f0      # 0 => raw waveform latent, 1 => fully attention-denoised latent
    attn_self_bias::Float32 = 2.0f0   # diagonal logit boost to preserve waveform-specific information
    num_pairs::Int = 1
    pair_codebook_mode::Symbol = :pair_specific_coarse_shared_detail
    pair_names::Union{Nothing,Vector{String}} = nothing
end

# ╔═╡ a0000003-0000-0000-0000-000000000001
md"## Conv Encoder"

# ╔═╡ 89599b3f-8c20-46c5-8f5c-ccbb71b26b36
begin
    struct ReshapeLayer
        dims::Tuple
    end
    Flux.@layer ReshapeLayer trainable = ()
    (m::ReshapeLayer)(x) = reshape(x, m.dims..., size(x)[end])

    struct Conv1DChain
        chain::Chain
    end
    Flux.@layer Conv1DChain trainable = (chain,)

    function (m::Conv1DChain)(::Nothing)
        return nothing
    end

    function (m::Conv1DChain)(x)
        x_flat = flatten_batch(x)
        x3 = add_dim3_reshape(x_flat)
        features = m.chain(x3)
        if ndims(features) == 3 && size(features, 2) == 1
            return reshape(features, size(features, 1), size(features, 3))
        elseif ndims(features) == 3
            return features
        end
        return reshape(features, size(features, 1), size(x_flat, 2))
    end

    struct SeqConv1DChain
        chain::Chain
    end
    Flux.@layer SeqConv1DChain trainable = (chain,)

    function (m::SeqConv1DChain)(::Nothing)
        return nothing
    end

    function (m::SeqConv1DChain)(x)
        x3 = ndims(x) == 3 ? x : add_dim3_reshape(flatten_batch(x))
        y = m.chain(x3)
        if ndims(y) == 3 && size(y, 2) == 1
            return reshape(y, size(y, 1), size(y, 3))
        end
        return y
    end

    struct PerPositionPreVQ{L,P}
        lstm::L
        proj::P
    end
    Flux.@layer PerPositionPreVQ trainable = (lstm, proj)

    function (m::PerPositionPreVQ)(::Nothing)
        return nothing
    end

    function (m::PerPositionPreVQ)(x)
        x3 = ndims(x) == 3 ? x : add_dim3_reshape(flatten_batch(x))
        z = if isnothing(m.lstm)
            x3
        else
            # Flux LSTM expects (channels, time, batch); latent path uses (time, channels, batch)
            Flux.reset!(m.lstm)
            y = m.lstm(permutedims(x3, (2, 1, 3)))
            permutedims(y, (2, 1, 3))
        end
        return m.proj(z)
    end
end

# ╔═╡ 2f7550d1-e854-4c2f-8efb-ad0bb70d5013
"""
Build a 1D convolutional encoder.

Returns `(encoder, flat_output_length)` or `(encoder, flat_output_length, outsize)`.
"""
function get_vq_conv_encoder(nt; kernels=[32, 16, 8, 4], filters=[8, 16, 32, 64],
    strides=[2, 2, 2, 2], use_bn::Bool=true,
    flatten_output::Bool=true, return_outsize::Bool=false)
    @assert length(kernels) == length(filters)
    layers = Any[]
    nin = 1
    for (i, k) in enumerate(kernels)
        nout = filters[i]
        s = i <= length(strides) ? strides[i] : 1
        push!(layers, Conv((k,), nin => nout, activation; pad=SamePad(), stride=s))
        if use_bn && i < length(kernels)
            push!(layers, BatchNorm(nout))
        end
        nin = nout
    end
    trunk = Chain(layers...)
    outsize = Flux.outputsize(trunk, (nt, 1); padbatch=true)
    flat_len = prod(outsize)
    if flatten_output
        push!(layers, Flux.flatten)
        enc = Conv1DChain(Chain(layers...))
    else
        enc = SeqConv1DChain(trunk)
    end
    return return_outsize ? (enc, flat_len, outsize) : (enc, flat_len)
end

# ╔═╡ 44e9c4cc-d02b-4e68-ad49-24f173556cbd
function infer_dec_upstrides(enc_strides::AbstractVector{<:Integer}, n_dec_layers::Int)
    vals = reverse(Int[s for s in enc_strides if s > 1])
    isempty(vals) && (vals = [1])
    while length(vals) > n_dec_layers
        vals[2] *= vals[1]
        deleteat!(vals, 1)
    end
    while length(vals) < n_dec_layers
        push!(vals, 1)
    end
    return vals
end

# ╔═╡ a0000005-0000-0000-0000-000000000001
md"## Conv Decoder"

# ╔═╡ 64430447-c267-4eec-8d38-63ccf91d82c4
"""
Build a 1D convolutional decoder (transposed convolutions).

Input dimension: `d_in` (latent dim fed to the decoder).
Output: reconstructed waveform of length `nt`.
"""
function get_vq_conv_decoder(nt, d_in; kernels=[4, 8, 16], filters=[64, 48, 16, 1],
    upstrides=[2, 2, 1], use_bn=false)
    @assert length(kernels) == length(filters) - 1
    @assert length(upstrides) == length(kernels)

    bottleneck_len = nt
    for s in upstrides
        bottleneck_len = cld(bottleneck_len, s)
    end
    bottleneck_channels = filters[1]

    layers = Any[]
    # Project from latent to spatial
    push!(layers, Dense(d_in, bottleneck_len * bottleneck_channels, activation))
    push!(layers, ReshapeLayer((bottleneck_len, bottleneck_channels)))

    nin = bottleneck_channels
    for (i, k) in enumerate(kernels)
        nout = filters[i+1]
        s = upstrides[i]
        if i < length(kernels)
            push!(layers, ConvTranspose((k,), nin => nout, activation; stride=s, pad=SamePad()))
            if use_bn
                push!(layers, BatchNorm(nout))
            end
        else
            # Final layer: no activation, output 1 channel
            push!(layers, ConvTranspose((k,), nin => nout; stride=s, pad=SamePad()))
        end
        nin = nout
    end

    return Conv1DChain(Chain(layers...)), bottleneck_len, bottleneck_channels
end

# ╔═╡ 8eb0be68-99e1-4df9-97bf-b29b99d8f759
function auto_dec_upstrides_for_nt(nt::Int, latent_len::Int, d::Int;
    dec_kernels::AbstractVector{<:Integer},
    dec_filters::AbstractVector{<:Integer},
    use_bn::Bool,
    enc_strides::AbstractVector{<:Integer})
    n_dec = length(dec_kernels)
    base = infer_dec_upstrides(enc_strides, n_dec)

    function outlen(strides::Vector{Int})
        dec, _, _ = get_vq_conv_decoder(nt, d;
            kernels=collect(dec_kernels), filters=collect(dec_filters),
            upstrides=strides, use_bn=use_bn)
        Flux.outputsize(dec.chain, (d,); padbatch=true)[1]
    end

    if outlen(base) == nt
        return base
    end

    kmax = collect(Int, dec_kernels)
    lo = max.(1, base .- 3)
    hi = min.(kmax, base .+ 3)
    best = copy(base)
    best_err = abs(outlen(base) - nt)
    ranges = [lo[i]:hi[i] for i in 1:n_dec]
    for cand in Iterators.product(ranges...)
        s = collect(Int, cand)
        any(s .> kmax) && continue
        olen = outlen(s)
        err = abs(olen - nt)
        if err < best_err
            best = s
            best_err = err
        end
        err == 0 && return s
    end
    return best
end

# ╔═╡ a0000006-0000-0000-0000-000000000001
md"""## Vector Quantizer (EMA)

Core VQ layer with:
- **Exponential Moving Average** codebook updates (no codebook gradient needed)
- **Dead entry reset**: re-initializes unused codebook entries from encoder outputs
- **Straight-through estimator**: gradients pass directly from z_q to z_e
"""

# ╔═╡ ee76dd74-feaa-4422-a9a8-5931eb817c24
"""Quantize without training updates (inference)."""
    function quantize_inference(vq, z_e::AbstractMatrix{Float32})
        return vq(z_e; training=false)
    end

# ╔═╡ 5fd449bf-d0d5-4944-8fc6-c0d47a0ed83f
# ── VQ helper functions (separate for JIT specialization) ─────────────────────

"""Compute squared-distance matrix between encoder outputs and codebook entries."""
function vq_distances(embedding::AbstractMatrix{Float32}, z_e::AbstractMatrix{Float32})
    z_sq = sum(abs2, z_e; dims=1)          # (1, N)
    e_sq = sum(abs2, embedding; dims=1)    # (1, K)
    return e_sq' .+ z_sq .- 2f0 .* (embedding' * z_e)  # (K, N)
end

# ╔═╡ 1f046042-0695-4a3d-a0ee-9ce4213c9b72
"""Nearest-neighbour lookup: returns indices (GPU CuVector) and one-hot encodings."""
function vq_lookup(vq, z_e::AbstractMatrix{Float32})
    dist = vq_distances(vq.embedding, z_e)
    indices_cart = dropdims(CUDA.argmin(dist; dims=1); dims=1)
    indices = getindex.(indices_cart, 1)
    encodings = Flux.onehotbatch(indices, 1:vq.K)
    z_q = vq.embedding * encodings
    return indices, encodings, z_q
end

# ╔═╡ 0a9a8028-2ba9-43d9-8b3d-8ada32ba314b
"""In-place EMA codebook update. Returns counts_cpu for reuse in dead-reset."""
function vq_ema_update!(vq, z_e::AbstractMatrix{Float32},
                        encodings)
    enc_sum_vec = vec(sum(encodings; dims=2))
    vq.ema_cluster_size .= vq.decay .* vq.ema_cluster_size .+ (1f0 - vq.decay) .* enc_sum_vec
    n = sum(vq.ema_cluster_size)
    vq.ema_cluster_size .= (vq.ema_cluster_size .+ vq.epsilon) ./ (n .+ Float32(vq.K) .* vq.epsilon) .* n
    dw = z_e * encodings'
    vq.ema_dw .= vq.decay .* vq.ema_dw .+ (1f0 - vq.decay) .* dw
    vq.embedding .= vq.ema_dw ./ reshape(vq.ema_cluster_size, 1, :)
    return cpu(enc_sum_vec)   # tiny K-vector, needed for dead tracking
end

# ╔═╡ b70d1b6c-2e5e-4548-9629-ea69011a2585
"""Reset dead codebook entries to perturbed donor encoder outputs (all GPU ops)."""
function vq_dead_reset!(vq, z_e,
                        counts_cpu)
    N = size(z_e, 2)
    dead_mask = counts_cpu .< 0.5f0
    vq.dead_count .= ifelse.(dead_mask, vq.dead_count .+ 1, 0)
    reset_mask = vq.dead_count .>= vq.dead_threshold
    n_reset = sum(reset_mask)
    n_reset == 0 && return nothing
    reset_idxs = CuArray(Int32.(findall(reset_mask)))
    donor_js   = CuArray(Int32.(rand(1:N, length(reset_idxs))))
    donor_cols = z_e[:, donor_js] .+ CUDA.randn(Float32, size(z_e, 1), length(reset_idxs)) .* 0.01f0
    vq.embedding[:, reset_idxs]      .= donor_cols
    vq.ema_dw[:, reset_idxs]         .= donor_cols
    vq.ema_cluster_size[reset_idxs]  .= 1f0
    vq.dead_count[cpu(reset_idxs)]   .= 0
    return nothing
end

# ╔═╡ 802aa1db-04b1-4444-ab9a-d9c4aee17dd9
"""Compute perplexity and entropy from per-entry probabilities."""
function vq_metrics(z_e, z_q,
                    avg_probs)
    vq_loss_val  = Flux.mse(z_e, z_q)
    avg_safe     = clamp.(avg_probs, 1f-10, 1f0)
    perplexity   = exp(-sum(avg_safe .* log.(avg_safe)))
    entropy_loss = sum(avg_safe .* log.(avg_safe))
    return vq_loss_val, perplexity, entropy_loss
end

# ╔═╡ 4cfb91c3-97e4-4006-9ec9-5eedd7de2849
begin
    mutable struct VectorQuantizerEMA{M<:AbstractMatrix{Float32}, V<:AbstractVector{Float32}}
        K::Int                                # number of codebook entries
        d::Int                                # embedding dimension
        embedding::M   # codebook: (d, K)
        ema_cluster_size::V  # (K,) EMA of assignment counts
        ema_dw::M      # (d, K) EMA of sum of assigned encoder outputs
        decay::Float32
        epsilon::Float32
        dead_count::Vector{Int}               # CPU counter: batches since last assignment
        dead_threshold::Int
    end
    Flux.@layer VectorQuantizerEMA trainable = ()  # no trainable params — EMA only

    function VectorQuantizerEMA(K::Int, d::Int;
        decay::Float32=0.99f0,
        epsilon::Float32=1f-5,
        dead_threshold::Int=2)
        embedding = xpu(randn(Float32, d, K) .* (1f0 / K))
        ema_cluster_size = xpu(ones(Float32, K))
        ema_dw = copy(embedding)
        dead_count = zeros(Int, K)
        return VectorQuantizerEMA(K, d, embedding, ema_cluster_size, ema_dw,
            decay, epsilon, dead_count, dead_threshold)
    end

    """
    Quantize encoder output `z_e` of shape (d, N) where N = T * batch_size.

    Returns `(; z_q, indices, encodings, vq_loss, commit_loss, perplexity, entropy_loss)`.
    - During training: updates codebook via EMA, resets dead entries.
    - `z_q` has straight-through gradient (straight-through estimator).
    """
    function (vq::VectorQuantizerEMA)(z_e::AbstractMatrix{Float32}; beta_commit::Float32=0.25f0, training::Bool=true)
        d, N = size(z_e)
        @assert d == vq.d "Expected embedding dim $(vq.d), got $d"

        z_q_detached, indices, encodings, vq_loss_val, perplexity, entropy_loss = Zygote.@ignore begin
            indices, encodings, z_q = vq_lookup(vq, z_e)

            if training
                counts_cpu = vq_ema_update!(vq, z_e, encodings)
                vq_dead_reset!(vq, z_e, counts_cpu)
                avg_probs = counts_cpu ./ N
            else
                avg_probs = cpu(vec(sum(encodings; dims=2))) ./ N
            end

            vq_loss_val, perplexity, entropy_loss = vq_metrics(z_e, z_q, avg_probs)
            (z_q, indices, encodings, vq_loss_val, perplexity, entropy_loss)
        end

        st_residual = Zygote.@ignore(z_q_detached .- z_e)
        z_q_st      = z_e .+ st_residual
        commit_loss = beta_commit * Flux.mse(z_e, Zygote.@ignore(z_q_detached))

        return (; z_q=z_q_st, indices, encodings, vq_loss=vq_loss_val,
            commit_loss, perplexity, entropy_loss)
    end

    struct PerSlotVectorQuantizer{Q}
        quantizers::Vector{Q}
        K::Int
        d::Int
    end
    Flux.@layer PerSlotVectorQuantizer trainable = ()

    function PerSlotVectorQuantizer(T::Int, K::Int, d::Int;
        decay::Float32=0.99f0,
        epsilon::Float32=1f-5,
        dead_threshold::Int=2)
        quantizers = [
            VectorQuantizerEMA(K, d; decay=decay, epsilon=epsilon, dead_threshold=dead_threshold)
            for _ in 1:T
        ]
        return PerSlotVectorQuantizer(quantizers, K, d)
    end

    function Base.getproperty(vq::PerSlotVectorQuantizer, s::Symbol)
        if s === :embedding
            return getfield(vq, :quantizers)[1].embedding
        end
        return getfield(vq, s)
    end

    function (vq::PerSlotVectorQuantizer)(z_e::AbstractMatrix{Float32}; beta_commit::Float32=0.25f0, training::Bool=true)
        d, N = size(z_e)
        @assert d == vq.d "Expected embedding dim $(vq.d), got $d"
        T = length(vq.quantizers)
        @assert mod(N, T) == 0 "Expected number of vectors $N to be divisible by number of slot codebooks $T"
        B = div(N, T)

        z_e_slots = reshape(z_e, d, T, B)
        results = [vq.quantizers[t](view(z_e_slots, :, t, :); beta_commit=beta_commit, training=training) for t in 1:T]
        z_q = reshape(cat([reshape(rt.z_q, d, 1, B) for rt in results]...; dims=2), d, T * B)
        # Preserve slot-major ordering so reshape(indices, T, B) maps row t -> slot t.
        idx_BT = hcat([Int.(cpu(rt.indices)) for rt in results]...)   # (B, T)
        indices = vec(permutedims(idx_BT))                            # (T*B,)
        encodings = hcat([rt.encodings for rt in results]...)
        vq_loss = mean([rt.vq_loss for rt in results])
        commit_loss = mean([rt.commit_loss for rt in results])
        perplexity = mean([rt.perplexity for rt in results])
        entropy_loss = mean([rt.entropy_loss for rt in results])

        return (; z_q,
            indices,
            encodings,
            vq_loss,
            commit_loss,
            perplexity,
            entropy_loss)
    end

    struct MultiscaleRVQQuantizer{Q}
        coarse::Q
        detail::Vector{Q}
        K::Int
        d::Int
    end
    Flux.@layer MultiscaleRVQQuantizer trainable = ()

    function MultiscaleRVQQuantizer(Kcoherent::Int, Knuisance::Int, d::Int;
        detail_stages::Int=1,
        decay_small::Float32=0.999f0,
        decay_large::Float32=0.99f0,
        epsilon::Float32=1f-5,
        dead_threshold::Int=2)
        detail_stages >= 1 || error("detail_stages must be >= 1, got $detail_stages")
        coarse = VectorQuantizerEMA(Kcoherent, d; decay=decay_small, epsilon=epsilon, dead_threshold=dead_threshold)
        detail = [VectorQuantizerEMA(Knuisance, d; decay=decay_large, epsilon=epsilon, dead_threshold=dead_threshold)
                  for _ in 1:detail_stages]
        return MultiscaleRVQQuantizer(coarse, detail, Kcoherent, d)
    end

    function Base.getproperty(vq::MultiscaleRVQQuantizer, s::Symbol)
        if s === :embedding
            return getfield(vq, :coarse).embedding
        end
        return getfield(vq, s)
    end

    struct PairSpecificMultiscaleRVQQuantizer{Q, U<:AbstractArray{Float32, 3}}
        coarse::Vector{Q}
        detail::Vector{Q}
        K::Int
        d::Int
        num_pairs::Int
        attn_U::U  # per-pair attention kernel (d × d × num_pairs); parametric so gpu() works
        attn_temperature::Union{Float32, Vector{Float32}}
        attn_blend::Float32
        attn_self_bias::Float32
    end
    Flux.@layer PairSpecificMultiscaleRVQQuantizer trainable = (attn_U,)

    function PairSpecificMultiscaleRVQQuantizer(num_pairs::Int, Kcoherent::Int, Knuisance::Int, d::Int;
        detail_stages::Int=1,
        decay_small::Float32=0.999f0,
        decay_large::Float32=0.99f0,
        epsilon::Float32=1f-5,
        dead_threshold::Int=2,
        attn_temperature::Union{Float32, Vector{Float32}}=0.1f0,
        attn_init_scale::Float32=1.0f0,
        attn_blend::Float32=0.25f0,
        attn_self_bias::Float32=2.0f0)
        num_pairs >= 1 || error("num_pairs must be >= 1, got $num_pairs")
        detail_stages >= 1 || error("detail_stages must be >= 1, got $detail_stages")
        attn_blend = clamp(attn_blend, 0f0, 1f0)
    coarse = [
        VectorQuantizerEMA(Kcoherent, d; decay=decay_small, epsilon=epsilon, dead_threshold=dead_threshold)
        for _ in 1:num_pairs
    ]
    detail = [VectorQuantizerEMA(Knuisance, d; decay=decay_large, epsilon=epsilon, dead_threshold=dead_threshold)
              for _ in 1:detail_stages]
    attn_U = xpu(cat([attn_init_scale * Matrix{Float32}(I, d, d) for _ in 1:num_pairs]...; dims=3))
    return PairSpecificMultiscaleRVQQuantizer(
        coarse, detail, Kcoherent, d, num_pairs,
        attn_U, attn_temperature, attn_blend, attn_self_bias)
    end

    function Base.getproperty(vq::PairSpecificMultiscaleRVQQuantizer, s::Symbol)
        if s === :embedding
            return getfield(vq, :coarse)[1].embedding
        end
        return getfield(vq, s)
    end

    pair_ids_to_cpu(pair_ids::AbstractVector{<:Integer}) = Int.(collect(pair_ids))
    pair_ids_to_cpu(pair_ids::CuArray{<:Integer}) = Int.(Array(pair_ids))

    function validate_pair_ids(pair_ids, num_pairs::Int, n_expected::Int)
        ids = pair_ids_to_cpu(pair_ids)
        length(ids) == n_expected || error("Expected $n_expected pair ids, got $(length(ids))")
        all(1 .<= ids .<= num_pairs) || error("pair_ids must lie in 1:$num_pairs")
        return ids
    end


    function (vq::MultiscaleRVQQuantizer)(z_e::AbstractMatrix{Float32}; beta_commit::Float32=0.25f0, training::Bool=true)
        d, N = size(z_e)
        @assert d == vq.d "Expected embedding dim $(vq.d), got $d"

        z_coarse_in = z_e

        rt_small = vq.coarse(z_coarse_in; beta_commit=beta_commit, training=training)
        z_q_total = rt_small.z_q
        residual = z_e .- Zygote.@ignore(rt_small.z_q)

        detail_indices = training ? nothing : Array{Int}(undef, length(vq.detail), N)
        vq_loss_detail = 0f0
        commit_detail = 0f0
        entropy_detail = 0f0
        perplexity_detail = 0f0

        for (s, q) in enumerate(vq.detail)
            rt = q(residual; beta_commit=beta_commit, training=training)
            z_q_total = z_q_total .+ rt.z_q
            residual = residual .- Zygote.@ignore(rt.z_q)
            if !training
                detail_indices[s, :] = Int.(cpu(rt.indices))
            end
            vq_loss_detail += rt.vq_loss
            commit_detail += rt.commit_loss
            entropy_detail += rt.entropy_loss
            perplexity_detail += rt.perplexity
        end

        n_detail = Float32(length(vq.detail))
        return (; z_q=z_q_total,
            indices=rt_small.indices,
            encodings=rt_small.encodings,
            vq_loss=rt_small.vq_loss + vq_loss_detail / n_detail,
            commit_loss=rt_small.commit_loss + commit_detail / n_detail,
            perplexity=rt_small.perplexity,
            entropy_loss=rt_small.entropy_loss + entropy_detail / n_detail,
            coarse_indices=Int.(cpu(rt_small.indices)),
            detail_indices=detail_indices,
            perplexity_coarse=rt_small.perplexity,
            perplexity_detail=perplexity_detail / n_detail)
    end

    function (vq::PairSpecificMultiscaleRVQQuantizer)(z_e::AbstractMatrix{Float32}, pair_ids;
        beta_commit::Float32=0.25f0, training::Bool=true)
        d, N = size(z_e)
        @assert d == vq.d "Expected embedding dim $(vq.d), got $d"
        ids = Zygote.@ignore validate_pair_ids(pair_ids, vq.num_pairs, N)
        # Per-pair self-attention pooling
        _buf = Zygote.Buffer(z_e)
        for pid in 1:vq.num_pairs
            sel = Zygote.@ignore findall(==(pid), ids)
            isempty(sel) && continue
            z_e_group = z_e[:, sel]
            z_norm = z_e_group ./ (sqrt.(sum(abs2, z_e_group; dims=1)) .+ 1f-8)
            U = vq.attn_U[:, :, pid]
            temp = isa(vq.attn_temperature, Vector) ? vq.attn_temperature[pid] : vq.attn_temperature
            logits = (U' * z_norm)' * z_norm / temp  # (n, n) — angular similarity
            if vq.attn_self_bias != 0f0
                n = size(logits, 1)
                logits = logits .+ xpu(Matrix{Float32}(I, n, n)) .* vq.attn_self_bias
            end
            W = Flux.softmax(logits; dims=1)  # (n, n)
            _buf[:, sel] = z_e_group * W  # weighted avg over unnormalized z_e
        end
        z_attn = copy(_buf)
        z_coarse_in = (1f0 - vq.attn_blend) .* z_e .+ vq.attn_blend .* z_attn

        z_q_coarse = zero(z_e)
        coarse_indices_acc = training ? nothing : zeros(Float32, N)
        vq_loss_small = 0f0
        commit_small = 0f0
        entropy_small = 0f0
        perplexity_small = 0f0
        num_active_pairs = 0
        for pid in 1:vq.num_pairs
            sel = findall(==(pid), ids)
            isempty(sel) && continue
            rt_small = vq.coarse[pid](z_coarse_in[:, sel]; beta_commit=beta_commit, training=training)
            sel_mask = xpu(Float32.(reshape(collect(1:N), N, 1) .== reshape(sel, 1, :)))
            z_q_coarse = z_q_coarse .+ rt_small.z_q * sel_mask'
            if !training
                coarse_row = reshape(Float32.(cpu(rt_small.indices)), 1, :)
                coarse_indices_acc = coarse_indices_acc .+ vec(coarse_row * cpu(sel_mask'))
            end
            vq_loss_small += rt_small.vq_loss
            commit_small += rt_small.commit_loss
            entropy_small += rt_small.entropy_loss
            perplexity_small += rt_small.perplexity
            num_active_pairs += 1
        end

        n_active = Float32(max(num_active_pairs, 1))
        residual = z_e .- Zygote.@ignore(z_q_coarse)
        z_q_total = z_q_coarse
        detail_indices = training ? nothing : Array{Int}(undef, 0, N)
        vq_loss_detail = 0f0
        commit_detail = 0f0
        entropy_detail = 0f0
        perplexity_detail = 0f0

        for (s, q) in enumerate(vq.detail)
            rt = q(residual; beta_commit=beta_commit, training=training)
            z_q_total = z_q_total .+ rt.z_q
            residual = residual .- Zygote.@ignore(rt.z_q)
            if !training
                detail_row = reshape(Int.(cpu(rt.indices)), 1, :)
                detail_indices = vcat(detail_indices, detail_row)
            end
            vq_loss_detail += rt.vq_loss
            commit_detail += rt.commit_loss
            entropy_detail += rt.entropy_loss
            perplexity_detail += rt.perplexity
        end

        n_detail = Float32(length(vq.detail))
        coarse_indices = training ? nothing : Int.(round.(coarse_indices_acc))
        indices_out = training ? ids : coarse_indices
        encodings = training ? nothing : Flux.onehotbatch(coarse_indices, 1:vq.K)
        return (; z_q=z_q_total,
            indices=indices_out,
            encodings,
            vq_loss=vq_loss_small / n_active + vq_loss_detail / n_detail,
            commit_loss=commit_small / n_active + commit_detail / n_detail,
            perplexity=perplexity_small / n_active,
            entropy_loss=entropy_small / n_active + entropy_detail / n_detail,
            coarse_indices=(training ? nothing : coarse_indices),
            detail_indices=detail_indices,
            perplexity_coarse=perplexity_small / n_active,
            perplexity_detail=perplexity_detail / n_detail,
            pair_ids=ids)
    end

    
end

# ╔═╡ a0000008-0000-0000-0000-000000000001
md"## VQ-VAE Model"

# ╔═╡ a0000009-0000-0000-0000-000000000001
begin
    struct VQVAE{E,P,VQ,D}
        encoder::E         # SeqConv1DChain: waveform → feature map (L, C, B)
        pre_vq::P          # Dense or (optional LSTM + 1x1 Conv): feature map/flat → latent vectors
        quantizer::VQ      # VectorQuantizerEMA, shared codebook across slots
        decoder::D         # Conv1DChain: (d*T or d) → waveform
        T::Int             # number of quantized vectors per waveform
        d::Int             # codebook embedding dimension
        latent_time_start::Union{Nothing,Int}  # if set, first latent index selected from the sequence encoder
        latent_time_end::Int  # last latent index selected from the sequence encoder (inclusive)
        per_position::Bool     # if true, slot index corresponds to latent-time position
    end
    Flux.@layer VQVAE trainable = (encoder, pre_vq, quantizer, decoder)

    codebook_size(m) = m.quantizer.K

    function select_latent_time_window(z_map::AbstractArray{Float32,3}, latent_time_start::Int, latent_time_end::Int)
        L, C, B = size(z_map)
        if latent_time_start > latent_time_end
            error("latent_time_start=$latent_time_start must be <= latent_time_end=$latent_time_end")
        end
        idx_start = clamp(latent_time_start, 1, L)
        idx_end = clamp(latent_time_end, 1, L)
        idxs = collect(idx_start:idx_end)
        return z_map[idxs, :, :]  # (window, C, B)
    end

    function dense_slot_latents(m, feat)
        z_pre = m.pre_vq(feat)
        if ndims(z_pre) != 2
            error("Dense path expected a 2D pre-VQ tensor, got ndims=$(ndims(z_pre))")
        end
        N_total = size(z_pre, 2)
        return reshape(z_pre, m.d, m.T, N_total), z_pre
    end

    function latent_time_window_slot_latents(m, feat)
        z_map = feat
        if ndims(z_map) != 3
            error("Latent-time path expected a 3D encoder tensor, got ndims=$(ndims(z_map))")
        end
        z_window = select_latent_time_window(z_map, m.latent_time_start, m.latent_time_end)
        z_window_flat = reshape(z_window, :, size(z_window, 3))
        z_pre = m.pre_vq(z_window_flat)
        if ndims(z_pre) != 2
            error("Latent-time window pre-VQ path expected a 2D tensor, got ndims=$(ndims(z_pre))")
        end
        N_total = size(z_pre, 2)
        return reshape(z_pre, m.d, m.T, N_total), z_pre, z_map
    end

    function per_position_slot_latents(m, feat)
        z_map = feat
        if ndims(z_map) != 3
            error("Per-position path expected a 3D encoder tensor, got ndims=$(ndims(z_map))")
        end
        # Quantize ALL latent time slots — no window restriction.
        # vmin/vmax window is used only for MSE upweighting in the loss function.
        z_proj = m.pre_vq(z_map)  # (L, d, B) using optional LSTM then Conv((1,), C=>d)
        if ndims(z_proj) != 3
            error("Per-position pre-VQ path expected a 3D tensor, got ndims=$(ndims(z_proj))")
        end
        W, d, N_total = size(z_proj)
        d == m.d || error("Per-position pre-VQ returned d=$d, expected $(m.d)")
        return permutedims(z_proj, (2, 1, 3)), z_map  # (d, W, B)
    end

    """
    Forward pass through VQ-VAE.

    Arguments:
    - `x`: input waveform (nt, batch)
    - `training`: whether to update EMA codebook

    Returns named tuple with reconstruction `xhat`, losses, codebook indices, etc.
    """
    function decode_from_latents(m, result)
        N_total = size(result.z_q, 3)
        z_q_for_dec = reshape(result.z_q, m.d * m.T, N_total)
        xhat = m.decoder(z_q_for_dec)
        return merge(result, (; xhat))
    end

    function (m::VQVAE)(x, pair_ids; beta_commit::Float32=0.25f0, training::Bool=true)
        return decode_from_latents(m, encode(m, x, pair_ids; beta_commit=beta_commit, training=training))
    end

    """
    Encode-only: get quantized codes and codebook indices for a batch of waveforms.
    No decoder pass.
    """
    function encode(m, x, pair_ids; beta_commit::Float32=0.25f0, training::Bool=false)
        x_flat = flatten_batch(x)
        feat_map = m.encoder(x_flat)
        if m.latent_time_start === nothing
            slot_latents, z_pre_flat = dense_slot_latents(m, feat_map)
            z_map = nothing
        elseif m.per_position
            slot_latents, z_map = per_position_slot_latents(m, feat_map)
            z_pre_flat = reshape(slot_latents, m.d, :)
        else
            slot_latents, z_pre_flat, z_map = latent_time_window_slot_latents(m, feat_map)
        end
        N_total = size(slot_latents, 3)
        pair_ids_cpu = Zygote.@ignore validate_pair_ids(pair_ids,
            m.quantizer isa PairSpecificMultiscaleRVQQuantizer ? m.quantizer.num_pairs : 1,
            N_total)
        z_e = reshape(slot_latents, m.d, m.T * N_total)
        rt = if m.quantizer isa PairSpecificMultiscaleRVQQuantizer
            m.quantizer(z_e, pair_ids_cpu; beta_commit, training)
        else
            m.quantizer(z_e; beta_commit, training)
        end
        codebook_indices = training ? nothing : reshape(Int.(cpu(rt.indices)), m.T, N_total)
        z_q = reshape(rt.z_q, m.d, m.T, N_total)
        coarse_codebook_indices = training ? nothing : begin
            if hasproperty(rt, :coarse_indices)
                reshape(Int.(rt.coarse_indices), m.T, N_total)
            else
                codebook_indices
            end
        end
        detail_codebook_indices = training ? nothing : (hasproperty(rt, :detail_indices) ? Int.(rt.detail_indices) : nothing)
        return (; z_map, z_pre_flat, z_e=reshape(z_e, m.d, m.T, N_total), z_q,
            pair_ids=pair_ids_cpu,
            z_q_flat=rt.z_q, codebook_indices,
            coarse_codebook_indices,
            detail_codebook_indices,
            vq_loss=rt.vq_loss,
            commit_loss=rt.commit_loss,
            perplexity=rt.perplexity,
            perplexity_coarse=(hasproperty(rt, :perplexity_coarse) ? rt.perplexity_coarse : rt.perplexity),
            perplexity_detail=(hasproperty(rt, :perplexity_detail) ? rt.perplexity_detail : rt.perplexity),
            entropy_loss=rt.entropy_loss)
    end

    """
    Get codebook prototypes: returns (d, K) matrix.
    """
    function get_codebook(m)
        if m.quantizer isa PerSlotVectorQuantizer
            return cat([cpu(q.embedding) for q in m.quantizer.quantizers]...; dims=3)
        elseif m.quantizer isa PairSpecificMultiscaleRVQQuantizer
            return cpu(m.quantizer.coarse[1].embedding)
        elseif m.quantizer isa MultiscaleRVQQuantizer
            return cpu(m.quantizer.coarse.embedding)
        end
        return cpu(m.quantizer.embedding)
    end

    function get_multiscale_codebooks(m)
        if m.quantizer isa PairSpecificMultiscaleRVQQuantizer
            return (; coarse=[cpu(q.embedding) for q in m.quantizer.coarse],
                detail=[cpu(q.embedding) for q in m.quantizer.detail])
        elseif !(m.quantizer isa MultiscaleRVQQuantizer)
            return (; coarse=get_codebook(m), detail=Any[])
        end
        return (; coarse=cpu(m.quantizer.coarse.embedding),
            detail=[cpu(q.embedding) for q in m.quantizer.detail])
    end

    function get_pair_coarse_codebook(m, pair_id::Int)
        m.quantizer isa PairSpecificMultiscaleRVQQuantizer || error("Pair-specific coarse codebooks are only available in v5 pair-specific mode.")
        1 <= pair_id <= m.quantizer.num_pairs || error("pair_id must lie in 1:$(m.quantizer.num_pairs)")
        return cpu(m.quantizer.coarse[pair_id].embedding)
    end

    function get_shared_detail_codebooks(m)
        if m.quantizer isa PairSpecificMultiscaleRVQQuantizer
            return [cpu(q.embedding) for q in m.quantizer.detail]
        elseif m.quantizer isa MultiscaleRVQQuantizer
            return [cpu(q.embedding) for q in m.quantizer.detail]
        end
        return Any[]
    end

    """
    Get cluster assignment histogram for a batch.
    Returns (K,) vector of percentages.
    """
    function combination_multipliers(K, T)
        [K^(t - 1) for t in 1:T]
    end

    function combination_index(digits::AbstractVector{Int}, multipliers::AbstractVector{Int})
        return sum((digits .- 1) .* multipliers) + 1
    end

    function combination_digits(idx::Int, K::Int, T::Int)
        multipliers = combination_multipliers(K, T)
        digits = zeros(Int, T)
        n = idx - 1
        for t in 1:T
            digits[t] = mod(div(n, multipliers[t]), K) + 1
        end
        return digits
    end

    function combination_labels(K, T)
        if T == 1
            return [string(k) for k in 1:K]
        end
        total = K^T
        labels = Vector{String}(undef, total)
        for idx in 1:total
            labels[idx] = join(combination_digits(idx, K, T), "-")
        end
        return labels
    end
    function get_cluster_percentages(m, x, pair_ids=fill(1, size(flatten_batch(x), 2)); return_labels::Bool=false)
        result = encode(m, x, pair_ids)
        K = codebook_size(m)
        if m.per_position
            counts = zeros(Float32, K)
            ci = result.codebook_indices
            for l in 1:size(ci, 1), j in 1:size(ci, 2)
                counts[ci[l, j]] += 1f0
            end
            labels = [string(k) for k in 1:K]
            percentages = counts ./ max(sum(counts), 1f-10) .* 100f0
            return return_labels ? (; percentages, labels) : percentages
        else
            counts = m.T == 1 ? zeros(Float32, K) : zeros(Float32, K^m.T)
            multipliers = m.T == 1 ? nothing : combination_multipliers(K, m.T)
            N_total = size(result.codebook_indices, 2)
            for j in 1:N_total
                if m.T == 1
                    counts[result.codebook_indices[1, j]] += 1f0
                else
                    digits = result.codebook_indices[:, j]
                    combo_idx = combination_index(digits, multipliers)
                    counts[combo_idx] += 1f0
                end
            end
            labels = m.T == 1 ? [string(k) for k in 1:K] : combination_labels(K, m.T)
            percentages = counts ./ max(sum(counts), 1f-10) .* 100f0
            return return_labels ? (; percentages, labels) : percentages
        end
    end

    """
    Get waveforms assigned to the combination `(k₁, k₂, …)` of length `T`.
    Returns `(data_in_cluster, indices_in_data)`.
    """
    function filter_cluster(m, x, pair_ids, ks::NTuple{N,Int}; time_index=nothing) where {N}
        if m.per_position
            if N != 1
                throw(ArgumentError("Per-position mode expects a single code tuple, e.g. (3,)"))
            end
        elseif N != m.T
            throw(ArgumentError("Expected tuple of length m.T=" * string(m.T)))
        end
        result = encode(m, x, pair_ids)
        x_flat = flatten_batch(x)
        ci_flat = result.codebook_indices
        ks_vec = collect(ks)
        if m.per_position
            k = ks_vec[1]
            if isnothing(time_index)
                selected = findall(j -> any(ci_flat[:, j] .== k), 1:size(ci_flat, 2))
            else
                if !(1 <= time_index <= size(ci_flat, 1))
                    throw(ArgumentError("time_index=" * string(time_index) * " out of range 1:" * string(size(ci_flat, 1))))
                end
                selected = findall(j -> ci_flat[time_index, j] == k, 1:size(ci_flat, 2))
            end
            return x_flat[:, selected], selected
        end
        selected = findall(j -> all(ci_flat[:, j] .== ks_vec), 1:size(ci_flat, 2))
        return x_flat[:, selected], selected
    end

    """
    Get cluster averages: mean waveform per cluster.

    - `time_index=nothing` (default):
      - per-position mode: pool all latent positions, return `(nt, K)`.
      - non per-position mode with `T>1`: return `(nt, K^T)` over code combinations.
    - `time_index=i`: use only latent slot `i` and return `(nt, K)`.
    """
    function get_cluster_averages(m, x, pair_ids=fill(1, size(flatten_batch(x), 2)); time_index=nothing)
        result = encode(m, x, pair_ids)
        K = codebook_size(m)
        nt = size(x, 1)
        x_flat = cpu(flatten_batch(x))
        ci_flat = result.codebook_indices
        N_total = size(result.codebook_indices, 2)

        if !isnothing(time_index)
            if !(1 <= time_index <= size(ci_flat, 1))
                throw(ArgumentError("time_index=" * string(time_index) * " out of range 1:" * string(size(ci_flat, 1))))
            end
            indices = vec(ci_flat[time_index, :])
            avgs = zeros(Float32, nt, K)
            counts = zeros(Int, K)
            for (j, k) in enumerate(indices)
                avgs[:, k] .+= x_flat[:, j]
                counts[k] += 1
            end
            for k in 1:K
                if counts[k] > 0
                    avgs[:, k] ./= counts[k]
                end
            end
            return avgs
        end

        if m.per_position
            avgs = zeros(Float32, nt, K)
            counts = zeros(Int, K)
            for j in 1:N_total, l in 1:size(ci_flat, 1)
                k = ci_flat[l, j]
                avgs[:, k] .+= x_flat[:, j]
                counts[k] += 1
            end
            for k in 1:K
                if counts[k] > 0
                    avgs[:, k] ./= counts[k]
                end
            end
            return avgs
        end

        if m.T == 1
            indices = vec(ci_flat)
            avgs = zeros(Float32, nt, K)
            counts = zeros(Int, K)
            for (j, k) in enumerate(indices)
                avgs[:, k] .+= x_flat[:, j]
                counts[k] += 1
            end
            for k in 1:K
                if counts[k] > 0
                    avgs[:, k] ./= counts[k]
                end
            end
        else
            num_combinations = K^m.T
            avgs = zeros(Float32, nt, num_combinations)
            counts = zeros(Int, num_combinations)
            multipliers = [K^(t - 1) for t in 1:m.T]
            for j in 1:N_total
                digits = ci_flat[:, j] .- 1
                combo_idx = sum(digits .* multipliers) + 1
                avgs[:, combo_idx] .+= x_flat[:, j]
                counts[combo_idx] += 1
            end
            for idx in 1:num_combinations
                if counts[idx] > 0
                    avgs[:, idx] ./= counts[idx]
                end
            end
        end
        return avgs
    end
end

# ╔═╡ a0000010-0000-0000-0000-000000000001
md"## Model Factory"

# ╔═╡ a0000011-0000-0000-0000-000000000001
# ╔═╡ a0000012-0000-0000-0000-000000000001
md"## Loss Functions"

# ╔═╡ 76930390-49e1-45da-b654-7d1e65c856b1
"""
VQ-VAE loss (unpaired, standard reconstruction):

    L = L_recon + L_commit + β_entropy · H

Train on pooled causal + acausal waveforms without pairing.
After training, examine codebook usage per branch.

Arguments:
- `model`: VQVAE model
- `x`: batch of waveforms (pooled causal + acausal)
- `para`: VQVAE_Para with loss weights
- `training`: whether to update EMA codebook
"""
function vqvae_forward(model, x, pair_ids, beta_commit::Float32, training::Bool)
    return model(x, pair_ids; beta_commit=beta_commit, training=training)
end

# ╔═╡ 8646155e-376f-44e6-89df-c3092d350046
function reconstruction_loss(xhat, x, loss_type::Symbol)
    if loss_type === :l1
        return mean(abs.(xhat .- x))
    elseif loss_type === :l2
        return Flux.mse(xhat, x)
    else
        error("Unsupported reconstruction_loss=$(loss_type). Use :l1 or :l2.")
    end
end

function vqvae_total_loss(result, x, entropy_weight::Float32, loss_type::Symbol)
    recon_loss = reconstruction_loss(result.xhat, x, loss_type)
    commit_loss  = result.commit_loss
    entropy_loss = result.entropy_loss
    total = recon_loss + commit_loss + entropy_weight * entropy_loss
    return total, recon_loss, commit_loss, entropy_loss
end

# ╔═╡ b4387329-5621-49ec-a01f-3fe0f686d399
function loss_vqvae(model, x, pair_ids, para; training::Bool=true)
    result = vqvae_forward(model, x, pair_ids, para.beta_commit, training)
    total, recon_loss, commit_loss, entropy_loss = vqvae_total_loss(result, x, para.entropy_weight, para.reconstruction_loss)
    return (; total, recon_loss, commit_loss, entropy_loss,
        perplexity=result.perplexity)
end

# ╔═╡ a0000015-0000-0000-0000-000000000001
md"## Data Iterator"

# ╔═╡ a0000017-0000-0000-0000-000000000001
md"## Training Loop"

# ╔═╡ eafe181e-19e9-409e-ad1d-ce859cf0e672
Base.@kwdef struct VQVAE_Training_Para
    pairs_per_batch::Int = 1
    per_pair_batchsize::Int = 128
    nepoch::Int = 30                           # epochs
    nprint::Int = 1                            # print frequency
    initial_learning_rate::Float64 = 0.001
    lr_decay::Float64 = 0.99                   # exponential LR decay per epoch
    stop_on_recon_loss::Union{Nothing,Float64} = nothing  # stop if recon loss below this
    sample_with_replacement::Bool = true
end

# ╔═╡ 7590d7a2-c8e9-4bb7-9438-aafdc72c84d7


# ╔═╡ 2a99c53c-d5fc-4ead-b7dc-3c6360548e57
function vq_eval_metrics(model, train_batch, test_batch, para)
    train_m = loss_vqvae(model, train_batch.x, train_batch.pair_ids, para; training=false)
    test_m  = loss_vqvae(model, test_batch.x, test_batch.pair_ids, para; training=false)
    return train_m, test_m
end

function pair_id_index_map(pair_ids)
    ids = Int.(pair_ids_to_cpu(pair_ids))
    mapping = Dict{Int,Vector{Int}}()
    for (idx, pid) in enumerate(ids)
        push!(get!(mapping, pid, Int[]), idx)
    end
    return mapping
end

function sample_pair_indices(indices::AbstractVector{Int}, nsample::Int; replace::Bool=true)
    n = length(indices)
    n > 0 || error("Cannot sample from an empty index set.")
    if replace || nsample > n
        return [indices[rand(1:n)] for _ in 1:nsample]
    end
    perm = Random.randperm(n)
    return indices[perm[1:nsample]]
end

function build_pairwise_batches(data, para, training_para; shuffle_pairs::Bool=true)
    x = data.x
    pair_ids = Int32.(pair_ids_to_cpu(data.pair_ids))
    pair_map = pair_id_index_map(pair_ids)
    pair_list = sort!(collect(keys(pair_map)))
    isempty(pair_list) && error("No pair ids found for batch construction.")

    training_para.pairs_per_batch >= 1 || error("pairs_per_batch must be >= 1, got $(training_para.pairs_per_batch)")
    training_para.per_pair_batchsize >= 1 || error("per_pair_batchsize must be >= 1, got $(training_para.per_pair_batchsize)")
    pairs_per_batch = min(training_para.pairs_per_batch, length(pair_list))
    training_para.pairs_per_batch > length(pair_list) &&
        @warn "pairs_per_batch=$(training_para.pairs_per_batch) exceeds available pairs=$(length(pair_list)); using $pairs_per_batch active pairs per batch."
    per_pair_batchsize = training_para.per_pair_batchsize

    batches = NamedTuple[]
    approx_total = sum(length(v) for v in values(pair_map))
    batch_columns = pairs_per_batch * per_pair_batchsize
    nbatches = shuffle_pairs ? max(1, cld(approx_total, batch_columns)) : 1
    for _ in 1:nbatches
        ordered_pairs = copy(pair_list)
        shuffle_pairs && Random.shuffle!(ordered_pairs)
        active_pairs = ordered_pairs[1:pairs_per_batch]
        batch_indices = Int[]
        batch_pair_ids = Int32[]
        for pid in active_pairs
            sampled = sample_pair_indices(pair_map[pid], per_pair_batchsize;
                replace=training_para.sample_with_replacement)
            append!(batch_indices, sampled)
            append!(batch_pair_ids, fill(Int32(pid), per_pair_batchsize))
        end
        push!(batches, (x=x[:, batch_indices], pair_ids=batch_pair_ids))
    end
    return batches
end

# ╔═╡ 5481cf12-dd58-4bf8-965e-0b68aaa7adf6
"""Append one epoch's metrics onto the loss_history vectors."""
function record_metrics!(loss_history, train_m, test_m)
    push!(loss_history.train_recon,      train_m.recon_loss)
    push!(loss_history.test_recon,       test_m.recon_loss)
    push!(loss_history.train_commit,     train_m.commit_loss)
    push!(loss_history.test_commit,      test_m.commit_loss)
    push!(loss_history.train_total,      train_m.total)
    push!(loss_history.test_total,       test_m.total)
    push!(loss_history.train_perplexity, train_m.perplexity)
    push!(loss_history.test_perplexity,  test_m.perplexity)
end

# ╔═╡ 1a8c1a9f-61d9-4591-b50f-cb7ebca78377
"""Run one full epoch of gradient updates. Returns nothing."""
function vq_train_epoch!(model, opt_state, train_loader, para)
    for batch in train_loader
        x_batch = batch.x
        pair_ids = batch.pair_ids
        g = Flux.gradient(model, x_batch, pair_ids) do m, xb, pidb
            loss_vqvae(m, xb, pidb, para; training=true).total
        end
        Optimisers.update!(opt_state, model, g[1])
    end
    GC.gc(false)
    CUDA.reclaim()
    return nothing
end

# ╔═╡ ce339f4a-0744-4bf1-9f3c-9aa9fb19eaf1
"""Log epoch summary if on a print interval."""
function vq_log_epoch(epoch::Int, train_m, test_m, nprint::Int)
    mod(epoch, nprint) == 0 || return nothing
    @info "Epoch $epoch" recon=train_m.recon_loss commit=train_m.commit_loss perplexity=train_m.perplexity test_recon=test_m.recon_loss
    return nothing
end

# ╔═╡ 7f3ff55b-0095-4d95-b3f0-c2371f0b7548
"""Check early stopping condition; returns true if training should stop."""
function vq_should_stop(epoch::Int, train_m, stop_threshold)
    isnothing(stop_threshold) && return false
    train_m.recon_loss < stop_threshold || return false
    @info "Early stopping: recon loss $(train_m.recon_loss) < $stop_threshold at epoch $epoch"
    return true
end

# ╔═╡ 40e94672-4fc6-4ffa-90e5-5f2cc3f7591d
"""
    update(model, loss_history, train_data, test_data, para, training_para)

Train the VQ-VAE on pooled (unpaired) waveforms.

Arguments:
- `model::VQVAE`: the VQ-VAE model
- `loss_history`: named tuple of loss vectors (mutated in-place)
- `D_train`: GPU matrix (nt, nwaveforms_train) — pooled causal + acausal
- `D_test`: GPU matrix (nt, nwaveforms_test)
- `para::VQVAE_Para`: architecture & loss parameters
- `training_para::VQVAE_Training_Para`: training hyperparameters, including
  `pairs_per_batch` and `per_pair_batchsize`
"""
function update(model, loss_history, train_data, test_data,
    para,
    training_para=VQVAE_Training_Para())

    opt_state = Optimisers.setup(Optimisers.Adam(eta=Float64(training_para.initial_learning_rate)), model)

    train_loader = build_pairwise_batches(train_data, para, training_para; shuffle_pairs=true)
    test_loader = build_pairwise_batches(test_data, para, training_para; shuffle_pairs=false)
    monitor_train = first(train_loader)
    monitor_test = first(test_loader)

    @progress name = "VQ-VAE training" for epoch = 1:training_para.nepoch
        train_loader = build_pairwise_batches(train_data, para, training_para; shuffle_pairs=true)
        monitor_train = first(train_loader)
        train_m, test_m = vq_eval_metrics(model, monitor_train, monitor_test, para)
        record_metrics!(loss_history, train_m, test_m)
        vq_train_epoch!(model, opt_state, train_loader, para)
        vq_log_epoch(epoch, train_m, test_m, training_para.nprint)
        vq_should_stop(epoch, train_m, training_para.stop_on_recon_loss) && break
    end
    return nothing
end

# ╔═╡ 8f8f3c1a-7c9a-4f3d-8e1f-2a1c1d2e9f5c
"""
    train_coarse_to_detail_classifier(coarse_codes, detail_codes; train_ratio=0.8, nepochs=100)

Train a linear classifier to predict detail code trajectories from coarse codes.

This is a simple mutual information proxy: if coarse and detail are disentangled,
knowing coarse should not predict detail well (accuracy ≈ 50% for binary or baseline).
If accuracy >> baseline, they are entangled.

Arguments:
- `coarse_codes::Vector`: coarse code indices (one per sample), e.g., from coarse_ac[1, :]
- `detail_codes::Matrix`: detail trajectories (T_detail × n_samples)
- `train_ratio`: fraction for training (default 0.8)
- `nepochs`: number of training epochs (default 100)

Returns:
- `train_acc::Float32`: training accuracy on detail prediction
- `test_acc::Float32`: test accuracy on detail prediction
- `null_baseline::Float32`: accuracy of always guessing most common detail combo
- `model`: trained Flux Dense layer
"""
function train_coarse_to_detail_classifier(coarse_codes::Vector, detail_codes::Matrix; 
    train_ratio=0.8, nepochs=100)
    
    n_samples = length(coarse_codes)
    @assert size(detail_codes, 2) == n_samples "Dimension mismatch: coarse_codes ($n_samples) vs detail_codes ($(size(detail_codes, 2)))"
    
    Kcoherent = maximum(coarse_codes)     # Coherent codebook size
    T_detail = size(detail_codes, 1)      # Nuisance trajectory length
    Knuisance = maximum(detail_codes)     # Nuisance codebook size
    
    # Convert detail trajectories into single class labels (combination index)
    # Each detail trajectory (e.g., [5, 3]) becomes a unique index
    if T_detail == 1
        target = vec(detail_codes)
        n_classes = Knuisance
    else
        target = zeros(Int32, n_samples)
        combo_mult = cumprod(vcat(1, fill(Knuisance, T_detail-1)))
        for j in 1:n_samples
            target[j] = sum((detail_codes[:, j] .- 1) .* combo_mult) + 1  # 1-indexed
        end
        n_classes = Knuisance^T_detail
    end
    
    # Null baseline: accuracy if always guessing most common detail combo
    counts = zeros(Int, n_classes)
    for j in 1:n_samples
        counts[target[j]] += 1
    end
    null_baseline = Float32(maximum(counts) / n_samples)
    
    # One-hot encode coarse codes as features
    X = Flux.onehotbatch(coarse_codes, 1:Kcoherent)  # (Kcoherent × n_samples)
    y = Flux.onehotbatch(target, 1:n_classes)     # (n_classes × n_samples)
    
    # Random train/test split
    perm = randperm(n_samples)
    n_train = ceil(Int, train_ratio * n_samples)
    train_idx = perm[1:n_train]
    test_idx  = perm[n_train+1:end]
    
    X_train = X[:, train_idx] |> gpu
    y_train = y[:, train_idx] |> gpu
    X_test  = X[:, test_idx]  |> gpu
    y_test  = y[:, test_idx]  |> gpu
    
    # Train linear classifier (single Dense layer)
    model = Chain(
        Dense(Kcoherent, n_classes, identity)
    ) |> gpu
    
    opt = Adam(0.01f0)
    opt_state = Optimisers.setup(opt, model)
    
    for epoch in 1:nepochs
        g = Flux.gradient(model) do m
            ŷ = m(X_train)
            Flux.logitcrossentropy(ŷ, y_train)
        end
        Optimisers.update!(opt_state, model, g[1])
    end
    
    # Compute accuracies
    function compute_accuracy(X, y, m)
        ŷ = m(X)
        pred = vec(argmax(cpu(ŷ), dims=1)[1, :])
        truth = vec(argmax(cpu(y), dims=1)[1, :])
        return Float32(mean(pred .== truth))
    end
    
    train_acc = compute_accuracy(X_train, y_train, model)
    test_acc  = compute_accuracy(X_test, y_test, model)
    
    return (train_acc=train_acc, test_acc=test_acc, null_baseline=null_baseline, model=model)
end

# ╔═╡ a0000020-0000-0000-0000-000000000001
md"## Plotting"

# ╔═╡ a0000022-0000-0000-0000-000000000001
"""
Plot VQ-VAE training dashboard: 3 vertically-stacked subplots sharing the epoch axis.
- Top: reconstruction loss (log scale)
- Middle: perplexity (linear)
- Bottom: commitment loss (log scale)
"""
function plot_training_dashboard(loss_history; title="VQ-VAE Training")
    epochs = collect(1:length(loss_history.train_recon))
    font_spec = attr(family="Computer Modern, Latin Modern Math, serif")
    grid_color = "rgba(128,128,128,0.2)"

    traces = [
        # Recon loss → subplot 1 (yaxis="y")
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_recon, mode="lines",
            name="Train recon", xaxis="x", yaxis="y",
            line=attr(color="#1f77b4", width=1.5)),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_recon, mode="lines",
            name="Test recon", xaxis="x", yaxis="y",
            line=attr(color="#1f77b4", width=1.5, dash="dash")),
        # Perplexity → subplot 2 (yaxis="y2")
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_perplexity, mode="lines",
            name="Train perplexity", xaxis="x", yaxis="y2",
            line=attr(color="#2ca02c", width=1.5)),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_perplexity, mode="lines",
            name="Test perplexity", xaxis="x", yaxis="y2",
            line=attr(color="#2ca02c", width=1.5, dash="dash")),
        # Commit loss → subplot 3 (yaxis="y3")
        PlutoPlotly.scatter(x=epochs, y=loss_history.train_commit, mode="lines",
            name="Train commit", xaxis="x", yaxis="y3",
            line=attr(color="#d62728", width=1.5)),
        PlutoPlotly.scatter(x=epochs, y=loss_history.test_commit, mode="lines",
            name="Test commit", xaxis="x", yaxis="y3",
            line=attr(color="#d62728", width=1.5, dash="dash")),
    ]

    layout = Layout(
        title=attr(text=title, font=merge(font_spec, attr(size=18))),
        height=700, width=900,
        plot_bgcolor="white", paper_bgcolor="white",
        # Shared x axis anchored at bottom panel
        xaxis=attr(title="Epoch", anchor="y3",
            showgrid=true, gridcolor=grid_color),
        # Top panel: recon loss
        yaxis=attr(title="Recon loss", domain=[0.70, 1.00], type="log",
            showgrid=true, gridcolor=grid_color,
            titlefont=attr(color="#1f77b4")),
        # Middle panel: perplexity
        yaxis2=attr(title="Perplexity", domain=[0.36, 0.64],
            showgrid=true, gridcolor=grid_color, rangemode="tozero",
            titlefont=attr(color="#2ca02c")),
        # Bottom panel: commit loss
        yaxis3=attr(title="Commit loss", domain=[0.00, 0.30], type="log",
            showgrid=true, gridcolor=grid_color,
            titlefont=attr(color="#d62728")),
        legend=attr(x=1.02, xanchor="left", y=0.5, orientation="v",
            font=merge(font_spec, attr(size=12))),
        margin=attr(r=160),
    )
    return PlutoPlotly.plot(traces, layout)
end



# ╔═╡ a0000023-0000-0000-0000-000000000001
"""
Plot cluster assignment histogram.
"""
function plot_cluster_histogram(pct_ac, pct_c; title="Cluster Usage", labels=nothing)
    xlabels = labels === nothing ? string.(1:length(pct_ac)) : labels
    traces = [
        PlutoPlotly.bar(x=xlabels, y=pct_ac, name="Acausal",
            marker=attr(color="rgba(31,119,180,0.7)")),
        PlutoPlotly.bar(x=xlabels, y=pct_c, name="Causal",
            marker=attr(color="rgba(214,39,40,0.7)")),
    ]
    layout = Layout(
        title=attr(text=title, font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
        barmode="group", height=400, width=800,
        xaxis=attr(title=labels === nothing ? "Cluster (pos, code)" : "Cluster combination",
            tickangle=labels === nothing ? 0 : -30),
        yaxis=attr(title="Percentage (%)"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ a0000024-0000-0000-0000-000000000001
md"## Codebook Analysis"

# ╔═╡ 09202321-5cf6-46f1-bd59-6069da6488c6
"""
Compute agreement rate: fraction of waveform windows where causal and acausal
map to the same codebook entry (for T=1; for T>1 uses majority vote).

High agreement = encoder extracts direction-invariant features for that window.
"""
function codebook_agreement(model, D_ac, D_c, pair_ids_ac=fill(1, size(D_ac, 2)), pair_ids_c=fill(1, size(D_c, 2)))
    res_ac = encode(model, D_ac, pair_ids_ac)
    res_c = encode(model, D_c, pair_ids_c)
    if model.T == 1
        idx_ac = vec(res_ac.codebook_indices)
        idx_c = vec(res_c.codebook_indices)
    else
        function majority(ci, K)
            ci_flat = reshape(ci, model.T, :)
            [
                begin
                    counts = zeros(Int, K)
                    for t in 1:model.T
                        counts[ci_flat[t, j]] += 1
                    end
                    argmax(counts)
                end for j in 1:size(ci_flat, 2)
            ]
        end
        K = codebook_size(model)
        idx_ac = majority(res_ac.codebook_indices, K)
        idx_c = majority(res_c.codebook_indices, K)
    end
    return mean(idx_ac .== idx_c)
end

# ╔═╡ faa3de6b-96eb-4aeb-b927-77d6771a1d0a
"""
    codebook_cross_analysis(model, D_ac, D_c)

Post-hoc analysis of codebook usage by branch across every code combination.
Returns a named tuple:
- `pct_ac`: (K^T,) percentage of acausal waveforms per combination
- `pct_c`:  (K^T,) percentage of causal waveforms per combination
- `confusion`: (K^T, K^T) matrix — confusion[i,j] = fraction of windows where
    acausal→combo i AND causal→combo j (only meaningful for paired data)
- `agreement`: scalar fraction where both branches share the same code
- `agreement`: scalar fraction where both branches share the same code
- `expected_agreement`: chance agreement from marginals
- `agreement_lift`: `agreement - expected_agreement`
- `cohens_kappa`: chance-corrected agreement
- `permutation_pvalue`: one-sided p-value against shuffled pairing
- `shared_codes`: combinations used by both branches (>5% usage each)
- `ac_only_codes`: combinations predominantly acausal (ac >5x causal)
- `c_only_codes`: combinations predominantly causal (c >5x acausal)
- `labels`: human-readable combination labels
"""
function codebook_cross_analysis(model, D_ac, D_c, pair_ids_ac=fill(1, size(D_ac, 2)), pair_ids_c=fill(1, size(D_c, 2)))
    K = codebook_size(model)
    T = model.T

    if model.per_position
        res_ac = encode(model, D_ac, pair_ids_ac)
        res_c = encode(model, D_c, pair_ids_c)
        ci_ac = res_ac.codebook_indices
        ci_c = res_c.codebook_indices
        W = size(ci_ac, 1)
        nw = min(size(ci_ac, 2), size(ci_c, 2))
        nobs = max(1, W * nw)

        confusion = zeros(Float32, K, K)
        counts_ac = zeros(Float32, K)
        counts_c = zeros(Float32, K)
        agreement_by_pos = zeros(Float32, W)
        expected_by_pos = zeros(Float32, W)
        kappa_by_pos = zeros(Float32, W)
        pval_by_pos = ones(Float32, W)

        for l in 1:W
            idx_ac = vec(ci_ac[l, 1:nw])
            idx_c = vec(ci_c[l, 1:nw])

            for i in 1:nw
                confusion[idx_ac[i], idx_c[i]] += 1f0
                counts_ac[idx_ac[i]] += 1f0
                counts_c[idx_c[i]] += 1f0
            end

            agreement_l = mean(idx_ac .== idx_c)
            p_ac_l = zeros(Float64, K)
            p_c_l = zeros(Float64, K)
            for i in 1:nw
                p_ac_l[idx_ac[i]] += 1.0
                p_c_l[idx_c[i]] += 1.0
            end
            p_ac_l ./= max(nw, 1)
            p_c_l ./= max(nw, 1)
            expected_l = sum(p_ac_l .* p_c_l)
            kappa_l = (agreement_l - expected_l) / max(1e-8, 1 - expected_l)

            nperm = 200
            idx_c_perm = copy(idx_c)
            ge_count = 0
            for _ in 1:nperm
                Random.shuffle!(idx_c_perm)
                ge_count += mean(idx_ac .== idx_c_perm) >= agreement_l
            end

            agreement_by_pos[l] = Float32(agreement_l)
            expected_by_pos[l] = Float32(expected_l)
            kappa_by_pos[l] = Float32(kappa_l)
            pval_by_pos[l] = Float32((ge_count + 1) / (nperm + 1))
        end

        confusion ./= max(sum(confusion), 1f-10)
        pct_ac = counts_ac ./ max(sum(counts_ac), 1f-10) .* 100f0
        pct_c = counts_c ./ max(sum(counts_c), 1f-10) .* 100f0

        agreement = mean(agreement_by_pos)
        expected_agreement = mean(expected_by_pos)
        agreement_lift = agreement - expected_agreement
        cohens_kappa = (agreement - expected_agreement) / max(1e-8, 1 - expected_agreement)
        permutation_pvalue = mean(pval_by_pos)

        threshold_pct = 5f0
        ratio_threshold = 5f0
        shared_codes = Int[]
        ac_only_codes = Int[]
        c_only_codes = Int[]
        for k in 1:K
            ac_used = pct_ac[k] > threshold_pct
            c_used = pct_c[k] > threshold_pct
            if ac_used && c_used
                push!(shared_codes, k)
            elseif pct_ac[k] > ratio_threshold * max(pct_c[k], 0.1f0)
                push!(ac_only_codes, k)
            elseif pct_c[k] > ratio_threshold * max(pct_ac[k], 0.1f0)
                push!(c_only_codes, k)
            end
        end

        return (; pct_ac, pct_c, confusion, agreement,
            expected_agreement, agreement_lift, cohens_kappa, permutation_pvalue,
            agreement_by_pos, expected_agreement_by_pos=expected_by_pos,
            cohens_kappa_by_pos=kappa_by_pos, permutation_pvalue_by_pos=pval_by_pos,
            shared_codes, ac_only_codes, c_only_codes,
            labels=[string(k) for k in 1:K])
    end

    # Per-branch percentages with combination labels
    pct_ac_res = get_cluster_percentages(model, D_ac, pair_ids_ac; return_labels=true)
    pct_ac = pct_ac_res.percentages
    combo_labels = pct_ac_res.labels
    pct_c = get_cluster_percentages(model, D_c, pair_ids_c)
    num_combinations = length(pct_ac)

    # Agreement and confusion (requires paired windows: same number of columns)
    res_ac = encode(model, D_ac, pair_ids_ac)
    res_c = encode(model, D_c, pair_ids_c)
    ci_ac = reshape(res_ac.codebook_indices, T, :)
    ci_c = reshape(res_c.codebook_indices, T, :)
    nw = min(size(ci_ac, 2), size(ci_c, 2))

    # Convert each waveform assignment to one combination index for robust statistics.
    idx_ac = Vector{Int}(undef, nw)
    idx_c = Vector{Int}(undef, nw)
    confusion = zeros(Float32, num_combinations, num_combinations)
    multipliers = combination_multipliers(K, T)
    for w in 1:nw
        idx_aw = combination_index(ci_ac[:, w], multipliers)
        idx_cw = combination_index(ci_c[:, w], multipliers)
        idx_ac[w] = idx_aw
        idx_c[w] = idx_cw
        confusion[idx_aw, idx_cw] += 1f0
    end
    confusion ./= max(sum(confusion), 1f-10)

    # Raw agreement can be high even for noise if both marginals are similar.
    agreement = mean(idx_ac .== idx_c)
    p_ac = zeros(Float64, num_combinations)
    p_c = zeros(Float64, num_combinations)
    for i in 1:nw
        p_ac[idx_ac[i]] += 1.0
        p_c[idx_c[i]] += 1.0
    end
    p_ac ./= max(nw, 1)
    p_c ./= max(nw, 1)
    expected_agreement = sum(p_ac .* p_c)
    agreement_lift = agreement - expected_agreement
    cohens_kappa = (agreement - expected_agreement) / max(1e-8, 1 - expected_agreement)

    # Permutation null: break pairing while preserving both branch marginals.
    nperm = 200
    idx_c_perm = copy(idx_c)
    ge_count = 0
    for _ in 1:nperm
        Random.shuffle!(idx_c_perm)
        ge_count += mean(idx_ac .== idx_c_perm) >= agreement
    end
    permutation_pvalue = (ge_count + 1) / (nperm + 1)

    # Classify codes
    threshold_pct = 5f0  # minimum percentage to count as "used"
    ratio_threshold = 5f0  # dominance ratio
    shared_codes = Int[]
    ac_only_codes = Int[]
    c_only_codes = Int[]
    for k in 1:num_combinations
        ac_used = pct_ac[k] > threshold_pct
        c_used = pct_c[k] > threshold_pct
        if ac_used && c_used
            push!(shared_codes, k)
        elseif pct_ac[k] > ratio_threshold * max(pct_c[k], 0.1f0)
            push!(ac_only_codes, k)
        elseif pct_c[k] > ratio_threshold * max(pct_ac[k], 0.1f0)
            push!(c_only_codes, k)
        end
    end

    return (; pct_ac, pct_c, confusion, agreement,
        expected_agreement, agreement_lift, cohens_kappa, permutation_pvalue,
        shared_codes, ac_only_codes, c_only_codes,
        labels=combo_labels)
end

# ╔═╡ 302a9921-c723-41d4-9d7a-234bf658a07e
"""
Plot confusion matrix of codebook assignments between acausal (rows) and causal (columns).
"""
function plot_codebook_confusion(confusion; title="Codebook Confusion: Acausal vs Causal", labels=nothing)
    KT = size(confusion, 1)
    xlabels = labels === nothing ? string.(1:KT) : labels
    ylabels = labels === nothing ? string.(1:KT) : labels
    text_vv = [
        [string(round(confusion[i, j] * 100; digits=1), "%") for j in 1:KT]
        for i in 1:KT
    ]
    trace = PlutoPlotly.heatmap(
        z=confusion,
        x=xlabels, y=ylabels,
        colorscale="Blues",
        text=text_vv,
        texttemplate="%{text}",
        hovertemplate="Acausal→%{y}, Causal→%{x}: %{text}<extra></extra>"
    )
    layout = Layout(
        title=attr(text=title, font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
        height=900, width=1000,
        xaxis=attr(title=labels === nothing ? "Causal Cluster (pos, code)" : "Causal combination",
            dtick=1, constrain="domain"),
        yaxis=attr(title=labels === nothing ? "Acausal Cluster (pos, code)" : "Acausal combination",
            dtick=1, scaleanchor="x", constrain="domain"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return PlutoPlotly.plot([trace], layout)
end

# ╔═╡ b1c2d3e4-f5a6-7890-abcd-ef1234567890
"""
        compute_latent_window(para) -> (latent_time_start, latent_time_end)

From physics parameters in `para`, compute:
- `latent_time_start`: latent-space index for arrival at `vmax` (fastest).
- `latent_time_end`: latent-space index for arrival at `vmin` (slowest).

Returns `(nothing, 1)` when `interstation_distance === nothing`.
"""
function compute_latent_window(para)
    para.interstation_distance === nothing && return (nothing, 1)
    total_stride = prod(para.enc_strides)
    vmin, vmax = para.velocity_range
    t_slow = para.interstation_distance / vmin / para.dt   # latest arrival (smallest velocity)
    t_fast = para.interstation_distance / vmax / para.dt   # earliest arrival (largest velocity)
        lstart = round(Int, t_fast / total_stride) + 1
        lend = round(Int, t_slow / total_stride) + 1
        return (lstart, lend)
end

# ╔═╡ a0000025-0000-0000-0000-000000000001
"""
Compute per-position, per-code waveform prototypes.

Returns:
- `prototypes`: (nt, W, K) mean input waveform for each latent position/code pair
- `counts`: (W, K) number of contributing waveforms
"""
function per_position_cluster_averages(model, x, pair_ids=fill(1, size(flatten_batch(x), 2)))
    model.per_position || error("per_position_cluster_averages requires model.per_position=true")
    result = encode(model, x, pair_ids)
    ci = result.codebook_indices # (W, N)
    x_flat = cpu(flatten_batch(x))
    nt, N = size(x_flat)
    W = size(ci, 1)
    K = codebook_size(model)
    prototypes = zeros(Float32, nt, W, K)
    counts = zeros(Int, W, K)
    for j in 1:N, l in 1:W
        k = ci[l, j]
        prototypes[:, l, k] .+= x_flat[:, j]
        counts[l, k] += 1
    end
    for l in 1:W, k in 1:K
        if counts[l, k] > 0
            prototypes[:, l, k] ./= counts[l, k]
        end
    end
    return (; prototypes, counts)
end

# ╔═╡ a0000026-0000-0000-0000-000000000001
"""
Estimate prototype SNR (dB) for each latent position/code pair.

SNR is computed as peak amplitude in the predicted arrival window divided by
RMS amplitude before the earliest arrival.
"""
function per_position_prototype_snr(model, x, para, pair_ids=fill(1, size(flatten_batch(x), 2)))
    proto = per_position_cluster_averages(model, x, pair_ids)
    protos = proto.prototypes
    nt, W, K = size(protos)

    para.interstation_distance === nothing && error("interstation_distance is required for SNR windowing")
    vmin, vmax = para.velocity_range
    t_fast = para.interstation_distance / vmax
    t_slow = para.interstation_distance / vmin
    i_fast = clamp(round(Int, t_fast / para.dt), 1, nt)
    i_slow = clamp(round(Int, t_slow / para.dt), 1, nt)
    if i_fast > i_slow
        i_fast, i_slow = i_slow, i_fast
    end
    i_noise_end = max(1, i_fast - 1)

    snr_db = fill(Float32(-Inf), W, K)
    for l in 1:W, k in 1:K
        w = protos[:, l, k]
        signal_peak = maximum(abs.(w[i_fast:i_slow]))
        noise_rms = sqrt(mean(abs2, w[1:i_noise_end])) + 1f-8
        snr_db[l, k] = 20f0 * log10(signal_peak / noise_rms + 1f-8)
    end
    return (; snr_db, counts=proto.counts, i_fast, i_slow)
end

# ╔═╡ a0000027-0000-0000-0000-000000000001
"""
Plot per-position prototype SNR as a heatmap (position x code).
"""
function plot_per_position_snr_heatmap(snr_db; title="Per-position prototype SNR (dB)")
    W, K = size(snr_db)
    trace = PlutoPlotly.heatmap(
        z=snr_db,
        x=string.(1:K),
        y=string.(1:W),
        colorscale="RdBu",
        colorbar=attr(title="SNR (dB)")
    )
    layout = Layout(
        title=attr(text=title, font=attr(size=18, family="Computer Modern, Latin Modern Math, serif")),
        width=850,
        height=500,
        xaxis=attr(title="Code"),
        yaxis=attr(title="Latent position"),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    return PlutoPlotly.plot([trace], layout)
end

# ╔═╡ a0000028-0000-0000-0000-000000000001
"""
Compute entropy statistics for each latent slot (position).

Returns:
- `entropy`: raw entropy per slot (nats)
- `entropy_norm`: entropy normalized by log(K), in [0, 1]
- `perplexity`: exp(entropy) per slot
- `probs`: (K, T) probability matrix where each column is a slot distribution
"""
function per_latent_slot_entropy(model, x, pair_ids=fill(1, size(flatten_batch(x), 2)))
    res = encode(model, x, pair_ids)
    ci = res.codebook_indices
    T = size(ci, 1)
    N = size(ci, 2)
    K = codebook_size(model)
    probs = zeros(Float32, K, T)
    entropy = zeros(Float32, T)
    entropy_norm = zeros(Float32, T)
    perplexity = zeros(Float32, T)

    logk = log(max(K, 2))
    for t in 1:T
        counts = zeros(Float32, K)
        for n in 1:N
            counts[ci[t, n]] += 1f0
        end
        p = counts ./ max(sum(counts), 1f-8)
        probs[:, t] .= p
        p_safe = clamp.(p, 1f-10, 1f0)
        h = -sum(p_safe .* log.(p_safe))
        entropy[t] = h
        entropy_norm[t] = h / max(logk, 1f-8)
        perplexity[t] = exp(h)
    end

    return (; entropy, entropy_norm, perplexity, probs)
end

# ╔═╡ a0000029-0000-0000-0000-000000000001
"""
Plot normalized entropy per latent slot.
"""
function plot_per_latent_slot_entropy(entropy_norm; title="Per-slot entropy")
    T = length(entropy_norm)
    xslot = collect(1:T)
    trace = PlutoPlotly.scatter(
        x=xslot,
        y=entropy_norm,
        mode="lines+markers",
        line=attr(color="#1f77b4", width=2),
        marker=attr(size=6),
        name="Normalized entropy"
    )
    layout = Layout(
        title=attr(text=title, font=attr(size=18, family="Computer Modern, Latin Modern Math, serif")),
        width=850,
        height=350,
        xaxis=attr(title="Latent slot"),
        yaxis=attr(title="Entropy / log(K)", range=[0, 1.05]),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    return PlutoPlotly.plot([trace], layout)
end

# ╔═╡ ce3070ae-405e-11f1-b1d1-cb34a215f45c
"""
    get_vqvae(para::VQVAE_Para)

Build a VQ-VAE model and loss history from parameters.

Returns `(model, loss_history)`.
"""
function get_vqvae(para)
    if para.seed !== nothing
        Random.seed!(para.seed)
    end

    if para.use_multiscale_rvq && para.per_position
        error("v4 multiscale mode expects dense whole-waveform quantization. Set per_position=false.")
    end

    latent_time_start, latent_time_end = compute_latent_window(para)

    if latent_time_start === nothing
        # Dense fallback: flatten the encoder output before pre-VQ projection.
        encoder, flat_len, enc_outsize = get_vq_conv_encoder(para.nt;
            kernels=para.enc_kernels, filters=para.enc_filters,
            strides=para.enc_strides, use_bn=para.use_bn,
            flatten_output=true, return_outsize=true)
        latent_len = enc_outsize[1]
        pre_vq = Dense(flat_len, para.d * para.T) |> xpu
        actual_T = para.T
    else
        # Latent-time mode: preserve the sequence and select [start:end] interval.
        encoder, flat_len, enc_outsize = get_vq_conv_encoder(para.nt;
            kernels=para.enc_kernels, filters=para.enc_filters,
            strides=para.enc_strides, use_bn=para.use_bn,
            flatten_output=false, return_outsize=true)
        latent_len = enc_outsize[1]
        if latent_time_start > latent_time_end
            error("latent_time_start=$latent_time_start must be <= latent_time_end=$latent_time_end")
        end
        if latent_time_start < 1 || latent_time_end > latent_len
            error("latent-time interval [$latent_time_start, $latent_time_end] is out of latent length $latent_len")
        end
        enc_channels = enc_outsize[2]
        window_len = latent_time_end - latent_time_start + 1
        if para.per_position
            if !para.per_position_use_lstm && !isnothing(para.per_position_lstm_hidden)
                @warn "per_position_lstm_hidden is set but per_position_use_lstm=false; hidden size will be ignored."
            end
            lstm_hidden = isnothing(para.per_position_lstm_hidden) ? enc_channels : para.per_position_lstm_hidden
            lstm_hidden > 0 || error("per_position_lstm_hidden must be > 0, got $lstm_hidden")
            lstm_layer = para.per_position_use_lstm ? (Flux.LSTM(enc_channels => lstm_hidden) |> xpu) : nothing
            proj_in_channels = para.per_position_use_lstm ? lstm_hidden : enc_channels
            proj_layer = SeqConv1DChain(Chain(Conv((1,), proj_in_channels => para.d))) |> xpu
            pre_vq = PerPositionPreVQ(lstm_layer, proj_layer)
            # All latent slots are quantized; T = full latent sequence length.
            actual_T = latent_len
            if para.T != 1 && para.T != latent_len
                @warn "per_position=true: overriding para.T=$(para.T) with computed latent_len=$(latent_len). " *
                      "In per-position mode, T is auto-set to the full encoder latent length. " *
                      "You can safely set T=1 in your parameters (it will be auto-computed)."
            end
        else
            pre_vq = Dense(window_len * enc_channels, para.d * para.T) |> xpu
            actual_T = para.T
        end
    end

    quantizer = if para.use_multiscale_rvq
        if actual_T != 1
            error("v4 multiscale mode expects T=1 (whole-waveform quantization). Current computed T=$(actual_T).")
        end
        if para.pair_codebook_mode == :pair_specific_coarse_shared_detail
            para.num_pairs >= 1 || error("num_pairs must be >= 1 for pair-specific coarse codebooks")
            if !isnothing(para.pair_names)
                length(para.pair_names) == para.num_pairs || error("Expected $(para.num_pairs) pair names, got $(length(para.pair_names))")
            end
            PairSpecificMultiscaleRVQQuantizer(para.num_pairs, para.Kcoherent, para.Knuisance, para.d;
                detail_stages=para.detail_stages,
                decay_small=para.ema_decay_small,
                decay_large=para.ema_decay_large,
                epsilon=para.epsilon,
                dead_threshold=para.dead_threshold,
                attn_temperature=para.attn_temperature,
                attn_init_scale=para.attn_init_scale,
                attn_blend=para.attn_blend,
                attn_self_bias=para.attn_self_bias)
        else
            MultiscaleRVQQuantizer(para.Kcoherent, para.Knuisance, para.d;
                detail_stages=para.detail_stages,
                decay_small=para.ema_decay_small,
                decay_large=para.ema_decay_large,
                epsilon=para.epsilon,
                dead_threshold=para.dead_threshold)
        end
    elseif para.per_slot_codebook
        actual_T > 1 || @warn "per_slot_codebook=true with T=$(actual_T): this is equivalent to a shared single-slot codebook."
        PerSlotVectorQuantizer(actual_T, para.Kcoherent, para.d;
            decay=para.ema_decay, epsilon=para.epsilon,
            dead_threshold=para.dead_threshold)
    else
        VectorQuantizerEMA(para.Kcoherent, para.d;
            decay=para.ema_decay, epsilon=para.epsilon,
            dead_threshold=para.dead_threshold)
    end

    # Decoder always sees the concatenated T slot vectors.
    d_in_dec = para.d * actual_T
    dec_upstrides = auto_dec_upstrides_for_nt(para.nt, latent_len, d_in_dec;
        dec_kernels=para.dec_kernels, dec_filters=para.dec_filters,
        use_bn=para.use_bn, enc_strides=para.enc_strides)
    decoder, _, _ = get_vq_conv_decoder(para.nt, d_in_dec;
        kernels=para.dec_kernels, filters=para.dec_filters,
        upstrides=dec_upstrides, use_bn=para.use_bn)

    dec_out_len = Flux.outputsize(decoder.chain, (d_in_dec,); padbatch=true)[1]
    if dec_out_len != para.nt
        error("Decoder geometry mismatch: output length $dec_out_len != nt $(para.nt). " *
              "Adjust enc_strides or decoder kernels/filters.")
    end
    @info "VQ-VAE geometry" nt = para.nt latent_len enc_outsize dec_upstrides dec_out_len latent_time_start latent_time_end

    model = VQVAE(xpu(encoder), xpu(pre_vq), quantizer, xpu(decoder), actual_T, para.d,
        latent_time_start, latent_time_end, para.per_position)

    loss_history = (;
        train_recon=Float32[],
        test_recon=Float32[],
        train_commit=Float32[],
        test_commit=Float32[],
        train_total=Float32[],
        test_total=Float32[],
        train_perplexity=Float32[],
        test_perplexity=Float32[],
    )

    return model, loss_history
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
ParameterSchedulers = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[compat]
BenchmarkTools = "~1.7.0"
CUDA = "~5.11.0"
DSP = "~0.8.4"
Distances = "~0.10.12"
Enzyme = "~0.13.138"
FFTW = "~1.10.0"
Flux = "~0.16.9"
Functors = "~0.5.2"
JLD2 = "~0.6.4"
MLUtils = "~0.4.8"
Optimisers = "~0.4.7"
ParameterSchedulers = "~0.4.3"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.80"
ProgressLogging = "~0.1.6"
StatsBase = "~0.34.10"
Zygote = "~0.7.10"
cuDNN = "~1.4.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "612fdbdf4ce3aeb69da7d4fa83eb1a10d25f9e77"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"
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
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

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

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "e0b47732a192dd59b9d079a06d04235e2f833963"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

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
git-tree-sha1 = "6876e30dc02dc69f0613cb6ece242144f2ca9e56"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.7.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

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

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "70dea6a7133d2100a143b515a00d6d887e208500"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "9.20.0+0"

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

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

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

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

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

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"

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
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

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
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

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

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "d6dd65421104fa9f7d5cc37283a998937f359a39"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.138"
weakdeps = ["ADTypes", "BFloat16s", "ChainRulesCore", "GPUArraysCore", "LogExpFunctions", "SpecialFunctions", "StaticArrays"]

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeGPUArraysCoreExt = "GPUArraysCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

[[deps.EnzymeCore]]
git-tree-sha1 = "24bbb6fc8fb87eb71c1f8d00184a60fc22c63903"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.19"
weakdeps = ["Adapt", "ChainRulesCore"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"
    EnzymeCoreChainRulesCoreExt = "ChainRulesCore"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "4c22000e08aaa862526d9a41cfb7003e4002e653"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.256+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

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

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

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

[[deps.Flux]]
deps = ["ADTypes", "Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "ea6715b3d7a95a07a62109df1c9ede2641a50706"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.9"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxFiniteDifferencesExt = "FiniteDifferences"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]
    FluxMooncakeExt = "Mooncake"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

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
git-tree-sha1 = "6487601563e4a1d1dab796e88b4548bf5544209e"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.4.1"
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
git-tree-sha1 = "fedfe5e7db7035271c3f58359007f971da1dde87"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.9.1"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "a589b6c1a0eff953571f5d8b0474f5020831114d"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.1.1"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.InfiniteArrays]]
deps = ["ArrayLayouts", "FillArrays", "Infinities", "LazyArrays", "LinearAlgebra"]
git-tree-sha1 = "e61675cbcf3ce57ea3566b0abcfe46e4a521bf6f"
uuid = "4858937d-0d70-526a-a4dd-2d5cb5dd786c"
version = "0.12.15"
weakdeps = ["Statistics"]

    [deps.InfiniteArrays.extensions]
    InfiniteArraysStatisticsExt = "Statistics"

[[deps.Infinities]]
git-tree-sha1 = "4495006c20b2fd27b8c453a1dd31d423654f3772"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.12"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

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
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "67c6f1f085cb2671c93fe34244c9cccde30f7a26"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

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

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays"]
git-tree-sha1 = "35079a6a869eecace778bcda8641f9a54ca3a828"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "1.10.0"
weakdeps = ["StaticArrays"]

    [deps.LazyArrays.extensions]
    LazyArraysStaticArraysExt = "StaticArrays"

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

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "5d4278f755440f70648d80cc6225f51e78e94094"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.1"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

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
git-tree-sha1 = "39a69ca451c3e78b9a6a2e42ef894fdf7505e629"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.5"

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

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "6731e0574fa5ee21c02733e397beb133df90de35"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "2.2.0"

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

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

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

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.ParameterSchedulers]]
deps = ["InfiniteArrays", "Optimisers"]
git-tree-sha1 = "c62f0da0663704d0472ae578c9bb802c44e70a4c"
uuid = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
version = "0.4.3"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "aea4eede5ab3ee188906d0cf3bbfa36eb543dccc"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.8"

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

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

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

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

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

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "5f4f629c085b87e71125eec6773f5f872c74a47a"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

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

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

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

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDA_Runtime_Discovery", "CUDNN_jll"]
git-tree-sha1 = "5494b0ae3ddc5ca0f64159d5ed3a396f36e0fcfe"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.4.7"

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
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╟─461f0505-2230-4b84-b6c6-1a9730808437
# ╠═cc11647d-1c56-4ceb-9677-703aca03c9f4
# ╠═d73472ff-9e09-45b0-8811-b7dd8d820358
# ╠═76dbf599-a9b3-459f-992b-16ab2f7b74f1
# ╠═4a95997e-5c12-4658-9b8e-a5065328e1c1
# ╠═97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
# ╠═26fb86d5-c844-469a-aef5-ed3c2a9ba949
# ╠═6affb3b3-9dc4-4bbc-a582-495fc1783a7a
# ╟─a0000001-0000-0000-0000-000000000001
# ╠═80f77b52-84e0-4664-8aa0-3d79fded40de
# ╠═6ba143e2-50df-441a-8f38-3ea8d9edd4d8
# ╟─a0000002-0000-0000-0000-000000000001
# ╠═91a25156-e121-4d53-a5a1-422f1230d235
# ╟─a0000003-0000-0000-0000-000000000001
# ╠═89599b3f-8c20-46c5-8f5c-ccbb71b26b36
# ╠═2f7550d1-e854-4c2f-8efb-ad0bb70d5013
# ╠═44e9c4cc-d02b-4e68-ad49-24f173556cbd
# ╠═8eb0be68-99e1-4df9-97bf-b29b99d8f759
# ╟─a0000005-0000-0000-0000-000000000001
# ╠═64430447-c267-4eec-8d38-63ccf91d82c4
# ╟─a0000006-0000-0000-0000-000000000001
# ╠═4cfb91c3-97e4-4006-9ec9-5eedd7de2849
# ╠═ee76dd74-feaa-4422-a9a8-5931eb817c24
# ╠═5fd449bf-d0d5-4944-8fc6-c0d47a0ed83f
# ╠═1f046042-0695-4a3d-a0ee-9ce4213c9b72
# ╠═0a9a8028-2ba9-43d9-8b3d-8ada32ba314b
# ╠═b70d1b6c-2e5e-4548-9629-ea69011a2585
# ╠═802aa1db-04b1-4444-ab9a-d9c4aee17dd9
# ╟─a0000008-0000-0000-0000-000000000001
# ╠═a0000009-0000-0000-0000-000000000001
# ╟─a0000010-0000-0000-0000-000000000001
# ╠═a0000011-0000-0000-0000-000000000001
# ╟─a0000012-0000-0000-0000-000000000001
# ╠═76930390-49e1-45da-b654-7d1e65c856b1
# ╠═8646155e-376f-44e6-89df-c3092d350046
# ╠═b4387329-5621-49ec-a01f-3fe0f686d399
# ╟─a0000015-0000-0000-0000-000000000001
# ╟─a0000017-0000-0000-0000-000000000001
# ╠═eafe181e-19e9-409e-ad1d-ce859cf0e672
# ╠═a0000018-0000-0000-0000-000000000001
# ╠═7590d7a2-c8e9-4bb7-9438-aafdc72c84d7
# ╠═2a99c53c-d5fc-4ead-b7dc-3c6360548e57
# ╠═5481cf12-dd58-4bf8-965e-0b68aaa7adf6
# ╠═1a8c1a9f-61d9-4591-b50f-cb7ebca78377
# ╠═ce339f4a-0744-4bf1-9f3c-9aa9fb19eaf1
# ╠═7f3ff55b-0095-4d95-b3f0-c2371f0b7548
# ╠═40e94672-4fc6-4ffa-90e5-5f2cc3f7591d
# ╟─a0000020-0000-0000-0000-000000000001
# ╠═a0000021-0000-0000-0000-000000000001
# ╠═a0000022-0000-0000-0000-000000000001
# ╠═a0000023-0000-0000-0000-000000000001
# ╟─a0000024-0000-0000-0000-000000000001
# ╠═09202321-5cf6-46f1-bd59-6069da6488c6
# ╠═faa3de6b-96eb-4aeb-b927-77d6771a1d0a
# ╠═302a9921-c723-41d4-9d7a-234bf658a07e
# ╠═b1c2d3e4-f5a6-7890-abcd-ef1234567890
# ╠═a0000025-0000-0000-0000-000000000001
# ╠═a0000026-0000-0000-0000-000000000001
# ╠═a0000027-0000-0000-0000-000000000001
# ╠═a0000028-0000-0000-0000-000000000001
# ╠═a0000029-0000-0000-0000-000000000001
# ╠═ce3070ae-405e-11f1-b1d1-cb34a215f45c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
