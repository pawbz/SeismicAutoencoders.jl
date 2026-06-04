using Random, Flux, CUDA, Statistics

Random.seed!(1234)

# Standalone minimal VQ-VAE (GPU) for testing Zygote.@ignore + STE
struct VectorQuantizer
    K::Int
    d::Int
    T::Int
    embedding::Flux.Embedding
end
Flux.@functor VectorQuantizer

function VectorQuantizer(K::Int, d::Int, T::Int)
    emb = Flux.Embedding(K => d, init=Flux.randn32)
    emb.weight .= randn(Float32, d, K) .* (1f0 / K)
    emb = xpu(emb)
    return VectorQuantizer(K, d, T, emb)
end

# Non-diff NN search (indices only)
function quantize_indices_shared(emb_layer::Flux.Embedding, z_e)
    emb_mat = xpu(emb_layer.weight)
    _, K = size(emb_mat)
    _, T, N = size(z_e)
    indices = Array{Int}(undef, T, N)
    for t in 1:T
        emb = emb_mat
        z_e_t = z_e[:, t, :]
        z_sq = sum(abs2, z_e_t; dims=1)
        e_sq = sum(abs2, emb; dims=1)
        dist = e_sq' .+ z_sq .- 2f0 .* (transpose(emb) * z_e_t)
        idx_cart = dropdims(CUDA.argmin(dist; dims=1); dims=1)
        idxs = getindex.(idx_cart, 1)
        indices[t, :] .= cpu(idxs)
    end
    return indices
end

function (vq::VectorQuantizer)(z_e::AbstractArray{Float32,3}; beta_commit::Float32=0.25f0, training::Bool=true)
    d, T, N = size(z_e)
    @assert d == vq.d
    @assert T == vq.T

    # Index search is non-differentiable
    indices = Zygote.@ignore quantize_indices_shared(vq.embedding, z_e)

    # Build encodings and quantized vectors in differentiable path (so embedding gets grads)
    emb = xpu(vq.embedding.weight)
    z_q = similar(z_e)
    encodings = similar(z_e, Float32, vq.K, vq.T, N)
    for t in 1:T
        idxs_cpu = indices[t, :]
        enc_t = Float32.(Flux.onehotbatch(idxs_cpu, 1:vq.K)) |> xpu
        z_q[:, t, :] .= emb * enc_t
        encodings[:, t, :] .= enc_t
    end

    # losses
    z_e_ig = Zygote.@ignore(z_e) |> xpu
    vq_loss = Flux.mse(z_q, z_e_ig)
    commit_loss = beta_commit * Flux.mse(z_e, (Zygote.@ignore(z_q) |> xpu))

    st_residual = xpu(Zygote.@ignore(z_q .- z_e))
    z_q_st = z_e .+ st_residual

    avg_probs = dropdims(mean(encodings; dims=3), dims=3)
    avg_probs_clamped = clamp.(avg_probs, 1f-10, 1f0)
    entropy_per_slot = vec(Array(sum(avg_probs_clamped .* log.(avg_probs_clamped); dims=1)))
    perplexity = mean(exp.(-entropy_per_slot))

    return (; z_q=z_q_st, indices, encodings, vq_loss, commit_loss, perplexity, entropy_loss=mean(entropy_per_slot))
end

# Minimal VQVAE model
struct VQVAE
    pre_vq::Dense
    quantizer::VectorQuantizer
    decoder::Dense
    d::Int
    T::Int
end
Flux.@functor VQVAE

function VQVAE(nt::Int, d::Int, K::Int, T::Int)
    pre_vq = Dense(nt, d * T) |> xpu
    quantizer = VectorQuantizer(K, d, T)
    decoder = Dense(d * T, nt) |> xpu
    return VQVAE(pre_vq, quantizer, decoder, d, T)
end

function (m::VQVAE)(x; beta_commit::Float32=0.25f0, training::Bool=true)
    # x: (nt, N)
    nt, N = size(x)
    z_pre = m.pre_vq(x)                # (d*T, N)
    z_e = reshape(z_pre, m.d, m.T, N)  # (d, T, N)
    vq_res = m.quantizer(z_e; beta_commit=beta_commit, training=training)
    z_q = reshape(vq_res.z_q, m.d * m.T, N)
    xhat = m.decoder(z_q)              # (nt, N)
    return (; xhat, vq_res, z_q)
end

# Test harness
para_nt = 32
d = 8; K = 16; T = 1
model = VQVAE(para_nt, d, K, T)

# tiny batch
x = rand(Float32, para_nt, 2) |> xpu

println("=== Standalone VQ-VAE smoke test ===")

lossfun() = begin
    res = model(x; beta_commit=0.25f0, training=true)
    recon = Flux.mse(res.xhat, x)
    total = recon + res.vq_res.commit_loss + 0.01f0 * res.vq_res.entropy_loss
    return total
end

try
    L = lossfun()
    @info "Forward OK" total=L

    # Collect trainable parameters (informational)
    tparams = Flux.trainable(model)
    println("trainable count: ", length(tparams))

    # Compute gradient w.r.t. the whole model (struct); extract field-wise grads
    g_model_tuple = Flux.gradient(m -> begin
            res = m(x; beta_commit=0.25f0, training=true)
            recon = Flux.mse(res.xhat, x)
            recon + res.vq_res.commit_loss + 0.01f0 * res.vq_res.entropy_loss
        end, model)

    g_model = g_model_tuple[1]

    # extract gradients for specific weights (may be `nothing` if not present)
    g_emb = getfield(getfield(getfield(g_model, :quantizer), :embedding), :weight)
    g_pre = getfield(getfield(g_model, :pre_vq), :weight)

    println("embedding grad norm: ", isnothing(g_emb) ? "(no grad)" : norm(Array(g_emb)))
    println("pre_vq grad norm:    ", isnothing(g_pre) ? "(no grad)" : norm(Array(g_pre)))
catch e
    @error "Smoke test failed" err=repr(e)
    rethrow(e)
end
