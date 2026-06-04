### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 9bb43b61-b218-40d4-b5e5-27040d587e75
using Random, Flux, CUDA, Statistics, LinearAlgebra

# ╔═╡ 9de1a92d-1892-4fcc-ba48-5466fd7e102d
using Zygote

# ╔═╡ 6b078df8-3343-4d41-9ab3-4010f7f2e4c0
using ADTypes, Enzyme, EnzymeCore, Lux, NNlib, Optimisers

# ╔═╡ 32fd3b7b-8407-4472-9e3b-21c749e67233
xpu = gpu

# ╔═╡ 5360ab64-be7d-477a-83a4-664a2b318933
Random.seed!(1234)

# ╔═╡ e0a22d76-93d7-45ec-a01e-7a5430902016
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

# ╔═╡ e520e7d7-a45e-45ea-b647-bd79125ae98b
begin
    # Standalone minimal VQ-VAE (GPU) for testing Zygote.@ignore + STE
    struct VectorQuantizer
        K::Int
        d::Int
        T::Int
        embedding::Flux.Embedding
    end
    Flux.@layer VectorQuantizer
    function VectorQuantizer(K::Int, d::Int, T::Int)
        emb = Flux.Embedding(K => d, init=Flux.randn32)
        emb.weight .= randn(Float32, d, K) .* (1f0 / K)
        emb = xpu(emb)
        return VectorQuantizer(K, d, T, emb)
    end
    function (vq::VectorQuantizer)(z_e::AbstractArray{Float32,3}; beta_commit::Float32=0.25f0, training::Bool=true)
        d, T, N = size(z_e)
        @assert d == vq.d
        @assert T == vq.T
        @show size(z_e)
        # Index search is non-differentiable
        indices = Zygote.@ignore quantize_indices_shared(vq.embedding, z_e)
        z_q = vq.embedding(indices)
        code_subscriptions = Flux.onehotbatch(indices, 1:vq.K)

        # perplexity
        enc_flat = reshape(code_subscriptions, vq.K, :)
        avg_probs = mean(enc_flat; dims=2)
        avg_probs_clamped = clamp.(avg_probs, 1f-10, 1f0)
        perplexity = exp(-sum(avg_probs_clamped .* log.(avg_probs_clamped)))


        # losses
        codebook_loss = Flux.mse(Zygote.@ignore(z_e), z_q)
        commit_loss = beta_commit * Flux.mse(z_e, Zygote.@ignore(z_q))
        embedding_loss = codebook_loss + commit_loss

        return (; z_q, codebook_loss, commit_loss, embedding_loss, perplexity, code_subscriptions, indices)
    end
end

# ╔═╡ 9be3deda-0091-4d0e-b06b-12c38ba1be15
begin
    # Minimal VQVAE model
    struct VQVAE
        pre_vq::Flux.Dense
        quantizer::VectorQuantizer
        decoder::Flux.Dense
        d::Int
        T::Int
    end
    Flux.@layer VQVAE trainable = (pre_vq, quantizer, decoder)
    function VQVAE(nt::Int, d::Int, K::Int, T::Int)
        pre_vq = Flux.Dense(nt, d * T) |> xpu
        quantizer = VectorQuantizer(K, d, T)
        decoder = Flux.Dense(d * T, nt) |> xpu
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
end

# ╔═╡ e470a196-ba0e-4b48-acb2-89138e316157
xpu(randn(10, 1))

# ╔═╡ af7e15d3-d779-4bc0-b219-c6ef863a94de
# Test harness
para_nt = 32

# ╔═╡ 231a5245-f1bf-4722-ba1d-c96eea0e7c0d
begin
	d = 8;
	K = 5;
	T = 2;
end

# ╔═╡ d7288bdf-53f6-4c58-b492-f7a9a92740c1
model = VQVAE(para_nt, d, K, T)

# ╔═╡ f80fe0e0-929a-4baa-a097-8f8a9e49547d
# tiny batch
x = rand(Float32, para_nt, 16) |> xpu

# ╔═╡ 9239e126-915e-4230-b73b-25486eb0394d
model(x).vq_res.code_subscriptions

# ╔═╡ 83afc7c4-7ed0-432b-af5e-d903ff173255
println("=== Standalone VQ-VAE smoke test ===")

# ╔═╡ 21b88114-7003-4179-bee3-91b03690afae
function lossfun(model, x)
    res = model(x; beta_commit=0.25f0, training=true)
    recon = Flux.mse(res.xhat, x)
    total = recon + res.vq_res.embedding_loss
    return total
end

# ╔═╡ 64a08df5-f08c-449f-bb4f-3d083e39950d
begin
    L = lossfun(model, x)
    @info "Forward OK" total = L
end

# ╔═╡ 34cdf4c9-8e01-40c5-836d-9b47d9f43002
xpu

# ╔═╡ 9fd80194-7112-4bbb-8550-0486fa8160b3
xpu(model)

# ╔═╡ b45ad394-958a-43ae-9f1a-2b8ca9df7dcd
grads = Flux.gradient(model, x) do m, x
    lossfun(m, x)
end

# ╔═╡ 2b7e8d1f-0f92-4a5f-a5c2-f67e5934619f
md"## Lux/Enzyme VQ-VAE Smoke Path"

# ╔═╡ 0f85b13f-4daf-4f0d-b3f7-576d8baaa43c
begin
    lux_rng = MersenneTwister(20260504)
    x_lux = Array(cpu(x))
end

# ╔═╡ d8a4e456-e264-4239-96ce-458b70ed098d
begin
    lux_argmin_col_indices(idx::AbstractArray{<:CartesianIndex}) = getindex.(idx, 1)
    lux_argmin_col_indices(idx::AbstractArray) = idx

    function lux_assignment_matrix(indices, K::Int)
        codes = reshape(collect(1:K), K, 1)
        return Float32.(codes .== reshape(indices, 1, :))
    end

    function lux_lookup_and_stats(embedding, z_e)
        d_local, T_local, N_local = size(z_e)
        z_flat = reshape(z_e, d_local, T_local * N_local)
        z_sq = sum(abs2, z_flat; dims=1)
        e_sq = sum(abs2, embedding; dims=1)
        dist = e_sq' .+ z_sq .- 2f0 .* (embedding' * z_flat)
        idx_raw = vec(argmin(dist; dims=1))
        indices = vec(lux_argmin_col_indices(idx_raw))
        enc = lux_assignment_matrix(indices, size(embedding, 2))
        z_q_flat = embedding * enc
        counts = vec(sum(enc; dims=2))
        sums = z_flat * enc'
        return reshape(z_q_flat, d_local, T_local, N_local),
            reshape(indices, T_local, N_local), counts, sums
    end

    function lux_perplexity(counts)
        total = max(sum(counts), 1f-8)
        p = counts ./ total
        psafe = clamp.(p, 1f-10, 1f0)
        return exp(-sum(psafe .* log.(psafe)))
    end
end

# ╔═╡ 9dc20d78-0781-4a55-91f8-3356a11ed677
function lux_quantize(m::LuxToyVQVAE, z_e, qst; training::Bool)
    z_q_detached, indices, counts, sums = EnzymeCore.ignore_derivatives(
        lux_lookup_and_stats(qst.embedding, z_e)
    )
    z_q = z_e .+ EnzymeCore.ignore_derivatives(z_q_detached .- z_e)
    commit_loss = m.beta_commit * mean(abs2, z_e .- EnzymeCore.ignore_derivatives(z_q_detached))
    perplexity = lux_perplexity(EnzymeCore.ignore_derivatives(counts))

    qst_new = if training
        ema_cluster_size = m.ema_decay .* qst.ema_cluster_size .+ (1f0 - m.ema_decay) .* counts
        ema_dw = m.ema_decay .* qst.ema_dw .+ (1f0 - m.ema_decay) .* sums
        embedding = ema_dw ./ reshape(max.(ema_cluster_size, 1f-5), 1, :)
        (; embedding=Float32.(embedding),
            ema_cluster_size=Float32.(ema_cluster_size),
            ema_dw=Float32.(ema_dw))
    else
        qst
    end

    return (; z_q, indices, commit_loss, perplexity), qst_new
end

# ╔═╡ 11134e4f-f83b-4ad0-8d17-100a837fe596
begin
    struct LuxToyVQVAE{P<:Lux.AbstractLuxLayer,D<:Lux.AbstractLuxLayer} <:
        Lux.AbstractLuxContainerLayer{(:pre_vq, :decoder)}
        pre_vq::P
        decoder::D
        K::Int
        d::Int
        T::Int
        beta_commit::Float32
        ema_decay::Float32
    end

    function Lux.initialparameters(rng::AbstractRNG, m::LuxToyVQVAE)
        return (;
            pre_vq=Lux.initialparameters(rng, m.pre_vq),
            decoder=Lux.initialparameters(rng, m.decoder),
        )
    end

    function Lux.initialstates(rng::AbstractRNG, m::LuxToyVQVAE)
        embedding = randn(rng, Float32, m.d, m.K) .* (1f0 / max(m.K, 1))
        return (;
            pre_vq=Lux.initialstates(rng, m.pre_vq),
            decoder=Lux.initialstates(rng, m.decoder),
            quantizer=(;
                embedding,
                ema_cluster_size=ones(Float32, m.K),
                ema_dw=copy(embedding),
            ),
        )
    end

	function (m::LuxToyVQVAE)(x, ps, st; training::Bool=true)
    z_pre, st_pre = m.pre_vq(x, ps.pre_vq, st.pre_vq)
    z_e = reshape(z_pre, m.d, m.T, size(x, 2))
    q, st_quantizer = lux_quantize(m, z_e, st.quantizer; training)
    z_q = reshape(q.z_q, m.d * m.T, size(x, 2))
    xhat, st_dec = m.decoder(z_q, ps.decoder, st.decoder)
    st_new = (; pre_vq=st_pre, decoder=st_dec, quantizer=st_quantizer)
    return (; xhat, z_e, z_q, commit_loss=q.commit_loss, perplexity=q.perplexity,
        indices=q.indices), st_new
end

	
end

# ╔═╡ c115d647-70d3-4749-9012-b40394a0bddf


# ╔═╡ 834836a5-01ea-4b18-a681-b74371b1255c
begin
    lux_model = LuxToyVQVAE(
        Lux.Dense(para_nt => d * T),
        Lux.Dense(d * T => para_nt),
        K, d, T, 0.25f0, 0.99f0,
    )
    lux_ps, lux_st = Lux.setup(lux_rng, lux_model)
end

# ╔═╡ 7590f568-3312-4c9f-88e4-7052d192d266
function lux_vqvae_loss(model, ps, st, batch)
    result, st_new = model(batch, ps, st; training=true)
    recon_loss = mean(abs2, result.xhat .- batch)
    total = recon_loss + result.commit_loss
    return total, st_new, (; recon_loss, commit_loss=result.commit_loss,
        perplexity=result.perplexity)
end

# ╔═╡ d339b18f-428e-4ec6-bc7e-9c33b74c10b2
function lux_vqvae_eval_loss(model, ps, st, batch)
    result, _ = model(batch, ps, st; training=false)
    recon_loss = mean(abs2, result.xhat .- batch)
    total = recon_loss + result.commit_loss
    return total, (; recon_loss, commit_loss=result.commit_loss,
        perplexity=result.perplexity)
end

# ╔═╡ 051e45c2-823e-4510-a4fa-5afaa0e260df
begin
    tree_copy(x::AbstractArray) = copy(Array(x))
    tree_copy(x::NamedTuple) = NamedTuple{keys(x)}(map(tree_copy, values(x)))
    tree_copy(x::Tuple) = map(tree_copy, x)
    tree_copy(x) = x

    tree_delta_sq(a::AbstractArray, b::AbstractArray) = sum(abs2, Array(a) .- Array(b))
    tree_delta_sq(a::NamedTuple, b::NamedTuple) =
        sum(tree_delta_sq(getfield(a, k), getfield(b, k)) for k in keys(a))
    tree_delta_sq(a::Tuple, b::Tuple) =
        sum(tree_delta_sq(a[i], b[i]) for i in eachindex(a))
    tree_delta_sq(a, b) = 0f0
    tree_delta(a, b) = sqrt(Float64(tree_delta_sq(a, b)))
end

# ╔═╡ 56720cc7-3eb1-4396-9502-d9848014d82a
function smoke_try(label, f)
    try
        value = f()
        return (; label, ok=true, value, error="")
    catch err
        return (; label, ok=false, value=nothing, error=sprint(showerror, err))
    end
end

# ╔═╡ e04d1604-6622-4263-b3f7-9248657c6c7f
function lux_forward_smoke(model, ps, st, batch)
    total, stats = lux_vqvae_eval_loss(model, ps, st, batch)
    return (; total, stats)
end

# ╔═╡ 4877f866-b98f-46bf-9b51-235566b83bf8
function lux_enzyme_gradient_smoke(model, ps, st, batch)
    tstate = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(1f-3))
    gs, loss, stats, tstate = Lux.Training.compute_gradients(
        ADTypes.AutoEnzyme(), lux_vqvae_loss, batch, tstate
    )
    return (; gs, loss, stats)
end

# ╔═╡ d4dbafef-a2ef-4bc6-86f5-c3db0a36b96e
function lux_enzyme_step_diagnostics(model, ps, st, batch)
    tstate = Lux.Training.TrainState(model, ps, st, Optimisers.Adam(1f-3))
    ps_before = tree_copy(tstate.parameters)
    st_before = tree_copy(tstate.states)
    loss_before, stats_before = lux_vqvae_eval_loss(model, tstate.parameters, tstate.states, batch)
    _, step_loss, step_stats, tstate = Lux.Training.single_train_step!(
        ADTypes.AutoEnzyme(), lux_vqvae_loss, batch, tstate
    )
    loss_after_one_step, stats_after = lux_vqvae_eval_loss(
        model, tstate.parameters, tstate.states, batch
    )
    return (;
        loss_before=Float32(loss_before),
        step_loss=Float32(step_loss),
        loss_after_one_step=Float32(loss_after_one_step),
        pre_vq_delta=tree_delta(ps_before.pre_vq, tstate.parameters.pre_vq),
        decoder_delta=tree_delta(ps_before.decoder, tstate.parameters.decoder),
        codebook_state_delta=tree_delta(st_before.quantizer.embedding, tstate.states.quantizer.embedding),
        perplexity_before=Float32(stats_before.perplexity),
        perplexity_after=Float32(stats_after.perplexity),
        recon_before=Float32(stats_before.recon_loss),
        recon_after=Float32(stats_after.recon_loss),
        commit_before=Float32(stats_before.commit_loss),
        commit_after=Float32(stats_after.commit_loss),
        stats=step_stats,
    )
end

# ╔═╡ 1ef9b5c6-8dd6-448a-9afe-7bc3b939ab72
begin
    fixed_codebook_payload = (; x=x_lux, embedding=copy(lux_st.quantizer.embedding))

    function lux_fixed_codebook_forward(model, ps, st, payload; training::Bool=true)
        batch = payload.x
        embedding = payload.embedding
        z_pre, st_pre = model.pre_vq(batch, ps.pre_vq, st.pre_vq)
        z_e = reshape(z_pre, model.d, model.T, size(batch, 2))
        z_q_detached, indices, counts, _ = EnzymeCore.ignore_derivatives(
            lux_lookup_and_stats(embedding, z_e)
        )
        z_q = z_e .+ EnzymeCore.ignore_derivatives(z_q_detached .- z_e)
        commit_loss = model.beta_commit *
            mean(abs2, z_e .- EnzymeCore.ignore_derivatives(z_q_detached))
        z_q_flat = reshape(z_q, model.d * model.T, size(batch, 2))
        xhat, st_dec = model.decoder(z_q_flat, ps.decoder, st.decoder)
        st_new = (; pre_vq=st_pre, decoder=st_dec)
        return (; xhat, commit_loss, perplexity=lux_perplexity(counts),
            indices), st_new
    end
end

# ╔═╡ 3f32f44d-3479-4ffc-ae7e-363e8f28a47d
function lux_fixed_codebook_loss(model, ps, st, payload)
    result, st_new = lux_fixed_codebook_forward(model, ps, st, payload; training=true)
    recon_loss = mean(abs2, result.xhat .- payload.x)
    total = recon_loss + result.commit_loss
    return total, st_new, (; recon_loss, commit_loss=result.commit_loss,
        perplexity=result.perplexity)
end

# ╔═╡ 44b33fed-119d-4385-a9c8-3c94a9f814a2
function lux_fixed_codebook_eval_loss(model, ps, st, payload)
    result, _ = lux_fixed_codebook_forward(model, ps, st, payload; training=false)
    recon_loss = mean(abs2, result.xhat .- payload.x)
    total = recon_loss + result.commit_loss
    return total, (; recon_loss, commit_loss=result.commit_loss,
        perplexity=result.perplexity)
end

# ╔═╡ e7debd35-07b4-4676-a3c2-365f5e588211
function lux_enzyme_fixed_codebook_gradient_smoke(model, ps, st, payload)
    train_st = (; pre_vq=st.pre_vq, decoder=st.decoder)
    tstate = Lux.Training.TrainState(model, ps, train_st, Optimisers.Adam(1f-3))
    gs, loss, stats, tstate = Lux.Training.compute_gradients(
        ADTypes.AutoEnzyme(), lux_fixed_codebook_loss, payload, tstate
    )
    return (; gs, loss, stats)
end

# ╔═╡ ed6b80bf-a6db-41b2-bca8-a260bdc2d707
function lux_enzyme_fixed_codebook_step_diagnostics(model, ps, st, payload)
    train_st = (; pre_vq=st.pre_vq, decoder=st.decoder)
    tstate = Lux.Training.TrainState(model, ps, train_st, Optimisers.Adam(1f-3))
    ps_before = tree_copy(tstate.parameters)
    loss_before, stats_before = lux_fixed_codebook_eval_loss(
        model, tstate.parameters, tstate.states, payload
    )
    _, step_loss, step_stats, tstate = Lux.Training.single_train_step!(
        ADTypes.AutoEnzyme(), lux_fixed_codebook_loss, payload, tstate
    )
    loss_after_one_step, stats_after = lux_fixed_codebook_eval_loss(
        model, tstate.parameters, tstate.states, payload
    )
    return (;
        loss_before=Float32(loss_before),
        step_loss=Float32(step_loss),
        loss_after_one_step=Float32(loss_after_one_step),
        pre_vq_delta=tree_delta(ps_before.pre_vq, tstate.parameters.pre_vq),
        decoder_delta=tree_delta(ps_before.decoder, tstate.parameters.decoder),
        perplexity_before=Float32(stats_before.perplexity),
        perplexity_after=Float32(stats_after.perplexity),
        recon_before=Float32(stats_before.recon_loss),
        recon_after=Float32(stats_after.recon_loss),
        commit_before=Float32(stats_before.commit_loss),
        commit_after=Float32(stats_after.commit_loss),
        parameters=tstate.parameters,
        states=tstate.states,
        stats=step_stats,
    )
end

# ╔═╡ a4614d4f-26fa-4dbd-9022-23da7910d264
function lux_external_codebook_update(model, ps, st, payload)
    batch = payload.x
    z_pre, _ = model.pre_vq(batch, ps.pre_vq, st.pre_vq)
    z_e = reshape(z_pre, model.d, model.T, size(batch, 2))
    _, _, counts, sums = lux_lookup_and_stats(st.quantizer.embedding, z_e)
    ema_cluster_size = model.ema_decay .* st.quantizer.ema_cluster_size .+
        (1f0 - model.ema_decay) .* counts
    ema_dw = model.ema_decay .* st.quantizer.ema_dw .+
        (1f0 - model.ema_decay) .* sums
    embedding = ema_dw ./ reshape(max.(ema_cluster_size, 1f-5), 1, :)
    qst_new = (; embedding=Float32.(embedding),
        ema_cluster_size=Float32.(ema_cluster_size),
        ema_dw=Float32.(ema_dw))
    return (; qst_new, codebook_state_delta=tree_delta(st.quantizer.embedding, qst_new.embedding),
        perplexity=lux_perplexity(counts))
end

# ╔═╡ 7fd86a7c-d3fd-4dc3-9784-4b55ef0e9216
function lux_plain_ae_loss(model, ps, st, batch)
    z_pre, st_pre = model.pre_vq(batch, ps.pre_vq, st.pre_vq)
    xhat, st_dec = model.decoder(z_pre, ps.decoder, st.decoder)
    loss = mean(abs2, xhat .- batch)
    return loss, (; pre_vq=st_pre, decoder=st_dec), (; recon_loss=loss)
end

# ╔═╡ 74a5177f-ad9a-4e2e-acb2-ea16d105e770
function lux_plain_ae_eval_loss(model, ps, st, batch)
    z_pre, _ = model.pre_vq(batch, ps.pre_vq, st.pre_vq)
    xhat, _ = model.decoder(z_pre, ps.decoder, st.decoder)
    loss = mean(abs2, xhat .- batch)
    return loss, (; recon_loss=loss)
end

# ╔═╡ 02d1a13f-e076-4b0a-b6e8-118540d8deec
function lux_enzyme_plain_ae_gradient_smoke(model, ps, st, batch)
    train_st = (; pre_vq=st.pre_vq, decoder=st.decoder)
    tstate = Lux.Training.TrainState(model, ps, train_st, Optimisers.Adam(1f-3))
    gs, loss, stats, tstate = Lux.Training.compute_gradients(
        ADTypes.AutoEnzyme(), lux_plain_ae_loss, batch, tstate
    )
    return (; gs, loss, stats)
end

# ╔═╡ b385ce59-28b9-411d-911e-bf79bacb76cf
function lux_enzyme_plain_ae_step_diagnostics(model, ps, st, batch)
    train_st = (; pre_vq=st.pre_vq, decoder=st.decoder)
    tstate = Lux.Training.TrainState(model, ps, train_st, Optimisers.Adam(1f-3))
    ps_before = tree_copy(tstate.parameters)
    loss_before, stats_before = lux_plain_ae_eval_loss(model, tstate.parameters, tstate.states, batch)
    _, step_loss, step_stats, tstate = Lux.Training.single_train_step!(
        ADTypes.AutoEnzyme(), lux_plain_ae_loss, batch, tstate
    )
    loss_after_one_step, stats_after = lux_plain_ae_eval_loss(
        model, tstate.parameters, tstate.states, batch
    )
    return (;
        loss_before=Float32(loss_before),
        step_loss=Float32(step_loss),
        loss_after_one_step=Float32(loss_after_one_step),
        pre_vq_delta=tree_delta(ps_before.pre_vq, tstate.parameters.pre_vq),
        decoder_delta=tree_delta(ps_before.decoder, tstate.parameters.decoder),
        recon_before=Float32(stats_before.recon_loss),
        recon_after=Float32(stats_after.recon_loss),
        stats=step_stats,
    )
end

# ╔═╡ 387b9bfb-28bf-4fd6-8608-599c78ea8e91
begin
    function lux_precomputed_ste_payload(model, ps, st, batch)
        z_pre, _ = model.pre_vq(batch, ps.pre_vq, st.pre_vq)
        z_e = reshape(z_pre, model.d, model.T, size(batch, 2))
        z_q_fixed, indices, counts, _ = lux_lookup_and_stats(st.quantizer.embedding, z_e)
        return (; x=batch, z_q_fixed=Float32.(z_q_fixed),
            perplexity=lux_perplexity(counts), indices)
    end

    precomputed_ste_payload = lux_precomputed_ste_payload(lux_model, lux_ps, lux_st, x_lux)
end

# ╔═╡ 73612870-4310-4791-a68f-fd25892a9c2a
function lux_precomputed_ste_loss(model, ps, st, payload)
    batch = payload.x
    z_pre, st_pre = model.pre_vq(batch, ps.pre_vq, st.pre_vq)
    z_e = reshape(z_pre, model.d, model.T, size(batch, 2))
    z_q_fixed = payload.z_q_fixed
    z_q = z_e .+ (EnzymeCore.ignore_derivatives(z_q_fixed) .-
        EnzymeCore.ignore_derivatives(z_e))
    commit_loss = model.beta_commit *
        mean(abs2, z_e .- EnzymeCore.ignore_derivatives(z_q_fixed))
    z_q_flat = reshape(z_q, model.d * model.T, size(batch, 2))
    xhat, st_dec = model.decoder(z_q_flat, ps.decoder, st.decoder)
    recon_loss = mean(abs2, xhat .- batch)
    total = recon_loss + commit_loss
    return total, (; pre_vq=st_pre, decoder=st_dec),
        (; recon_loss, commit_loss, perplexity=payload.perplexity)
end

# ╔═╡ 2f24d41d-ae04-4e0d-b9d4-97d3814c905c
function lux_precomputed_ste_eval_loss(model, ps, st, payload)
    total, _, stats = lux_precomputed_ste_loss(model, ps, st, payload)
    return total, stats
end

# ╔═╡ e4c55070-916d-48b9-b34d-a3ab34ad4062
function lux_enzyme_precomputed_ste_gradient_smoke(model, ps, st, payload)
    train_st = (; pre_vq=st.pre_vq, decoder=st.decoder)
    tstate = Lux.Training.TrainState(model, ps, train_st, Optimisers.Adam(1f-3))
    gs, loss, stats, tstate = Lux.Training.compute_gradients(
        ADTypes.AutoEnzyme(), lux_precomputed_ste_loss, payload, tstate
    )
    return (; gs, loss, stats)
end

# ╔═╡ 671bf2d2-76eb-48e1-9396-f8e383425339
function lux_enzyme_precomputed_ste_step_diagnostics(model, ps, st, payload)
    train_st = (; pre_vq=st.pre_vq, decoder=st.decoder)
    tstate = Lux.Training.TrainState(model, ps, train_st, Optimisers.Adam(1f-3))
    ps_before = tree_copy(tstate.parameters)
    loss_before, stats_before = lux_precomputed_ste_eval_loss(
        model, tstate.parameters, tstate.states, payload
    )
    _, step_loss, step_stats, tstate = Lux.Training.single_train_step!(
        ADTypes.AutoEnzyme(), lux_precomputed_ste_loss, payload, tstate
    )
    loss_after_one_step, stats_after = lux_precomputed_ste_eval_loss(
        model, tstate.parameters, tstate.states, payload
    )
    return (;
        loss_before=Float32(loss_before),
        step_loss=Float32(step_loss),
        loss_after_one_step=Float32(loss_after_one_step),
        pre_vq_delta=tree_delta(ps_before.pre_vq, tstate.parameters.pre_vq),
        decoder_delta=tree_delta(ps_before.decoder, tstate.parameters.decoder),
        perplexity=Float32(stats_after.perplexity),
        recon_before=Float32(stats_before.recon_loss),
        recon_after=Float32(stats_after.recon_loss),
        commit_before=Float32(stats_before.commit_loss),
        commit_after=Float32(stats_after.commit_loss),
        stats=step_stats,
    )
end

# ╔═╡ cdc3957f-381b-4946-80e6-a90e9f585044
smoke_results = [
    smoke_try("Flux control forward", () -> lossfun(model, x)),
    smoke_try("Flux/Zygote gradient", () -> Flux.gradient(model, x) do m, xb
        lossfun(m, xb)
    end),
    smoke_try("Lux/Enzyme plain AE gradient", () ->
        lux_enzyme_plain_ae_gradient_smoke(lux_model, lux_ps, lux_st, x_lux)),
    smoke_try("Lux/Enzyme plain AE train step", () ->
        lux_enzyme_plain_ae_step_diagnostics(lux_model, lux_ps, lux_st, x_lux)),
    smoke_try("Lux/Enzyme precomputed STE gradient", () ->
        lux_enzyme_precomputed_ste_gradient_smoke(lux_model, lux_ps, lux_st, precomputed_ste_payload)),
    smoke_try("Lux/Enzyme precomputed STE train step", () ->
        lux_enzyme_precomputed_ste_step_diagnostics(lux_model, lux_ps, lux_st, precomputed_ste_payload)),
    smoke_try("Lux forward", () -> lux_forward_smoke(lux_model, lux_ps, lux_st, x_lux)),
    smoke_try("Lux/Enzyme gradient", () -> lux_enzyme_gradient_smoke(lux_model, lux_ps, lux_st, x_lux)),
    smoke_try("Lux/Enzyme train step", () -> lux_enzyme_step_diagnostics(lux_model, lux_ps, lux_st, x_lux)),
    smoke_try("Lux/Enzyme fixed-codebook gradient", () ->
        lux_enzyme_fixed_codebook_gradient_smoke(lux_model, lux_ps, lux_st, fixed_codebook_payload)),
    smoke_try("Lux/Enzyme fixed-codebook train step", () ->
        lux_enzyme_fixed_codebook_step_diagnostics(lux_model, lux_ps, lux_st, fixed_codebook_payload)),
    smoke_try("External codebook update", () ->
        lux_external_codebook_update(lux_model, lux_ps, lux_st, fixed_codebook_payload)),
]

# ╔═╡ 6896023b-ed14-4378-9251-7cfba17c9137
smoke_status_text = join([
    "- $(r.label): $(r.ok ? "ok" : "failed: $(r.error)")"
    for r in smoke_results
], "\n")

# ╔═╡ a119c1d5-2819-4c72-aefa-bd0e2cfb8604
result_by_label(label) = smoke_results[findfirst(r -> r.label == label, smoke_results)]

# ╔═╡ b3b09a8c-3bb0-45e8-a6c9-26598e17049f
lux_step_row = result_by_label("Lux/Enzyme train step")

# ╔═╡ 8dc1484a-0fd6-4f09-8953-a9f30a50b266
lux_step_metrics = lux_step_row.ok ? lux_step_row.value : nothing

# ╔═╡ 278431a0-256d-4a8a-8795-238b3c6c1d2f
plain_ae_step_row = result_by_label("Lux/Enzyme plain AE train step")

# ╔═╡ af63db60-b2a4-47a8-bb39-819f704270f2
plain_ae_step_metrics = plain_ae_step_row.ok ? plain_ae_step_row.value : nothing

# ╔═╡ 5a6fe71d-1210-4c44-90c2-96321231991d
precomputed_ste_step_row = result_by_label("Lux/Enzyme precomputed STE train step")

# ╔═╡ 44113888-23fc-4166-abab-ae29d06a09ce
precomputed_ste_step_metrics = precomputed_ste_step_row.ok ? precomputed_ste_step_row.value : nothing

# ╔═╡ 91ae0d18-7d9c-42f1-bb83-61789c41ae8a
plain_ae_metric_text = if isnothing(plain_ae_step_metrics)
    "Lux/Enzyme plain AE train step failed, so plain-AE deltas are unavailable."
else
    join([
        "- plain_ae_loss_before: $(plain_ae_step_metrics.loss_before)",
        "- plain_ae_loss_after_one_step: $(plain_ae_step_metrics.loss_after_one_step)",
        "- plain_ae_pre_vq_delta: $(plain_ae_step_metrics.pre_vq_delta)",
        "- plain_ae_decoder_delta: $(plain_ae_step_metrics.decoder_delta)",
        "- plain_ae_recon_before: $(plain_ae_step_metrics.recon_before)",
        "- plain_ae_recon_after: $(plain_ae_step_metrics.recon_after)",
    ], "\n")
end

# ╔═╡ 010b27b4-c7c9-4180-b34b-ae327e85c95b
precomputed_ste_metric_text = if isnothing(precomputed_ste_step_metrics)
    "Lux/Enzyme precomputed STE train step failed, so precomputed-STE deltas are unavailable."
else
    join([
        "- ste_loss_before: $(precomputed_ste_step_metrics.loss_before)",
        "- ste_loss_after_one_step: $(precomputed_ste_step_metrics.loss_after_one_step)",
        "- ste_pre_vq_delta: $(precomputed_ste_step_metrics.pre_vq_delta)",
        "- ste_decoder_delta: $(precomputed_ste_step_metrics.decoder_delta)",
        "- ste_perplexity: $(precomputed_ste_step_metrics.perplexity)",
        "- ste_recon_before: $(precomputed_ste_step_metrics.recon_before)",
        "- ste_recon_after: $(precomputed_ste_step_metrics.recon_after)",
        "- ste_commit_before: $(precomputed_ste_step_metrics.commit_before)",
        "- ste_commit_after: $(precomputed_ste_step_metrics.commit_after)",
    ], "\n")
end

# ╔═╡ 59e58140-8741-484c-95e4-e79214a0df77
fixed_step_row = result_by_label("Lux/Enzyme fixed-codebook train step")

# ╔═╡ f4099841-fc40-4c4f-84bf-5e79e8fd927e
fixed_step_metrics = fixed_step_row.ok ? fixed_step_row.value : nothing

# ╔═╡ c22f4459-c732-4437-84d7-16e774f667d2
external_codebook_row = result_by_label("External codebook update")

# ╔═╡ f8e98ef1-b2e0-411d-a083-9027867668b9
external_codebook_metrics = external_codebook_row.ok ? external_codebook_row.value : nothing

# ╔═╡ 773f4779-86fc-4eff-8db2-bc3284de274f
lux_metric_text = if isnothing(lux_step_metrics)
    "Lux/Enzyme train step failed, so numeric deltas are unavailable."
else
    join([
        "- loss_before: $(lux_step_metrics.loss_before)",
        "- loss_after_one_step: $(lux_step_metrics.loss_after_one_step)",
        "- pre_vq_delta: $(lux_step_metrics.pre_vq_delta)",
        "- decoder_delta: $(lux_step_metrics.decoder_delta)",
        "- codebook_state_delta: $(lux_step_metrics.codebook_state_delta)",
        "- perplexity_before: $(lux_step_metrics.perplexity_before)",
        "- perplexity_after: $(lux_step_metrics.perplexity_after)",
        "- recon_before: $(lux_step_metrics.recon_before)",
        "- recon_after: $(lux_step_metrics.recon_after)",
        "- commit_before: $(lux_step_metrics.commit_before)",
        "- commit_after: $(lux_step_metrics.commit_after)",
    ], "\n")
end

# ╔═╡ fa01e2d1-e197-44fa-9184-1521b6c2ec95
fixed_metric_text = if isnothing(fixed_step_metrics)
    "Lux/Enzyme fixed-codebook train step failed, so fixed-codebook deltas are unavailable."
else
    external_delta = isnothing(external_codebook_metrics) ? NaN :
        external_codebook_metrics.codebook_state_delta
    join([
        "- fixed_loss_before: $(fixed_step_metrics.loss_before)",
        "- fixed_loss_after_one_step: $(fixed_step_metrics.loss_after_one_step)",
        "- fixed_pre_vq_delta: $(fixed_step_metrics.pre_vq_delta)",
        "- fixed_decoder_delta: $(fixed_step_metrics.decoder_delta)",
        "- external_codebook_state_delta: $(external_delta)",
        "- fixed_perplexity_before: $(fixed_step_metrics.perplexity_before)",
        "- fixed_perplexity_after: $(fixed_step_metrics.perplexity_after)",
        "- fixed_recon_before: $(fixed_step_metrics.recon_before)",
        "- fixed_recon_after: $(fixed_step_metrics.recon_after)",
    ], "\n")
end

# ╔═╡ 8c365a6b-52c0-47f8-93c4-98d4cbd07004
lux_interpretation_text = if isnothing(lux_step_metrics)
    if !isnothing(precomputed_ste_step_metrics)
        join([
            "- full in-loss VQ lookup: fail",
            "- precomputed STE training path: pass",
            "- implication: keep nearest-neighbor lookup and codebook updates outside the Enzyme loss",
        ], "\n")
    else
        "- Lux/Enzyme path failed before diagnostics could run."
    end
else
    param_ok = lux_step_metrics.pre_vq_delta > 0 || lux_step_metrics.decoder_delta > 0
    codebook_ok = lux_step_metrics.codebook_state_delta > 0
    loss_ok = lux_step_metrics.loss_after_one_step < lux_step_metrics.loss_before
    join([
        "- parameter update: $(param_ok ? "pass" : "fail")",
        "- codebook state update: $(codebook_ok ? "pass" : "fail")",
        "- one-step fixed-batch loss decrease: $(loss_ok ? "pass" : "not guaranteed / inspect after multiple steps")",
    ], "\n")
end

# ╔═╡ b3fd20d5-36dd-47ce-bd08-d26450e02638
fixed_interpretation_text = let 
if isnothing(fixed_step_metrics)
    "- Fixed-codebook Lux/Enzyme path failed before diagnostics could run."
else
    param_ok = fixed_step_metrics.pre_vq_delta > 0 || fixed_step_metrics.decoder_delta > 0
    codebook_ok = !isnothing(external_codebook_metrics) &&
        external_codebook_metrics.codebook_state_delta > 0
    loss_ok = fixed_step_metrics.loss_after_one_step < fixed_step_metrics.loss_before
    join([
        "- fixed-codebook parameter update: $(param_ok ? "pass" : "fail")",
        "- external codebook update: $(codebook_ok ? "pass" : "fail")",
        "- fixed-codebook one-step loss decrease: $(loss_ok ? "pass" : "not guaranteed / inspect after multiple steps")",
    ], "\n")
end
end

# ╔═╡ c047f29c-0282-4c1d-a798-deb3be5389a8
md"""
## Side-by-side Smoke Status

$(smoke_status_text)

## Plain AE Lux/Enzyme Diagnostics

$(plain_ae_metric_text)

## Precomputed STE Lux/Enzyme Diagnostics

$(precomputed_ste_metric_text)

## Lux/Enzyme Numeric Diagnostics

$(lux_metric_text)

## Fixed-codebook Lux/Enzyme Diagnostics

$(fixed_metric_text)

## Interpretation

$(lux_interpretation_text)

$(fixed_interpretation_text)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Lux = "b2108857-7c20-44ae-9111-449ecde12c47"
NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[compat]
ADTypes = "~1.22.0"
CUDA = "~6.0.0"
Enzyme = "~0.13.140"
EnzymeCore = "~0.8.20"
Flux = "~0.16.10"
Lux = "~1.31.4"
NNlib = "~0.9.34"
Optimisers = "~0.4.7"
Zygote = "~0.7.10"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "52d7cf29b4b6e6ed60dc35e61bccb544f9231e88"

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
git-tree-sha1 = "0761717147821d696c9470a7a86364b2fbd22fd8"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.2"
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
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

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

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "42cd00edaac86f941815fe557c1d01e11913e07c"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.28"
weakdeps = ["ChainRulesCore", "EnzymeCore"]

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

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

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.ExpressionExplorer]]
git-tree-sha1 = "5f1c005ed214356bbe41d442cc1ccd416e510b7e"
uuid = "21656369-7473-754a-2065-74616d696c43"
version = "1.1.4"

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

[[deps.Flux]]
deps = ["ADTypes", "Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "GPUArrays", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "cb318a415a089c337d0c15000d1608cee8434ebf"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.10"

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

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "2dfe3b4b96c6ecbea7c798dfbe96d493fd7a1848"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.17.8"

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
git-tree-sha1 = "9510d7008275fc5b33fc72a73f8fddef0b5430c6"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.11"

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

[[deps.ReactantCore]]
deps = ["ExpressionExplorer", "MacroTools"]
git-tree-sha1 = "5b9e0fe7fb2cf3794fd96ac32bf2732aa4bb9776"
uuid = "a3311ec8-5e00-46d5-b541-4f83e724a433"
version = "0.1.19"

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
# ╠═9bb43b61-b218-40d4-b5e5-27040d587e75
# ╠═32fd3b7b-8407-4472-9e3b-21c749e67233
# ╠═9de1a92d-1892-4fcc-ba48-5466fd7e102d
# ╠═6b078df8-3343-4d41-9ab3-4010f7f2e4c0
# ╠═5360ab64-be7d-477a-83a4-664a2b318933
# ╠═e520e7d7-a45e-45ea-b647-bd79125ae98b
# ╠═e0a22d76-93d7-45ec-a01e-7a5430902016
# ╠═9be3deda-0091-4d0e-b06b-12c38ba1be15
# ╠═e470a196-ba0e-4b48-acb2-89138e316157
# ╠═af7e15d3-d779-4bc0-b219-c6ef863a94de
# ╠═231a5245-f1bf-4722-ba1d-c96eea0e7c0d
# ╠═d7288bdf-53f6-4c58-b492-f7a9a92740c1
# ╠═f80fe0e0-929a-4baa-a097-8f8a9e49547d
# ╠═9239e126-915e-4230-b73b-25486eb0394d
# ╠═83afc7c4-7ed0-432b-af5e-d903ff173255
# ╠═21b88114-7003-4179-bee3-91b03690afae
# ╠═64a08df5-f08c-449f-bb4f-3d083e39950d
# ╠═34cdf4c9-8e01-40c5-836d-9b47d9f43002
# ╠═9fd80194-7112-4bbb-8550-0486fa8160b3
# ╠═b45ad394-958a-43ae-9f1a-2b8ca9df7dcd
# ╟─2b7e8d1f-0f92-4a5f-a5c2-f67e5934619f
# ╠═0f85b13f-4daf-4f0d-b3f7-576d8baaa43c
# ╠═11134e4f-f83b-4ad0-8d17-100a837fe596
# ╠═d8a4e456-e264-4239-96ce-458b70ed098d
# ╠═9dc20d78-0781-4a55-91f8-3356a11ed677
# ╠═c115d647-70d3-4749-9012-b40394a0bddf
# ╠═834836a5-01ea-4b18-a681-b74371b1255c
# ╠═7590f568-3312-4c9f-88e4-7052d192d266
# ╠═d339b18f-428e-4ec6-bc7e-9c33b74c10b2
# ╠═051e45c2-823e-4510-a4fa-5afaa0e260df
# ╠═56720cc7-3eb1-4396-9502-d9848014d82a
# ╠═e04d1604-6622-4263-b3f7-9248657c6c7f
# ╠═4877f866-b98f-46bf-9b51-235566b83bf8
# ╠═d4dbafef-a2ef-4bc6-86f5-c3db0a36b96e
# ╠═1ef9b5c6-8dd6-448a-9afe-7bc3b939ab72
# ╠═3f32f44d-3479-4ffc-ae7e-363e8f28a47d
# ╠═44b33fed-119d-4385-a9c8-3c94a9f814a2
# ╠═e7debd35-07b4-4676-a3c2-365f5e588211
# ╠═ed6b80bf-a6db-41b2-bca8-a260bdc2d707
# ╠═a4614d4f-26fa-4dbd-9022-23da7910d264
# ╠═7fd86a7c-d3fd-4dc3-9784-4b55ef0e9216
# ╠═74a5177f-ad9a-4e2e-acb2-ea16d105e770
# ╠═02d1a13f-e076-4b0a-b6e8-118540d8deec
# ╠═b385ce59-28b9-411d-911e-bf79bacb76cf
# ╠═387b9bfb-28bf-4fd6-8608-599c78ea8e91
# ╠═73612870-4310-4791-a68f-fd25892a9c2a
# ╠═2f24d41d-ae04-4e0d-b9d4-97d3814c905c
# ╠═e4c55070-916d-48b9-b34d-a3ab34ad4062
# ╠═671bf2d2-76eb-48e1-9396-f8e383425339
# ╠═cdc3957f-381b-4946-80e6-a90e9f585044
# ╠═6896023b-ed14-4378-9251-7cfba17c9137
# ╠═a119c1d5-2819-4c72-aefa-bd0e2cfb8604
# ╠═b3b09a8c-3bb0-45e8-a6c9-26598e17049f
# ╠═8dc1484a-0fd6-4f09-8953-a9f30a50b266
# ╠═278431a0-256d-4a8a-8795-238b3c6c1d2f
# ╠═af63db60-b2a4-47a8-bb39-819f704270f2
# ╠═5a6fe71d-1210-4c44-90c2-96321231991d
# ╠═44113888-23fc-4166-abab-ae29d06a09ce
# ╠═91ae0d18-7d9c-42f1-bb83-61789c41ae8a
# ╠═010b27b4-c7c9-4180-b34b-ae327e85c95b
# ╠═59e58140-8741-484c-95e4-e79214a0df77
# ╠═f4099841-fc40-4c4f-84bf-5e79e8fd927e
# ╠═c22f4459-c732-4437-84d7-16e774f667d2
# ╠═f8e98ef1-b2e0-411d-a083-9027867668b9
# ╠═773f4779-86fc-4eff-8db2-bc3284de274f
# ╠═fa01e2d1-e197-44fa-9184-1521b6c2ec95
# ╠═8c365a6b-52c0-47f8-93c4-98d4cbd07004
# ╠═b3fd20d5-36dd-47ce-bd08-d26450e02638
# ╟─c047f29c-0282-4c1d-a798-deb3be5389a8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
