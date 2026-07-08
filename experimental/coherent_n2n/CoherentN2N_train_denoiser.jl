# Noise2Noise training loop for the complex denoiser.
#
# Structurally follows train_phase_aligner's plumbing
# (experimental/phase_aligner/PhaseAligner_architecture.jl:422-477):
# Optimisers.setup + Adam, cosine warm-restart LR, NamedTuple loss history,
# periodic logging. Not a copy of its two-phase equivariance semantics —
# just the training-loop skeleton (optimizer/schedule/logging plumbing).

using Flux
using Optimisers
using Statistics: mean

"""
    complex_mse_loss(pred, target) -> Float32

Mean squared error over complex values: `mean(abs2, pred .- target)`.
"""
complex_mse_loss(pred::AbstractMatrix{<:Complex}, target::AbstractMatrix{<:Complex}) =
    Float32(mean(abs2, pred .- target))

Base.@kwdef struct CoherentN2N_Denoiser_Training_Para
    n_samples_per_epoch::Int = 512
    batchsize::Int            = 64
    nepoch::Int                = 200
    initial_lr::Float64        = 0.001
    restart_period::Int         = 50
    nprint::Int                 = 20
end

"""Log if on print interval."""
function log_denoiser_epoch(epoch::Int, loss::Real, nprint::Int)
    mod(epoch, nprint) == 0 || return nothing
    @info "Denoiser epoch $epoch" loss = round(Float64(loss); digits=6)
    return nothing
end

"""
    train_denoiser!(model::ComplexDenoiser, X̂_aligned::AbstractMatrix{ComplexF32}, training_para) -> loss_history

Train `model` with Noise2Noise pairs randomly resampled from `X̂_aligned`'s
`R` aligned earthquake spectra (see `sample_n2n_pairs`). Each epoch: redraw
`n_samples_per_epoch` (input, target) pairs, then take gradient steps over
minibatches of them. Returns a `(; train_loss)` NamedTuple history.
"""
function train_denoiser!(model::ComplexDenoiser, X̂_aligned::AbstractMatrix{ComplexF32},
                          training_para::CoherentN2N_Denoiser_Training_Para=CoherentN2N_Denoiser_Training_Para();
                          rng::AbstractRNG=Random.default_rng())
    opt_state = Optimisers.setup(Optimisers.Adam(training_para.initial_lr), model)
    T = training_para.restart_period
    h = (; train_loss=Float32[])

    for epoch in 1:training_para.nepoch
        input, target = sample_n2n_pairs(X̂_aligned, training_para.n_samples_per_epoch; rng=rng)
        # Reshuffle which look is input vs. target each epoch.
        if rand(rng, Bool)
            input, target = target, input
        end

        epoch_losses = Float32[]
        n = size(input, 2)
        bs = min(training_para.batchsize, n)
        for start in 1:bs:n
            stop = min(start + bs - 1, n)
            xb = input[:, start:stop]
            yb = target[:, start:stop]
            loss, grads = Flux.withgradient(model) do m
                complex_mse_loss(m(xb), yb)
            end
            Optimisers.update!(opt_state, model, grads[1])
            push!(epoch_losses, loss)
        end

        train_loss = mean(epoch_losses)
        push!(h.train_loss, train_loss)
        log_denoiser_epoch(epoch, train_loss, training_para.nprint)

        t_cos = mod1(epoch, T)
        eta = training_para.initial_lr * (1 + cos(π * t_cos / T)) / 2
        Optimisers.adjust!(opt_state; eta=eta)
    end

    return h
end
