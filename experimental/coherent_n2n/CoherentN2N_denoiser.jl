# Complex-valued conv encoder/decoder denoiser for CoherentN2N.
#
# Unlike PhaseEncoder (experimental/phase_aligner/PhaseAligner_architecture.jl:
# 147-180), which maps a real waveform to a scalar phase, this network maps a
# complex spectrum to a denoised complex spectrum: 2 input channels
# (real, imag) -> conv encoder -> conv decoder -> 2 output channels
# (real, imag). No global-average-pool / scalar head — this is a full
# reconstruction network, trained with Noise2Noise pairs (see
# CoherentN2N_n2n_pairs.jl / CoherentN2N_train_denoiser.jl).

using Flux

const activation_cn2n = x -> leakyrelu(x, 0.1f0)

"""
    ComplexDenoiser

`model(X̂)` maps a `(nt, B)` ComplexF32 batch of spectra to a denoised
`(nt, B)` ComplexF32 batch, by stacking real/imag as 2 input channels,
running a Flux `Chain` (encoder+decoder), and recombining the 2 output
channels back into a complex spectrum.
"""
struct ComplexDenoiser{C}
    chain::C
end
Flux.@layer ComplexDenoiser trainable = (chain,)

function (m::ComplexDenoiser)(X̂::AbstractMatrix{ComplexF32})
    nt, B = size(X̂)
    # Non-mutating construction (reshape + cat) so this stays Zygote-differentiable
    # (in-place setindex! on arrays is not supported by Zygote's reverse-mode AD).
    re = reshape(real.(X̂), nt, 1, B)
    im_ = reshape(imag.(X̂), nt, 1, B)
    x2 = cat(re, im_; dims=2)   # (nt, 2, B)

    y2 = m.chain(x2)   # (nt', 2, B) — nt' may exceed nt when nt isn't evenly
    # divisible by the encoder's total stride (SamePad's ceil(nt/2) per
    # stage isn't exactly inverted by Upsample(2)'s exact doubling; e.g.
    # nt=300 -> 304 with 3 stride-2 stages). Crop back to nt rather than
    # constraining callers to power-of-2-friendly trace lengths.
    y2 = y2[1:nt, :, :]
    return complex.(y2[:, 1, :], y2[:, 2, :])
end

"""
    build_complex_denoiser(nt; enc_kernels, enc_filters, dec_kernels, dec_filters) -> ComplexDenoiser

Build a `ComplexDenoiser`. Encoder: strided 1D `Conv`+`BatchNorm` chain
(mirrors the conv-chain-building pattern of `build_phase_encoder`,
`PhaseAligner_architecture.jl:164-180`, but keeps 2 channels throughout and
does not pool to a scalar). Decoder: mirrored upsample+conv chain back to
`nt` samples, 2 output channels (real, imag), no final activation (linear
output — a spectrum has no natural bound).

`enc_kernels`/`enc_filters` and `dec_kernels`/`dec_filters` should have the
same length (number of stages); the decoder upsamples by the same total
factor the encoder strides down by.
"""
function build_complex_denoiser(nt::Int;
                                 enc_kernels::Vector{Int}=[32, 16, 8],
                                 enc_filters::Vector{Int}=[16, 32, 64],
                                 dec_kernels::Vector{Int}=[8, 16, 32],
                                 dec_filters::Vector{Int}=[32, 16, 2])
    @assert length(enc_kernels) == length(enc_filters)
    @assert length(dec_kernels) == length(dec_filters)

    layers = Any[]
    nin = 2
    for (k, nout) in zip(enc_kernels, enc_filters)
        push!(layers, Conv((k,), nin => nout, activation_cn2n; pad=SamePad(), stride=2))
        push!(layers, BatchNorm(nout))
        nin = nout
    end
    for (i, (k, nout)) in enumerate(zip(dec_kernels, dec_filters))
        is_last = i == length(dec_kernels)
        push!(layers, Upsample(2))
        act = is_last ? identity : activation_cn2n
        push!(layers, Conv((k,), nin => nout, act; pad=SamePad()))
        is_last || push!(layers, BatchNorm(nout))
        nin = nout
    end
    chain = Chain(layers...)
    return ComplexDenoiser(chain)
end
