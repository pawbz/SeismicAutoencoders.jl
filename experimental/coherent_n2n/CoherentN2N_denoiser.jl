# Complex-valued conv U-Net denoiser for CoherentN2N.
#
# Unlike PhaseEncoder (experimental/phase_aligner/PhaseAligner_architecture.jl:
# 147-180), which maps a real waveform to a scalar phase, this network maps a
# complex spectrum to a denoised complex spectrum: 2 input channels
# (real, imag) -> conv encoder -> conv decoder -> 2 output channels
# (real, imag). No global-average-pool / scalar head — this is a full
# reconstruction network, trained with Noise2Noise pairs (see
# CoherentN2N_n2n_pairs.jl / CoherentN2N_train_denoiser.jl).
#
# Real/imag are kept as 2 ordinary real-valued channels throughout (Flux has
# no built-in complex Conv/BatchNorm) rather than genuinely complex-valued
# layers (cf. Deep Complex U-Net / DCUNet for speech enhancement, which also
# predicts a multiplicative mask rather than the spectrum directly) — this
# network predicts the denoised spectrum directly. That's a deliberate scope
# choice for this project's data scale (nt ~128-300, tens of traces/station),
# not an oversight.
#
# Encoder/decoder feature maps at matching resolutions are concatenated
# (U-Net skip connections) so the decoder isn't forced to reconstruct fine
# timing/phase detail purely from the bottleneck.

using Flux

const activation_cn2n = x -> leakyrelu(x, 0.1f0)

"""
    ComplexDenoiser

`model(X̂)` maps a `(nt, B)` ComplexF32 batch of spectra to a denoised
`(nt, B)` ComplexF32 batch. Real/imag are stacked as 2 input channels, run
through a symmetric strided-conv encoder / upsample-conv decoder U-Net with
skip connections between matching-resolution stages, then the 2 output
channels are recombined into a complex spectrum.

`encoder[i]` is stage `i` counting from the input (finest resolution) down to
the bottleneck; `decoder[i]` is stage `i` counting from the bottleneck back up
to the output. `decoder[i]` concatenates the skip from encoder stage
`n_stages - i` (one resolution finer than what `decoder[i]` starts from), for
`i < n_stages`. The last decoder stage (`i == n_stages`, which restores full
resolution) has **no** skip — deliberately not a canonical U-Net, which would
skip-connect the raw input itself at the finest level: doing so let the
decoder's output layer see the unprocessed noisy input directly, which
leaked per-trace noise past the bottleneck and measurably hurt the
alternating loop's blind source recovery (`test_outer_loop.jl`) even though
it didn't hurt the isolated per-trace denoising test. Both fields are plain
`Vector`s of `Chain`s — Flux's `@layer` machinery walks `Vector`s of layers
the same way it walks a `Chain`'s internals, so no custom `Functors.jl` code
is needed for parameter collection / `gpu`/`cpu` transfer.
"""
struct ComplexDenoiser{E,D}
    encoder::E
    decoder::D
    n_stages::Int
end
Flux.@layer ComplexDenoiser trainable = (encoder, decoder)

function (m::ComplexDenoiser)(X̂::AbstractMatrix{ComplexF32})
    nt, B = size(X̂)
    # Non-mutating construction (reshape + cat) so this stays Zygote-differentiable
    # (in-place setindex! on arrays is not supported by Zygote's reverse-mode AD).
    re = reshape(real.(X̂), nt, 1, B)
    im_ = reshape(imag.(X̂), nt, 1, B)
    x2 = cat(re, im_; dims=2)   # (nt, 2, B)

    # Skip connections require encoder/decoder feature maps to line up exactly
    # at every stage, not just at the final output — pad up front to the next
    # multiple of 2^n_stages (each stage halves the length under SamePad, so
    # this is the smallest length every stage divides evenly), run the whole
    # U-Net at that padded length, and crop back to nt only at the very end.
    stride_total = 2^m.n_stages
    nt_pad = cld(nt, stride_total) * stride_total
    nt_pad > nt && (x2 = Flux.pad_zeros(x2, (0, nt_pad - nt); dims=1))

    # skips[i] is encoder stage i's output, for i in 1:n_stages-1 (encoder
    # stage n_stages is the bottleneck and is never read as a skip — the
    # first decoder stage starts from it as `h`, not as a concatenated skip;
    # the raw pre-encoder input x2 is deliberately not a skip either — see
    # ComplexDenoiser's docstring). Built as a Tuple via functional
    # accumulation (each step produces a new tuple rather than mutating a
    # shared container) since Zygote's reverse-mode AD doesn't support
    # mutating array assignment, including push!.
    skips = ()
    h = x2
    for i in 1:m.n_stages
        h = m.encoder[i](h)
        i < m.n_stages && (skips = (skips..., h))
    end

    for i in 1:m.n_stages
        h = Flux.upsample_nearest(h; size=(size(h, 1) * 2,))
        if i < m.n_stages
            skip = skips[m.n_stages - i]
            h = cat(h, skip; dims=2)
        end
        h = m.decoder[i](h)
    end

    y2 = h[1:nt, :, :]
    return complex.(y2[:, 1, :], y2[:, 2, :])
end

"""
    build_complex_denoiser(nt; kernels, filters) -> ComplexDenoiser

Build a `ComplexDenoiser` U-Net. `kernels`/`filters` describe the encoder
stages from input to bottleneck (`filters[end]` is the bottleneck width); the
decoder mirrors them in reverse, so it is symmetric by construction — there
is no independent decoder configuration, since every decoder stage must pair
with exactly one matching-resolution encoder stage to concatenate its skip.

Each encoder stage is `Conv(stride=2) + BatchNorm` (halves the length, cf.
`build_complex_denoiser`'s padding-to-`2^n_stages` in `ComplexDenoiser`'s
forward pass). Each decoder stage nearest-upsamples by 2, concatenates the
matching encoder skip (doubling that stage's input channel count — hence
`conv_in = nin + skip_ch` below), then applies `Conv + BatchNorm` (linear,
no `BatchNorm`, on the final stage — a spectrum has no natural bound).
"""
function build_complex_denoiser(nt::Int;
                                 kernels::Vector{Int}=[32, 16, 8],
                                 filters::Vector{Int}=[16, 32, 64])
    @assert length(kernels) == length(filters)
    n_stages = length(kernels)

    encoder = Any[]
    nin = 2
    for (k, nout) in zip(kernels, filters)
        push!(encoder, Chain(Conv((k,), nin => nout, activation_cn2n; pad=SamePad(), stride=2),
                              BatchNorm(nout)))
        nin = nout
    end

    # Decoder stage i (bottleneck -> output) concatenates the skip from
    # skips[n_stages - i] in the forward pass (see ComplexDenoiser) — encoder
    # stage (n_stages - i)'s output, filters[n_stages - i] channels wide —
    # for i < n_stages. The last stage (i == n_stages) has no skip (see
    # ComplexDenoiser's docstring for why), so its Conv input is just the
    # previous decoder stage's output width, unchanged by concatenation.
    rev_kernels = reverse(kernels)
    decoder = Any[]
    for i in 1:n_stages
        is_last = i == n_stages
        out_ch = is_last ? 2 : filters[n_stages - i]
        conv_in = is_last ? nin : nin + filters[n_stages - i]
        act = is_last ? identity : activation_cn2n
        layers = Any[Conv((rev_kernels[i],), conv_in => out_ch, act; pad=SamePad())]
        is_last || push!(layers, BatchNorm(out_ch))
        push!(decoder, Chain(layers...))
        nin = out_ch
    end

    return ComplexDenoiser(encoder, decoder, n_stages)
end
