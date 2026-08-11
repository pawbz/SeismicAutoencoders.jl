# Complex-spectrum denoiser for CoherentN2N — dilated, circular, stride-1 stack.
#
# Maps a complex spectrum to a denoised complex spectrum: 2 input channels
# (real, imag) -> dilated residual conv stack -> 2 output channels (real, imag).
# Real/imag are kept as 2 ordinary real-valued channels throughout (Flux has no
# built-in complex Conv); the network predicts a MULTIPLICATIVE COMPLEX MASK
# applied to the input spectrum, as in Deep Complex U-Net / DCUNet for speech.
# See the mask comment in the ComplexDenoiser call for why the mask formulation
# (rather than direct or residual prediction) is what keeps the time-domain
# response free of a first-sample delta.
# Trained with Noise2Noise pairs (CoherentN2N_n2n_pairs.jl /
# CoherentN2N_train_denoiser.jl).
#
# WHY DILATED + CIRCULAR + STRIDE-1 (not a strided U-Net):
# The previous design was a strided (stride-2) conv U-Net. Its stride-2
# downsampling samples a coarser lattice, which BREAKS translation
# equivariance: a fixed, input-independent position bias appears along the
# frequency axis, and because we `ifft` the output spectrum to time, that bias
# `ifft`s to a NODE at the centre time sample (the trace centre) — an untrained
# net suppressed the centre by ~30x relative to the edges (see the
# untrained-architecture per-sample-RMS diagnostic in CoherentN2N_main.jl).
# The audio/speech field (DCUNet) tolerates the same stride-2 bias because a 2D
# STFT + overlap-add reconstruction averages it away over many frames and huge
# data learns around it — neither of which we have (single global FFT of the
# whole trace, tens of traces/station).
#
# The fix is a WaveNet-style stack of DILATED (1,2,4,8,…) STRIDE-1 convolutions
# with CIRCULAR padding:
#   - stride-1 + dilation preserves translation equivariance (dilation commutes
#     with shift), so there is no privileged position and the centre node is
#     removed by construction (untrained centre/edge RMS ~0.03 -> ~0.9);
#   - dilations grow the receptive field exponentially WITHOUT subsampling, so
#     we keep the multiscale context the U-Net wanted at full resolution;
#   - circular padding is the physically honest boundary for a periodic DFT
#     spectrum (the same periodic/FFT world `shift_spectrum` already lives in),
#     and closes the boundary inequivalence.
# Dilations grow until the receptive field ~covers the trace but stay BELOW
# nt/4 — a dilation approaching nt/2 aliases against the circular period and
# re-introduces a bias (empirically dil..32 at nt=300 regressed the centre).

using Flux

const activation_cn2n = x -> leakyrelu(x, 0.1f0)

"""
    CircDilConv(conv, k, dilation)

A stride-1 1-D `Conv` with **circular** boundary padding at a given `dilation`.
Flux's `Conv(pad=SamePad())` only zero-pads; circular padding makes the
convolution translation-equivariant on the periodic DFT spectrum (no privileged
position -> no centre-time node). The wrapped `conv` is built with `pad=0`;
this layer circular-pads the input by the dilated half-span
(`span = (k-1)*dilation`, split `span÷2` / `span-span÷2`) so the output length
equals the input length exactly.
"""
struct CircDilConv{C}
    conv::C
    k::Int
    dilation::Int
end
Flux.@layer CircDilConv

function (c::CircDilConv)(x)
    span = (c.k - 1) * c.dilation
    pl = span ÷ 2
    pr = span - pl
    L = size(x, 1)
    # pad_circular requires each pad amount < L. For a pathological dilation >=
    # trace length, fall back to zero padding rather than erroring (length is
    # unchanged either way) — the builder keeps dilations below nt/4 so this
    # branch is not hit in normal use.
    #
    # Circular is correct here because ComplexDenoiser runs over the FULL DFT
    # spectrum, which really is periodic (bin nt neighbours bin 1 across DC).
    xp = (pl < L && pr < L) ? Flux.pad_circular(x, (pl, pr); dims=1) :
                              Flux.pad_zeros(x, (pl, pr); dims=1)
    return c.conv(xp)
end

"""
Circular, stride-1 `Conv((k,), cin=>cout, act; dilation)` — length-preserving.

`bias` defaults to `true` (Flux's default) but the OUTPUT projection must pass
`bias=false`; see `build_complex_denoiser`. A conv bias is added identically at
every position along the axis, and this axis is FREQUENCY — so a bias is a
constant offset across all bins, which is exactly a delta function at t=1 in the
time domain. Left in place on the output layer it puts a large spike on the first
sample of every denoised trace.
"""
circ_dil_conv(k::Int, ch::Pair{Int,Int}, act=identity; dilation::Int=1, bias::Bool=true) =
    CircDilConv(Conv((k,), ch, act; pad=0, dilation=dilation, bias=bias), k, dilation)

"""
    ComplexDenoiser

`model(X̂)` maps a `(nt, B)` ComplexF32 batch of spectra to a denoised
`(nt, B)` ComplexF32 batch. Real/imag are stacked as 2 input channels, run
through: an input projection (2 -> width), a stack of dilated **residual**
blocks (each `CircDilConv + BatchNorm + activation`, added to its input), and a
linear output projection (width -> 2), then recombined into a complex spectrum.

All convolutions are stride-1 + circular + dilated, so the whole operator is
translation-equivariant and length-preserving (no down/upsampling, no
crop-to-`2^n`). `blocks` is a plain `Vector` of `Chain`s — Flux's `@layer`
machinery walks `Vector`s of layers like a `Chain`'s internals, so no custom
`Functors.jl` code is needed for parameter collection / `gpu`/`cpu` transfer.
`in_proj` / `out_proj` are single `CircDilConv`s (dilation 1). The output
projection is linear (no BatchNorm) — a spectrum has no natural bound.
"""
struct ComplexDenoiser{I,BL,O}
    in_proj::I
    blocks::BL
    out_proj::O
end
Flux.@layer ComplexDenoiser trainable = (in_proj, blocks, out_proj)

"""
    hermitian_project(Y) -> (nt, B) ComplexF32

Project an arbitrary spectrum onto the Hermitian subspace — the set of spectra
that are FFTs of REAL signals — by averaging each bin with the conjugate of its
mirror partner:

    Y'[k] = (Y[k] + conj(Y[nt-k+2])) / 2

DC (and Nyquist for even `nt`) are their own partners, so the projection reduces
to taking their real part there. This is an orthogonal projection: it discards
exactly the anti-Hermitian component, which is precisely the part
`real(ifft(...))` would otherwise throw away silently.

Applying it means the network CANNOT emit a spectrum whose inverse transform has
an imaginary part — the guarantee we need — while still letting the conv stack
run over the FULL periodic frequency axis, where bin `nt` and bin 1 are genuine
neighbours (they are conjugate partners across DC). That periodicity is what
makes circular padding physically meaningful and keeps the operator's time-domain
response flat; running on a half spectrum instead turns both ends into hard
boundaries and measured ~3.5x WORSE edge bias.
"""
function hermitian_project(Y::AbstractMatrix{ComplexF32})
    nt = size(Y, 1)
    # Mirror index map: 1 -> 1, k -> nt-k+2 for k >= 2. Built by slicing + vcat
    # (non-mutating) so this stays Zygote-differentiable.
    Ymir = vcat(Y[1:1, :], reverse(Y[2:nt, :]; dims=1))
    return (Y .+ conj.(Ymir)) ./ 2f0
end

function (m::ComplexDenoiser)(X̂::AbstractMatrix{ComplexF32})
    nt, B = size(X̂)
    # Run over the FULL spectrum: the frequency axis is genuinely periodic (bin
    # nt neighbours bin 1 across DC), so the circular padding in CircDilConv is
    # physically correct and there is no boundary for artifacts to accumulate at.
    # Hermitian symmetry is imposed afterwards by projection rather than by
    # slicing to a half spectrum — see hermitian_project for why that ordering
    # matters for the edge response.
    #
    # Non-mutating construction (reshape + cat) so this stays Zygote-differentiable
    # (in-place setindex! on arrays is not supported by Zygote's reverse-mode AD).
    re = reshape(real.(X̂), nt, 1, B)
    im_ = reshape(imag.(X̂), nt, 1, B)
    x2 = cat(re, im_; dims=2)   # (nt, 2, B)

    h = m.in_proj(x2)           # (nt, width, B)
    for blk in m.blocks
        h = h .+ blk(h)         # residual dilated block (length + width preserved)
    end
    y2 = m.out_proj(h)          # (nt, 2, B)

    # MULTIPLICATIVE COMPLEX MASK — the network emits a per-bin complex gain that
    # multiplies the input spectrum, rather than emitting the spectrum (or a
    # correction to it) directly.
    #
    # This is the same formulation as Deep Complex U-Net speech denoising
    # (`output = mask * orig_x` in madhavmk/Noise2Noise-audio_denoising), and it
    # is what structurally removes the first-sample artifact:
    #
    #   - direct  Y = f(X̂)      — output unconstrained; a constant across all
    #                             bins IS a delta at t=1, and nothing forbids it.
    #   - residual Y = X̂ + f(X̂) — better (starts near identity) but f can still
    #                             add a constant across bins, so the delta
    #                             survives.
    #   - mask    Y = m(X̂) ⊙ X̂ — output is pointwise PROPORTIONAL to the input.
    #                             Where the data has no energy the output has
    #                             none, so a frequency-flat offset cannot be
    #                             manufactured and the t=1 delta is impossible.
    #
    # This matters for tapered receiver functions specifically: the Tukey window
    # forces the data to ~zero at both trace edges, and a mask cannot invent
    # energy there, whereas direct/residual prediction demonstrably did.
    #
    # The mask must be BOUNDED, not just centred on 1. `out_proj` is a linear
    # conv whose raw output is O(1)-or-larger at initialization, so a bare
    # `1 .+ y2` mask still swings wildly bin-to-bin and reintroduces the very
    # first-sample artifact the mask is meant to prevent (measured spike 3.9 on
    # real data). DCUNet bounds its mask for the same reason.
    #
    # Parameterize as magnitude ⊙ unit-phase, with BOTH parts bounded:
    #   |mask| = 2·sigmoid(·)  ∈ (0, 2), = 1 when the channel is 0
    #   ∠mask  = π·tanh(·)     ∈ (-π, π), = 0 when the channel is 0
    #
    # Bounding the PHASE matters as much as bounding the magnitude. `out_proj` is
    # a linear conv: at initialization its raw output spans roughly ±20, so feeding
    # that straight into cos/sin means the phase sweeps ~5.6 FULL TURNS across the
    # spectrum before a single training step. That is not "identity at init" — it
    # is a random phase scramble, exactly the thing that produced the time-domain
    # edge artifacts, and its oscillating derivative makes the loss surface
    # pathological (observed: NaN from epoch 2 on real data).
    #
    # tanh keeps the phase in one principal branch, so the derivative is smooth
    # and bounded, and the whole operator is ≈ identity at init (both channels ≈ 0
    # ⇒ mask ≈ 1). The mask can still attenuate a bin to zero, pass it through,
    # amplify up to 2x, and rotate to any phase — the full useful range — it just
    # cannot wind arbitrarily or explode.
    mag = 2f0 .* Flux.sigmoid.(y2[:, 1, :])
    ph = Float32(π) .* tanh.(y2[:, 2, :])
    mask = complex.(mag .* cos.(ph), mag .* sin.(ph))
    return hermitian_project(mask .* X̂)
end

"""
    dilation_schedule(nt; base=2) -> Vector{Int}

Exponentially growing dilations `1, base, base^2, …` for a trace of length
`nt`, truncated so the largest dilation stays **at or below `nt÷8`**. A
dilation approaching `nt/2` aliases against the circular period and
re-introduces a position bias; empirically the flat-response sweet spot for
`nt≈128–300` is a max dilation around `nt÷8` (both `nt÷4` — too large,
aliasing — and `nt÷16` — receptive field too small — gave a worse per-sample
RMS flatness). Always returns at least `[1]`.
"""
function dilation_schedule(nt::Int; base::Int=2)
    cap = max(1, nt ÷ 8)
    dils = Int[1]
    d = base
    while d <= cap
        push!(dils, d)
        d *= base
    end
    return dils
end

"""
    build_complex_denoiser(nt; kernels, filters, kernel_size, dilations) -> ComplexDenoiser

Build a dilated-circular `ComplexDenoiser`. Kept the same positional/`kernels`/
`filters` signature as the previous strided-U-Net builder for drop-in
compatibility with existing callers (notebook, tests), but the internals are a
WaveNet-style dilated residual stack (see the file header / `ComplexDenoiser`
for why). The legacy args are reinterpreted:

- `filters` — channel widths; the stack runs at `width = maximum(filters)`
  channels (a single width, since there is no encoder/decoder pyramid now).
- `kernels` — accepted for signature compatibility but **ignored**: a small
  kernel (`kernel_size`, default 3) is used regardless, because the receptive
  field must come from DILATION, not kernel width. Empirically larger kernels
  hurt the flat response (nt=300: centre/edge ~0.59 at k=3 vs ~0.19 at k=5).

New keyword args (preferred going forward):

- `kernel_size` — conv kernel length (default 3; keep small).
- `dilations` — explicit dilation schedule; defaults to `dilation_schedule(nt)`
  (`1,2,4,8,…` capped at `nt÷8`). The number of residual blocks equals
  `length(dilations)`.
"""
function build_complex_denoiser(nt::Int;
                                 kernels::Vector{Int}=[32, 16, 8],
                                 filters::Vector{Int}=[16, 32, 64],
                                 kernel_size::Int=3,
                                 dilations::Union{Nothing,Vector{Int}}=nothing)
    width = maximum(filters)
    k = kernel_size
    dils = dilations === nothing ? dilation_schedule(nt) : dilations

    in_proj = circ_dil_conv(k, 2 => width, activation_cn2n; dilation=1)

    blocks = Any[]
    for d in dils
        push!(blocks, Chain(circ_dil_conv(k, width => width, activation_cn2n; dilation=d),
                            BatchNorm(width)))
    end

    # Linear, no BatchNorm, no bias. The convolution axis is FREQUENCY, so a bias
    # is a constant added to every bin — exactly a delta at t=1 in the time
    # domain. Dropping it removes that failure mode by construction. (It is not on
    # its own sufficient: the dominant artifact came from an unbounded, hot output
    # projection — see below and the mask comment in the forward pass.)
    out_proj = circ_dil_conv(k, width => 2, identity; dilation=1, bias=false)

    # SMALL-INIT the output projection so the mask starts at ≈ identity.
    #
    # With Flux's default (Glorot) init the raw out_proj output spans ≈ ±20 at
    # initialization. Pushed through the mask's sigmoid/tanh that is fully
    # saturated — the "identity at init" property is lost (measured
    # ‖Y-X‖/‖X‖ = 2.0, i.e. the untrained operator distorts the input more than
    # it preserves it) and training starts on a pathological loss surface.
    # Scaling the initial weights by 0.01 brings the pre-activation to ≈ ±0.3, so
    # sigmoid/tanh sit in their linear region: mask ≈ 1, ‖Y-X‖/‖X‖ ≈ 0.14, and
    # gradients flow. This changes only the STARTING POINT — every weight remains
    # free to grow, so no expressivity is given up.
    out_proj = CircDilConv(
        Conv(out_proj.conv.weight .* 0.01f0, out_proj.conv.bias, identity;
             pad=0, dilation=out_proj.dilation),
        out_proj.k, out_proj.dilation)

    return ComplexDenoiser(in_proj, blocks, out_proj)
end
