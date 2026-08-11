# Bispectrum-inversion multireference alignment (MRA) for coherent signal recovery.
#
# The model: every trace is a circularly shifted, noisy copy of one unknown
# source, `y_i = T_{ℓ_i} x + ξ_i`. The CoherentN2N path attacks this by
# *estimating the shifts* `ℓ_i` and stacking; that is provably fragile at low
# SNR, because the per-trace lag estimate degrades faster than stacking recovers.
#
# This module never estimates a shift. It recovers `x` from two shift-INVARIANT
# moments of the data:
#   - the power spectrum   P[k] = E|X̂(k)|²                → Fourier MAGNITUDES a[k]
#   - the bispectrum       B[k1,k2] = E X̂(k1)X̂(k2)conj(X̂(k1+k2)) → Fourier PHASES φ[k]
# Both are invariant because a shift multiplies X̂(k) by a unit phase that cancels
# in each expression. Reference: Bendory, Boumal, Ma, Zhao & Singer,
# "Bispectrum Inversion with Application to Multireference Alignment",
# IEEE TSP 2018 (arXiv:1705.00641).
#
# THE NO-WRAP TRIPLE RULE (the load-bearing detail — do not "optimize" it away).
# A sub-sample shift is a phase ramp exp(-i2π f_k τ), and it cancels in
# `p[k1]p[k2]conj(p[k1+k2])` only where the frequency axis is ADDITIVE, i.e.
# f[k1] + f[k2] == f[k1+k2]. FFTW's `fftfreq` is SIGNED (+0.5 wraps to -0.5), so
# this fails for every triple whose `k1+k2` crosses Nyquist — measured 1024 of
# 4096 index pairs at nt=64, worst error 1.0. Capping `kmax` at nt÷2-1 does NOT
# fix it (measured relative error 0.84, identical to the full band). What fixes
# it is restricting to triples with `k1 + k2 <= kmax`, so the third index is used
# DIRECTLY and never wraps: measured error 7e-16 for every shift tested. Those
# are exactly the triples frequency marching consumes anyway, so the restriction
# is free. Integer circular shifts are invariant regardless (measured 3.7e-16).
#
# LIMITATION — per-trace polarity. The bispectrum is degree 3 in the data, so a
# per-trace gain `g` scales it by `g³`. For a polarity flip `g = ±1` that is
# `g³ = g`, i.e. flips do NOT cancel — the third-order average cancels toward
# zero instead of accumulating. Measured max|B| at 1.7-9.5% of its unit-gain
# value, with the ratio SHRINKING as more traces are added: a bias that data
# cannot fix. Signed recovery is correspondingly unreliable (measured correlation
# ~0.36 against a clean 0.99).
#
# Note the failure is subtler than "output is noise": the recovered waveform
# SHAPE often survives up to a global sign (|corr| ~0.98), and because the power
# spectrum is degree 2 (`g² = 1`) it is untouched, so `a` still looks perfectly
# healthy. What is actually lost is the absolute polarity and phase — precisely
# the scientific target here. Resolve polarity upstream, or restrict to
# unit-gain data.
#
# Everything here is CPU Float32/ComplexF32: this is moment estimation plus a
# small smooth optimization, so there is nothing for a GPU to do.

using FFTW
using Statistics: mean, median

"""
    fidx(k, nt) -> Int

Map a **0-based** DFT frequency index `k` to its **1-based** Julia array index,
wrapping circularly. Uses `mod` rather than `%` so negative `k` wraps correctly:
`fidx(-1, nt) == nt`, which is what makes Hermitian symmetrization a one-liner.

Throughout this file frequencies are reasoned about 0-based (`k = 0` is DC) and
stored 1-based (`a[k+1]`, `B[k1+1, k2+1]`). This helper is the ONLY place the
wrap-around is expressed.
"""
@inline fidx(k::Integer, nt::Integer) = mod(k, nt) + 1

"""
    power_spectrum_estimate(Y; noise_ps=nothing, sigma2=nothing, tail_frac=0.25f0)
        -> (; a, P, sigma2)

Shift-invariant estimate of the Fourier **magnitudes** of the common signal.

`Y` is either an `(nt, N)` `ComplexF32` matrix of trace spectra (used as-is) or
an `(nt, N)` `Float32` matrix of real traces (FFT'd internally along dim 1).
Columns are traces, matching the `(nt, R)` gather convention used elsewhere.

Computes `P[k] = mean_i |Y[k,i]|²`, removes the additive-noise bias, and returns
`a = sqrt.(max.(P, 0f0))` (length `nt`).

Debiasing, in precedence order:
- `noise_ps` — an explicit length-`nt` noise power spectrum `E|ξ̂(k)|²` (use this
  for colored noise), subtracted directly.
- `sigma2` — white noise of per-sample variance σ². **The subtracted quantity is
  `nt*sigma2`, not `sigma2`.** FFTW's `fft` is unnormalized, so white noise has
  `E|ξ̂(k)|² = nt·σ²` at every `k` (verified numerically: ratio 1.0004). Dropping
  the `nt` under-subtracts by a factor of `nt` and is the classic bug here.
- neither — σ² is estimated from the high-frequency tail: the **median** of
  `P[k]/nt` over the top `tail_frac` of the band up to Nyquist. Median rather
  than mean so residual signal leaking into the tail does not inflate it. This
  assumes the signal is band-limited below the tail; if it is not, σ̂² is too
  large and `max(P, 0)` clamps those bins to zero magnitude, which is a safe
  failure (they simply get zero weight downstream).

The `max.(P, 0f0)` clamp is mandatory, not cosmetic: noise makes `P` slightly
negative at suppressed frequencies and `sqrt` of a negative `Float32` throws.
"""
function power_spectrum_estimate(Y::AbstractMatrix{ComplexF32};
                                  noise_ps::Union{Nothing,AbstractVector}=nothing,
                                  sigma2::Union{Nothing,Real}=nothing,
                                  tail_frac::Real=0.25f0)
    nt = size(Y, 1)
    P = Float32.(vec(mean(abs2.(Y), dims=2)))

    σ2_used = 0f0
    if noise_ps !== nothing
        @assert length(noise_ps) == nt "noise_ps must have length nt=$nt"
        P .-= Float32.(noise_ps)
    elseif sigma2 !== nothing
        σ2_used = Float32(sigma2)
        P .-= Float32(nt) * σ2_used            # nt factor: fft is unnormalized
    else
        nyq = nt ÷ 2
        lo = max(2, round(Int, (1 - Float32(tail_frac)) * nyq))
        σ2_used = Float32(median(@view P[lo+1:nyq+1]) / nt)
        P .-= Float32(nt) * σ2_used
    end

    return (; a=sqrt.(max.(P, 0f0)), P, sigma2=σ2_used)
end

function power_spectrum_estimate(D::AbstractMatrix{Float32}; kwargs...)
    return power_spectrum_estimate(ComplexF32.(fft(D, 1)); kwargs...)
end

"""
    resolve_kmax(a, nt; kmax=nothing, band_fraction=0.01f0) -> Int

Choose the reliable frequency band. The bispectrum is meaningless where the
magnitude is ~0 (its phase is pure noise), and the cost is `O(kmax²·N)`, so
bandlimiting buys both accuracy and speed.

`kmax === nothing` picks the largest `k` whose magnitude still exceeds
`band_fraction * maximum(a)`. The result is hard-capped at `nt÷2 - 1`, keeping
the third index of every triple strictly below Nyquist.
"""
function resolve_kmax(a::AbstractVector{Float32}, nt::Integer;
                      kmax::Union{Nothing,Integer}=nothing,
                      band_fraction::Real=0.01f0)
    cap = nt ÷ 2 - 1
    k = if kmax !== nothing
        min(Int(kmax), cap)
    else
        thr = Float32(band_fraction) * maximum(a)
        last = 2
        for kk in 2:cap
            a[kk+1] >= thr && (last = kk)
        end
        last
    end
    k >= 2 || throw(ArgumentError("kmax resolved to $k; need >= 2 to form any phase equation"))
    return k
end

"""
    bispectrum_estimate(Y; kmax=nothing, a=nothing, band_fraction=0.01f0,
                        noise_bias=nothing) -> (; B, kmax)

Shift-invariant estimate of the third-order moment carrying the Fourier
**phases**: `B[k1,k2] = mean_i Y[k1,i] Y[k2,i] conj(Y[k1+k2,i])`, returned as a
`(kmax+1, kmax+1)` `ComplexF32` matrix indexed `B[k1+1, k2+1]` for 0-based
`k1, k2`.

Only **no-wrap** triples are filled: `k1 >= 1`, `k2 >= k1`, and `k1 + k2 <= kmax`,
so the third index `k1+k2+1` is used directly and never wraps past Nyquist. See
the file header for why this — and not a `kmax` cap — is what makes sub-sample
shifts exactly invariant. Entries outside that set stay zero; `k1 = 0` is
excluded because `B[0,k2] = X̂₀|X̂_{k2}|²` carries only `φ₀`, which the gauge
already pins.

`B[k1,k2] == B[k2,k1]` exactly, so only the upper triangle is computed and then
mirrored — halving the work. The inner loop runs over traces, touching three
fixed rows of `Y`, which is cache-friendly and allocation-free.

`noise_bias`, if given, is subtracted from `B`. The default `nothing` is
correct for additive noise that is zero-mean and independent across traces: such
noise has zero third moment, so it contributes no bias — only `O(1/√N)` variance.
That is precisely why the bispectrum is the right statistic for this problem.
"""
function bispectrum_estimate(Y::AbstractMatrix{ComplexF32};
                              kmax::Union{Nothing,Integer}=nothing,
                              a::Union{Nothing,AbstractVector{Float32}}=nothing,
                              band_fraction::Real=0.01f0,
                              noise_bias::Union{Nothing,AbstractMatrix}=nothing)
    nt, N = size(Y)
    kk = if kmax !== nothing
        min(Int(kmax), nt ÷ 2 - 1)
    elseif a !== nothing
        resolve_kmax(a, nt; band_fraction=band_fraction)
    else
        nt ÷ 2 - 1
    end
    kk >= 2 || throw(ArgumentError("kmax=$kk too small; need >= 2"))

    B = zeros(ComplexF32, kk + 1, kk + 1)
    invN = 1f0 / N
    @inbounds for k1 in 1:kk, k2 in k1:kk
        k1 + k2 > kk && continue          # no-wrap only — direct index, no mod
        i1 = k1 + 1; i2 = k2 + 1; i3 = k1 + k2 + 1
        acc = zero(ComplexF32)
        for i in 1:N
            acc += Y[i1, i] * Y[i2, i] * conj(Y[i3, i])
        end
        v = acc * invN
        B[i1, i2] = v
        B[i2, i1] = v                     # symmetry B[k1,k2] == B[k2,k1]
    end

    if noise_bias !== nothing
        @assert size(noise_bias) == size(B) "noise_bias must be $(size(B))"
        B .-= ComplexF32.(noise_bias)
    end
    return (; B, kmax=kk)
end

function bispectrum_estimate(D::AbstractMatrix{Float32}; kwargs...)
    return bispectrum_estimate(ComplexF32.(fft(D, 1)); kwargs...)
end

"""
    invert_phases_freq_marching(B, a; kmax=size(B,1)-1) -> Vector{Float32}

Recover Fourier phases `φ[k]` (0-based `k`, stored at `φ[k+1]`, length `kmax+1`)
by **frequency marching**.

Gauge: `φ[0] = φ[1] = 0`. Fixing `φ[0]` removes a global sign/DC phase; fixing
`φ[1]` removes the linear phase ramp, i.e. the global circular shift — which is
genuinely unrecoverable, since every shifted copy of the signal has the same
bispectrum. Recovery is therefore always "up to a circular shift".

For each `k = 2:kmax`, every split `k = k1 + k2` gives an independent estimate
`φ[k] ≈ φ[k1] + φ[k2] - angle(B[k1,k2])`. Rather than trusting one pair, all
splits are averaged as **unit complex numbers** — the correct circular mean, and
immune to the ±π wrapping that would wreck a plain arithmetic mean of angles.
Each split is weighted by `a[k1]a[k2]a[k]`, the expected `|B|` magnitude, so
low-amplitude splits whose phase is unreliable contribute little.

Looping `k1 in 1:(k÷2)` with `k2 = k - k1` enumerates each unordered split
exactly once, so no `k2 >= k1` guard is needed.

Weakness worth knowing: marching is sequential, so an error at low `k`
propagates into every higher `k`. It is fast and an excellent initializer, but
it is greedy, not a global optimum — see [`invert_phases_sync`](@ref).
"""
function invert_phases_freq_marching(B::AbstractMatrix{ComplexF32},
                                      a::AbstractVector{Float32};
                                      kmax::Integer=size(B, 1) - 1)
    φ = zeros(Float32, kmax + 1)
    @inbounds for k in 2:kmax
        num = zero(ComplexF32)
        den = 0f0
        for k1 in 1:(k ÷ 2)
            k2 = k - k1
            w = a[k1+1] * a[k2+1] * a[k+1]
            w <= 0 && continue
            num += w * cis(φ[k1+1] + φ[k2+1] - angle(B[k1+1, k2+1]))
            den += w
        end
        φ[k+1] = den > 0 ? Float32(angle(num)) : 0f0
    end
    return φ
end

"""
    invert_phases_sync(B, a; kmax=size(B,1)-1, n_iters=300, init=nothing)
        -> (; φ, n_steps, obj)

Globally-refined alternative to frequency marching: minimize the weighted
phase-consistency objective over all no-wrap triples `(k1, k2, k3=k1+k2)`,

    f(θ) = Σ_n 2 w_n (1 - cos(θ_{k1} + θ_{k2} - θ_{k3} - β_n)),
    β_n = angle(B[k1,k2]),   w_n = a[k1]a[k2]a[k3]  (normalized to sum 1)

which is the standard smooth relaxation of `Σ w_n |u_{k1}u_{k2}conj(u_{k3}) - e^{iβ_n}|²`
restricted to the unit circle. Gradient: `∂f/∂θ_{k1} = ∂f/∂θ_{k2} = +2w_n sin(δ_n)`
and `∂f/∂θ_{k3} = -2w_n sin(δ_n)`, with `δ_n = θ_{k1}+θ_{k2}-θ_{k3}-β_n`.

**Design note.** The literature reaches for a leading-eigenvector
"phase-synchronization" initializer here. That does not apply cleanly: the
relation `u_{k1}u_{k2}conj(u_{k1+k2})` is irreducibly *three-body*, so forcing it
into a two-body Hermitian matrix `M[i,j] ~ u_i conj(u_j)` leaves a third unknown
factor on the right-hand side — it does not close without already knowing the
phases. A genuine rank-1 completion needs the `(kmax+1)²`-dimensional lifted
variable, out of proportion for this problem size. Instead we initialize from
[`invert_phases_freq_marching`](@ref) (or `init`) and refine, which measured
strictly better than random restarts in every trial.

Parametrizing directly in phase `θ` keeps `|u| = 1` implicit, so there is no
projection step and no Wirtinger-derivative sign trap.

The gauge `θ[0] = θ[1] = 0` is held by zeroing those two gradient entries every
iteration, which is cleaner than re-projecting afterwards.

**Armijo backtracking is required** — with a fixed step this iteration diverges
and can return something worse than its own initializer. The step is re-initialized
to 1 on entry (never carried across calls) and the loop breaks once it collapses
below 1e-14. `n_steps` counts accepted steps and is returned so a caller can
detect a stalled solve rather than silently receiving the initializer back.
"""
function invert_phases_sync(B::AbstractMatrix{ComplexF32},
                             a::AbstractVector{Float32};
                             kmax::Integer=size(B, 1) - 1,
                             n_iters::Integer=300,
                             init::Union{Nothing,AbstractVector{Float32}}=nothing)
    # Flatten the no-wrap triples once: recomputing angle(B) inside the loop
    # would dominate the runtime.
    k1s = Int[]; k2s = Int[]; k3s = Int[]; ws = Float32[]; βs = Float32[]
    for k1 in 1:kmax, k2 in k1:kmax
        k1 + k2 > kmax && continue
        w = a[k1+1] * a[k2+1] * a[k1+k2+1]
        w <= 0 && continue
        push!(k1s, k1 + 1); push!(k2s, k2 + 1); push!(k3s, k1 + k2 + 1)
        push!(ws, w); push!(βs, Float32(angle(B[k1+1, k2+1])))
    end
    θ = init === nothing ? invert_phases_freq_marching(B, a; kmax=kmax) : copy(init)
    isempty(ws) && return (; φ=θ, n_steps=0, obj=0f0)
    ws ./= sum(ws)

    function objective(t)
        s = 0f0
        @inbounds for n in eachindex(ws)
            s += 2f0 * ws[n] * (1f0 - cos(t[k1s[n]] + t[k2s[n]] - t[k3s[n]] - βs[n]))
        end
        s
    end
    function gradient!(g, t)
        fill!(g, 0f0)
        @inbounds for n in eachindex(ws)
            d = 2f0 * ws[n] * sin(t[k1s[n]] + t[k2s[n]] - t[k3s[n]] - βs[n])
            g[k1s[n]] += d; g[k2s[n]] += d; g[k3s[n]] -= d
        end
        g[1] = 0f0                      # gauge φ[0] = 0
        length(g) >= 2 && (g[2] = 0f0)  # gauge φ[1] = 0
        g
    end

    g = zeros(Float32, kmax + 1)
    f = objective(θ)
    step = 1f0                          # re-initialized per call, never carried
    n_steps = 0
    θtry = similar(θ)
    for _ in 1:n_iters
        gradient!(g, θ)
        gn = sum(abs2, g)
        gn <= 0 && break
        accepted = false
        while step > 1f-14
            @. θtry = θ - step * g
            ft = objective(θtry)
            if ft <= f - 1f-4 * step * gn      # Armijo sufficient decrease
                copyto!(θ, θtry); f = ft
                step *= 2f0; n_steps += 1; accepted = true
                break
            end
            step *= 0.5f0
        end
        accepted || break
    end
    return (; φ=θ, n_steps, obj=f)
end

"""
    gauge_phases!(φ; kmax=length(φ)-1) -> φ

Fix the two-parameter gauge in place: subtract the constant `φ[0]` and the linear
ramp `k·φ[1]`, then wrap to `(-π, π]`. Removing a linear phase ramp *is* removing
a circular shift, so this pins the otherwise-unrecoverable global alignment.

Caveat: pinning the ramp via `φ[1]` alone is only well-conditioned when `a[1]` is
significant. For a bandpass wavelet it need not be (measured `a[1]/max(a) ≈ 0.21`),
which is why comparisons against a known truth should fit the residual ramp over
a magnitude-weighted band rather than reading it off `k = 1`.
"""
function gauge_phases!(φ::AbstractVector{Float32}; kmax::Integer=length(φ) - 1)
    c = φ[1]
    r = length(φ) >= 2 ? φ[2] - c : 0f0
    @inbounds for k in 0:kmax
        φ[k+1] -= c + k * r
        φ[k+1] = mod(φ[k+1] + Float32(π), 2f0 * Float32(π)) - Float32(π)
    end
    return φ
end

"""
    center_signal(x) -> (; x, lag)

Rotate a recovered signal so its energy sits at the **middle sample**, `nt÷2`.

The absolute time origin is *not identifiable* from the moments: the power
spectrum and the bispectrum are both exactly invariant to a global circular
shift — that invariance is what lets the method skip alignment — so a gather of
`T_{ℓᵢ}x + ξᵢ` constrains `x` only up to a shift. Nothing can recover the origin;
a convention can only *declare* one.

Centring is that declaration, and it is a good one: it depends on the recovered
signal alone, so it introduces no estimation error and cannot be corrupted by
noise. The centre of mass is computed *circularly* (via the phase of the first
harmonic of `x²`), which is the only correct way to average positions on a ring —
a plain arithmetic mean of sample indices would be meaningless for a wavelet
straddling the wrap point. Energy `x²` is used rather than `x` so that an
oscillatory wavelet with near-zero mean still has a well-defined location.

Returns the rotated signal and the `lag` applied, so the rotation is auditable
and reversible. Only the position changes, never the shape.
"""
function center_signal(x::AbstractVector{Float32})
    nt = length(x)
    # Circular centre of mass: treat sample t as angle 2πt/nt on the unit circle,
    # weight by energy, and read the resultant's angle. Immune to wrap-around.
    w = Float32.(abs2.(x))
    z = sum(w[t+1] * cis(2f0 * Float32(π) * t / nt) for t in 0:nt-1)
    com = mod(angle(z) * nt / (2f0 * Float32(π)), nt)     # in [0, nt)
    lag = round(Int, nt ÷ 2 - com)
    return (; x=circshift(x, lag), lag)
end

"""
    anchor_to_data(x, traces) -> (; x, lag)

Rotate a recovered signal into the **data's own time coordinates**, by matching
it against the raw stack of `traces`.

Alternative to [`center_signal`](@ref) for when the input gather is already
roughly windowed on the arrival (common for real data an operator has cut), so
that the data frame — not the trace centre — is the meaningful reference. The lag
maximises the circular cross-correlation between `x` and the raw stack.

Caveat, and the reason this is not the default: the anchor is only as good as the
stack it is read from. When the shifts are large and near-uniform the raw stack is
destroyed by incoherent summation and the lag becomes noise. `center_signal` has
no such failure mode. Inspect the returned `lag` if you use this.
"""
function anchor_to_data(x::AbstractVector{Float32}, traces::AbstractMatrix{Float32})
    ref = vec(mean(traces; dims=2))
    # Circular cross-correlation via FFT: cc[l+1] = Σₜ x[t] ref[t+l].
    cc = real(ifft(conj(fft(Float32.(x))) .* fft(Float32.(ref))))
    lag = argmax(cc) - 1
    return (; x=circshift(x, lag), lag)
end

"""
    reconstruct_signal(a, φ; nt, kmax=length(a)-1) -> (; x, X̂)

Assemble the complex spectrum from recovered magnitudes and phases and return
both it and the real time-domain signal.

`X̂[0] = a[0]` (DC is real, its phase gauged to 0); for `k = 1:kmax`,
`X̂[k] = a[k]e^{iφ[k]}` with `X̂[-k] = conj(X̂[k])` enforcing Hermitian symmetry so
`ifft` is real. When `nt` is even and the band reaches it, the Nyquist bin is
forced real (it is its own conjugate partner).

Bins above `kmax` are left at zero — a deliberate low-pass, since `a` is
unreliable there. This is why even noiseless recovery correlates ~0.998 rather
than 1.0 against a broadband truth: the reconstruction is band-limited by
construction.

Both outputs are returned because the **phase is the scientific target**; the
waveform is often just how it gets inspected.
"""
function reconstruct_signal(a::AbstractVector{Float32}, φ::AbstractVector{Float32};
                             nt::Integer, kmax::Integer=length(a) - 1)
    X̂ = zeros(ComplexF32, nt)
    X̂[1] = ComplexF32(a[1])
    @inbounds for k in 1:kmax
        z = a[k+1] * cis(φ[k+1])
        X̂[fidx(k, nt)] = z
        X̂[fidx(-k, nt)] = conj(z)
    end
    if iseven(nt) && kmax >= nt ÷ 2
        X̂[nt ÷ 2 + 1] = ComplexF32(real(X̂[nt ÷ 2 + 1]))
    end
    return (; x=Float32.(real(ifft(X̂))), X̂)
end

"""
    recover_mra(traces; method=:marching, kmax=nothing, band_fraction=0.01f0,
                sigma2=nothing, noise_ps=nothing, noise_bias=nothing, n_iters=300)
        -> (; x, X̂, a, φ, B, kmax, sigma2)

Top-level alignment-free recovery. `traces` is an `(nt, N)` `Float32` gather
(columns are traces) or an `(nt, N)` `ComplexF32` matrix of spectra.

Pipeline: FFT → [`power_spectrum_estimate`](@ref) → [`resolve_kmax`](@ref) →
[`bispectrum_estimate`](@ref) → phase inversion → [`gauge_phases!`](@ref) →
[`reconstruct_signal`](@ref). No shift is ever estimated.

`method`:
- `:marching` — [`invert_phases_freq_marching`](@ref); fast, greedy.
- `:sync` — marching, then the weighted least-squares refinement of
  [`invert_phases_sync`](@ref); better under heavy noise.

`x` is recovered **up to a global circular shift** — the moments are shift
invariant, so the absolute origin is not identifiable from the data. `origin`
therefore *declares* one, and the returned `origin_lag` records the rotation
applied:

- `:center` (default) — [`center_signal`](@ref): put the wavelet's circular
  centre of energy at the middle sample, `nt÷2`. Depends only on the recovered
  signal, so it adds no estimation error. Use this to compare results directly,
  without a `circshift`.
- `:data` — [`anchor_to_data`](@ref): match the raw stack of the input gather,
  landing in the data's own time coordinates. Better when the gather is already
  windowed on the arrival; degrades if the stack is incoherent.
- `:gauge` — leave the inversion's raw `φ[0]=φ[1]=0` frame (the previous
  behaviour). Compare with a best-lag correlation.

None of these change the recovered *shape* — only where it sits.

`x` is *not* ambiguous up to reflection: for real `x`, time reversal maps
`B → conj(B)`, which is measurably different from `B`, so orientation is
determined.

!!! warning "Per-trace polarity must be resolved upstream"
    The bispectrum is degree 3, so a per-trace gain scales it by `g³`; for a
    polarity flip `g = ±1` that is `g³ = g`, which does **not** cancel. Mixed
    polarity drives `B` toward zero — measured at 1.7-9.5% of its unit-gain
    magnitude, shrinking further as `N` grows — so the recovered polarity and
    phase are not trustworthy. The waveform *shape* may still survive up to a
    global sign, and the power spectrum (degree 2) is untouched, so `a` looks
    healthy either way: neither is evidence the result is sound. Check polarity
    before trusting `φ`.
"""
function recover_mra(traces::AbstractMatrix{ComplexF32};
                     method::Symbol=:marching,
                     kmax::Union{Nothing,Integer}=nothing,
                     band_fraction::Real=0.01f0,
                     sigma2::Union{Nothing,Real}=nothing,
                     noise_ps::Union{Nothing,AbstractVector}=nothing,
                     noise_bias::Union{Nothing,AbstractMatrix}=nothing,
                     origin::Symbol=:center,
                     n_iters::Integer=300)
    origin in (:center, :data, :gauge) ||
        throw(ArgumentError("origin must be :center, :data or :gauge, got :$origin"))
    method in (:marching, :sync) ||
        throw(ArgumentError("method must be :marching or :sync, got :$method"))
    nt = size(traces, 1)

    ps = power_spectrum_estimate(traces; noise_ps=noise_ps, sigma2=sigma2)
    kk = resolve_kmax(ps.a, nt; kmax=kmax, band_fraction=band_fraction)
    bs = bispectrum_estimate(traces; kmax=kk, noise_bias=noise_bias)

    a_band = ps.a[1:kk+1]
    φ = if method === :marching
        invert_phases_freq_marching(bs.B, a_band; kmax=kk)
    else
        invert_phases_sync(bs.B, a_band; kmax=kk, n_iters=n_iters).φ
    end
    gauge_phases!(φ; kmax=kk)

    rec = reconstruct_signal(a_band, φ; nt=nt, kmax=kk)

    # Place the result on a declared time origin. The inversion's own gauge
    # (φ[0]=φ[1]=0) is arbitrary, so without this the output sits in a frame
    # unrelated to the input and every comparison needs a circshift.
    x, lag = if origin === :center
        c = center_signal(rec.x); (c.x, c.lag)
    elseif origin === :data
        # The mean spectrum inverse-transforms to exactly the raw stack, so the
        # data anchor is available here without needing the real traces back.
        ref = Float32.(real(ifft(vec(mean(traces; dims=2)))))
        a = anchor_to_data(rec.x, reshape(ref, :, 1)); (a.x, a.lag)
    else
        (rec.x, 0)
    end
    X̂ = lag == 0 ? rec.X̂ : rec.X̂ .* cis.(-2f0 * Float32(π) * lag .* (0:nt-1) ./ nt)

    return (; x, X̂, a=ps.a, φ, B=bs.B, kmax=kk, sigma2=ps.sigma2, origin_lag=lag)
end

function recover_mra(traces::AbstractMatrix{Float32}; kwargs...)
    return recover_mra(ComplexF32.(fft(traces, 1)); kwargs...)
end

"""
    recover_mra_grouped(groups; min_events=2, kwargs...) -> (; groups, valid)

Apply [`recover_mra`](@ref) to each gather in `groups` (a `Vector` of `(nt, R_g)`
`Float32` matrices — one per station or condition bin).

Returns results **index-aligned 1:1 with the input**, plus a `valid::BitVector`
marking which groups were actually processed. Groups with fewer than
`max(min_events, 2)` columns are warned about and skipped, receiving a
zero-filled result of the correct shapes so downstream indexing and plotting
never hit a shape error. This mirrors `run_coherent_n2n_grouped`'s contract.

`min_events = 2` is the structural floor inherited from that precedent, **not** a
statistical recommendation: bispectrum estimates are high-variance and their
accuracy is strongly sample-limited (measured mean correlation 0.46 → 0.85 going
from 100 to 6400 traces at σ=0.8). Expect to need hundreds of traces.
"""
function recover_mra_grouped(groups::Vector{<:AbstractMatrix{Float32}};
                              min_events::Integer=2, kwargs...)
    G = length(groups)
    G > 0 || throw(ArgumentError("groups must be non-empty"))
    need = max(Int(min_events), 2)
    valid = BitVector(size(groups[g], 2) >= need for g in 1:G)

    results = Vector{NamedTuple}(undef, G)
    for g in 1:G
        if valid[g]
            results[g] = recover_mra(groups[g]; kwargs...)
        else
            @warn "Group $g has only $(size(groups[g], 2)) column(s) (< min_events=$need); skipped"
            nt = size(groups[g], 1)
            kk = max(2, nt ÷ 2 - 1)
            results[g] = (; x=zeros(Float32, nt), X̂=zeros(ComplexF32, nt),
                          a=zeros(Float32, nt), φ=zeros(Float32, kk + 1),
                          B=zeros(ComplexF32, kk + 1, kk + 1), kmax=kk, sigma2=0f0,
                          origin_lag=0)
        end
    end
    return (; groups=results, valid)
end
