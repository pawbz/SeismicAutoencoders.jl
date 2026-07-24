# Location-gauge fixing for per-earthquake shift vectors.
#
# The alternating loop's shift updates (Block B) have an inherent gauge
# freedom: adding a constant to every τ_r for a given station's earthquake
# set doesn't change any pairwise alignment. We fix this gauge by pinning a
# location statistic of the shift distribution to zero. The default is the
# *mode* (KDE peak): it anchors zero on the densest cluster of shifts — the
# dominant coherent population — and lets the remaining tail be arbitrarily
# skewed (we do NOT force equal counts either side of zero). Pass
# `center=median` for a symmetric-count gauge, or `center=mean` for the classic
# zero-sum gauge. Whatever is subtracted is stored as the "anchor" so absolute
# timing can be restored at the end (see run_coherent_n2n).

using Statistics: mean, median, std

"""
    mode_kde(τ::AbstractVector{<:Real}; bandwidth=nothing, niter=50, tol=1f-4) -> Float32

Estimate the mode (peak of the distribution) of `τ` via Gaussian-kernel
mean-shift *seeded at the median*. Starting from `median(τ)`, iterate the
KDE-weighted mean (a Gaussian mean-shift step) until it converges to the nearest
density peak. `bandwidth` defaults to Silverman's rule of thumb from the spread
of `τ`; degenerate inputs (fewer than 2 points, or zero spread) fall back to the
mean.

Seeding at the median — rather than taking a global grid argmax — makes the
estimate **stable across outer-loop iterations**: when the distribution is
sharply peaked it climbs to the true mode; when it is broad/flat (no dominant
cluster) mean-shift barely moves and the result gracefully degrades to ≈median,
instead of an argmax that jitters between iterations and destabilizes the loop.

This is the location that best represents the *dominant cluster* of shifts,
robust to a skewed/heavy tail — unlike the mean (dragged by the tail) or the
median (which forces equal counts either side).
"""
function mode_kde(τ::AbstractVector{<:Real}; bandwidth=nothing,
                  niter::Int=50, tol::Float32=1f-4)
    n = length(τ)
    n == 0 && return 0f0
    n == 1 && return Float32(τ[1])

    lo, hi = extrema(τ)
    # Silverman's rule; guard against zero/degenerate spread.
    σ = Float32(std(τ))
    h = bandwidth === nothing ?
        1.06f0 * max(σ, 1f-6) * Float32(n)^(-1f0/5f0) : Float32(bandwidth)
    # All shifts identical (or bandwidth collapsed): nothing to peak-find.
    (hi ≈ lo || h ≤ 0f0) && return Float32(mean(τ))

    inv2h2 = 1f0 / (2f0 * h * h)
    x = Float32(median(τ))          # robust seed
    @inbounds for _ in 1:niter
        wsum = 0f0
        xw = 0f0
        for t in τ
            tf = Float32(t)
            Δ = x - tf
            w = exp(-Δ * Δ * inv2h2)
            wsum += w
            xw += w * tf
        end
        wsum == 0f0 && break        # x drifted off the support; keep last
        xnew = xw / wsum
        abs(xnew - x) < tol && (x = xnew; break)
        x = xnew
    end
    return x
end

"""
    enforce_location_gauge!(τ::AbstractVector{Float32}; center=mode_kde) -> Float32

Subtract `center(τ)` from `τ` in place (location gauge). The default
`center=mode_kde` pins the *dominant cluster* of shifts to zero, allowing an
arbitrarily skewed tail (counts either side of zero need not balance). Pass
`center=median` for a symmetric-count gauge, or `center=mean` for the classic
zero-sum/zero-mean gauge.

Note on iterative use: a nonlinear gauge (mode/median) is only a fixed point of
the alternating shift-and-stack loop if the loop pins the source's time origin
to a fixed template each iteration (as `run_coherent_n2n` does by re-anchoring ŝ
to the previous iteration's source). Without that, only the linear `mean` gauge
keeps the origin from wandering.

Returns the subtracted value — the "anchor" — which the caller should accumulate
and reapply at the end of the outer loop to restore an absolute time reference
(anchored to whatever the initial coarse cross-correlation lags implied).
"""
function enforce_location_gauge!(τ::AbstractVector{Float32}; center=mode_kde)
    m = Float32(center(τ))
    τ .-= m
    return m
end
