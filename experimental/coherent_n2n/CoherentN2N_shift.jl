# Fourier-domain shift machinery for CoherentN2N.
#
# Complex-domain sibling of shift_traces_Fourier (experimental/phase_aligner/
# PhaseAligner_architecture.jl:117-131). We stay in the frequency domain
# throughout the alternating loop, so this operates directly on spectra
# (no fft/ifft wrapping, no real(...) projection).

using FFTW
using LinearAlgebra: dot

"""
    shift_spectrum(X̂, τ, grid) -> ComplexF32 matrix

Apply a sub-sample Fourier shift theorem directly to spectra.

- `X̂`:    `(nt, B)` ComplexF32 matrix (already FFT'd along dim 1)
- `τ`:    `(1, B)` Float32 matrix — shifts in **samples**
- `grid`: `(nt,)` complex vector `= -im * 2π * fftfreq(nt)`

`shift_spectrum(X̂, τ, grid) == fft(shift_traces_Fourier(real(ifft(X̂,1)), τ, grid), 1)`
but skips both FFT round-trips.
"""
function shift_spectrum(X̂::AbstractMatrix{ComplexF32}, τ::AbstractMatrix{Float32},
                         grid::AbstractVector)
    phase = exp.(grid .* τ)
    return X̂ .* phase
end

function shift_spectrum(X̂::AbstractMatrix{ComplexF32}, τ_scalar::Float32, grid::AbstractVector)
    τ_mat = fill(τ_scalar, 1, size(X̂, 2))
    return shift_spectrum(X̂, τ_mat, grid)
end

"""
    estimate_shift_xcorr_coarse(x_ref, x_trace; polarity_agnostic=true) -> Int

Integer-sample coarse pre-alignment via FFT-based cross-correlation peak.
Bootstraps `estimate_shift_phase_slope` so the fine fit doesn't cycle-skip
on shifts approaching ±nt/2. Returns the shift `τ` (samples) such that
`x_trace` best matches `x_ref` shifted by `τ`, i.e. positive `τ` means
`x_trace` lags `x_ref`.

`polarity_agnostic=true` (default) searches the *magnitude* of the
cross-correlation (`argmax(abs.(cc))`), not the signed peak: earthquake
traces can carry an unknown per-earthquake polarity flip (radiation-pattern
sign), which would otherwise produce a large *negative* peak at the true lag
that a plain `argmax` misses entirely. Use this when polarity is resolved
downstream by `estimate_earthquake_gain` (`use_polarity_gain=true`).

`polarity_agnostic=false` searches the *signed* peak (`argmax(cc)`) instead:
a polarity-flipped trace then no longer aligns cleanly to the wrong sign — it
fails to find a strong positive peak at the true lag, surfacing the flip as a
bad shift estimate. Use this when nothing downstream corrects sign
(`use_polarity_gain=false`) and a flip should not be silently absorbed.

`max_shift` (default `Inf`) restricts the peak search to lags within
`max_shift` samples **of `center`** (i.e. `|τ - center| <= max_shift`): only
those circular lags are considered, so the coarse pick cannot cycle-skip to a
far lag. This is the principled place to bound shifts — it constrains
estimation itself rather than clamping a bad estimate after the fact.

`center` (default `0`) is the shift the window is centred on — the trace's
**expected** shift, e.g. its value from the previous Block-B iteration. This
is what makes `max_shift` a *step* bound rather than an absolute-lag bound: the
argmax then tracks the true peak near where the trace already is, instead of a
lag-0 window snapping to the nearest in-window lobe (a half-period cycle-skip)
when the true peak lies outside `±max_shift` of the origin. At init (no prior)
pass `center=0`.

`shift_prior_p` (default `0` → OFF, current hard-window behaviour) turns the
hard `±max_shift` window into a **soft super-Gaussian MAP prior** on the lag:
the score is multiplied by `prior(lag) = exp(-(|lag - center| / max_shift)^p)`
before the argmax, so lags near `center` are preferred but a distant lag can
still win if its correlation is decisively stronger. This is flat inside
`±max_shift` (the plateau of the super-Gaussian) and then decays smoothly, so
shifts no longer pile up at a hard edge; it reduces low-SNR cycle-skip
outliers by biasing toward the plausible band. Requires a finite `max_shift`
(the prior width) — with `max_shift = Inf` the prior is skipped (no width).
`shift_prior_backstop_mult` (default `3`) sets a *generous* hard search bound
at `center ± ceil(shift_prior_backstop_mult · max_shift)` so a pathological far
cycle-skip can never be argmax'd; at `3·max_shift` the weight is `exp(-3^p)`
(≈0 for p≥4), so the backstop only forbids what the prior already kills.
"""
function estimate_shift_xcorr_coarse(x_ref::AbstractVector{<:Real}, x_trace::AbstractVector{<:Real};
                                      polarity_agnostic::Bool=true, max_shift::Real=Inf,
                                      center::Real=0, shift_prior_p::Real=0,
                                      shift_prior_backstop_mult::Real=3)
    nt = length(x_ref)
    @assert length(x_trace) == nt "x_ref and x_trace must have equal length"
    X̂_ref   = fft(Float32.(x_ref))
    X̂_trace = fft(Float32.(x_trace))
    # cc[k] = sum_n x_trace[n] * x_ref[n-k]  -> peaks at k=τ when x_trace(t) = x_ref(t-τ),
    # matching the shift_spectrum/grid convention (positive τ means x_trace lags x_ref).
    cc = real(ifft(conj.(X̂_ref) .* X̂_trace))  # circular cross-correlation
    score = polarity_agnostic ? abs.(cc) : cc
    # Signed lag for each array index k (0-based), wrapping k > nt/2 to negative.
    signed_lag(k) = k > nt ÷ 2 ? k - nt : k
    c = round(Int, center)
    prior_on = shift_prior_p > 0 && isfinite(shift_prior_p) && isfinite(max_shift) && max_shift > 0
    if prior_on
        # Soft super-Gaussian MAP prior on the lag, inside a generous hard backstop.
        w = Float32(max_shift)
        p = Float32(shift_prior_p)
        backstop = ceil(Int, shift_prior_backstop_mult * max_shift)
        cand = [k for k in 0:(nt-1) if abs(signed_lag(k) - c) <= backstop]
        isempty(cand) && return Float32(c)
        weight = Float32[exp(-(abs(signed_lag(k) - c) / w)^p) for k in cand]
        best = cand[argmax(score[cand .+ 1] .* weight)]
        return signed_lag(best)
    elseif isfinite(max_shift)
        # Hard window: only consider indices whose circular lag is within max_shift of center.
        m = floor(Int, max_shift)
        cand = [k for k in 0:(nt-1) if abs(signed_lag(k) - c) <= m]
        isempty(cand) && return Float32(c)   # window fell entirely outside; keep center
        best = cand[argmax(score[cand .+ 1])]
        return signed_lag(best)
    end
    lag0 = argmax(score) - 1  # 0-based lag
    return signed_lag(lag0)
end

"""
    unwrap_phase(phi) -> Vector{Float32}

Simple cumulative 2π-jump-correction phase unwrap along a 1D vector.
"""
function unwrap_phase(phi::AbstractVector{<:Real})
    out = Float32.(copy(phi))
    for i in 2:length(out)
        d = out[i] - out[i-1]
        while d > π
            out[i] -= 2π
            d = out[i] - out[i-1]
        end
        while d < -π
            out[i] += 2π
            d = out[i] - out[i-1]
        end
    end
    return out
end

"""
    estimate_shift_phase_slope(X̂_ref, X̂_trace, freqs; weight=:coherence, freq_band=nothing,
                                zero_intercept=false) -> Float32

Sub-sample delay via cross-spectrum phase-slope fit.

Cross-spectrum `C = X̂_ref .* conj.(X̂_trace)`. For a pure delay `τ`,
`angle(C(ω)) ≈ ω*τ` (since `X̂_trace(ω) = X̂_ref(ω) * exp(-iωτ)` when
`x_trace` lags `x_ref` by `τ`). Phase is unwrapped across frequency and
fit with weighted least squares (weight = cross-spectrum magnitude by
default, downweighting low-SNR bins), giving the slope `τ`.

`freqs` are angular or cyclic frequencies matching the convention of
`grid`/`fftfreq`; only the sign convention needs to stay consistent with
`shift_spectrum`. Positive `τ` means `x_trace` lags `x_ref`.

`freq_band`, if given as `(flo, fhi)`, restricts the fit to `flo <= |freq| <= fhi`.

`zero_intercept=false` (default) fits `phi ≈ ω*τ + c`, tolerating a residual
constant phase offset `c`. A polarity flip between `x_ref`/`x_trace`
contributes exactly such an offset (`c ≈ ±π`, since a sign flip multiplies
the cross-spectrum by -1), which this mode absorbs into `c`, leaving `τ`
correct regardless of polarity. `zero_intercept=true` fits `phi ≈ ω*τ`
through the origin instead: a `±π` offset then corrupts the fit directly, so
a polarity-flipped trace no longer returns a clean, wrong-signed-but-accurate
`τ` — it returns a visibly bad one. Use `zero_intercept=true` when nothing
downstream corrects sign (`use_polarity_gain=false`) and a flip should
surface as a bad alignment rather than be silently tolerated.
"""
function estimate_shift_phase_slope(X̂_ref::AbstractVector{<:Complex}, X̂_trace::AbstractVector{<:Complex},
                                     freqs::AbstractVector{<:Real};
                                     weight::Symbol=:coherence, freq_band=nothing,
                                     zero_intercept::Bool=false)
    @assert length(X̂_ref) == length(X̂_trace) == length(freqs)
    C = X̂_ref .* conj.(X̂_trace)
    mag = abs.(C)

    keep = trues(length(freqs))
    if freq_band !== nothing
        flo, fhi = freq_band
        keep .= flo .<= abs.(freqs) .<= fhi
    end
    # Always drop the DC bin (zero frequency carries no delay information).
    keep[1] = false
    @assert any(keep) "No frequency bins left after freq_band filtering"

    # fftfreq is not monotonic (wraps from +Nyquist to -Nyquist mid-array);
    # unwrapping requires monotonic frequency order, so sort first.
    idx = findall(keep)
    sort!(idx, by = i -> freqs[i])

    ω = 2π .* freqs[idx]
    phi = unwrap_phase(angle.(C[idx]))
    w = weight === :coherence ? mag[idx] ./ (maximum(mag[idx]) + eps(Float32)) : ones(Float32, length(idx))

    τ = if zero_intercept
        # Weighted least squares through the origin: phi ≈ ω*τ (no tolerance
        # for a constant phase offset, so a polarity-flip's ±π contribution
        # corrupts the fit instead of being absorbed).
        sum(w .* ω .* phi) / sum(w .* ω .^ 2)
    else
        # Weighted least squares with intercept: phi ≈ ω*τ + c (robust to any
        # residual constant phase term, including a polarity flip's ±π).
        Sw   = sum(w)
        Swx  = sum(w .* ω)
        Swy  = sum(w .* phi)
        Swxx = sum(w .* ω .^ 2)
        Swxy = sum(w .* ω .* phi)
        (Sw * Swxy - Swx * Swy) / (Sw * Swxx - Swx^2)
    end
    return Float32(τ)
end

"""
    estimate_shift_two_stage(x_ref, x_trace, freqs; freq_band=nothing, polarity_agnostic=true) -> Float32

Coarse integer-sample xcorr bootstrap + fine phase-slope fit. This is the
recommended shift-estimation entry point (used both for initialization and
Block B). `x_ref`/`x_trace` are real-valued time-domain traces; `freqs` are
`fftfreq(nt)`-convention frequencies for the fine fit.

`polarity_agnostic` (default `true`) is threaded through to both stages: the
coarse xcorr peak search (`estimate_shift_xcorr_coarse`) and the fine fit's
intercept handling (`estimate_shift_phase_slope`, via `zero_intercept =
!polarity_agnostic`). Set to `false` when nothing downstream corrects
polarity (`use_polarity_gain=false`), so a polarity-flipped trace fails to
align cleanly instead of being silently aligned with the wrong sign. The
coarse stage is the primary, robust polarity gate here (a flipped trace's
cross-correlation at the true lag is large and *negative*, so the signed
`argmax` reliably misses it); `zero_intercept` on the fine fit is a
secondary safeguard against the same failure mode but, being a linear fit
over a possibly non-ideal phase spectrum, doesn't always corrupt the result
as dramatically.

`max_shift` (default `Inf`) bounds the returned delay to `center ± max_shift`
samples: it restricts the coarse xcorr peak search to that range (preventing a
cycle-skip to a far lag) AND clamps the final coarse+fine result to the
boundary, so a trace whose true shift exceeds the bound saturates at the edge
rather than being estimated out of range.

`center` (default `0`) is the shift the `±max_shift` window is centred on — the
trace's **expected** shift (e.g. its previous Block-B value). This makes
`max_shift` a *step* bound around where the trace already is, not an absolute
bound around lag 0: with `center=0` a trace whose true shift is outside
`±max_shift` of the origin gets a lag-0 window that snaps to the nearest lobe
inside the window — for an oscillatory waveform, a half-period-shifted,
polarity-inverted match (a cycle-skip). Passing `center=τ_prev` keeps the
window over the true peak. Pass `center=0` only at init (no prior).

`shift_prior_p` / `shift_prior_backstop_mult` (defaults `0` / `3`) turn the hard
`±max_shift` window into a soft super-Gaussian MAP prior on the lag — see
`estimate_shift_xcorr_coarse`. With `shift_prior_p > 0` the coarse search is
bounded only by the generous backstop `center ± ceil(mult·max_shift)`, so the
final result is clamped to that backstop (plus the ~0.5 fine overshoot), NOT to
`±max_shift`: the prior, not a hard wall, is what keeps shifts small, and the
distribution decays smoothly instead of piling up at an edge.
"""
function estimate_shift_two_stage(x_ref::AbstractVector{<:Real}, x_trace::AbstractVector{<:Real},
                                   freqs::AbstractVector{<:Real}; freq_band=nothing,
                                   polarity_agnostic::Bool=true, max_shift::Real=Inf,
                                   center::Real=0, shift_prior_p::Real=0,
                                   shift_prior_backstop_mult::Real=3)
    nt = length(x_ref)
    τ_coarse = estimate_shift_xcorr_coarse(x_ref, x_trace; polarity_agnostic=polarity_agnostic,
                                           max_shift=max_shift, center=center,
                                           shift_prior_p=shift_prior_p,
                                           shift_prior_backstop_mult=shift_prior_backstop_mult)

    # Undo the coarse shift on x_trace (circular) before the fine fit, so the
    # residual fine delay is small and safely within ±0.5 samples. x_trace
    # lags x_ref by τ_coarse (shift_spectrum convention: X̂_trace = X̂_ref *
    # exp(-iωτ)), so removing it means applying exp(+iωτ_coarse).
    X̂_trace_shifted = fft(Float32.(x_trace)) .* exp.(im .* 2π .* freqs .* Float32(τ_coarse))
    X̂_ref = fft(Float32.(x_ref))

    τ_fine = estimate_shift_phase_slope(X̂_ref, X̂_trace_shifted, freqs; freq_band=freq_band,
                                         zero_intercept=!polarity_agnostic)
    τ = Float32(τ_coarse) + τ_fine
    # The bound is enforced by the coarse stage's IN-WINDOW argmax; τ_fine is only
    # a sub-sample (|·|<~0.5) refinement, so clamp only that harmless overshoot.
    # Bound depends on mode: soft prior → clamp to the generous backstop (the
    # prior, not the clamp, keeps shifts small, so no ±edge pile-up); hard window
    # → clamp to center ± max_shift as before.
    isfinite(max_shift) || return τ
    prior_on = shift_prior_p > 0 && isfinite(shift_prior_p) && max_shift > 0
    bound = prior_on ? Float32(ceil(shift_prior_backstop_mult * max_shift)) + 0.5f0 :
                       Float32(max_shift)
    return clamp(τ, Float32(center) - bound, Float32(center) + bound)
end
