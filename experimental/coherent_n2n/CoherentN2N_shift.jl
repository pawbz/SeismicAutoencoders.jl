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
    estimate_shift_xcorr_coarse(x_ref, x_trace) -> Int

Integer-sample coarse pre-alignment via FFT-based cross-correlation peak.
Bootstraps `estimate_shift_phase_slope` so the fine fit doesn't cycle-skip
on shifts approaching ±nt/2. Returns the shift `τ` (samples) such that
`x_trace` best matches `x_ref` shifted by `τ`, i.e. positive `τ` means
`x_trace` lags `x_ref`.

Searches the *magnitude* of the cross-correlation (`argmax(abs.(cc))`), not
the signed peak: earthquake traces can carry an unknown per-earthquake
polarity flip (radiation-pattern sign), which would otherwise produce a
large *negative* peak at the true lag that a plain `argmax` misses
entirely. Polarity itself is resolved later by `estimate_earthquake_gain`.
"""
function estimate_shift_xcorr_coarse(x_ref::AbstractVector{<:Real}, x_trace::AbstractVector{<:Real})
    nt = length(x_ref)
    @assert length(x_trace) == nt "x_ref and x_trace must have equal length"
    X̂_ref   = fft(Float32.(x_ref))
    X̂_trace = fft(Float32.(x_trace))
    # cc[k] = sum_n x_trace[n] * x_ref[n-k]  -> peaks at k=τ when x_trace(t) = x_ref(t-τ),
    # matching the shift_spectrum/grid convention (positive τ means x_trace lags x_ref).
    cc = real(ifft(conj.(X̂_ref) .* X̂_trace))  # circular cross-correlation
    lag0 = argmax(abs.(cc)) - 1                  # 0-based lag, peak of |cc| (polarity-agnostic)
    # Wrap lags > nt/2 into negative range (circular shift convention)
    return lag0 > nt ÷ 2 ? lag0 - nt : lag0
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
    estimate_shift_phase_slope(X̂_ref, X̂_trace, freqs; weight=:coherence, freq_band=nothing) -> Float32

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
"""
function estimate_shift_phase_slope(X̂_ref::AbstractVector{<:Complex}, X̂_trace::AbstractVector{<:Complex},
                                     freqs::AbstractVector{<:Real};
                                     weight::Symbol=:coherence, freq_band=nothing)
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

    # Weighted least squares: phi ≈ ω*τ + c   (no intercept needed for pure delay,
    # but fit with intercept for robustness to any residual constant phase term)
    Sw   = sum(w)
    Swx  = sum(w .* ω)
    Swy  = sum(w .* phi)
    Swxx = sum(w .* ω .^ 2)
    Swxy = sum(w .* ω .* phi)
    denom = Sw * Swxx - Swx^2
    τ = (Sw * Swxy - Swx * Swy) / denom
    return Float32(τ)
end

"""
    estimate_shift_two_stage(x_ref, x_trace, freqs; freq_band=nothing) -> Float32

Coarse integer-sample xcorr bootstrap + fine phase-slope fit. This is the
recommended shift-estimation entry point (used both for initialization and
Block B). `x_ref`/`x_trace` are real-valued time-domain traces; `freqs` are
`fftfreq(nt)`-convention frequencies for the fine fit.
"""
function estimate_shift_two_stage(x_ref::AbstractVector{<:Real}, x_trace::AbstractVector{<:Real},
                                   freqs::AbstractVector{<:Real}; freq_band=nothing)
    nt = length(x_ref)
    τ_coarse = estimate_shift_xcorr_coarse(x_ref, x_trace)

    # Undo the coarse shift on x_trace (circular) before the fine fit, so the
    # residual fine delay is small and safely within ±0.5 samples. x_trace
    # lags x_ref by τ_coarse (shift_spectrum convention: X̂_trace = X̂_ref *
    # exp(-iωτ)), so removing it means applying exp(+iωτ_coarse).
    X̂_trace_shifted = fft(Float32.(x_trace)) .* exp.(im .* 2π .* freqs .* Float32(τ_coarse))
    X̂_ref = fft(Float32.(x_ref))

    τ_fine = estimate_shift_phase_slope(X̂_ref, X̂_trace_shifted, freqs; freq_band=freq_band)
    return Float32(τ_coarse) + τ_fine
end
