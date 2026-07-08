# Synthetic ground-truth generator for CoherentN2N testing.
# Adapted from the Ricker-wavelet convention used by
# experimental/phase_aligner/Training_PhaseAligner.jl:81 (make_ricker_dataset_syn).

using Random
using FFTW

"""
    ricker_wavelet(nt, f0; dt=1.0, t0=nt÷2) -> Vector{Float32}

Classic Ricker (Mexican-hat) wavelet of length `nt`, dominant frequency `f0`,
peak centered at sample `t0`.
"""
function ricker_wavelet(nt::Int, f0::Real; dt::Real=1.0, t0::Real=nt ÷ 2)
    t = (collect(0:nt-1) .- t0) .* dt
    a = (π * f0) .^ 2
    return Float32.((1 .- 2 .* a .* t .^ 2) .* exp.(-a .* t .^ 2))
end

"""
    broadband_wavelet(nt, f0s; dt=1.0, t0=nt÷2) -> Vector{Float32}

Sum of Ricker wavelets at several dominant frequencies `f0s`. A single
narrowband Ricker wavelet has a very broad, slowly-decaying autocorrelation
(poor time resolution), which makes coarse cross-correlation lag-picking
genuinely ill-conditioned even at low noise. Real seismic arrivals — and
this pipeline's own spectral-whitening preprocessing step — are broadband,
so this is the more representative test source for shift-estimator tests.
"""
function broadband_wavelet(nt::Int, f0s; dt::Real=1.0, t0::Real=nt ÷ 2)
    s = zeros(Float32, nt)
    for f0 in f0s
        s .+= ricker_wavelet(nt, f0; dt=dt, t0=t0)
    end
    return s ./ Float32(length(f0s))
end

"""
    make_synthetic_gather(; nt, R, f0, true_shifts, true_gains, noise_std, rng, source_kind) -> (s_true, D, freqs)

Build `R` synthetic single-station "earthquake" traces:
`d_r = gain_r * Shift_{τ_r}[s_true] + noise_r`

Returns:
- `s_true`: `(nt,)` clean source wavelet (single Ricker, or broadband if
  `source_kind=:broadband`)
- `D`: `(nt, R)` Float32 matrix of noisy shifted/scaled traces
- `freqs`: `(nt,)` `fftfreq(nt)`-convention frequency vector, for use with
  `estimate_shift_two_stage` / `estimate_shift_phase_slope`.

`true_shifts` and `true_gains` are also returned via the same call for
ground-truth comparison in tests.
"""
function make_synthetic_gather(; nt::Int=300, R::Int=40, f0::Real=0.05,
                                true_shifts::Union{Nothing,Vector{Float32}}=nothing,
                                true_gains::Union{Nothing,Vector{ComplexF32}}=nothing,
                                noise_std::Real=0.05, rng::AbstractRNG=Random.default_rng(),
                                source_kind::Symbol=:ricker)
    s_true = source_kind === :broadband ? broadband_wavelet(nt, (f0, 2f0, 4f0)) : ricker_wavelet(nt, f0)
    freqs = Float32.(fftfreq(nt))
    grid = -im .* 2π .* freqs

    τ = true_shifts === nothing ? Float32.(rand(rng, R) .* 10 .- 5) : true_shifts
    # Real-valued polarity/gain (±1 scaled) so the time-domain trace stays real;
    # a genuinely complex gain only makes sense applied to a spectrum (used
    # separately in frequency-domain gain-recovery tests).
    g = true_gains === nothing ? ComplexF32.(rand(rng, (-1.0, 1.0), R)) : true_gains

    Ŝ = fft(s_true)
    D = Matrix{Float32}(undef, nt, R)
    for r in 1:R
        shifted = real(ifft(Ŝ .* exp.(grid .* τ[r])))
        D[:, r] = Float32.(real(g[r]) .* shifted .+ noise_std .* randn(rng, nt))
    end
    return s_true, D, freqs, τ, g
end
