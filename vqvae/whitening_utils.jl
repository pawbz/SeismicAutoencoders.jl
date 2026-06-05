# Lightweight whitening utilities — NO heavy ML dependencies (Lux, Reactant, Enzyme, CUDA)
# Only dependencies: DSP, FFTW, Statistics (already in Project.toml)

using DSP, FFTW, Statistics

"""
    compute_whitening_fir(X::AbstractMatrix{Float32}; kernel_length=64, min_power_fraction=0.05)

Compute per-column FIR whitening filters from a data matrix X.

Each column of X gets its own whitening filter based on its power spectrum.
The filter is designed in the frequency domain as W = P^(-0.25), where P is
the power spectrum. When applied via filtfilt (forward+backward), the net
result is P^(-0.5) whitening on the magnitude spectrum.

# Arguments
- `X::AbstractMatrix{Float32}`: Data matrix (time × waveforms)
- `kernel_length::Int=64`: Length of FIR filter taps
- `min_power_fraction::Float64=0.05`: Noise floor as fraction of max power

# Returns
Matrix of shape (kernel_length, size(X,2)) containing per-column FIR coefficients
"""
function compute_whitening_fir(X_cpu::AbstractMatrix{Float32};
    kernel_length::Int=64, min_power_fraction::Float64=0.05)
    nt = size(X_cpu, 1)
    n_waveforms = size(X_cpu, 2)

    # Compute power spectrum for each waveform
    P = abs2.(fft(Float64.(X_cpu), 1))
    P[1, :] .= 0.0  # zero DC component

    # Apply noise floor per waveform
    P .= max.(P, min_power_fraction .* maximum(P, dims=1))

    # Frequency-domain whitening filter: W = P^(-0.25)
    # When filtfilt is applied (forward + backward pass), magnitude response is squared,
    # so net effect is P^(-0.5) whitening
    W = P .^ (-0.25)
    W[1, :] .= 0.0  # keep DC zero

    # Transform to time domain via IFFT
    w_full = real.(ifft(W, 1))

    # Shift and extract central taps
    w_shift = fftshift(w_full, 1)
    center = div(nt, 2) + 1
    lo = center - div(kernel_length, 2)
    taps = w_shift[lo : lo + kernel_length - 1, :]

    # Apply Hann window and normalize
    hann_window = DSP.hann(kernel_length)
    taps .*= hann_window
    taps ./= sum(abs.(taps), dims=1)  # normalize each column

    return Float32.(taps)
end

"""
    apply_whitening_fir(X::AbstractMatrix{Float32}, fir::AbstractMatrix{Float32})

Apply per-column FIR whitening filters to data via zero-phase filtfilt.

Each column of X is filtered with the corresponding column of fir.
Uses DSP.filtfilt for zero-phase filtering (forward+backward).

# Arguments
- `X::AbstractMatrix{Float32}`: Data matrix (time × waveforms)
- `fir::AbstractMatrix{Float32}`: FIR filters, shape (kernel_length, waveforms)

# Returns
Whitened data matrix, same shape as X
"""
function apply_whitening_fir(X_cpu::AbstractMatrix{Float32}, fir::AbstractMatrix{Float32})
    n_waveforms = size(X_cpu, 2)
    X_whitened = similar(X_cpu, Float32)

    for i in 1:n_waveforms
        fir_taps = Float64.(fir[:, i])
        X_col = Float64.(X_cpu[:, i])
        X_whitened[:, i] = Float32.(DSP.filtfilt(fir_taps, [1.0], X_col))
    end

    return X_whitened
end
