### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 7ebf7c4e-353d-4cfd-8094-c07343589e4e
using PlutoUI, Printf, FFTW, DSP, LinearAlgebra

# ╔═╡ a1b2c3d4-0001-0000-0000-000000000001
using Peaks

# ╔═╡ bbcb5888-9968-49f5-97f0-db8e5c53c121
using PlutoPlotly

# ╔═╡ de53b0d8-61aa-4b68-8fd3-a8d775ba8c95
using StatsBase

# ╔═╡ a7d1a9e8-c1dd-4fbe-a1c7-3b7d6a1fe3a1
using ColorSchemes, Colors

# ╔═╡ 120b0f0e-2b75-447b-aeeb-caf9844e0290
using Markdown

# ╔═╡ 0f0ffc2f-3754-4b4a-9b69-27f9702c9541
using Test

# ╔═╡ a210884b-23a6-4e03-9028-cd4949c0304d
TableOfContents(include_definitions=true)

# ╔═╡ 25162009-a8a6-4983-a89c-a6be9769f1f7
md"""
## Multiple Filter Technique
This notebook implements a Multiple Filter Technique (MFT) workflow for seismic traces. It defines data types (SeismicTrace, MFTResult) and core signal-processing utilities: narrow-band Gaussian filtering in frequency domain, analytic-envelope computation, and a simple arrival picker (find_group_arrival). The main routine perform_mft_analysis applies the narrowband filters across a user-specified period range, computes envelopes, detects group arrivals within a velocity-based window, estimates group/phase velocities, spectral amplitudes and a basic quality factor (peak/mean)
"""

# ╔═╡ 161e84fc-db09-4a69-ab53-c73eaa4a1486
md"## Core"

# ╔═╡ e4d5e20b-2bcc-4a22-a2d1-aae7974b02cf
xsynthetic = zeros(1000); xsynthetic[500] = 1.0

# ╔═╡ 5ecefa6f-2a86-46bf-b91b-9ce4c0a32e08
md"""
Period range (s): $(@bind period_min Slider(1.0:1.0:20.0, default=5.0, show_value=true)) to $(@bind period_max Slider(20.0:5.0:200.0, default=50.0, show_value=true))

Number of periods: $(@bind n_periods Slider(10:5:100, default=100, show_value=true))

Gaussian bandwidth (% of centre frequency): $(@bind bandwidth_percent Slider(5.0:1.0:100.0, default=28.0, show_value=true))

FFT zero-padding: $(@bind zero_pad_factor Select([1 => "1× (none)", 2 => "2×", 4 => "4×", 8 => "8×"]))

Phase velocity mode: $(@bind phvel_mode Select(["none" => "None (group only)", "single" => "Single station", "noise_cc" => "Ambient noise CC (π/4)"]))
"""

# ╔═╡ 9ce67356-5ff7-4a22-b244-75e35350f959
md"## Structs"

# ╔═╡ b6d23222-ba2a-11f0-bcd7-8f110c4a4cd6
# Data structures for MFT analysis
"""
    MFTResult

Structure to hold Multiple Filter Technique analysis results.
"""
struct MFTResult
    periods::Vector{Float64}                    # Analysis periods [s]
    frequencies::Vector{Float64}                # Analysis frequencies [Hz]
    group_velocities::Vector{Float64}           # Group velocities [km/s]
    phase_velocities::Vector{Float64}           # Phase velocities [km/s]
    phase_branch_numbers::Vector{Int}           # Relative phase-branch indices around the unwrapped branch
    phase_velocity_branches::Matrix{Float64}    # Phase velocities [period, branch]
    amplitudes::Vector{Float64}                 # Spectral amplitudes
    filtered_traces::Matrix{Float64}            # Filtered time series [time, frequency]
    envelopes::Matrix{Float64}                  # Amplitude envelopes [time, frequency]
    arrival_times::Vector{Float64}              # Group arrival times [s]
    distance::Float64                           # Source-receiver distance [km]
    quality_factors::Vector{Float64}  # Quality assessment for each measurement
	all_peaks::Vector{Vector{Tuple{Float64,Float64}}}  # Up to 4 strongest envelope peaks per period: (time, amplitude)
	time::Vector{Float64 }
end

# ╔═╡ cee98fdd-f0d9-4ea8-9256-2492a649a512
begin
	"""
	    SeismicTrace
	
	Structure to hold seismic time series data.
	"""
	struct SeismicTrace
	    data::Vector{Float64}
		dt::Float64   # Sampling interval [s]
		distance::Float64                          # Epicentral distance [km]
	
	    function SeismicTrace(data, dt, distance)
	        @warn "SeismicTrace assumes the first sample occurs at time dt = $(dt) s, not at 0.0 s. MFT will therefore construct its time axis starting at dt."
	        return new(data, dt, distance)
	    end
	end
	
	SeismicTrace(; data, dt, distance) = SeismicTrace(data, dt, distance)
end

# ╔═╡ 8816d460-fa4d-4b46-803c-5e0c4b80fb9e
Xsynthetic = SeismicTrace(xsynthetic, 1.0, 5.0 * 500)

# ╔═╡ 72db9d05-2714-4e86-bf36-a0c8ee13c040
md"## Core Functions"

# ╔═╡ fcf48627-0b80-4c6b-b204-f40da981aaa5
function perform_mft_analysis(trace::SeismicTrace, period_range, n_periods; kwargs...)
	period_min, period_max = Float64.(period_range)
	periods_analysis = collect(exp10.(range(log10(period_min), log10(period_max), length=n_periods)))
	return perform_mft_analysis(trace, periods_analysis; kwargs...)
end

# ╔═╡ d5ca213a-b085-4d65-88fa-d221bf5826e1
"""
    narrow_band_filter(data::Vector{Float64}, dt::Float64, center_freq::Float64, 
                      bandwidth_factor::Float64=0.1) -> Vector{ComplexF64}

Apply narrowband Gaussian filter centered at specified frequency.

# Arguments
- `data`: Input time series
- `dt`: Sampling interval [s]
- `center_freq`: Center frequency [Hz]
- `bandwidth_factor`: Fractional bandwidth (default 0.1 = 10%)

# Returns
- Complex filtered time series
"""
function narrow_band_filter(data::AbstractArray, dt::Float64, center_freq::Float64, 
                           bandwidth_factor::Float64=0.1)
    npts = size(data, 1)
    
    # FFT parameters
    fft_data = fft(data, 1)
    freqs = cat(collect(fftfreq(npts, 1.0/dt)), dims=ndims(data))
    
    # Gaussian filter parameters
    sigma_freq = center_freq * bandwidth_factor / 2.0  # Standard deviation
    
    # Create Gaussian filter in frequency domain
    filter_response = exp.(-0.5 * ((freqs .- center_freq) ./ sigma_freq).^2)
    filter_response += exp.(-0.5 * ((freqs .+ center_freq) ./ sigma_freq).^2)  # Negative frequencies
    
    # Apply filter
    filtered_fft = fft_data .* filter_response
    
    # Transform back to time domain
    filtered_data = real.(ifft(filtered_fft, 1))
    
    return filtered_data
end

# ╔═╡ 4192fcef-4662-4944-b354-71e8e00285b1
md"""
narrow_band_filter_analytic(data, dt, center_freq, bandwidth_factor=sqrt(2/25)) -> Vector{ComplexF64}

Apply a one-sided Gaussian narrowband filter and return the **complex analytic signal**.

The filter is parameterized by a single fractional bandwidth:

    σ_f = f₀ · bandwidth_factor / 2
    H(f) = exp( -0.5 ((f - f₀)/σ_f)^2 )   for f ≥ 0
    H(f) = 0                              for f < 0

This is equivalent to the sacmft96 form

    H(f) = exp( -α ((f - f₀)/f₀)^2 )

with

    α = 2 / bandwidth_factor²

so `bandwidth_factor = sqrt(2/25) ≈ 0.283` reproduces the old `α = 25` default.

The one-sided spectrum is multiplied by 2 so that `abs(z(t))` equals the
two-sided envelope (Hilbert-transform convention).

# Arguments
- `data`             : input time series
- `dt`               : sampling interval [s]
- `center_freq`      : centre frequency f₀ [Hz]
- `bandwidth_factor` : fractional bandwidth relative to f₀

# Returns
- `Vector{ComplexF64}` — complex analytic signal;
  `abs.(z)` = envelope,  `angle.(z)` = instantaneous phase
"""


# ╔═╡ 2c1a711e-261e-4e82-ade2-2fb6bc0e6b93
function narrow_band_filter_analytic(data::AbstractVector{<:Real}, dt::Float64,
                                     center_freq::Float64,
                                     bandwidth_factor::Float64=sqrt(2.0 / 25.0))
    npts  = length(data)
    spec  = fft(data)
    freqs = fftfreq(npts, 1.0 / dt)
    out   = zeros(ComplexF64, npts)
    sigma_freq = center_freq * bandwidth_factor / 2.0

    for k in eachindex(freqs)
        f = freqs[k]
        if f > 0.0
            H = exp(-0.5 * ((f - center_freq) / sigma_freq)^2)
            out[k] = 2.0 * H * spec[k]   # ×2: one-sided → analytic convention
        elseif f == 0.0
            # DC: near-zero contribution at any usable f₀
            H = exp(-0.5 * (center_freq / sigma_freq)^2)
            out[k] = H * spec[k]
        end
        # f < 0: out[k] stays zero (one-sided spectrum)
    end
    return ifft(out)
end

# ╔═╡ e9bfb21b-f29d-410f-827f-e66d1117020f
"""
    compute_phase_velocity(analytic_signal, time, t_group, freq, distance;
                           phvel_correction=0.0) -> Float64

Estimate phase velocity from the instantaneous phase of the analytic signal at
the group arrival time (single-frequency, n = 0 branch only).

> **Note:** `perform_mft_analysis` uses cross-frequency phase unwrapping
> (DSP.unwrap across the frequency axis) to resolve the 2πn branch ambiguity,
> consistent with sacmft96.  This function is a single-point utility; call it
> only when you already know the correct branch or when the distance is short
> enough that n = 0 is guaranteed.

The total phase accumulated travelling distance Δ at phase velocity c is:

    φ_total = 2π f₀ Δ / c

The instantaneous phase of the analytic signal at the group arrival:

    φ(tᵍ) = angle( z(tᵍ) ) = 2π f₀ tᵍ − φ_total + φ_correction

Rearranging (n = 0 branch):

    c = 2π f₀ Δ / ( 2π f₀ tᵍ − φ(tᵍ) + φ_correction )

# Arguments
- `analytic_signal`  : complex analytic signal from `narrow_band_filter_analytic`
- `time`             : time vector [s] (same grid)
- `t_group`          : group arrival time [s]
- `freq`             : centre frequency f₀ [Hz]
- `distance`         : source–receiver distance [km]
- `phvel_correction` : additional phase shift [rad]
                         0.0 – single station / two-station same great circle
                               (sacmft96 dophvel = 2)
                         π/4 – interstation ambient-noise cross-correlation
                               (sacmft96 dophvel = 1)

# Returns
- Phase velocity [km/s], or `NaN` if undefined or implausible
"""
function compute_phase_velocity(analytic_signal::AbstractVector{<:Complex},
                                time::Vector{Float64},
                                t_group::Float64,
                                freq::Float64,
                                distance::Float64;
                                phvel_correction::Float64=0.0)
    (isnan(t_group) || t_group <= 0.0) && return NaN
    dt_local = time[2] - time[1]
    i_g = clamp(round(Int, (t_group - time[1]) / dt_local) + 1, 1, length(time))

    # Instantaneous phase at group arrival (wrapped to (−π, π])
    phi_measured = angle(analytic_signal[i_g])

    omega = 2π * freq
    denom = omega * t_group - phi_measured + phvel_correction
    abs(denom) < 1e-6 && return NaN

    c = omega * distance / denom
    # Sanity check: plausible seismic phase velocity (0.5 – 20 km/s)
    (c < 0.5 || c > 20.0) && return NaN
    return c
end

# ╔═╡ f71ee0c9-1e34-49c2-9385-4589c561f4e4
"""
    multiple_narrow_band_filters(trace::SeismicTrace, periods::Vector{Float64}, bandwidth_percent::Float64)

Apply multiple narrowband filters at different central periods and sum them.
Useful for enhancing specific frequency bands before CSS.
"""
function multiple_narrow_band_filters(traces, dt, periods::Vector{Float64}, bandwidth_percent)
   
    filtered_components = []
    
    # Filter at each central period and accumulate
     filtered_components = map(periods) do T
        center_freq = 1.0 / T
        filtered_data = narrow_band_filter(traces, dt, center_freq, bandwidth_percent / 100.0)
    end
    
    return filtered_components
end

# ╔═╡ aae31a73-fef6-48dc-a2b8-fb9b12c03099
"""
    find_group_arrivals(envelope, time, search_window; max_peaks=4) -> Vector{Tuple{Float64,Float64}}

Find up to `max_peaks` largest envelope peaks in `envelope` within `search_window`
using `Peaks.findmaxima`.

Peaks are returned as `(time, amplitude)` tuples sorted by descending amplitude.
If no strict local maxima are found, the absolute maximum in the window is returned.
Returns an empty vector when the window is empty.
"""
function find_group_arrivals(envelope::Vector{Float64}, time::Vector{Float64},
                             search_window::Tuple{Float64,Float64};
                             max_peaks::Int=4)
    t_start, t_end = search_window
    idx_window = findall(t -> t_start <= t <= t_end, time)
    isempty(idx_window) && return Tuple{Float64,Float64}[]

    env_win = envelope[idx_window]

    # All strict local maxima in the windowed envelope
    peak_idx, peak_vals = findmaxima(env_win)

    if isempty(peak_idx)
        # Fall back: absolute maximum (flat or monotone envelope in window)
        i_best = argmax(env_win)
        return [(time[idx_window[i_best]], env_win[i_best])]
    end

    order = sortperm(peak_vals, rev=true)
    nkeep = min(max_peaks, length(order))
    peaks = Tuple{Float64,Float64}[]
    for j in 1:nkeep
        idx = peak_idx[order[j]]
        push!(peaks, (time[idx_window[idx]], env_win[idx]))
    end
    return peaks
end

# ╔═╡ 7088299d-e267-42e3-9c87-1a238332f0f4
"""
    perform_mft_analysis(trace::SeismicTrace, periods::Vector{Float64};
                        velocity_range=(2.0, 6.0),
                        bandwidth_factor=sqrt(2/25),
                        zero_pad_factor=1,
                        phvel_correction=0.0, compute_phase=true) -> MFTResult

Perform Multiple Filter Technique analysis on a seismic trace.

The Gaussian narrowband filter is controlled by a single fractional bandwidth,
with the equivalent sacmft96 parameter given by

    α = 2 / bandwidth_factor²

so `bandwidth_factor ≈ 0.283` reproduces the previous `α = 25` default.

# Arguments
- `trace`            : Input seismic trace
- `periods`          : Analysis periods [s]
- `velocity_range`   : Expected group velocity range [km/s]
- `bandwidth_factor` : Fractional Gaussian bandwidth relative to centre frequency
- `zero_pad_factor`  : FFT zero-padding factor (1, 2, 4, 8). Improves frequency resolution
                       of the Gaussian filtering while keeping picks on the original time window
- `phvel_correction` : Additional phase [rad]: 0.0 = single-station, π/4 = noise CC
- `compute_phase`    : Compute phase velocity if true

# Returns
- `MFTResult`
"""
function perform_mft_analysis(trace::SeismicTrace, periods::Vector{Float64};
                             velocity_range::Tuple{Float64,Float64}=(2.0, 6.0),
                             bandwidth_factor::Float64=sqrt(2.0 / 25.0),
                             zero_pad_factor::Int=4,
                             phvel_correction::Float64=0.0,
                             compute_phase::Bool=true)

    # 10× oversampling for better arrival-time and phase resolution
    data_unpadded = DSP.resample(trace.data, 10.0)
    dt            = trace.dt / 10.0

    if !(zero_pad_factor in (1, 2, 4, 8))
        throw(ArgumentError("zero_pad_factor must be one of 1, 2, 4, 8"))
    end

    npts_original = length(data_unpadded)
    npts_padded   = npts_original * zero_pad_factor
    data = if zero_pad_factor == 1
        data_unpadded
    else
        vcat(data_unpadded, zeros(Float64, npts_padded - npts_original))
    end

    nfreq       = length(periods)
    frequencies = 1.0 ./ periods

    # Output arrays
    filtered_traces  = zeros(Float64, npts_original, nfreq)
    envelopes        = zeros(Float64, npts_original, nfreq)
    arrival_times    = fill(NaN,     nfreq)
    group_velocities = fill(NaN,     nfreq)
    phase_velocities = fill(NaN,     nfreq)
    phase_branch_numbers = collect(-3:3)
    phase_velocity_branches = fill(NaN, nfreq, length(phase_branch_numbers))
    amplitudes       = zeros(Float64, nfreq)
    quality_factors  = zeros(Float64, nfreq)
    raw_phases       = fill(NaN,     nfreq)   # wrapped φ(tᵍ) per frequency — unwrapped after loop
    all_peaks        = [Tuple{Float64,Float64}[] for _ in 1:nfreq]
    time             = collect(range(dt, step=dt, length=npts_original))

    min_vel, max_vel = velocity_range

    for (i, freq) in enumerate(frequencies)
        # --- Complex analytic signal via one-sided Gaussian ---
        z = narrow_band_filter_analytic(data, dt, freq, bandwidth_factor)[1:npts_original]
        filtered_traces[:, i] = real.(z)

        # Envelope = modulus of analytic signal
        envelope = abs.(z)
        envelopes[:, i] = envelope

        # Velocity-bounded search window
        t_min = trace.distance / max_vel
        t_max = trace.distance / min_vel
        search_window = (t_min, t_max)

        # Up to four strongest peaks in the search window; keep the strongest
        # one as the canonical group-velocity pick for backward compatibility.
        peaks = find_group_arrivals(envelope, time, search_window; max_peaks=4)
        all_peaks[i] = peaks
        t_group, amp_peak = isempty(peaks) ? (NaN, 0.0) : first(peaks)
        arrival_times[i] = t_group

        if !isnan(t_group) && t_group > 0.0
            group_velocities[i] = trace.distance / t_group
        end

        amplitudes[i] = isnan(t_group) ? maximum(envelope) : amp_peak

        # Quality factor: peak / mean (SNR proxy, same as sacmft96)
        mean_env = mean(envelope)
        quality_factors[i] = mean_env > 0.0 ? amplitudes[i] / mean_env : 0.0

        # Collect wrapped phase at group arrival for post-loop unwrapping
        if compute_phase && !isnan(t_group) && t_group > 0.0
            i_g = clamp(round(Int, (t_group - time[1]) / dt) + 1, 1, npts_original)
            raw_phases[i] = angle(z[i_g])
        end
    end

    # --- Phase velocity via cross-frequency phase unwrapping (sacmft96 method) ---
    # sacmft96 resolves the 2πn branch ambiguity by treating the array
    # [ω·tᵍ(f) − φ_wrapped(f) + correction] as a smooth function of frequency
    # and unwrapping it across the frequency axis.  This is equivalent to tracking
    # the phase dispersion curve continuously rather than picking each frequency
    # independently (which would be stuck on the n=0 branch).
    if compute_phase
        valid = .!isnan.(raw_phases) .& .!isnan.(arrival_times) .& (arrival_times .> 0.0)
        valid_idxs = findall(valid)
        if length(valid_idxs) >= 2
            omegas = 2π .* frequencies
            # Match Fortran-style traversal: unwrap from low frequency to high frequency
            # (equivalently long period to short period), independent of user period ordering.
            order = sortperm(frequencies[valid_idxs])
            valid_sorted = valid_idxs[order]
            # Phase differences before unwrapping (each has a 2πn ambiguity)
            phase_diffs_valid = [
                omegas[i] * arrival_times[i] - raw_phases[i] + phvel_correction
                for i in valid_sorted
            ]
            # DSP.unwrap removes jumps > π between consecutive entries,
            # resolving the branch-number ambiguity continuously across frequency.
            phase_diffs_unwrapped = DSP.unwrap(phase_diffs_valid)
            for (j, i) in enumerate(valid_sorted)
                denom0 = phase_diffs_unwrapped[j]
                for (ib, branch) in enumerate(phase_branch_numbers)
                    denom = denom0 + 2π * branch
                    if abs(denom) > 1e-6
                        c = omegas[i] * trace.distance / denom
                        phase_velocity_branches[i, ib] = (0.5 ≤ c ≤ 20.0) ? c : NaN
                    end
                end
                phase_velocities[i] = phase_velocity_branches[i, findfirst(==(0), phase_branch_numbers)]
            end
        end
    end

    return MFTResult(periods, frequencies, group_velocities, phase_velocities,
                     phase_branch_numbers, phase_velocity_branches,
                     amplitudes, filtered_traces, envelopes, arrival_times,
                     trace.distance, quality_factors, all_peaks, time)
end

# ╔═╡ 2de6446f-d07c-4151-8e13-0719c3056826
res = perform_mft_analysis(Xsynthetic, (period_min, period_max), n_periods;
    bandwidth_factor = bandwidth_percent / 100.0,
    zero_pad_factor = zero_pad_factor,
	phvel_correction = phvel_mode == "noise_cc" ? π/4 : 0.0,
	compute_phase    = phvel_mode != "none")

# ╔═╡ 74f5430a-8d8a-41cf-b683-52cc29bb12c0
"""
    find_group_arrival(envelope, time, search_window) -> (t_group, amplitude)

Return the strongest envelope peak in the search window. This preserves the
original group-velocity pick behaviour while `find_group_arrivals` exposes the
additional overtone candidates.
"""
function find_group_arrival(envelope::Vector{Float64}, time::Vector{Float64},
                            search_window::Tuple{Float64,Float64})
    peaks = find_group_arrivals(envelope, time, search_window; max_peaks=1)
    isempty(peaks) && return (NaN, 0.0)
    return first(peaks)
end

# ╔═╡ 0c8a6b9e-5e71-48d3-bff0-072ae085fd5a
"""
    compute_envelope(analytic_signal::Vector{ComplexF64}) -> Vector{Float64}

Compute amplitude envelope from analytic signal.
"""
function compute_envelope(x)
    return abs.(hilbert(x));
end

# ╔═╡ a9eed5d0-7811-4210-999c-ef28ef7769bb
md"## Appendix"

# ╔═╡ 6cf5d72c-2380-4585-86ed-dfd2afda5bd0
md"### Plots"

# ╔═╡ d779b115-911a-44f2-8cb9-6f1962a5d4a3

"""
Create a publication-quality interactive MFT heatmap with arrival time picks.

Features:
- Clean layout suitable for presentations/papers
- Overlaid arrival time curve from group velocity picks
- Proper axis labels and colorbar
- Configurable sizing for journal figures
"""
function plot_envelopes(res::MFTResult;
                  colorscale="Reds",
                  width=800,
                  height=500,
                  font_family="Arial, sans-serif",
                  font_size=20,
                  title="Multiple Filter Technique Analysis",
                  show_picks=true,
                  show_branches=true,
                  pick_color="white",
                  pick_width=3)

	vsticks = range(1, stop=8, step=0.5)
	tticks = res.distance ./ vsticks
	tmin = res.distance ./ 2.0
	tmax = res.distance ./ 6.0
	
    layout = Layout(
        title = attr(
            text = title,
            font = attr(size = font_size + 2, family = font_family)
        ),
        yaxis = attr(
            title = "Velocity (km/s)",
			range = (tmin, tmax),
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            zeroline = false,
            titlefont = attr(size = font_size, family = font_family),
            tickfont = attr(size = font_size - 2, family = font_family),
			tickmode = "array",
        	tickvals = tticks,
        	ticktext = vsticks
        ),
        xaxis = attr(
            title = "Period (s)",
            type = "linear",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            zeroline = false,
            titlefont = attr(size = font_size, family = font_family),
            tickfont = attr(size = font_size - 2, family = font_family)
        ),
        coloraxis = attr(
            colorbar = attr(
                title = attr(
                    text = "Envelope<br>Amplitude",
                    font = attr(size = font_size - 1, family = font_family)
                ),
                tickfont = attr(size = font_size - 2, family = font_family),
                len = 0.85,
                thickness = 20
            ),
            colorscale = colorscale,
            cmin = 0,  # Start from 0 for envelope
            cmax = quantile(vec(res.envelopes), 1.0)  # Clip to a percentile
        ),
        width = width,
        height = height,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin = attr(l=80, r=120, t=80, b=80),
        showlegend = show_picks
    )
    E = res.envelopes ./ maximum(res.envelopes, dims=1)
    # Create heatmap trace
    heatmap_trace = contour(
        y = res.time,
        x = res.periods,
        z = E,
        coloraxis = "coloraxis",
        hovertemplate = "Time: %{x:.2f} s<br>Period: %{y:.2f} s<br>Amplitude: %{z:.3f}<extra></extra>"
    )
    
    traces = [heatmap_trace]

    # Add overtone / multiple-branch picks if requested
    if show_branches && !isempty(res.all_peaks)
        branch_colors = ["cyan", "deepskyblue", "lime", "yellow"]
        nbranches = maximum(length.(res.all_peaks); init=0)
        for branch in 1:nbranches
            x_branch = Float64[]
            y_branch = Float64[]
            for (ip, peaks) in enumerate(res.all_peaks)
                if length(peaks) >= branch
                    push!(x_branch, res.periods[ip])
                    push!(y_branch, peaks[branch][1])
                end
            end
            if !isempty(x_branch)
                push!(traces, scatter(
                    x = x_branch,
                    y = y_branch,
                    mode = "lines+markers",
                    line = attr(color = branch_colors[mod1(branch, length(branch_colors))], width = 1.5, dash = branch == 1 ? "solid" : "dot"),
                    marker = attr(size = branch == 1 ? 6 : 5, color = branch_colors[mod1(branch, length(branch_colors))], symbol = branch == 1 ? "circle-open" : "x"),
                    name = branch == 1 ? "Peak branch 1 (max)" : "Peak branch $(branch)",
                    hovertemplate = "Period: %{x:.2f} s<br>Arrival: %{y:.2f} s<br>Branch: $(branch)<extra></extra>",
                    showlegend = true
                ))
            end
        end
    end
    
    # Add arrival time picks if requested
    if show_picks
        # Filter out NaN values
        valid_indices = findall(!isnan, res.arrival_times)
        if !isempty(valid_indices)
            # Color the picks by the per-period quality factor and add a colorbar
            pick_colors = res.quality_factors[valid_indices]
            # Use grayscale: black -> white (reversed Greys so that higher quality is lighter)
            pick_trace = scatter(
                x = res.periods[valid_indices],
                y = res.arrival_times[valid_indices],
                mode = "lines+markers",
                line = attr(color = pick_color, width = pick_width),
                marker = attr(
                    color = pick_colors,
                    colorscale = "Greys",
                    size = 8,
                    symbol = "circle",
                    line = attr(color = "black", width = 1),
                    showscale = false,
                    colorbar = attr(
                        title = attr(text = "Quality Factor", font = attr(size = font_size - 2)),
                        tickfont = attr(size = font_size - 4),
                        len = 0.7,
                        thickness = 15
                    )
                ),
                name = "Group Velocity Picks",
                hovertemplate = "Period: %{x:.2f} s<br>Arrival: %{y:.2f} s<br>Velocity: " .* 
                               string.(round.(res.group_velocities[valid_indices], digits=2)) .* 
                               " km/s<br>Quality: %{marker.color:.2f}<extra></extra>",
                showlegend = true
            )
            push!(traces, pick_trace)
        end
    end
    
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 09d3b269-c8bd-425a-ad4a-1ecd29150507
plot_envelopes(res, title="Envelopes")

# ╔═╡ df761225-7a00-4a7c-b56f-611243a75e97
function plot_dispersion_curve(results::AbstractVector{MFTResult};
						 names = [string("Result ", i) for i in 1:length(results)],
                         colorscale="Viridis",
                         width=800,
                         height=500,
                         font_family="Arial, sans-serif",
                         font_size=20,
                         title="Group Velocity Dispersion (multiple)",
                         color_by_quality=true,
                         velocity_range=nothing,  # (vmin, vmax) or nothing for auto
                         show_phase=false)  # optionally overlay phase velocity

    # Allow passing a single result as a convenience
    if length(results) == 1
        return plot_dispersion_curve(results[1]; colorscale=colorscale, width=width, height=height,
                                     font_family=font_family, font_size=font_size, title=title,
                                     color_by_quality=color_by_quality, velocity_range=velocity_range,
                                     show_phase=show_phase)
    end

    # Collect valid group velocities across all results to determine autoscaling
    all_group_vals = Float64[]
    for r in results
        append!(all_group_vals, filter(!isnan, r.group_velocities))
    end

    if isempty(all_group_vals)
        @warn "No valid group velocity measurements found across results"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No valid data"]))
    end

    # Determine velocity range
    if isnothing(velocity_range)
        vel_min = 0.9 * minimum(all_group_vals)
        vel_max = 1.1 * maximum(all_group_vals)
    else
        vel_min, vel_max = velocity_range
    end

    layout = Layout(
        title = attr(
            text = title,
            font = attr(size = font_size + 2, family = font_family)
        ),
        xaxis = attr(
            title = "Period (s)",
            type = "linear",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            zeroline = false,
            titlefont = attr(size = font_size, family = font_family),
            tickfont = attr(size = font_size - 2, family = font_family)
        ),
        yaxis = attr(
            title = "Group Velocity (km/s)",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            zeroline = false,
            range = [vel_min, vel_max],
            titlefont = attr(size = font_size, family = font_family),
            tickfont = attr(size = font_size - 2, family = font_family)
        ),
        width = width,
        height = height,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin = attr(l=80, r=120, t=80, b=80),
        showlegend = true
    )

    traces = [scatter()]

    # line color palette to cycle through for different results
    palette = ["steelblue", "firebrick", "darkgreen", "purple", "orange", "brown", "teal", "magenta"]

    # Plot each result using a distinct line color (no markers)
    for (i, r) in enumerate(results)
        valid_idx = findall(!isnan, r.group_velocities)
        if isempty(valid_idx)
            continue
        end

        periods_valid = r.periods[valid_idx]
        group_vel_valid = r.group_velocities[valid_idx]
        color = palette[fld(i+1, 2)]

        # alternate between solid and dotted line styles while keeping the same color
        dash_style = isodd(i) ? "solid" : "dash"
        group_trace = scatter(
            x = periods_valid,
            y = group_vel_valid,
            mode = "lines",
            line = attr(color = color, width = 2, dash = dash_style),
            name = names[i],
            hovertemplate = "Period: %{x:.2f} s<br>Velocity: %{y:.3f} km/s<extra></extra>"
        )

        push!(traces, group_trace)
    end

    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ ee08ecac-5010-4a64-820e-43ee80823939
function plot_dispersion_curve(res::MFTResult;
                         colorscale="Viridis",
                         width=800,
                         height=500,
                         font_family="Arial, sans-serif",
                         font_size=20,
                         title="Group Velocity Dispersion",
                         color_by_quality=true,
                         velocity_range=nothing,  # (vmin, vmax) or nothing for auto
                         show_phase=false)  # optionally overlay phase velocity
    
    # Filter valid measurements (non-NaN velocities)
    valid_idx = findall(!isnan, res.group_velocities)
    
    if isempty(valid_idx)
        @warn "No valid group velocity measurements found"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No valid data"]))
    end
    
    periods_valid = res.periods[valid_idx]
    group_vel_valid = res.group_velocities[valid_idx]
    quality_valid = res.quality_factors[valid_idx]
    
    # Determine velocity range
    if isnothing(velocity_range)
        vel_min = 0.9 * minimum(group_vel_valid)
        vel_max = 1.1 * maximum(group_vel_valid)
    else
        vel_min, vel_max = velocity_range
    end
    
    layout = Layout(
        title = attr(
            text = title,
            font = attr(size = font_size + 2, family = font_family)
        ),
        xaxis = attr(
            title = "Period (s)",
            type = "linear",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            zeroline = false,
            titlefont = attr(size = font_size, family = font_family),
            tickfont = attr(size = font_size - 2, family = font_family)
        ),
        yaxis = attr(
            title = "Group Velocity (km/s)",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            zeroline = false,
            range = [vel_min, vel_max],
            titlefont = attr(size = font_size, family = font_family),
            tickfont = attr(size = font_size - 2, family = font_family)
        ),
        width = width,
        height = height,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin = attr(l=80, r=120, t=80, b=80),
        showlegend = true
    )
    
    traces = [scatter()]
    
    if color_by_quality
        # Normalize quality factors for color mapping
        q_norm = (quality_valid .- minimum(quality_valid)) ./ 
                 (maximum(quality_valid) - minimum(quality_valid) .+ 1e-8)
        
        group_trace = scatter(
            x = periods_valid,
            y = group_vel_valid,
            mode = "markers+lines",
            marker = attr(
                size = 10,
                color = quality_valid,
                colorscale = colorscale,
                showscale = true,
                colorbar = attr(
                    title = "Quality<br>Factor",
                    titlefont = attr(size = font_size - 2),
                    tickfont = attr(size = font_size - 4),
                    len = 0.7,
                    thickness = 15
                ),
                line = attr(color = "white", width = 1)
            ),
            line = attr(color = "gray", width = 1, dash = "dot"),
            name = "Group Velocity",
            hovertemplate = "Period: %{x:.2f} s<br>Velocity: %{y:.3f} km/s<br>Quality: %{marker.color:.1f}<extra></extra>"
        )
    else
        group_trace = scatter(
            x = periods_valid,
            y = group_vel_valid,
            mode = "markers+lines",
            marker = attr(
                size = 10,
                color = "steelblue",
                line = attr(color = "white", width = 1)
            ),
            line = attr(color = "steelblue", width = 2),
            name = "Group Velocity",
            hovertemplate = "Period: %{x:.2f} s<br>Velocity: %{y:.3f} km/s<extra></extra>"
        )
    end
    
    push!(traces, group_trace)

    # --- Phase velocity overlay (sacmft96 dophvel implementation) ---
    if show_phase
        phv_idx = findall(i -> !isnan(res.phase_velocities[i]), 1:length(res.periods))
        if !isempty(phv_idx)
            phase_trace = scatter(
                x = res.periods[phv_idx],
                y = res.phase_velocities[phv_idx],
                mode = "markers+lines",
                marker = attr(
                    size = 8, color = "firebrick", symbol = "diamond",
                    line = attr(color = "white", width = 1)
                ),
                line = attr(color = "firebrick", width = 2, dash = "dash"),
                name = "Phase Velocity",
                hovertemplate = "Period: %{x:.2f} s<br>Phase Vel: %{y:.3f} km/s<extra></extra>"
            )
            push!(traces, phase_trace)
        end
    end

    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 1585e739-23c2-4a23-8e80-6f132d503765
plot_dispersion_curve([res, res], title=string("Group Velocity"))

# ╔═╡ 3bab4eab-92da-4539-b3df-d5c637d1ce77
# Plot all individual CCs with the stack overlaid
function plot_crosscorr_bundle(result; title="Cross-Correlations")
    traces = [scatter()]
    
    # Individual CCs (semi-transparent)
    for (i, cc) in enumerate(result.cc_all)
        push!(traces, scatter(
            x = result.lags,
            y = cc,
            mode = "lines",
            line = attr(color = "lightgray", width = 0.5),
            showlegend = false,
            hoverinfo = "skip"
        ))
    end
    
    # Stacked average (bold)
    push!(traces, scatter(
        x = result.lags,
        y = result.cc_average,
        mode = "lines",
        line = attr(color = "red", width = 3),
        name = "Stack (n=$(result.n_events))"
    ))
    
    layout = Layout(
        title = title,
        xaxis = attr(title = "Time Lag (s)", zeroline = true),
        yaxis = attr(title = "Normalized Amplitude", zeroline = true),
        width = 900,
        height = 500,
        plot_bgcolor = "white"
    )
    
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ c7a19591-be69-423d-a510-7ec0c0cedf27
md"## Tests"

# ╔═╡ c322f226-d679-454a-9bcb-4f67f46b2bb2
"""
    multi_frequency_cosine_sum(t, x; freqs, c0, α, freq_weights=nothing)

Generate a synthetic signal by directly summing monochromatic cosines with
dispersive phase delay at distance `x`:

    u(t) = Σ wᵢ cos(2π fᵢ t - k(fᵢ) x)

where `k(fᵢ)` satisfies `2πf = c0*k + α*k²`.

Returns `(u, cg, cp, tg)` with theoretical references derived from the same
dispersion relation.
"""
function multi_frequency_cosine_sum(t::Vector{Float64}, x::Float64;
                                    freqs::Vector{Float64},
                                    c0::Float64,
                                    α::Float64,
                                    freq_weights=nothing)
    u = zeros(Float64, length(t))
    if freq_weights === nothing
        freq_weights = ones(length(freqs))
    end

    cg = similar(freqs)
    cp = similar(freqs)
    tg = similar(freqs)

    for (i, fc) in enumerate(freqs)
        ω0 = 2π * fc
        k0 = if abs(α) < 1e-12
            ω0 / c0
        else
            disc = c0^2 + 4α*ω0
            disc <= 0 && throw(ArgumentError("Invalid (c0, α, f): non-positive discriminant for k0"))
            (-c0 + sqrt(disc)) / (2α)
        end
        cg[i] = c0 + 2α*k0
        cp[i] = ω0 / k0
        tg[i] = x / cg[i]

        u .+= freq_weights[i] .* cos.(ω0 .* t .- k0 * x)
    end

    return u, cg, cp, tg
end

# ╔═╡ 0e5573f5-5322-4784-8c53-97230373dc13
inv(50)

# ╔═╡ f0db8807-ddd7-47b8-a792-54b15fb07c7e
# synthetic-controls
md"""
### Synthetic Experiment Controls

**Distance (km):** $(@bind synth_distance Slider(50:10:500, default=240, show_value=true))

**MFT bandwidth (% of centre frequency):** $(@bind synth_bandwidth Slider(5.0:1.0:100.0, default=28.0, show_value=true))

**FFT zero-padding:** $(@bind synth_zero_pad_factor Select([1 => "1× (none)", 2 => "2×", 4 => "4×", 8 => "8×"]))

**Noise (%):** $(@bind noise_perc Slider(0:1:100, default=5, show_value=true))
"""

# ╔═╡ ccfe0c61-2be2-4bf0-b01d-b3fa06bf2f05
# prefilter-controls
md"""
### Pre-filtering Controls (before MFT)

**Apply pre-filter:** $(@bind apply_prefilter_flag CheckBox(default=false))


**Number of log-spaced bands (3–50 s):** $(@bind prefilter_nbands Slider(2:1:8, default=3, show_value=true))

**Pre-filter bandwidth (%):** $(@bind prefilter_bw Slider(5:1:100, default=80, show_value=true))
"""

# ╔═╡ c2d1790c-cbac-447a-a234-dfb4d6d188e0
@bind resample_noise CounterButton("Resample noise")

# ╔═╡ 78ba3876-d4d0-4eb8-8300-1c3cef28db58
synthetic_test_data = let
resample_noise
	
	x  = Float64(synth_distance)     # 120 km
	c0 = 3.5
	α  = 1.0
	
	freqs = collect(0.02:0.001:0.3)   # primary microseism band
	dt = inv(2.0 * maximum(freqs))
	t  = collect(0:dt:300)
	
    u, cg, cp, tg = multi_frequency_cosine_sum(t, x;
        freqs=freqs,
        c0=c0,
        α=α)
    (; data=u./maximum(abs, u) + noise_perc / 100.0 * randn(length(u)), t, cg, cp, tg, distance=x, dt, periods = inv.(freqs), freqs )
end

# ╔═╡ 7a64a743-db7b-405c-a334-793d2b8a36db
plot(synthetic_test_data.t, synthetic_test_data.data)

# ╔═╡ 875e8a94-2a80-4ad9-8cc6-c48adaab8e4f
# ╠═╡ disabled = true
#=╠═╡
# comparison-plot
let
	res_synthetic_test = res_synthetic_test.res
    # Create comparison plot of expected vs measured group velocities
    
    # Filter valid measurements
    valid_idx = findall(!isnan, res_synthetic_test.group_velocities)
    
    traces = [scatter()]
    
    # Plot 1: Theoretical group velocity (from synthetic data generation)
    push!(traces, scatter(
        x = synthetic_test_data.periods,
        y = synthetic_test_data.cg,
        mode = "lines+markers",
        name = "Expected (Theoretical)",
        line = attr(color = "black", width = 3),
        marker = attr(size = 8, symbol = "circle"),
        hovertemplate = "Period: %{x:.2f} s<br>Velocity: %{y:.3f} km/s<extra></extra>"
    ))

    # Plot 1b: Theoretical phase velocity
    push!(traces, scatter(
        x = synthetic_test_data.periods,
        y = synthetic_test_data.cp,
        mode = "lines+markers",
        name = "Expected Phase (Theoretical)",
        line = attr(color = "dimgray", width = 2, dash = "dot"),
        marker = attr(size = 7, symbol = "triangle-up"),
        hovertemplate = "Period: %{x:.2f} s<br>Phase Vel: %{y:.3f} km/s<extra></extra>"
    ))
    
    # Plot 2: MFT measured group velocity
    push!(traces, scatter(
        x = res_synthetic_test.periods[valid_idx],
        y = res_synthetic_test.group_velocities[valid_idx],
        mode = "lines+markers",
        name = "MFT Group Velocity",
        line = attr(color = "red", width = 2, dash = "dash"),
        marker = attr(size = 10, symbol = "diamond", color = "red"),
        hovertemplate = "Period: %{x:.2f} s<br>Group Vel: %{y:.3f} km/s<extra></extra>"
    ))

    # Plot 3: MFT phase velocity (when computed)
    phv_idx = findall(!isnan, res_synthetic_test.phase_velocities)
    if !isempty(phv_idx)
        push!(traces, scatter(
            x = res_synthetic_test.periods[phv_idx],
            y = res_synthetic_test.phase_velocities[phv_idx],
            mode = "lines+markers",
            name = "MFT Phase Velocity",
            line = attr(color = "steelblue", width = 2, dash = "dot"),
            marker = attr(size = 8, symbol = "triangle-up", color = "steelblue"),
            hovertemplate = "Period: %{x:.2f} s<br>Phase Vel: %{y:.3f} km/s<extra></extra>"
        ))
    end

    # Compute errors against theoretical group and phase references
    metrics = compute_velocity_error_metrics(
        res_synthetic_test,
        synthetic_test_data.periods,
        synthetic_test_data.cg,
        synthetic_test_data.cp
    )
    @printf("Group mean relative error: %.2f%%\n", metrics.group_mean)
    @printf("Group max  relative error: %.2f%%\n", metrics.group_max)
    @printf("Phase mean relative error: %.2f%%\n", metrics.phase_mean)
    @printf("Phase max  relative error: %.2f%%\n", metrics.phase_max)
    @printf("Valid group picks: %d / %d\n", metrics.n_group, length(res_synthetic_test.periods))
    @printf("Valid phase picks: %d / %d\n", metrics.n_phase, length(res_synthetic_test.periods))
    
    layout = Layout(
        title = attr(
            text = "MFT Group Velocity Recovery: Expected vs Measured",
            font = attr(size = 20)
        ),
        xaxis = attr(
            title = "Period (s)",
            type = "linear",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            titlefont = attr(size = 18),
            tickfont = attr(size = 14)
        ),
        yaxis = attr(
            title = "Group Velocity (km/s)",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            titlefont = attr(size = 18),
            tickfont = attr(size = 14)
        ),
        width = 900,
        height = 600,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        showlegend = true,
        legend = attr(
            x = 0.02,
            y = 0.98,
            bgcolor = "rgba(255,255,255,0.8)",
            bordercolor = "black",
            borderwidth = 1
        )
    )
    
   WideCell( PlutoPlotly.plot(traces, layout))
end
  ╠═╡ =#

# ╔═╡ 1293cef0-186d-4a43-86f5-181372ef59e0
log_period_bands = begin
    Tmin, Tmax = 10.0, 40.0
    exp10.(range(log10(Tmin), log10(Tmax), length=prefilter_nbands))
end

# ╔═╡ 7b8ea90d-a09d-4952-bd66-943808dc9d3b
res_synthetic_test = let
	# plot(synthetic_data)
			
	       

	        # Apply multi-band pre-filter if enabled
	        filtered_components = nothing
	        D = if apply_prefilter_flag

	            filtered_components = multiple_narrow_band_filters(synthetic_test_data.data, synthetic_test_data.dt, log_period_bands, prefilter_bw)
	            @info "Applied multi-band pre-filter: T=$(log_period_bands) s, BW=$(prefilter_bw)%"
				mean(filtered_components)
			else
				synthetic_test_data.data
	        end

	 # Create seismic trace
	        trace = SeismicTrace(
	            data=D,
	            dt=synthetic_test_data.dt,
	            distance=synthetic_test_data.distance
	        )

	
	        # Perform MFT analysis
	        periods_analysis = synthetic_test_data.periods
	        (; res=perform_mft_analysis(
	            trace,
	            periods_analysis,
	            velocity_range = (2.0 - 0.5, 5.0 + 0.5),
                bandwidth_factor = synth_bandwidth / 100.0,
                zero_pad_factor = synth_zero_pad_factor,
	            compute_phase  = true
	        ), filtered_trace=trace, filtered_components=filtered_components)
end

# ╔═╡ c169578d-3dd2-42f8-8e57-4146a8dca2cd
plot_dispersion_curve(res_synthetic_test.res)

# ╔═╡ 2ba1a355-d9fa-4268-b046-0b201191c11e
plot_envelopes(res_synthetic_test.res)

# ╔═╡ 44fad356-ff66-4bbc-a83b-5d8d996d95a5
plot(res_synthetic_test.filtered_trace.data)

# ╔═╡ 8dc0f4bc-00cb-11f1-9443-a59380edd23b
let
	# Show original vs pre-filtered data with individual components
    if apply_prefilter_flag
        trace_original = SeismicTrace(
            data=synthetic_test_data.data,
            dt=synthetic_test_data.dt,
            distance=synthetic_test_data.distance
        )
        
        central_periods = log_period_bands
        
        # Get filtered components and sum
        filtered_trace = res_synthetic_test.filtered_trace
        filtered_components = res_synthetic_test.filtered_components
        
        # Plot comparison
        traces = [scatter()]
        
        # Original signal
        push!(traces, scatter(
            x = synthetic_test_data.t,
            y = synthetic_test_data.data ./ maximum(abs.(synthetic_test_data.data)),
            mode = "lines",
            name = "Original",
            line = attr(color = "lightgray", width = 1),
            opacity = 0.3
        ))
        
        # Individual filtered components
        if !isnothing(filtered_components)
            colors = ["blue", "green", "orange"]
            for (i, (T, filt, color)) in enumerate(zip(central_periods, filtered_components, colors))
                # Normalize for display
                filt_norm = filt ./ maximum(abs.(filt)) .* 0.5
                push!(traces, scatter(
                    x = synthetic_test_data.t,
                    y = filt_norm,
                    mode = "lines",
                    name = "T=$(T)s band",
                    line = attr(color = color, width = 1.5, dash = "dot"),
                    opacity = 0.6
                ))
            end
        end
        
        # Summed filtered signal
        push!(traces, scatter(
            x = synthetic_test_data.t,
            y = filtered_trace.data ./ maximum(abs.(filtered_trace.data)),
            mode = "lines",
            name = "Sum of 3 bands",
            line = attr(color = "red", width = 2.5)
        ))
        
        layout = Layout(
            title = "Multi-band Pre-filtering: Individual Components + Sum",
            xaxis = attr(title = "Time (s)"),
            yaxis = attr(title = "Amplitude"),
            width = 900,
            height = 500,
            plot_bgcolor = "white",
            legend = attr(x = 0.02, y = 0.98)
        )
        
        PlutoPlotly.plot(traces, layout)
    else
        md"**Pre-filter disabled.** Check the box above to enable."
    end
end

# ╔═╡ 8dc0fc46-00cb-11f1-9407-453086ae541f
let
	if apply_prefilter_flag
        # Compute FFT of original and filtered data
        original_fft = fft(synthetic_test_data.data)
        filtered_fft = fft(res_synthetic_test.filtered_trace.data)
        
        # Frequency vector
        n = length(synthetic_test_data.data)
        freqs = fftfreq(n, 1.0/synthetic_test_data.dt)
        
        # Only positive frequencies
        pos_idx = freqs .> 0
        freqs_pos = freqs[pos_idx]
        
        # Amplitude spectrum
        amp_original = abs.(original_fft[pos_idx])
        amp_filtered = abs.(filtered_fft[pos_idx])
        
        # Normalize
        amp_original ./= maximum(amp_original)
        amp_filtered ./= maximum(amp_filtered)
        
        traces_spec = [scatter()]
        
        push!(traces_spec, scatter(
            x = freqs_pos,
            y = amp_original,
            mode = "lines",
            name = "Original Spectrum",
            line = attr(color = "gray", width = 2),
            opacity = 0.5
        ))
        
        push!(traces_spec, scatter(
            x = freqs_pos,
            y = amp_filtered,
            mode = "lines",
            name = "Filtered Spectrum (sum)",
            line = attr(color = "red", width = 2.5)
        ))
        
        # Mark center frequencies for all three periods
        central_periods = log_period_bands
        colors = ["blue", "green", "orange"]
        for (T, color) in zip(central_periods, colors)
            fc = 1.0 / T
            push!(traces_spec, scatter(
                x = [fc, fc],
                y = [1e-4, 1],
                mode = "lines",
                name = "fc=$(round(fc, digits=3)) Hz (T=$(T)s)",
                line = attr(color = color, width = 2, dash = "dash")
            ))
        end
        
        layout_spec = Layout(
            title = "Frequency Domain: 3-Band Pre-filtering",
            xaxis = attr(
                title = "Frequency (Hz)",
                range = [-2, log10(0.3)],
                type = "log"
            ),
            yaxis = attr(
                title = "Normalized Amplitude",
                type = "log",
                range = [-4, 0]
            ),
            width = 900,
            height = 500,
            plot_bgcolor = "white",
            legend = attr(x = 0.65, y = 0.98, font=attr(size=10))
        )
        
        PlutoPlotly.plot(traces_spec, layout_spec)
    else
        md"Enable pre-filter to see spectrum comparison."
    end
end

# ╔═╡ 64882ed3-eea4-45a7-bf0f-6d0c1e14d5af


# ╔═╡ 6f3cb4dc-11ec-4e8d-a9f6-c4c2301cbcf1
"""
    compute_velocity_error_metrics(res::MFTResult, periods_true, cg_true, cp_true)

Compute mean/max relative errors (%) for recovered group and phase velocities
against theoretical synthetic references at nearest periods.
"""
function compute_velocity_error_metrics(res::MFTResult,
                                        periods_true::Vector{Float64},
                                        cg_true::Vector{Float64},
                                        cp_true::Vector{Float64})
    group_errors = Float64[]
    phase_errors = Float64[]

    for i in eachindex(res.periods)
        idx_ref = argmin(abs.(periods_true .- res.periods[i]))

        gv = res.group_velocities[i]
        if !isnan(gv)
            gref = cg_true[idx_ref]
            push!(group_errors, 100.0 * abs(gv - gref) / max(abs(gref), 1e-8))
        end

        pv = res.phase_velocities[i]
        if !isnan(pv)
            pref = cp_true[idx_ref]
            push!(phase_errors, 100.0 * abs(pv - pref) / max(abs(pref), 1e-8))
        end
    end

    group_mean = isempty(group_errors) ? NaN : mean(group_errors)
    group_max = isempty(group_errors) ? NaN : maximum(group_errors)
    phase_mean = isempty(phase_errors) ? NaN : mean(phase_errors)
    phase_max = isempty(phase_errors) ? NaN : maximum(phase_errors)

    return (; group_mean, group_max, phase_mean, phase_max,
            n_group=length(group_errors), n_phase=length(phase_errors))
end




# ╔═╡ 7fd65030-f541-4fd4-aea8-8ab6d9cc6d0e
@bind run_synth_suite CounterButton("Run synthetic MFT suite")

# ╔═╡ 78a3288f-6b85-4ed1-a399-b4528799f95a
let
    run_synth_suite

    distances = [100.0, 250.0, 500.0, 1000.0]
    noises = [0.0, 5.0, 10.0]
    freqs = collect(0.02:0.01:0.30)
    periods = inv.(freqs)

    rows = String[]
    push!(rows, "Synthetic MFT benchmark (group + phase errors)")
    push!(rows, "dist[km] noise[%] g_mean[%] g_max[%] p_mean[%] p_max[%] n_g n_p")

    for dist in distances
        for noise in noises
            dt = inv(2.0 * maximum(freqs))
            t = collect(0:dt:300)

            u, cg, cp, tg = multi_frequency_cosine_sum(
                t, dist;
                freqs=freqs,
                c0=3.5,
                α=1.0
            )
            data = u ./ maximum(abs, u) .+ noise / 100.0 * randn(length(u))
            trace = SeismicTrace(data=data, dt=dt, distance=dist)

            res_case = perform_mft_analysis(
                trace,
                periods;
                velocity_range=(1.5, 6.5),
                bandwidth_factor=synth_bandwidth / 100.0,
                zero_pad_factor=synth_zero_pad_factor,
                compute_phase=true
            )

            m = compute_velocity_error_metrics(res_case, periods, cg, cp)
            push!(rows,
                @sprintf("%7.1f %8.1f %8.2f %8.2f %8.2f %8.2f %4d %4d",
                         dist, noise, m.group_mean, m.group_max,
                         m.phase_mean, m.phase_max, m.n_group, m.n_phase)
            )
        end
    end

    print("```\n" * join(rows, "\n") * "\n```")
end

# ╔═╡ 871ae074-31c8-4b62-8a94-3aa018b1de87


# ╔═╡ 6d5ab2a2-8ce2-4b1d-96ea-0aa1c27e83fd
function plot_phase_velocities(res::MFTResult;
                         width=900,
                         height=550,
                         font_family="Arial, sans-serif",
                         font_size=20,
                         title="Phase Velocity Branches",
                         velocity_range=nothing,
                         show_group=true)

    valid_phase = .!isnan.(res.phase_velocity_branches)
    if !any(valid_phase)
        @warn "No valid phase velocity branch measurements found"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No valid phase branch data"]))
    end

    all_phase_vals = vec(res.phase_velocity_branches[valid_phase])
    all_vel_vals = copy(all_phase_vals)
    if show_group
        append!(all_vel_vals, filter(!isnan, res.group_velocities))
    end

    if isempty(all_vel_vals)
        @warn "No valid velocity measurements found"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No valid velocity data"]))
    end

    if isnothing(velocity_range)
        vel_min = 0.9 * minimum(all_vel_vals)
        vel_max = 1.1 * maximum(all_vel_vals)
    else
        vel_min, vel_max = velocity_range
    end

    layout = Layout(
        title = attr(
            text = title,
            font = attr(size = font_size + 2, family = font_family)
        ),
        xaxis = attr(
            title = "Period (s)",
            type = "linear",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            zeroline = false,
            titlefont = attr(size = font_size, family = font_family),
            tickfont = attr(size = font_size - 2, family = font_family)
        ),
        yaxis = attr(
            title = "Velocity (km/s)",
            showgrid = true,
            gridcolor = "rgba(128,128,128,0.2)",
            zeroline = false,
            range = [vel_min, vel_max],
            titlefont = attr(size = font_size, family = font_family),
            tickfont = attr(size = font_size - 2, family = font_family)
        ),
        width = width,
        height = height,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin = attr(l=80, r=120, t=80, b=80),
        showlegend = true
    )

    traces = [scatter()]
    branch_palette = ["#2c7bb6", "#00a6ca", "#00ccbc", "#d7191c", "#fdae61", "#abd9e9", "#7b3294"]
    zero_idx = findfirst(==(0), res.phase_branch_numbers)

    for (ib, branch) in enumerate(res.phase_branch_numbers)
        valid_idx = findall(i -> !isnan(res.phase_velocity_branches[i, ib]), 1:length(res.periods))
        isempty(valid_idx) && continue

        is_canonical = ib == zero_idx
        push!(traces, scatter(
            x = res.periods[valid_idx],
            y = res.phase_velocity_branches[valid_idx, ib],
            mode = "lines+markers",
            marker = attr(
                size = is_canonical ? 8 : 6,
                color = branch_palette[mod1(ib, length(branch_palette))],
                symbol = is_canonical ? "diamond" : "circle-open",
                line = attr(color = "white", width = 1)
            ),
            line = attr(
                color = branch_palette[mod1(ib, length(branch_palette))],
                width = is_canonical ? 3 : 1.5,
                dash = is_canonical ? "solid" : (branch < 0 ? "dot" : "dash")
            ),
            name = branch == 0 ? "Phase branch 0 (unwrapped)" : "Phase branch $(branch > 0 ? "+" : "")$(branch)",
            hovertemplate = "Period: %{x:.2f} s<br>Phase Vel: %{y:.3f} km/s<br>Branch: $(branch)<extra></extra>"
        ))
    end

    if show_group
        valid_group = findall(!isnan, res.group_velocities)
        if !isempty(valid_group)
            push!(traces, scatter(
                x = res.periods[valid_group],
                y = res.group_velocities[valid_group],
                mode = "lines+markers",
                marker = attr(size = 7, color = "black", symbol = "x"),
                line = attr(color = "black", width = 2, dash = "dot"),
                name = "Group velocity",
                hovertemplate = "Period: %{x:.2f} s<br>Group Vel: %{y:.3f} km/s<extra></extra>"
            ))
        end
    end

    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 9db0d269-d3ca-48fc-9583-91b5e11822d2
WideCell(plot_phase_velocities(res_synthetic_test.res))

# ╔═╡ 76f5c114-3cdb-11f1-917c-2b05ca51409f
"""
    MultimodalDispersion

Structure to hold a single continuous dispersion curve representing one mode (fundamental or overtone).

Fields:
- `periods::Vector{Float64}` — Analysis periods [s]
- `arrival_times::Vector{Float64}` — Group arrival times [s] at each period
- `group_velocities::Vector{Float64}` — Computed group velocities [km/s]
- `peak_amplitudes::Vector{Float64}` — Envelope peak amplitudes at each period (quality proxy)
- `mode_index::Int` — Mode label (1=fundamental, 2=first overtone, etc.)

All vectors must have equal length. Represents one continuous mode across all analysis periods.
"""
struct MultimodalDispersion
    periods::Vector{Float64}
    arrival_times::Vector{Float64}
    group_velocities::Vector{Float64}
    peak_amplitudes::Vector{Float64}
    mode_index::Int
    
    function MultimodalDispersion(periods, arrival_times, group_velocities, peak_amplitudes, mode_index)
        n = length(periods)
        @assert length(arrival_times) == n "arrival_times length must match periods"
        @assert length(group_velocities) == n "group_velocities length must match periods"
        @assert length(peak_amplitudes) == n "peak_amplitudes length must match periods"
        new(periods, arrival_times, group_velocities, peak_amplitudes, mode_index)
    end
end

# ╔═╡ 82c85400-3cdb-11f1-9af8-1da7c901570c
"""
    extract_all_peaks_matrix(res::MFTResult, distance::Float64)

Convert sparse all_peaks storage into dense matrices for multipeak analysis.

Returns: `(periods, peak_times_matrix, peak_amps_matrix)`
- `peak_times_matrix[i, j]` — arrival time (s) of j-th peak at i-th period, or NaN if peak not found
- `peak_amps_matrix[i, j]` — amplitude of j-th peak, or NaN if peak not found
"""
function extract_all_peaks_matrix(res::MFTResult, distance::Float64)
    nperiods = length(res.periods)
    
    # Find maximum number of peaks across all periods
    max_peaks = maximum(length(peaks) for peaks in res.all_peaks; init=0)
    max_peaks = max(max_peaks, 1)  # Ensure at least 1
    
    # Initialize matrices with NaN
    peak_times = fill(NaN, nperiods, max_peaks)
    peak_amps = fill(NaN, nperiods, max_peaks)
    
    # Populate matrices from sparse storage
    for (i, peaks) in enumerate(res.all_peaks)
        for (j, (t_arrival, amplitude)) in enumerate(peaks)
            if j <= max_peaks
                peak_times[i, j] = t_arrival
                peak_amps[i, j] = amplitude
            end
        end
    end
    
    return (res.periods, peak_times, peak_amps)
end

# ╔═╡ 82c857c8-3cdb-11f1-8b14-3f44a70d8c9a
"""
    match_peaks_continuous(periods::Vector{Float64}, peak_times::Matrix{Float64},
                          peak_amps::Matrix{Float64}; velocity_range::Tuple{Float64,Float64}=(2.0, 6.0),
                          distance::Float64=1.0, start_peak_idx::Int=1) -> MultimodalDispersion

Extract one continuous dispersion curve via greedy period-to-period peak matching.

**Algorithm:** For each period, select the peak closest (in time) to the previous period's pick.
This minimizes period-to-period velocity jumps, producing a smooth single-mode dispersion curve.

# Arguments
- `periods` : Analysis periods [s]
- `peak_times` : Matrix of arrival times [period × peak], NaN for missing peaks
- `peak_amps` : Matrix of peak amplitudes [period × peak], for quality tracking
- `velocity_range` : Expected group velocity bounds for validation (currently unused; for validation only)
- `distance` : Source-receiver distance [km] for group velocity computation
- `start_peak_idx` : Which peak (column) to start from (default: 1 = strongest peak)

# Returns
- `MultimodalDispersion` — Continuous mode with periods, arrival times, velocities, amplitudes
"""
function match_peaks_continuous(periods::Vector{Float64}, peak_times::Matrix{Float64},
                               peak_amps::Matrix{Float64}; 
                               velocity_range::Tuple{Float64,Float64}=(2.0, 6.0),
                               distance::Float64=1.0,
                               start_peak_idx::Int=1)
    nperiods = length(periods)
    
    arrival_times = fill(NaN, nperiods)
    amplitudes = fill(NaN, nperiods)
    
    # Start with specified peak from first period
    if start_peak_idx <= size(peak_times, 2) && !isnan(peak_times[1, start_peak_idx])
        arrival_times[1] = peak_times[1, start_peak_idx]
        amplitudes[1] = peak_amps[1, start_peak_idx]
    else
        # Fallback: use first non-NaN peak in first period
        for j in 1:size(peak_times, 2)
            if !isnan(peak_times[1, j])
                arrival_times[1] = peak_times[1, j]
                amplitudes[1] = peak_amps[1, j]
                break
            end
        end
    end
    
    # Return NaN if first period has no valid peaks
    if isnan(arrival_times[1])
        group_vels = fill(NaN, nperiods)
        return MultimodalDispersion(periods, arrival_times, group_vels, amplitudes, 0)
    end
    
    # Greedy matching for subsequent periods
    for i in 2:nperiods
        prev_time = arrival_times[i-1]
        
        # Find closest non-NaN peak in current period
        best_j = 0
        best_dist = Inf
        
        for j in 1:size(peak_times, 2)
            t_candidate = peak_times[i, j]
            if !isnan(t_candidate)
                dist = abs(t_candidate - prev_time)
                if dist < best_dist
                    best_dist = dist
                    best_j = j
                end
            end
        end
        
        if best_j > 0
            arrival_times[i] = peak_times[i, best_j]
            amplitudes[i] = peak_amps[i, best_j]
        end
    end
    
    # Compute group velocities: v_g = distance / arrival_time
    group_vels = similar(arrival_times)
    for i in 1:nperiods
        if !isnan(arrival_times[i]) && arrival_times[i] > 0
            group_vels[i] = distance / arrival_times[i]
        else
            group_vels[i] = NaN
        end
    end
    
    return MultimodalDispersion(periods, arrival_times, group_vels, amplitudes, 1)
end

# ╔═╡ 82c86256-3cdb-11f1-befc-c7b40eeb6b06
"""
    extract_all_modes(res::MFTResult; distance::Float64=1.0, max_modes::Int=4,
                     velocity_range::Tuple{Float64,Float64}=(2.0, 6.0)) -> Vector{MultimodalDispersion}

Extract all continuous dispersion modes from MFT result.

**Strategy:** Iteratively extract modes by greedy peak matching, starting from each unused initial peak.
Marks peaks as "used" to prevent duplicate mode extraction.

# Arguments
- `res` : MFTResult from perform_mft_analysis
- `distance` : Source-receiver distance [km]
- `max_modes` : Maximum number of modes to extract (default: 4)
- `velocity_range` : Velocity bounds for validation (default: 2–6 km/s)

# Returns
- `Vector{MultimodalDispersion}` — Extracted modes, sorted by initial peak amplitude (strongest first)
"""
function extract_all_modes(res::MFTResult; distance::Float64=1.0, max_modes::Int=4,
                          velocity_range::Tuple{Float64,Float64}=(2.0, 6.0))
    periods, peak_times, peak_amps = extract_all_peaks_matrix(res, distance)
    
    modes = MultimodalDispersion[]
    used_peaks = falses(size(peak_times))
    
    for mode_idx in 1:max_modes
        # Find next unused peak in first period
        start_peak_idx = 0
        for j in 1:size(peak_times, 2)
            if !used_peaks[1, j] && !isnan(peak_times[1, j])
                start_peak_idx = j
                break
            end
        end
        
        # No more peaks to extract
        if start_peak_idx == 0
            break
        end
        
        # Extract continuous mode starting from this peak
        mode = match_peaks_continuous(periods, peak_times, peak_amps; 
                                       distance=distance, 
                                       velocity_range=velocity_range,
                                       start_peak_idx=start_peak_idx)
        
        # Mark used peaks
        for (i, t_arrival) in enumerate(mode.arrival_times)
            if !isnan(t_arrival)
                # Find which column was used
                for j in 1:size(peak_times, 2)
                    if !isnan(peak_times[i, j]) && abs(peak_times[i, j] - t_arrival) < 1e-6
                        used_peaks[i, j] = true
                        break
                    end
                end
            end
        end
        
        # Assign mode index and append
        mode = MultimodalDispersion(mode.periods, mode.arrival_times, mode.group_velocities, 
                                   mode.peak_amplitudes, length(modes) + 1)
        push!(modes, mode)
    end
    
    return modes
end

# ╔═╡ 347f96a4-3cdc-11f1-9cee-ab8ff408ac98
"""
    BranchAnalysisResult

Structure to hold MFT analysis results for both causal and acausal branches.

Fields:
- `causal_result::MFTResult` — MFT result from causal branch
- `acausal_result::MFTResult` — MFT result from acausal branch
- `causal_modes::Vector{MultimodalDispersion}` — Extracted modes from causal branch
- `acausal_modes::Vector{MultimodalDispersion}` — Extracted modes from acausal branch
- `branch_correlation::Vector{Float64}` — Zero-lag correlation at each period (causal vs acausal)
- `periods::Vector{Float64}` — Analysis periods [s]
- `distance::Float64` — Source-receiver distance [km]

Enables comparison of dispersion curves between causal (positive time lags) and acausal (negative, time-reversed) 
branches of ambient noise cross-correlations for validation and multimodal tracking.
"""
struct BranchAnalysisResult
    causal_result::MFTResult
    acausal_result::MFTResult
    causal_modes::Vector{MultimodalDispersion}
    acausal_modes::Vector{MultimodalDispersion}
    branch_correlation::Vector{Float64}
    periods::Vector{Float64}
    distance::Float64
end

# ╔═╡ 347f9a3c-3cdc-11f1-8f82-9d9b48d7858b
"""
    zero_lag_correlation(filtered_traces_causal::Matrix{Float64}, filtered_traces_acausal::Matrix{Float64})

Compute normalized zero-lag cross-correlation between causal and acausal filtered traces at each frequency.

**Purpose:** Validates consistency between causal and acausal branches of ambient noise correlations.
High correlation (>0.85) indicates reliable, physically consistent wave propagation across both branches.

# Arguments
- `filtered_traces_causal` : Filtered narrowband time series [time × frequency] from causal branch
- `filtered_traces_acausal` : Filtered narrowband time series [time × frequency] from acausal branch

# Returns
- `Vector{Float64}` — Correlation coefficient [-1,1] at each frequency (period)
"""
function zero_lag_correlation(filtered_traces_causal::Matrix{Float64}, filtered_traces_acausal::Matrix{Float64})
    nfreq = size(filtered_traces_causal, 2)
    correlations = Float64[]
    
    for i in 1:nfreq
        fc = filtered_traces_causal[:, i]
        fa = filtered_traces_acausal[:, i]
        
        # Normalize
        fc_norm = fc .- mean(fc)
        fa_norm = fa .- mean(fa)
        
        # Compute correlation
        numerator = sum(fc_norm .* fa_norm)
        denominator = sqrt(sum(fc_norm.^2) * sum(fa_norm.^2))
        
        if denominator > 0
            corr = numerator / denominator
            push!(correlations, clamp(corr, -1.0, 1.0))  # Clamp to [-1, 1] range
        else
            push!(correlations, NaN)
        end
    end
    
    return correlations
end

# ╔═╡ 347f9eb0-3cdc-11f1-a7ce-4d4b3258205b
"""
    analyze_causal_acausal_branches(trace_causal::SeismicTrace, trace_acausal::SeismicTrace,
                                   periods::Vector{Float64}; max_modes=4,
                                   compute_correlation=true, kwargs...) -> BranchAnalysisResult

Single-source-state branch analysis (backward-compatible API).
"""
function analyze_causal_acausal_branches(trace_causal::SeismicTrace,
                                         trace_acausal::SeismicTrace,
                                         periods::Vector{Float64};
                                         max_modes::Int=4,
                                         compute_correlation::Bool=true,
                                         kwargs...)
    # Validate input
    @assert length(trace_causal.data) == length(trace_acausal.data) "Causal and acausal traces must have same length"
    @assert trace_causal.dt ≈ trace_acausal.dt "Causal and acausal traces must have same sampling interval"
    
    # Perform MFT analysis on both branches
    result_causal = perform_mft_analysis(trace_causal, periods; kwargs...)
    result_acausal = perform_mft_analysis(trace_acausal, periods; kwargs...)
    
    # Extract modes from both branches
    modes_causal = extract_all_modes(result_causal; distance=trace_causal.distance, max_modes=max_modes)
    modes_acausal = extract_all_modes(result_acausal; distance=trace_causal.distance, max_modes=max_modes)
    
    # Compute branch correlation if requested
    correlations = fill(NaN, length(periods))
    if compute_correlation
        correlations = zero_lag_correlation(result_causal.filtered_traces, result_acausal.filtered_traces)
    end
    
    return BranchAnalysisResult(result_causal, result_acausal, 
                               modes_causal, modes_acausal,
                               correlations, periods, trace_causal.distance)
end

# ╔═╡ 4c915e4e-3cdc-11f1-9235-33f20a13a658
"""
    plot_branch_correlation(result::BranchAnalysisResult;
                           title="Causal-Acausal Envelope Correlation",
                           width=900, height=500,
                           font_family="Arial, sans-serif",
                           font_size=14,
                           correlation_threshold=0.85)

Plot the zero-lag correlation between causal and acausal branches across periods.

**Interpretation:**
- Correlation > threshold (default 0.85): Consistent, reliable arrivals on both branches
- Correlation < threshold: Possible mode/branch inconsistency; manually check
- NaN: Unable to compute (e.g., zero-amplitude envelope)

# Arguments
- `result` : BranchAnalysisResult from analyze_causal_acausal_branches
- `title` : Plot title
- `width`, `height` : Plot dimensions
- `correlation_threshold` : Horizontal threshold line to highlight acceptable correlations (default: 0.85)

# Returns
- PlutoPlotly plot object
"""
function plot_branch_correlation(result::BranchAnalysisResult;
                                title="Causal-Acausal Correlation",
                                width=900,
                                height=500,
                                font_family="Arial, sans-serif",
                                font_size=14,
                                correlation_threshold=0.85)
    
    # Filter out NaN values for plotting
    valid_idx = findall(!isnan, result.branch_correlation)
    
    if isempty(valid_idx)
        @warn "No valid correlation values found"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No correlation data"]))
    end
    
    periods_valid = result.periods[valid_idx]
    corr_valid = result.branch_correlation[valid_idx]
    
    # Color code by threshold
    colors = [c >= correlation_threshold ? "green" : "orange" for c in corr_valid]
    
    traces = [
        scatter(
            x=periods_valid,
            y=corr_valid,
            mode="markers+lines",
            marker=attr(size=10, color=colors, line=attr(color="darkgray", width=1)),
            line=attr(color="lightgray", width=2),
            name="Correlation",
            hovertemplate="Period: %{x:.2f} s<br>Correlation: %{y:.3f}<extra></extra>"
        ),
        scatter(
            x=[minimum(periods_valid), maximum(periods_valid)],
            y=[correlation_threshold, correlation_threshold],
            mode="lines",
            line=attr(color="red", width=2, dash="dash"),
            name="Threshold ($correlation_threshold)",
            hoverinfo="skip"
        )
    ]
    
    layout = Layout(
        title=attr(text=title, font=attr(size=font_size+2, family=font_family)),
        xaxis=attr(title="Period (s)", type="linear", range=[minimum(result.periods), maximum(result.periods)], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=attr(title="Correlation Coefficient", range=[-0.1, 1.05], 
                  showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        width=width, height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=attr(l=80, r=80, t=80, b=80),
        showlegend=true,
        legend=attr(x=0.02, y=0.98, font=attr(size=font_size-2),
                   bgcolor="rgba(255,255,255,0.8)", borderwidth=1)
    )
    
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 91b2f6bc-c0f9-4b84-a3d1-37ff6a4bf02d
"""
    _sample_scheme_colors(name::AbstractString, n::Int) -> Vector{String}

Sample `n` colors from ColorSchemes.jl and return Plotly-compatible `"rgb(r,g,b)"` strings.
Falls back to `viridis` when the requested scheme name is not found.
"""
function _sample_scheme_colors(name::AbstractString, n::Int)
    if n <= 0
        return String[]
    end

    candidates = (
        Symbol(name),
        Symbol(lowercase(name)),
        Symbol(replace(name, " " => "_")),
        Symbol(lowercase(replace(name, " " => "_"))),
    )

    cs = nothing
    for key in candidates
        if haskey(ColorSchemes.colorschemes, key)
            cs = ColorSchemes.colorschemes[key]
            break
        end
    end
    cs === nothing && (cs = ColorSchemes.colorschemes[:viridis])

    ts = n == 1 ? [0.5] : collect(range(0.0, 1.0, length=n))
    return [
        let c = get(cs, t)
            "rgb($(round(Int, 255 * red(c))),$(round(Int, 255 * green(c))),$(round(Int, 255 * blue(c))))"
        end
        for t in ts
    ]
end

# ╔═╡ 8cd3a49a-3cdb-11f1-bf54-4dd564d7b55a
"""
    plot_multipeak_dispersion(modes::Vector{MultimodalDispersion}; 
                             title="Multimodal Group Velocity Dispersion",
                             colorscale="Viridis",
                             width=900, height=600,
                             font_family="Arial, sans-serif",
                             font_size=16,
                             show_amplitudes=true)

Create publication-quality interactive plot of multiple dispersion modes.

**Features:**
- Each mode rendered as separate trace with distinct color
- Mode index labeled in legend
- Peak amplitudes visualized via marker size or line opacity
- Interactive hover shows period, velocity, amplitude per mode

# Arguments
- `modes` : Vector of MultimodalDispersion structures
- `title` : Plot title
- `colorscale` : Plotly colorscale name ("Viridis", "Plasma", "RdYlBu", etc.)
- `width`, `height` : Plot dimensions in pixels
- `show_amplitudes` : If true, marker size reflects peak amplitude

# Returns
- PlutoPlotly plot object
"""
function plot_multipeak_dispersion(modes::Vector{MultimodalDispersion}; 
                                  title="Multimodal Group Velocity Dispersion",
                                  colorscale="Viridis",
                                  width=900,
                                  height=600,
                                  font_family="Arial, sans-serif",
                                  font_size=16,
                                  show_amplitudes=true)
    
    if isempty(modes)
        @warn "No modes provided for plotting"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No modes"]))
    end
    
    # Collect all valid velocity measurements to determine axis range
    all_vels = Float64[]
    for mode in modes
        append!(all_vels, filter(!isnan, mode.group_velocities))
    end
    
    if isempty(all_vels)
        @warn "No valid velocity measurements found in modes"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No valid data"]))
    end
    
    vel_min = 0.9 * minimum(all_vels)
    vel_max = 1.1 * maximum(all_vels)
    
    # Generate color palette for modes
    n_modes = length(modes)
    colors = _sample_scheme_colors(colorscale, n_modes)
    
    # Build traces for each mode
    traces = []
    
    for (mode_idx, mode) in enumerate(modes)
        # Filter valid measurements
        valid_idx = findall(!isnan, mode.group_velocities)
        
        if isempty(valid_idx)
            continue
        end
        
        periods_valid = mode.periods[valid_idx]
        vels_valid = mode.group_velocities[valid_idx]
        amps_valid = mode.peak_amplitudes[valid_idx]
        
        # Normalize amplitudes for marker sizing
        amp_min = minimum(amps_valid)
        amp_max = maximum(amps_valid)
        marker_sizes = if amp_max > amp_min
            8 .+ 12 .* (amps_valid .- amp_min) ./ (amp_max - amp_min)
        else
            fill(12.0, length(amps_valid))
        end
        
        # Create trace for this mode
        color = colors[mode_idx]
        trace = scatter(
            x=periods_valid,
            y=vels_valid,
            mode="markers+lines",
            marker=attr(
                size=marker_sizes,
                color=color,
                opacity=0.8,
                line=attr(color="white", width=1)
            ),
            line=attr(
                color=color,
                width=2,
                dash="solid"
            ),
            name="Mode $(mode.mode_index) (n=$(length(valid_idx)))",
            hovertemplate="<b>Mode $(mode.mode_index)</b><br>" *
                          "Period: %{x:.2f} s<br>" *
                          "Group velocity: %{y:.3f} km/s<br>" *
                          "Amplitude: %{customdata:.2f}<extra></extra>",
            customdata=amps_valid
        )
        
        push!(traces, trace)
    end
    
    if isempty(traces)
        @warn "No valid traces to plot"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No valid traces"]))
    end
    
    # Create layout
    layout = Layout(
        title=attr(
            text=title,
            font=attr(size=font_size+2, family=font_family)
        ),
        xaxis=attr(
            title="Period (s)", type="linear",
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=false,
            titlefont=attr(size=font_size, family=font_family),
            tickfont=attr(size=font_size-2, family=font_family)
        ),
        yaxis=attr(
            title="Group Velocity (km/s)",
            showgrid=true,
            gridcolor="rgba(128,128,128,0.2)",
            zeroline=false,
            range=[vel_min, vel_max],
            titlefont=attr(size=font_size, family=font_family),
            tickfont=attr(size=font_size-2, family=font_family)
        ),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=attr(l=80, r=120, t=80, b=80),
        showlegend=true,
        legend=attr(
            x=1.02,
            y=1.0,
            font=attr(size=font_size-2),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
    )
    
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 4c914cc4-3cdc-11f1-9254-6500ce058f12
"""
    plot_branch_comparison(result::BranchAnalysisResult; 
                          title="Causal vs Acausal Dispersion Comparison",
                          colorscale="Viridis",
                          width=1200, height=600,
                          font_family="Arial, sans-serif",
                          font_size=14)

Create side-by-side comparison plot of causal and acausal dispersion modes.

**Features:**
- Left panel: Causal branch modes (one color per mode)
- Right panel: Acausal branch modes (same colors for direct comparison)
- Shared y-axis for velocity, separate x-axes for clarity
- Interactive hover with mode indices and amplitudes

# Arguments
- `result` : BranchAnalysisResult from analyze_causal_acausal_branches
- `title` : Plot title
- `colorscale` : Plotly colorscale for mode coloring
- `width`, `height` : Plot dimensions in pixels
- `font_family`, `font_size` : Typography settings

# Returns
- PlutoPlotly plot object with subplots
"""
function plot_branch_comparison(result::BranchAnalysisResult; 
                               title="Causal vs Acausal Dispersion Comparison",
                               colorscale="Viridis",
                               width=1200,
                               height=600,
                               font_family="Arial, sans-serif",
                               font_size=14)
    
    # Collect all valid velocities for axis scaling
    all_vels = Float64[]
    for modes in [result.causal_modes, result.acausal_modes]
        for mode in modes
            append!(all_vels, filter(!isnan, mode.group_velocities))
        end
    end
    
    if isempty(all_vels)
        @warn "No valid velocity measurements in branch results"
        return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No data"]))
    end
    
    vel_min = 0.9 * minimum(all_vels)
    vel_max = 1.1 * maximum(all_vels)
    
    # Generate colors for modes
    n_max_modes = max(length(result.causal_modes), length(result.acausal_modes))
    colors = _sample_scheme_colors(colorscale, max(n_max_modes, 2))
    
    # Build causal branch traces
    traces_causal = [scatter()]
    for (mode_idx, mode) in enumerate(result.causal_modes)
        valid_idx = findall(!isnan, mode.group_velocities)
        if !isempty(valid_idx)
            periods_valid = mode.periods[valid_idx]
            vels_valid = mode.group_velocities[valid_idx]
            amps_valid = mode.peak_amplitudes[valid_idx]
            
            amp_min = minimum(amps_valid)
            amp_max = maximum(amps_valid)
            marker_sizes = (amp_max > amp_min) ? 
                8 .+ 8 .* (amps_valid .- amp_min) ./ (amp_max - amp_min) : 
                fill(10.0, length(amps_valid))
            
            trace = scatter(
                x=periods_valid, y=vels_valid,
                xaxis="x1", yaxis="y1",
                mode="markers+lines",
                marker=attr(size=marker_sizes, color=colors[mode_idx], opacity=0.7, 
                           line=attr(color="white", width=1)),
                line=attr(color=colors[mode_idx], width=2),
                name="C-Mode $(mode.mode_index)",
                legendgroup="mode_$(mode_idx)",
                hovertemplate="<b>Causal Mode $(mode.mode_index)</b><br>" *
                             "Period: %{x:.2f} s<br>Velocity: %{y:.3f} km/s<extra></extra>",
                showlegend=true
            )
            push!(traces_causal, trace)
        end
    end
    
    # Build acausal branch traces
    traces_acausal = [scatter()]
    for (mode_idx, mode) in enumerate(result.acausal_modes)
        valid_idx = findall(!isnan, mode.group_velocities)
        if !isempty(valid_idx)
            periods_valid = mode.periods[valid_idx]
            vels_valid = mode.group_velocities[valid_idx]
            amps_valid = mode.peak_amplitudes[valid_idx]
            
            amp_min = minimum(amps_valid)
            amp_max = maximum(amps_valid)
            marker_sizes = (amp_max > amp_min) ? 
                8 .+ 8 .* (amps_valid .- amp_min) ./ (amp_max - amp_min) : 
                fill(10.0, length(amps_valid))
            
            trace = scatter(
                x=periods_valid, y=vels_valid,
                xaxis="x2", yaxis="y1",
                mode="markers+lines",
                marker=attr(size=marker_sizes, color=colors[mode_idx], opacity=0.7, 
                           line=attr(color="white", width=1)),
                line=attr(color=colors[mode_idx], width=2, dash="dot"),
                name="A-Mode $(mode.mode_index)",
                legendgroup="mode_$(mode_idx)",
                hovertemplate="<b>Acausal Mode $(mode.mode_index)</b><br>" *
                             "Period: %{x:.2f} s<br>Velocity: %{y:.3f} km/s<extra></extra>",
                showlegend=true
            )
            push!(traces_acausal, trace)
        end
    end
    
    all_traces = vcat(traces_causal, traces_acausal)
    
    layout = Layout(
        title=attr(text=title, font=attr(size=font_size+2, family=font_family)),
        xaxis1=attr(title="Period (s)", type="linear", domain=[0, 0.45], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        xaxis2=attr(title="Period (s)", type="linear", domain=[0.55, 1], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        yaxis1=attr(title="Group Velocity (km/s)", range=[vel_min, vel_max], 
                   showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        width=width, height=height,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=attr(l=80, r=80, t=80, b=80),
        showlegend=true,
        legend=attr(x=1.02, y=1.0, font=attr(size=font_size-2))
    )
    
    return PlutoPlotly.plot(all_traces, layout)
end

# ╔═╡ 60f8c1a1-5c76-4cda-a257-84bc5bf8c791
"""
    BranchBatchAnalysisResult

Container for multi-source-state causal/acausal branch analysis.

Fields:
- `state_results::Vector{BranchAnalysisResult}`: Per-source-state branch analysis results
- `branch_correlation::Matrix{Float64}`: Correlation matrix [period × source_state]
- `periods::Vector{Float64}`: Analysis periods [s]
- `state_labels::Vector{String}`: Labels for each source state
"""
struct BranchBatchAnalysisResult
    state_results::Vector{BranchAnalysisResult}
    branch_correlation::Matrix{Float64}
    periods::Vector{Float64}
    state_labels::Vector{String}
end

# ╔═╡ 7f0eb497-46c0-4a43-95d5-39f6af0d783e
"""
    analyze_causal_acausal_branches(traces_causal::AbstractVector{<:SeismicTrace},
                                   traces_acausal::AbstractVector{<:SeismicTrace},
                                   periods::Vector{Float64};
                                   state_labels=nothing,
                                   max_modes=4,
                                   compute_correlation=true,
                                   kwargs...) -> BranchBatchAnalysisResult

Batch branch analysis across multiple source states.

Each element of `traces_causal` and `traces_acausal` is treated as one source state.
The returned correlation matrix has shape `[period × source_state]`, enabling
direct visualization of correlation curves for all source states.
"""
function analyze_causal_acausal_branches(traces_causal::AbstractVector{<:SeismicTrace},
                                         traces_acausal::AbstractVector{<:SeismicTrace},
                                         periods::Vector{Float64};
                                         state_labels=nothing,
                                         max_modes::Int=4,
                                         compute_correlation::Bool=true,
                                         kwargs...)
    nstates = length(traces_causal)
    @assert nstates == length(traces_acausal) "Causal and acausal trace vectors must have the same number of source states"
    @assert nstates > 0 "At least one source state is required"

    labels = if isnothing(state_labels)
        ["State $(i)" for i in 1:nstates]
    else
        @assert length(state_labels) == nstates "state_labels length must match number of source states"
        string.(state_labels)
    end

    state_results = BranchAnalysisResult[]
    corr_matrix = fill(NaN, length(periods), nstates)

    for i in 1:nstates
        st_res = analyze_causal_acausal_branches(traces_causal[i], traces_acausal[i], periods;
                                                 max_modes=max_modes,
                                                 compute_correlation=compute_correlation,
                                                 kwargs...)
        push!(state_results, st_res)
        corr_matrix[:, i] = st_res.branch_correlation
    end

    return BranchBatchAnalysisResult(state_results, corr_matrix, periods, labels)
end

# ╔═╡ 5da0af71-cfbc-4598-bf6f-cf38f8fce6fe
"""
    plot_branch_correlation(result::BranchBatchAnalysisResult;
                           title="Causal-Acausal Correlation Across Source States",
                           colorscale="Viridis",
                           width=1000, height=550,
                           font_family="Arial, sans-serif",
                           font_size=14,
                           correlation_threshold=0.85)

Plot branch-correlation curves for all source states in a single figure.
"""
function plot_branch_correlation(result::BranchBatchAnalysisResult;
                                 title="Causal-Acausal Correlation Across Source States",
                                 colorscale="Viridis",
                                 width=1000,
                                 height=550,
                                 font_family="Arial, sans-serif",
                                 font_size=14,
                                 correlation_threshold=0.85,
                                 reference_results::AbstractVector{<:BranchAnalysisResult}=BranchAnalysisResult[],
                                 reference_labels::AbstractVector{<:AbstractString}=String[])
    nstates = size(result.branch_correlation, 2)
    nstates == 0 && return PlutoPlotly.plot(scatter(x=[0], y=[0], text=["No source states"]))

    colors = _sample_scheme_colors(colorscale, max(nstates, 2))
    traces = [scatter()]

    for i in 1:nstates
        corr_i = result.branch_correlation[:, i]
        valid_idx = findall(!isnan, corr_i)
        isempty(valid_idx) && continue

        push!(traces, scatter(
            x = result.periods[valid_idx],
            y = corr_i[valid_idx],
            mode = "lines+markers",
            marker = attr(size = 7, color = colors[i], line = attr(color = "white", width = 0.8)),
            line = attr(color = colors[i], width = 1.8),
            name = result.state_labels[i],
            hovertemplate = "State: $(result.state_labels[i])<br>Period: %{x:.2f} s<br>Correlation: %{y:.3f}<extra></extra>"
        ))
    end

    for (iref, ref) in enumerate(reference_results)
        corr_ref = ref.branch_correlation
        valid_idx = findall(c -> !isnan(c), corr_ref)
        isempty(valid_idx) && continue
        ref_label = iref <= length(reference_labels) ? String(reference_labels[iref]) : "Global average"
        push!(traces, scatter(
            x = ref.periods[valid_idx],
            y = corr_ref[valid_idx],
            mode = "lines+markers",
            marker = attr(size = 8, color = "#222222", symbol = "circle-open", line = attr(color = "#222222", width = 1.5)),
            line = attr(color = "#222222", width = 3.0),
            name = ref_label,
            hovertemplate = "$(ref_label)<br>Period: %{x:.2f} s<br>Correlation: %{y:.3f}<extra></extra>"
        ))
    end

    # Threshold guide
    push!(traces, scatter(
        x = [minimum(result.periods), maximum(result.periods)],
        y = [correlation_threshold, correlation_threshold],
        mode = "lines",
        line = attr(color = "red", width = 2, dash = "dash"),
        name = "Threshold ($(correlation_threshold))",
        hoverinfo = "skip"
    ))

    layout = Layout(
        title = attr(text = title, font = attr(size = font_size + 2, family = font_family)),
        xaxis = attr(title = "Period (s)",
            type = "linear", showgrid = true, gridcolor = "rgba(128,128,128,0.2)"),
        yaxis = attr(title = "Correlation Coefficient", range = [-0.1, 1.05], showgrid = true, gridcolor = "rgba(128,128,128,0.2)"),
        width = width,
        height = height,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin = attr(l = 80, r = 80, t = 80, b = 80),
        showlegend = true,
        legend = attr(x = 1.02, y = 1.0, font = attr(size = font_size - 2), bgcolor = "rgba(255,255,255,0.8)", borderwidth = 1)
    )

    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ c0fb74a4-5f74-4451-a097-b0ddb4e79f2e
"""
    high_correlation_indices(result::BranchAnalysisResult; correlation_threshold=0.85)

Return period indices where branch correlation is finite and above threshold.
"""
function high_correlation_indices(result::BranchAnalysisResult; correlation_threshold::Float64=0.85)
    return findall(c -> isfinite(c) && c >= correlation_threshold, result.branch_correlation)
end

# ╔═╡ 4f47e6c2-0f17-4f71-a7da-08f651176632
"""
    high_correlation_indices(result::BranchBatchAnalysisResult; correlation_threshold=0.85)

Return a boolean mask [period, state] selecting finite correlation values above threshold.
"""
function high_correlation_indices(result::BranchBatchAnalysisResult; correlation_threshold::Float64=0.85)
    return map(c -> isfinite(c) && c >= correlation_threshold, result.branch_correlation)
end

# ╔═╡ 1a67f649-4fd7-4f63-bf4a-28d5cb66f52d
"""
    _resolve_period_index(periods; period=nothing, period_index=nothing)

Resolve a target period index from either a period value or a 1-based index.
Returns `(idx, period_value)`.
"""
function _resolve_period_index(periods::AbstractVector{<:Real};
                               period::Union{Nothing,Real}=nothing,
                               period_index::Union{Nothing,Integer}=nothing)
    if !isnothing(period) && !isnothing(period_index)
        throw(ArgumentError("Provide either `period` or `period_index`, not both"))
    end

    if !isnothing(period_index)
        idx = Int(period_index)
        (idx < 1 || idx > length(periods)) && throw(BoundsError(periods, idx))
        return idx, Float64(periods[idx])
    end

    if isnothing(period)
        idx = max(1, Int(cld(length(periods), 2)))
        return idx, Float64(periods[idx])
    end

    idx = argmin(abs.(Float64.(periods) .- Float64(period)))
    return idx, Float64(periods[idx])
end

# ╔═╡ 88284a1d-00ff-4f69-ac6e-a39dbec70863
"""
    plot_filtered_traces_by_period(result::BranchBatchAnalysisResult;
                                   period=nothing,
                                   period_index=nothing,
                                   correlation_threshold=nothing,
                                   normalize_each=true,
                                   scale=0.7,
                                   spacing=2.2,
                                   colorscale="Viridis",
                                   width=1000,
                                   height=900,
                                   font_family="Arial, sans-serif",
                                   font_size=12,
                                   title="Filtered Traces Across Source States")

Plot filtered causal+acausal traces for all source states at one selected period.

For each source state, the displayed trace is:
- negative lag side: reversed acausal filtered trace
- positive lag side: causal filtered trace

Selection can be provided as a period value (`period`) or index (`period_index`).
If `correlation_threshold` is provided, only source states with correlation
at the selected period above threshold are shown.
"""
function plot_filtered_traces_by_period(result::BranchBatchAnalysisResult;
                                        period::Union{Nothing,Real}=nothing,
                                        period_index::Union{Nothing,Integer}=nothing,
                                        correlation_threshold::Union{Nothing,Float64}=nothing,
                                        normalize_each::Bool=true,
                                        scale::Float64=0.7,
                                        spacing::Float64=2.2,
                                        colorscale::String="Viridis",
                                        width::Int=1000,
                                        height::Int=900,
                                        font_family::String="Arial, sans-serif",
                                        font_size::Int=12,
                                        title::String="Filtered Traces Across Source States")
    nstates = length(result.state_results)
    nstates == 0 && return PlutoPlotly.plot(scatter(x=[0.0], y=[0.0], text=["No source states"]))

    idx, period_used = _resolve_period_index(result.periods; period=period, period_index=period_index)
    state_colors = _sample_scheme_colors(colorscale, max(nstates, 2))

    traces = [scatter()]
    plotted = 0
    max_abs_y = 0.0

    for i in 1:nstates
        st = result.state_results[i]
        label = result.state_labels[i]
        corr = st.branch_correlation[idx]

        if !isnothing(correlation_threshold)
            if !(isfinite(corr) && corr >= correlation_threshold)
                continue
            end
        end

        ac = st.acausal_result.filtered_traces[:, idx]
        ca = st.causal_result.filtered_traces[:, idx]
        if isempty(ac) || isempty(ca)
            continue
        end

        t_neg = -reverse(st.acausal_result.time)
        t_pos = st.causal_result.time
        t_full = vcat(t_neg, t_pos)
        x_full = vcat(reverse(ac), ca)

        if normalize_each
            amp = maximum(abs, x_full)
            x_full = amp > 0 ? x_full ./ amp : x_full
        end

        plotted += 1
        offset = (plotted - 1) * spacing
        y = scale .* x_full .+ offset
        max_abs_y = max(max_abs_y, maximum(abs, y))

        push!(traces, scatter(
            x=t_full,
            y=y,
            mode="lines",
            line=attr(color=state_colors[i], width=1.7),
            name=label,
            hovertemplate="State: $(label)<br>Period: $(round(period_used; digits=3)) s<br>Corr: $(isfinite(corr) ? round(corr; digits=3) : NaN)<br>Lag: %{x:.2f} s<extra></extra>"
        ))
    end

    if plotted == 0
        msg = isnothing(correlation_threshold) ? "No valid traces at selected period" : "No states pass threshold at selected period"
        @warn msg
        return PlutoPlotly.plot(scatter(x=[0.0], y=[0.0], text=[msg]))
    end

    subtitle = "period=$(round(period_used; digits=3)) s (index=$(idx))"
    if !isnothing(correlation_threshold)
        subtitle *= ", threshold=$(correlation_threshold)"
    end

    layout = Layout(
        title=attr(text="$(title) [$(subtitle)]", font=attr(size=font_size + 2, family=font_family)),
        xaxis=attr(title="Lag Time (s)", showgrid=true, gridcolor="rgba(128,128,128,0.2)", zeroline=true, zerolinecolor="rgba(100,100,100,0.45)"),
        yaxis=attr(title="Source State (offset filtered traces)", showgrid=false, zeroline=false, range=[-scale * 1.3, max_abs_y + scale]),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=attr(l=85, r=120, t=85, b=75),
        showlegend=true,
        legend=attr(x=1.02, y=1.0, font=attr(size=font_size - 2), bgcolor="rgba(255,255,255,0.85)", borderwidth=1)
    )

    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ f26e8f7a-80d8-4f99-ab64-8e13a870e145
"""
    _periods_from_nyquist(dt; dT=0.5, period_max=60.0)

Build analysis periods in seconds from the Nyquist period `2dt` up to `period_max`
with increment `dT`.
"""
function _periods_from_nyquist(dt::Float64; period_min::Union{Float64, Nothing}=nothing, dT::Float64=0.5, period_max::Float64=60.0)
    dT > 0.0 || throw(ArgumentError("dT must be positive"))
    period_max > 0.0 || throw(ArgumentError("period_max must be positive"))

    period_min = period_min === nothing ? 2.0 * dt : period_min
    period_max >= period_min || throw(ArgumentError("period_max=$(period_max) must be >= Nyquist period $(period_min)"))

    periods = collect(exp10.(range(log10(period_min), log10(period_max), length=400)))
    isempty(periods) && (periods = [period_min])
    return periods
end

# ╔═╡ d10d4b32-4e4e-4198-b08d-084dffeb3694
"""
    analyze_causal_acausal_branches(trace_causal::SeismicTrace, trace_acausal::SeismicTrace;
                                    dT=0.5, period_max=60.0, max_modes=4,
                                    compute_correlation=true, kwargs...) -> BranchAnalysisResult

Nyquist-default API: periods are auto-generated from `2dt:dT:period_max`.
"""
function analyze_causal_acausal_branches(trace_causal::SeismicTrace,
                                         trace_acausal::SeismicTrace;
                                         period_min::Union{Float64, Nothing}=nothing,
                                         dT::Float64=0.5,
                                         period_max::Float64=60.0,
                                         max_modes::Int=4,
                                         compute_correlation::Bool=true,
                                         kwargs...)
    periods = _periods_from_nyquist(trace_causal.dt; period_min=period_min, dT=dT, period_max=period_max)
    return analyze_causal_acausal_branches(trace_causal, trace_acausal, periods;
                                           max_modes=max_modes,
                                           compute_correlation=compute_correlation,
                                           kwargs...)
end

# ╔═╡ 95f715de-a6aa-4ec5-a2ce-4f7bb241ce34
"""
    analyze_causal_acausal_branches(traces_causal::AbstractVector{<:SeismicTrace},
                                    traces_acausal::AbstractVector{<:SeismicTrace};
                                    state_labels=nothing,
                                    dT=0.5,
                                    period_max=60.0,
                                    max_modes=4,
                                    compute_correlation=true,
                                    kwargs...) -> BranchBatchAnalysisResult

Batch Nyquist-default API: one common period axis is auto-generated from
`2dt:dT:period_max` using the first state's `dt`.
"""
function analyze_causal_acausal_branches(traces_causal::AbstractVector{<:SeismicTrace},
                                         traces_acausal::AbstractVector{<:SeismicTrace};
                                         state_labels=nothing,
                                         period_min::Union{Float64, Nothing}=nothing,
                                         dT::Float64=0.5,
                                         period_max::Float64=60.0,
                                         max_modes::Int=4,
                                         compute_correlation::Bool=true,
                                         kwargs...)
    nstates = length(traces_causal)
    @assert nstates == length(traces_acausal) "Causal and acausal trace vectors must have the same number of source states"
    @assert nstates > 0 "At least one source state is required"

    dt_ref = traces_causal[1].dt
    for i in 1:nstates
        @assert traces_causal[i].dt ≈ traces_acausal[i].dt "State $(i): causal and acausal dt must match"
        @assert traces_causal[i].dt ≈ dt_ref "All source states must share the same dt for batch Nyquist-default analysis"
    end

    periods = _periods_from_nyquist(dt_ref; period_min=period_min, dT=dT, period_max=period_max)
    return analyze_causal_acausal_branches(traces_causal, traces_acausal, periods;
                                           state_labels=state_labels,
                                           max_modes=max_modes,
                                           compute_correlation=compute_correlation,
                                           kwargs...)
end

# ╔═╡ c9597868-5ae3-4629-b96d-d25f9a7d316d
"""
    _matched_peak_average_velocities(causal_peaks, acausal_peaks, distance;
                                     velocity_tolerance_fraction=0.10)

Match causal and acausal peak-derived group velocities using relative-difference
tolerance and return averaged matched velocities.
"""
function _matched_peak_average_velocities(causal_peaks::Vector{Tuple{Float64,Float64}},
                                          acausal_peaks::Vector{Tuple{Float64,Float64}},
                                          distance::Float64;
                                          velocity_tolerance_fraction::Float64=0.10)
    velocity_tolerance_fraction >= 0.0 || throw(ArgumentError("velocity_tolerance_fraction must be >= 0"))

    causal_vels = Float64[]
    acausal_vels = Float64[]

    for (t, _) in causal_peaks
        if isfinite(t) && t > 0.0
            v = distance / t
            isfinite(v) && v > 0.0 && push!(causal_vels, v)
        end
    end
    for (t, _) in acausal_peaks
        if isfinite(t) && t > 0.0
            v = distance / t
            isfinite(v) && v > 0.0 && push!(acausal_vels, v)
        end
    end

    isempty(causal_vels) && return Float64[]
    isempty(acausal_vels) && return Float64[]

    used_acausal = falses(length(acausal_vels))
    matched_avg = Float64[]

    for vc in causal_vels
        best_j = 0
        best_rel = Inf
        for (j, va) in enumerate(acausal_vels)
            used_acausal[j] && continue
            denom = max((abs(vc) + abs(va)) / 2.0, eps(Float64))
            rel = abs(vc - va) / denom
            if rel <= velocity_tolerance_fraction && rel < best_rel
                best_rel = rel
                best_j = j
            end
        end

        if best_j > 0
            used_acausal[best_j] = true
            push!(matched_avg, 0.5 * (vc + acausal_vels[best_j]))
        end
    end

    sort!(matched_avg)
    return matched_avg
end

# ╔═╡ 07f578f5-f269-45f1-a6e6-c90f4d0f5c2f
"""
    plot_all_highcorr_groupvelocity_picks(result::BranchAnalysisResult;
                                          correlation_threshold=0.85,
                                          width=1000,
                                          height=600,
                                          font_family="Arial, sans-serif",
                                          font_size=14,
                                          title="All High-Correlation Group-Velocity Picks")

Plot all causal and acausal envelope-peak group-velocity picks, restricted to periods
where branch correlation is above `correlation_threshold`.
"""
function plot_all_highcorr_groupvelocity_picks(result::BranchAnalysisResult;
                                               correlation_threshold::Float64=0.85,
                                               pair_and_average::Bool=false,
                                               velocity_tolerance_fraction::Float64=0.10,
                                               width::Int=1000,
                                               height::Int=600,
                                               font_family::String="Arial, sans-serif",
                                               font_size::Int=14,
                                               title::String="All High-Correlation Group-Velocity Picks")
    hi_idx = high_correlation_indices(result; correlation_threshold=correlation_threshold)
    if isempty(hi_idx)
        @warn "No periods satisfy the correlation threshold $(correlation_threshold)"
        return PlutoPlotly.plot(scatter(x=[0.0], y=[0.0], text=["No high-correlation periods"]))
    end

    traces = [scatter()]
    all_vels = Float64[]

    if pair_and_average
        matched_by_period = Dict{Int,Vector{Float64}}()
        max_matched = 0

        for ip in hi_idx
            causal_peaks = result.causal_result.all_peaks[ip]
            acausal_peaks = result.acausal_result.all_peaks[ip]
            vavg = _matched_peak_average_velocities(causal_peaks, acausal_peaks, result.distance;
                                                    velocity_tolerance_fraction=velocity_tolerance_fraction)
            isempty(vavg) && continue
            matched_by_period[ip] = vavg
            max_matched = max(max_matched, length(vavg))
            append!(all_vels, vavg)
        end

        for pidx in 1:max_matched
            xp = Float64[]
            yp = Float64[]
            cp = Float64[]

            for ip in hi_idx
                if haskey(matched_by_period, ip)
                    vals = matched_by_period[ip]
                    if pidx <= length(vals)
                        push!(xp, result.periods[ip])
                        push!(yp, vals[pidx])
                        push!(cp, result.branch_correlation[ip])
                    end
                end
            end

            isempty(xp) && continue
            push!(traces, scatter(
                x=xp,
                y=yp,
                mode="markers+lines",
                marker=attr(size=8, color="#2ca02c", symbol="circle", line=attr(color="white", width=0.8)),
                line=attr(color="#2ca02c", width=1.8),
                name="Avg matched peak $(pidx)",
                hovertemplate="Matched Avg<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Corr: %{customdata:.3f}<extra></extra>",
                customdata=cp
            ))
        end
    else
        branches = [
            ("Causal", result.causal_result, "#1f77b4", "circle"),
            ("Acausal", result.acausal_result, "#d62728", "diamond"),
        ]

        for (branch_name, mft_res, color, marker_symbol) in branches
            max_peaks = maximum(length.(mft_res.all_peaks); init=0)
            max_peaks == 0 && continue

            for pidx in 1:max_peaks
                xp = Float64[]
                yp = Float64[]
                cp = Float64[]

                for ip in hi_idx
                    peaks = mft_res.all_peaks[ip]
                    if pidx <= length(peaks)
                        tpk, _ = peaks[pidx]
                        if isfinite(tpk) && tpk > 0.0
                            vpk = result.distance / tpk
                            if isfinite(vpk) && vpk > 0.0
                                push!(xp, mft_res.periods[ip])
                                push!(yp, vpk)
                                push!(cp, result.branch_correlation[ip])
                                push!(all_vels, vpk)
                            end
                        end
                    end
                end

                isempty(xp) && continue

                push!(traces, scatter(
                    x=xp,
                    y=yp,
                    mode="markers+lines",
                    marker=attr(size=8, color=color, symbol=marker_symbol, line=attr(color="white", width=0.8)),
                    line=attr(color=color, width=1.8),
                    name="$(branch_name) peak $(pidx)",
                    hovertemplate="$(branch_name)<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Corr: %{customdata:.3f}<extra></extra>",
                    customdata=cp
                ))
            end
        end
    end

    if length(traces) == 1 || isempty(all_vels)
        @warn "No valid peak-derived group-velocity picks found in high-correlation periods"
        return PlutoPlotly.plot(scatter(x=[0.0], y=[0.0], text=["No valid high-correlation picks"]))
    end

    y_min = 0.9 * minimum(all_vels)
    y_max = 1.1 * maximum(all_vels)
    n_hi = length(hi_idx)

    layout = Layout(
        title=attr(text="$(title) (N=$(n_hi), threshold=$(correlation_threshold))", font=attr(size=font_size + 2, family=font_family)),
        xaxis=attr(title="Period (s)", type="linear", range=[minimum(result.periods), maximum(result.periods)], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=attr(title="Group Velocity (km/s)", range=[y_min, y_max], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=attr(l=80, r=80, t=80, b=80),
        showlegend=true,
        legend=attr(x=1.02, y=1.0, font=attr(size=font_size - 2), bgcolor="rgba(255,255,255,0.8)", borderwidth=1)
    )

    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 58d6ecb6-2c42-4a53-9cbc-b72ebf48356c
"""
    plot_all_highcorr_groupvelocity_picks(result::BranchBatchAnalysisResult;
                                          correlation_threshold=0.85,
                                          colorscale="Viridis",
                                          width=1200,
                                          height=700,
                                          font_family="Arial, sans-serif",
                                          font_size=13,
                                          title="All High-Correlation Group-Velocity Picks Across Source States")

Plot all envelope-peak group-velocity picks for every source state, restricted to
periods where that state's branch correlation is above `correlation_threshold`.
"""
function plot_all_highcorr_groupvelocity_picks(result::BranchBatchAnalysisResult;
                                               correlation_threshold::Float64=0.85,
                                               pair_and_average::Bool=false,
                                               velocity_tolerance_fraction::Float64=0.10,
                                               reference_results::AbstractVector{<:BranchAnalysisResult}=BranchAnalysisResult[],
                                               reference_labels::AbstractVector{<:AbstractString}=String[],
                                               colorscale::String="Viridis",
                                               width::Int=1200,
                                               height::Int=700,
                                               font_family::String="Arial, sans-serif",
                                               font_size::Int=13,
                                               title::String="All High-Correlation Group-Velocity Picks Across Source States")
    nstates = length(result.state_results)
    nstates == 0 && return PlutoPlotly.plot(scatter(x=[0.0], y=[0.0], text=["No source states"]))

    state_colors = _sample_scheme_colors(colorscale, max(nstates, 2))
    traces = [scatter()]
    all_vels = Float64[]
    total_hi = 0

    for i in 1:nstates
        st = result.state_results[i]
        label = result.state_labels[i]
        hi_idx = high_correlation_indices(st; correlation_threshold=correlation_threshold)
        total_hi += length(hi_idx)
        isempty(hi_idx) && continue

        if pair_and_average
            matched_by_period = Dict{Int,Vector{Float64}}()
            max_matched = 0

            for ip in hi_idx
                causal_peaks = st.causal_result.all_peaks[ip]
                acausal_peaks = st.acausal_result.all_peaks[ip]
                vavg = _matched_peak_average_velocities(causal_peaks, acausal_peaks, st.distance;
                                                        velocity_tolerance_fraction=velocity_tolerance_fraction)
                isempty(vavg) && continue
                matched_by_period[ip] = vavg
                max_matched = max(max_matched, length(vavg))
            end

            for pidx in 1:max_matched
                xp = Float64[]
                yp = Float64[]
                cp = Float64[]

                for ip in hi_idx
                    if haskey(matched_by_period, ip)
                        vals = matched_by_period[ip]
                        if pidx <= length(vals)
                            push!(xp, st.periods[ip])
                            push!(yp, vals[pidx])
                            push!(cp, st.branch_correlation[ip])
                            push!(all_vels, vals[pidx])
                        end
                    end
                end

                isempty(xp) && continue

                push!(traces, scatter(
                    x=xp,
                    y=yp,
                    mode="markers",
                    marker=attr(
                        size=7,
                        color=state_colors[i],
                        opacity=0.82,
                        symbol="circle",
                        line=attr(color="white", width=0.7)
                    ),
                    name="$(label) | Avg matched p$(pidx)",
                    hovertemplate="State: $(label)<br>Type: Avg matched<br>Peak: $(pidx)<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Corr: %{customdata:.3f}<extra></extra>",
                    customdata=cp
                ))
            end

            continue
        end

        branch_defs = [
            ("Causal", st.causal_result, "circle"),
            ("Acausal", st.acausal_result, "diamond"),
        ]

        for (branch_name, mft_res, marker_symbol) in branch_defs
            max_peaks = maximum(length.(mft_res.all_peaks); init=0)
            max_peaks == 0 && continue

            for pidx in 1:max_peaks
                xp = Float64[]
                yp = Float64[]
                cp = Float64[]

                for ip in hi_idx
                    peaks = mft_res.all_peaks[ip]
                    if pidx <= length(peaks)
                        tpk, _ = peaks[pidx]
                        if isfinite(tpk) && tpk > 0.0
                            vpk = st.distance / tpk
                            if isfinite(vpk) && vpk > 0.0
                                push!(xp, mft_res.periods[ip])
                                push!(yp, vpk)
                                push!(cp, st.branch_correlation[ip])
                                push!(all_vels, vpk)
                            end
                        end
                    end
                end

                isempty(xp) && continue

                push!(traces, scatter(
                    x=xp,
                    y=yp,
                    mode="markers",
                    marker=attr(
                        size=7,
                        color=state_colors[i],
                        opacity=0.78,
                        symbol=marker_symbol,
                        line=attr(color="white", width=0.7)
                    ),
                    name="$(label) | $(branch_name) p$(pidx)",
                    hovertemplate="State: $(label)<br>Branch: $(branch_name)<br>Peak: $(pidx)<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Corr: %{customdata:.3f}<extra></extra>",
                    customdata=cp
                ))
            end
        end
    end

    reference_styles = [
        ("Causal", "#08306b", "circle-open"),
        ("Acausal", "#7f0000", "diamond-open"),
    ]

    for (iref, ref) in enumerate(reference_results)
        ref_label = iref <= length(reference_labels) ? String(reference_labels[iref]) : "Global avg"
        ref_idx = eachindex(ref.periods)
        for (branch_name, mft_res, color, marker_symbol) in [
                (reference_styles[1][1], ref.causal_result, reference_styles[1][2], reference_styles[1][3]),
                (reference_styles[2][1], ref.acausal_result, reference_styles[2][2], reference_styles[2][3])]
            max_peaks = maximum(length.(mft_res.all_peaks); init=0)
            max_peaks == 0 && continue
            for pidx in 1:max_peaks
                xp = Float64[]
                yp = Float64[]
                cp = Float64[]
                for ip in ref_idx
                    peaks = mft_res.all_peaks[ip]
                    if pidx <= length(peaks)
                        tpk, _ = peaks[pidx]
                        if isfinite(tpk) && tpk > 0.0
                            vpk = ref.distance / tpk
                            if isfinite(vpk) && vpk > 0.0
                                push!(xp, mft_res.periods[ip])
                                push!(yp, vpk)
                                push!(cp, ref.branch_correlation[ip])
                                push!(all_vels, vpk)
                            end
                        end
                    end
                end
                isempty(xp) && continue
                push!(traces, scatter(
                    x=xp,
                    y=yp,
                    mode="lines+markers",
                    marker=attr(
                        size=9,
                        color=color,
                        opacity=1.0,
                        symbol=marker_symbol,
                        line=attr(color=color, width=1.4)
                    ),
                    line=attr(color=color, width=2.6, dash="dash"),
                    name="$(ref_label) $(lowercase(branch_name)) peak $(pidx)",
                    hovertemplate="$(ref_label)<br>Branch: $(branch_name)<br>Peak: $(pidx)<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Corr: %{customdata:.3f}<extra></extra>",
                    customdata=cp
                ))
            end
        end
    end

    if length(traces) == 1 || isempty(all_vels)
        @warn "No valid high-correlation picks across source states"
        return PlutoPlotly.plot(scatter(x=[0.0], y=[0.0], text=["No valid high-correlation picks"]))
    end

    y_min = 0.9 * minimum(all_vels)
    y_max = 1.1 * maximum(all_vels)

    layout = Layout(
        title=attr(text="$(title) (threshold=$(correlation_threshold), high-corr samples=$(total_hi))", font=attr(size=font_size + 2, family=font_family)),
        xaxis=attr(title="Period (s)", type="linear", range=[minimum(result.periods), maximum(result.periods)], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=attr(title="Group Velocity (km/s)", range=[y_min, y_max], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=attr(l=80, r=120, t=80, b=80),
        showlegend=true,
        legend=attr(x=1.02, y=1.0, font=attr(size=font_size - 2), bgcolor="rgba(255,255,255,0.8)", borderwidth=1)
    )

    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ a079a952-8252-4f56-a4c7-2a9ae58e0f11
"""
    SourceStateConsensusPick

Consensus group-velocity picks combined across source states for one receiver pair.

Fields:
- `periods`: Analysis periods [s]
- `group_velocities`: Final consensus group velocity per period [km/s]
- `arrival_times`: Final arrival time per period [s]
- `confidence`: Confidence score per period [0,1]
- `support`: Number of unique source states supporting the selected cluster
- `candidate_velocities`: All causal/acausal-matched candidate velocities per period
- `accepted_candidate_velocities`: Candidate velocities inside the selected cluster
- `rejected_candidate_velocities`: Candidate velocities outside the selected cluster
- `selected_cluster_index`: Selected cluster index per period, or 0 when no pick is accepted
- `candidate_group_velocities`: Smooth consensus candidate branches [period, candidate]
- `candidate_arrival_times`: Candidate arrival times [period, candidate]
- `candidate_confidence`: Candidate confidence scores [period, candidate]
- `candidate_support`: Candidate source-state support counts [period, candidate]
- `candidate_cluster_index`: Selected cluster index for each candidate branch [period, candidate]
"""
struct SourceStateConsensusPick
    periods::Vector{Float64}
    group_velocities::Vector{Float64}
    arrival_times::Vector{Float64}
    confidence::Vector{Float64}
    support::Vector{Int}
    candidate_velocities::Vector{Vector{Float64}}
    accepted_candidate_velocities::Vector{Vector{Float64}}
    rejected_candidate_velocities::Vector{Vector{Float64}}
    selected_cluster_index::Vector{Int}
    candidate_group_velocities::Matrix{Float64}
    candidate_arrival_times::Matrix{Float64}
    candidate_confidence::Matrix{Float64}
    candidate_support::Matrix{Int}
    candidate_cluster_index::Matrix{Int}
end

# ╔═╡ f2c83612-d161-4fef-a728-c2dcbfa48b1c
struct _ConsensusCandidate
    velocity::Float64
    state_index::Int
    branch_correlation::Float64
end

# ╔═╡ 1c36878c-f684-46a8-a2ab-4203a01d0169
struct _ConsensusCluster
    velocity::Float64
    confidence::Float64
    support::Int
    candidate_indices::Vector{Int}
end

# ╔═╡ 7f9cbf19-46b3-403b-90f8-a4591fd5ef4f
function _relative_difference(a::Float64, b::Float64)
    denom = max((abs(a) + abs(b)) / 2.0, eps(Float64))
    return abs(a - b) / denom
end

# ╔═╡ f7f4e17f-0784-43db-9a6b-c2b07a963f94
function _median_sorted(vals::Vector{Float64})
    isempty(vals) && return NaN
    s = sort(vals)
    n = length(s)
    mid = fld(n + 1, 2)
    return isodd(n) ? s[mid] : 0.5 * (s[mid] + s[mid + 1])
end

# ╔═╡ c17d7f26-3d82-44b8-860c-b12df0bf9293
function _mean_finite(vals::Vector{Float64})
    good = [v for v in vals if isfinite(v)]
    isempty(good) && return NaN
    return sum(good) / length(good)
end

# ╔═╡ 1a7d4cbb-a1ea-4319-a404-2293faf72069
function _cluster_consensus_candidates(candidates::Vector{_ConsensusCandidate},
                                       nstates::Int;
                                       cluster_tolerance_fraction::Float64=0.08)
    cluster_tolerance_fraction >= 0.0 || throw(ArgumentError("cluster_tolerance_fraction must be >= 0"))
    isempty(candidates) && return _ConsensusCluster[]

    order = sortperm([c.velocity for c in candidates])
    groups = Vector{Vector{Int}}()

    for idx in order
        v = candidates[idx].velocity
        if isempty(groups)
            push!(groups, [idx])
            continue
        end

        last_group = groups[end]
        center = _median_sorted([candidates[j].velocity for j in last_group])
        if _relative_difference(v, center) <= cluster_tolerance_fraction
            push!(last_group, idx)
        else
            push!(groups, [idx])
        end
    end

    clusters = _ConsensusCluster[]
    for group in groups
        velocities = [candidates[j].velocity for j in group]
        corrs = [candidates[j].branch_correlation for j in group]
        states = unique([candidates[j].state_index for j in group])

        v_med = _median_sorted(velocities)
        v_avg = _mean_finite(velocities)
        mean_corr = clamp(_mean_finite(corrs), 0.0, 1.0)
        support_fraction = nstates > 0 ? clamp(length(states) / nstates, 0.0, 1.0) : 0.0

        compactness = 1.0
        if length(velocities) > 1 && isfinite(v_med) && v_med > 0.0
            rel_spread = maximum(abs.(velocities .- v_med)) / v_med
            compactness = clamp(1.0 - rel_spread / max(cluster_tolerance_fraction, eps(Float64)), 0.0, 1.0)
        end

        confidence = clamp(0.45 * support_fraction + 0.35 * mean_corr + 0.20 * compactness, 0.0, 1.0)
        push!(clusters, _ConsensusCluster(v_avg, confidence, length(states), group))
    end

    return clusters
end

# ╔═╡ c7600895-0b53-477d-aaca-19925dec0d91
function _select_consensus_clusters(period_clusters::Vector{Vector{_ConsensusCluster}};
                                    min_support::Int=1,
                                    min_confidence::Float64=0.0,
                                    smoothness_weight::Float64=1.0)
    min_support >= 1 || throw(ArgumentError("min_support must be >= 1"))
    0.0 <= min_confidence <= 1.0 || throw(ArgumentError("min_confidence must be in [0, 1]"))
    smoothness_weight >= 0.0 || throw(ArgumentError("smoothness_weight must be >= 0"))

    nperiods = length(period_clusters)
    selected = zeros(Int, nperiods)
    prev_velocity = NaN

    for ip in 1:nperiods
        clusters = period_clusters[ip]
        best_j = 0
        best_score = -Inf

        for (j, cluster) in enumerate(clusters)
            cluster.support < min_support && continue
            cluster.confidence < min_confidence && continue

            smooth_penalty = 0.0
            if isfinite(prev_velocity) && isfinite(cluster.velocity) && cluster.velocity > 0.0
                smooth_penalty = smoothness_weight * _relative_difference(cluster.velocity, prev_velocity)
            end

            score = cluster.confidence + 0.08 * cluster.support - smooth_penalty
            if score > best_score
                best_score = score
                best_j = j
            end
        end

        selected[ip] = best_j
        if best_j > 0
            prev_velocity = clusters[best_j].velocity
        end
    end

    return selected
end

# ╔═╡ 07914f7d-70fb-477d-bb05-dfa2a68d03ff
function _eligible_cluster_indices(clusters::Vector{_ConsensusCluster},
                                   used::AbstractVector{Bool};
                                   min_support::Int,
                                   min_confidence::Float64)
    return [j for j in eachindex(clusters)
            if !used[j] &&
               clusters[j].support >= min_support &&
	               clusters[j].confidence >= min_confidence]
end

# ╔═╡ 2e76ea52-ad8e-4f32-9b01-b88d7d1163f9
function _local_low_velocity_bonus(clusters::Vector{_ConsensusCluster}, j::Int,
                                   eligible::Vector{Int})
    isempty(eligible) && return 0.0
    vals = [clusters[k].velocity for k in eligible if isfinite(clusters[k].velocity)]
    isempty(vals) && return 0.0
    vmin = minimum(vals)
    vmax = maximum(vals)
    span = vmax - vmin
    span <= eps(Float64) && return 0.5
    return clamp((vmax - clusters[j].velocity) / span, 0.0, 1.0)
end

# ╔═╡ bdccfc35-4436-4716-a868-bec5a18317d3
function _select_smooth_consensus_candidate_branches(period_clusters::Vector{Vector{_ConsensusCluster}};
                                                     max_candidates::Int=3,
                                                     min_support::Int=1,
                                                     min_confidence::Float64=0.0,
                                                     smoothness_weight::Float64=1.0,
                                                     max_smooth_jump_fraction::Float64=0.12,
                                                     max_gap_periods::Int=1,
                                                     selection_mode::Symbol=:low_velocity)
    max_candidates >= 1 || throw(ArgumentError("max_candidates must be >= 1"))
    min_support >= 1 || throw(ArgumentError("min_support must be >= 1"))
    0.0 <= min_confidence <= 1.0 || throw(ArgumentError("min_confidence must be in [0, 1]"))
    smoothness_weight >= 0.0 || throw(ArgumentError("smoothness_weight must be >= 0"))
    max_smooth_jump_fraction >= 0.0 || throw(ArgumentError("max_smooth_jump_fraction must be >= 0"))
    max_gap_periods >= 0 || throw(ArgumentError("max_gap_periods must be >= 0"))
    selection_mode in (:low_velocity, :confidence) || throw(ArgumentError("selection_mode must be :low_velocity or :confidence"))

    nperiods = length(period_clusters)
    selected = zeros(Int, nperiods, max_candidates)
    used = [falses(length(clusters)) for clusters in period_clusters]

    for icand in 1:max_candidates
        seed_ip = 0
        seed_j = 0
        seed_score = -Inf

        for ip in 1:nperiods
            eligible = _eligible_cluster_indices(period_clusters[ip], used[ip];
                                                 min_support=min_support,
                                                 min_confidence=min_confidence)
            for j in eligible
                cluster = period_clusters[ip][j]
                low_bonus = _local_low_velocity_bonus(period_clusters[ip], j, eligible)
                score = selection_mode == :low_velocity ?
                        (cluster.confidence + 0.08 * cluster.support + 0.12 * low_bonus) :
                        (cluster.confidence + 0.08 * cluster.support)
                if score > seed_score
                    seed_score = score
                    seed_ip = ip
                    seed_j = j
                end
            end
        end

        seed_j == 0 && break

        selected[seed_ip, icand] = seed_j
        used[seed_ip][seed_j] = true

        for direction in (-1, 1)
            prev_velocity = period_clusters[seed_ip][seed_j].velocity
            gap_count = 0
            ip_range = direction == -1 ? ((seed_ip - 1):-1:1) : ((seed_ip + 1):nperiods)

            for ip in ip_range
                best_j = 0
                best_score = -Inf

                eligible = _eligible_cluster_indices(period_clusters[ip], used[ip];
                                                     min_support=min_support,
                                                     min_confidence=min_confidence)

                for j in eligible
                    cluster = period_clusters[ip][j]
                    rel_jump = _relative_difference(cluster.velocity, prev_velocity)
                    rel_jump <= max_smooth_jump_fraction || continue

                    low_bonus = _local_low_velocity_bonus(period_clusters[ip], j, eligible)
                    score = selection_mode == :low_velocity ?
                            (cluster.confidence + 0.08 * cluster.support + 0.12 * low_bonus - smoothness_weight * rel_jump) :
                            (cluster.confidence + 0.08 * cluster.support - smoothness_weight * rel_jump)
                    if score > best_score
                        best_score = score
                        best_j = j
                    end
                end

                if best_j == 0
                    gap_count += 1
                    gap_count > max_gap_periods && break
                    continue
                end

                selected[ip, icand] = best_j
                used[ip][best_j] = true
                prev_velocity = period_clusters[ip][best_j].velocity
                gap_count = 0
            end
        end
    end

    return selected
end

# ╔═╡ 0eed3963-d6e2-4c6d-938b-7e233811364c
"""
    consensus_group_velocity_picks(result::BranchBatchAnalysisResult;
        correlation_threshold=0.85,
        velocity_tolerance_fraction=0.10,
        cluster_tolerance_fraction=nothing,
        min_support=1,
        min_confidence=0.0,
        smoothness_weight=1.0,
        max_candidates=3,
        max_smooth_jump_fraction=0.12,
        max_gap_periods=1,
        selection_mode=:low_velocity,
        min_candidate_periods=2)

Combine causal/acausal-matched group-velocity candidates across source states
for one receiver pair. Causal/acausal agreement is required inside each source
state, while source states are pooled with OR-style support. The output keeps
up to `max_candidates` smooth candidate branches with gaps allowed.
	Use `selection_mode=:low_velocity` to trace the local minimum group-velocity
	envelope; use `selection_mode=:confidence` for the older confidence-first
	branch ordering. Candidate branches that cover more periods are ordered ahead
	of shorter branches; set `min_candidate_periods=1` to keep single-period islands.
	"""
function consensus_group_velocity_picks(result::BranchBatchAnalysisResult;
                                        correlation_threshold::Float64=0.85,
                                        velocity_tolerance_fraction::Float64=0.10,
                                        cluster_tolerance_fraction::Union{Float64,Nothing}=nothing,
                                        min_support::Int=1,
                                        min_confidence::Float64=0.0,
                                        smoothness_weight::Float64=1.0,
                                        max_candidates::Int=3,
                                        max_smooth_jump_fraction::Float64=0.12,
                                        max_gap_periods::Int=1,
                                        selection_mode::Symbol=:low_velocity,
                                        min_candidate_periods::Int=2)
    nstates = length(result.state_results)
    nstates > 0 || throw(ArgumentError("At least one source state is required"))
    min_candidate_periods >= 1 || throw(ArgumentError("min_candidate_periods must be >= 1"))
    cluster_tol = isnothing(cluster_tolerance_fraction) ? velocity_tolerance_fraction : cluster_tolerance_fraction
    cluster_tol >= 0.0 || throw(ArgumentError("cluster_tolerance_fraction must be >= 0"))
    nperiods = length(result.periods)
    distance_ref = result.state_results[1].distance
    for (istate, st) in enumerate(result.state_results)
        st.distance ≈ distance_ref || throw(ArgumentError("State $(istate) distance $(st.distance) does not match receiver-pair distance $(distance_ref)"))
    end

    period_candidates = [Vector{_ConsensusCandidate}() for _ in 1:nperiods]
    candidate_velocities = [Float64[] for _ in 1:nperiods]

    for (istate, st) in enumerate(result.state_results)
        for ip in 1:nperiods
            corr = st.branch_correlation[ip]
            (isfinite(corr) && corr >= correlation_threshold) || continue

            causal_peaks = st.causal_result.all_peaks[ip]
            acausal_peaks = st.acausal_result.all_peaks[ip]
            matched_velocities = _matched_peak_average_velocities(causal_peaks, acausal_peaks, st.distance;
                                                                  velocity_tolerance_fraction=velocity_tolerance_fraction)
            for velocity in matched_velocities
                isfinite(velocity) && velocity > 0.0 || continue
                push!(period_candidates[ip], _ConsensusCandidate(velocity, istate, corr))
                push!(candidate_velocities[ip], velocity)
            end
        end
    end

    period_clusters = [_cluster_consensus_candidates(period_candidates[ip], nstates;
                                                     cluster_tolerance_fraction=cluster_tol)
                       for ip in 1:nperiods]
    candidate_cluster_index = _select_smooth_consensus_candidate_branches(period_clusters;
                                                                          max_candidates=max_candidates,
                                                                          min_support=min_support,
                                                                          min_confidence=min_confidence,
                                                                          smoothness_weight=smoothness_weight,
                                                                          max_smooth_jump_fraction=max_smooth_jump_fraction,
                                                                          max_gap_periods=max_gap_periods,
                                                                          selection_mode=selection_mode)
    selected_cluster_index = vec(candidate_cluster_index[:, 1])

    group_velocities = fill(NaN, nperiods)
    arrival_times = fill(NaN, nperiods)
    confidence = fill(0.0, nperiods)
    support = zeros(Int, nperiods)
    candidate_group_velocities = fill(NaN, nperiods, max_candidates)
    candidate_arrival_times = fill(NaN, nperiods, max_candidates)
    candidate_confidence = fill(0.0, nperiods, max_candidates)
    candidate_support = zeros(Int, nperiods, max_candidates)
    accepted_candidate_velocities = [Float64[] for _ in 1:nperiods]
    rejected_candidate_velocities = [copy(candidate_velocities[ip]) for ip in 1:nperiods]

    for icand in 1:max_candidates
        for ip in 1:nperiods
            j = candidate_cluster_index[ip, icand]
            j == 0 && continue

            cluster = period_clusters[ip][j]
            candidate_group_velocities[ip, icand] = cluster.velocity
            candidate_confidence[ip, icand] = cluster.confidence
            candidate_support[ip, icand] = cluster.support

            if isfinite(distance_ref) && distance_ref > 0.0 && isfinite(cluster.velocity) && cluster.velocity > 0.0
                candidate_arrival_times[ip, icand] = distance_ref / cluster.velocity
            end
        end
    end

    candidate_counts = [count(v -> isfinite(v) && v > 0.0, candidate_group_velocities[:, icand])
                        for icand in 1:max_candidates]
    for icand in 1:max_candidates
        if candidate_counts[icand] < min_candidate_periods
            candidate_group_velocities[:, icand] .= NaN
            candidate_arrival_times[:, icand] .= NaN
            candidate_confidence[:, icand] .= 0.0
            candidate_support[:, icand] .= 0
            candidate_cluster_index[:, icand] .= 0
        end
    end

    candidate_counts = [count(v -> isfinite(v) && v > 0.0, candidate_group_velocities[:, icand])
                        for icand in 1:max_candidates]
    candidate_medians = [_median_sorted([v for v in candidate_group_velocities[:, icand] if isfinite(v) && v > 0.0])
                         for icand in 1:max_candidates]
    candidate_mean_conf = [_mean_finite([v for v in candidate_confidence[:, icand] if isfinite(v) && v > 0.0])
                           for icand in 1:max_candidates]
    for icand in 1:max_candidates
        isnan(candidate_medians[icand]) && (candidate_medians[icand] = Inf)
        isnan(candidate_mean_conf[icand]) && (candidate_mean_conf[icand] = 0.0)
    end

    order = if selection_mode == :low_velocity
        sortperm(1:max_candidates, by=icand -> (-candidate_counts[icand],
                                                candidate_medians[icand],
                                                -candidate_mean_conf[icand]))
    else
        sortperm(1:max_candidates, by=icand -> (-candidate_counts[icand],
                                                -candidate_mean_conf[icand],
                                                candidate_medians[icand]))
    end

    candidate_group_velocities = candidate_group_velocities[:, order]
    candidate_arrival_times = candidate_arrival_times[:, order]
    candidate_confidence = candidate_confidence[:, order]
    candidate_support = candidate_support[:, order]
    candidate_cluster_index = candidate_cluster_index[:, order]

    for ip in 1:nperiods
        group_velocities[ip] = candidate_group_velocities[ip, 1]
        arrival_times[ip] = candidate_arrival_times[ip, 1]
        confidence[ip] = candidate_confidence[ip, 1]
        support[ip] = candidate_support[ip, 1]

        accepted_all = Set{Int}()
        for icand in 1:max_candidates
            j = candidate_cluster_index[ip, icand]
            j == 0 && continue
            union!(accepted_all, period_clusters[ip][j].candidate_indices)
        end

        accepted_candidate_velocities[ip] = [period_candidates[ip][k].velocity for k in sort(collect(accepted_all))]
        rejected_candidate_velocities[ip] = [period_candidates[ip][k].velocity
                                             for k in eachindex(period_candidates[ip])
                                             if !(k in accepted_all)]
    end

    return SourceStateConsensusPick(result.periods, group_velocities, arrival_times,
                                    confidence, support, candidate_velocities,
                                    accepted_candidate_velocities, rejected_candidate_velocities,
                                    selected_cluster_index,
                                    candidate_group_velocities, candidate_arrival_times,
                                    candidate_confidence, candidate_support,
                                    candidate_cluster_index)
end

# ╔═╡ a669a82b-b147-45d8-bf4e-c2e4d97c1bd3
"""
    plot_consensus_groupvelocity_picks(batch_result, consensus_result; ...)

Plot all causal/acausal-matched source-state candidates as faint markers and
overlay the accepted OR-style source-state consensus group-velocity curve.
"""
function plot_consensus_groupvelocity_picks(batch_result::BranchBatchAnalysisResult,
                                            consensus_result::SourceStateConsensusPick;
                                            correlation_threshold::Float64=0.85,
                                            velocity_tolerance_fraction::Float64=0.10,
                                            colorscale::String="Viridis",
                                            width::Int=1200,
                                            height::Int=700,
                                            font_family::String="Arial, sans-serif",
                                            font_size::Int=13,
                                            title::String="Source-State Consensus Group-Velocity Picks")
    nperiods = length(consensus_result.periods)
    nstates = length(batch_result.state_results)
    state_colors = _sample_scheme_colors(colorscale, max(nstates, 2))
    traces = [scatter()]
    all_vels = Float64[]

    for istate in 1:nstates
        xp = Float64[]
        yp = Float64[]
        cp = Float64[]
        label = batch_result.state_labels[istate]

        for ip in 1:nperiods
            st = batch_result.state_results[istate]
            corr = st.branch_correlation[ip]
            (isfinite(corr) && corr >= correlation_threshold) || continue
            causal_peaks = st.causal_result.all_peaks[ip]
            acausal_peaks = st.acausal_result.all_peaks[ip]
            matched_velocities = _matched_peak_average_velocities(causal_peaks, acausal_peaks, st.distance;
                                                                  velocity_tolerance_fraction=velocity_tolerance_fraction)
            for velocity in matched_velocities
                isfinite(velocity) && velocity > 0.0 || continue
                push!(xp, consensus_result.periods[ip])
                push!(yp, velocity)
                push!(cp, corr)
                push!(all_vels, velocity)
            end
        end

        isempty(xp) && continue
        push!(traces, scatter(
            x=xp,
            y=yp,
            mode="markers",
            marker=attr(size=6, color=state_colors[istate], opacity=0.28, symbol="circle"),
            name="$(label) candidates",
            hovertemplate="State: $(label)<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Corr: %{customdata:.3f}<extra></extra>",
            customdata=cp
        ))
    end

    candidate_colors = ["#111111", "#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    ncandidates = size(consensus_result.candidate_group_velocities, 2)

    for icand in 1:ncandidates
        vels = consensus_result.candidate_group_velocities[:, icand]
        valid = findall(v -> isfinite(v) && v > 0.0, vels)
        isempty(valid) && continue

        append!(all_vels, vels[valid])
        marker_sizes = [8 + 3 * consensus_result.candidate_support[ip, icand] for ip in valid]
        custom = [[consensus_result.candidate_confidence[ip, icand],
                   consensus_result.candidate_support[ip, icand],
                   icand] for ip in valid]
        color = candidate_colors[mod1(icand, length(candidate_colors))]

        push!(traces, scatter(
            x=consensus_result.periods,
            y=vels,
            mode="lines+markers",
            connectgaps=false,
            line=attr(color=color, width=icand == 1 ? 3.0 : 2.2, dash=icand == 1 ? "solid" : "dash"),
            marker=attr(
                size=[isfinite(vels[ip]) && vels[ip] > 0.0 ? 8 + 3 * consensus_result.candidate_support[ip, icand] : 0 for ip in 1:nperiods],
                color=consensus_result.candidate_confidence[:, icand],
                colorscale="Viridis",
                cmin=0.0,
                cmax=1.0,
                colorbar=icand == 1 ? attr(title="Confidence") : nothing,
                symbol=icand == 1 ? "diamond" : "circle",
                line=attr(color="white", width=1.0)
            ),
            name="Consensus candidate $(icand)",
            hovertemplate="Consensus candidate %{customdata[2]}<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Confidence: %{customdata[0]:.3f}<br>Support: %{customdata[1]} states<extra></extra>",
            customdata=[[consensus_result.candidate_confidence[ip, icand],
                         consensus_result.candidate_support[ip, icand],
                         icand] for ip in 1:nperiods]
        ))
    end

    if length(traces) == 1 || isempty(all_vels)
        @warn "No source-state consensus candidates available"
        return PlutoPlotly.plot(scatter(x=[0.0], y=[0.0], text=["No consensus candidates"]))
    end

    y_min = 0.9 * minimum(all_vels)
    y_max = 1.1 * maximum(all_vels)

    layout = Layout(
        title=attr(text=title, font=attr(size=font_size + 2, family=font_family)),
        xaxis=attr(title="Period (s)", type="linear", range=[minimum(consensus_result.periods), maximum(consensus_result.periods)], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        yaxis=attr(title="Group Velocity (km/s)", range=[y_min, y_max], showgrid=true, gridcolor="rgba(128,128,128,0.2)"),
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=attr(l=80, r=120, t=80, b=80),
        showlegend=true,
        legend=attr(x=1.02, y=1.0, font=attr(size=font_size - 2), bgcolor="rgba(255,255,255,0.8)", borderwidth=1)
    )

    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ 6e91e0bb-ef16-48bc-a9c9-f5b2d691ee1a
"""
    ReceiverPairGeometry

Endpoint and ray-path summary for one receiver pair.
"""
struct ReceiverPairGeometry
    station1::String
    station2::String
    lat1::Float64
    lon1::Float64
    lat2::Float64
    lon2::Float64
    midpoint_lat::Float64
    midpoint_lon::Float64
    distance_km::Float64
    azimuth_deg::Float64
end

# ╔═╡ 19ac99e3-f2b0-4d78-af7d-b688d548724b
"""
    PairConsensusForTomography

Geometry plus source-state consensus candidates for one receiver pair.
"""
struct PairConsensusForTomography
    label::String
    pair::Tuple{String,String}
    geometry::ReceiverPairGeometry
    consensus::SourceStateConsensusPick
end

# ╔═╡ c50a86d5-7f3e-4d03-9583-0a33a238e227
"""
    TomographyCandidateMix

One candidate curve for tomography. `candidate_indices` may contain several
non-overlapping consensus candidate columns from the same receiver pair.
"""
struct TomographyCandidateMix
    pair_index::Int
    label::String
    candidate_indices::Vector{Int}
    group_velocities::Vector{Float64}
    confidence::Vector{Float64}
    support::Vector{Int}
    coverage_count::Int
    mean_confidence::Float64
    neighbor_agreement::Float64
    total_score::Float64
end

# ╔═╡ b2460f88-a89c-4402-a3ee-6f9190624932
function _haversine_km(lat1::Real, lon1::Real, lat2::Real, lon2::Real)
    r = 6371.0
    φ1, φ2 = deg2rad(Float64(lat1)), deg2rad(Float64(lat2))
    dφ = deg2rad(Float64(lat2 - lat1))
    dλ = deg2rad(Float64(lon2 - lon1))
    a = sin(dφ / 2)^2 + cos(φ1) * cos(φ2) * sin(dλ / 2)^2
    return 2.0 * r * asin(min(1.0, sqrt(a)))
end

# ╔═╡ 9da0699a-8146-44b7-8eef-89f3e7cf6882
function _azimuth_deg(lat1::Real, lon1::Real, lat2::Real, lon2::Real)
    φ1, φ2 = deg2rad(Float64(lat1)), deg2rad(Float64(lat2))
    dλ = deg2rad(Float64(lon2 - lon1))
    y = sin(dλ) * cos(φ2)
    x = cos(φ1) * sin(φ2) - sin(φ1) * cos(φ2) * cos(dλ)
    return mod(rad2deg(atan(y, x)) + 360.0, 360.0)
end

# ╔═╡ b67deee3-1244-4f49-81bd-05c86b0c9084
function _axial_angle_difference_deg(a::Real, b::Real)
    d = abs(mod(Float64(a - b) + 180.0, 360.0) - 180.0)
    return min(d, 180.0 - d)
end

# ╔═╡ 40e5fc55-8f4e-43fa-8ac7-eaa9608a18c7
function receiver_pair_geometry(pair::Tuple{<:AbstractString,<:AbstractString},
                                latitudes::AbstractVector{<:Real},
                                longitudes::AbstractVector{<:Real};
                                distance::Union{Nothing,Real}=nothing)
    length(latitudes) >= 2 || throw(ArgumentError("latitudes must contain two endpoints"))
    length(longitudes) >= 2 || throw(ArgumentError("longitudes must contain two endpoints"))

    lat1, lat2 = Float64(latitudes[1]), Float64(latitudes[2])
    lon1, lon2 = Float64(longitudes[1]), Float64(longitudes[2])
    midpoint_lat = 0.5 * (lat1 + lat2)
    midpoint_lon = 0.5 * (lon1 + lon2)
    dist = isnothing(distance) ? _haversine_km(lat1, lon1, lat2, lon2) : Float64(distance)
    az = _azimuth_deg(lat1, lon1, lat2, lon2)

    return ReceiverPairGeometry(String(pair[1]), String(pair[2]),
                                lat1, lon1, lat2, lon2,
                                midpoint_lat, midpoint_lon, dist, az)
end

# ╔═╡ f203a997-875b-42df-a0af-196d11e7c0af
function tomography_pair_consensus(pair::Tuple{<:AbstractString,<:AbstractString},
                                   consensus::SourceStateConsensusPick;
                                   latitudes,
                                   longitudes,
                                   distance=nothing,
                                   label::Union{Nothing,String}=nothing)
    geom = receiver_pair_geometry(pair, latitudes, longitudes; distance=distance)
    lbl = isnothing(label) ? "$(pair[1])-$(pair[2])" : label
    return PairConsensusForTomography(lbl, (String(pair[1]), String(pair[2])), geom, consensus)
end

# ╔═╡ a05f6476-568b-4c82-9512-f74ae6d6c8f3
function similar_ray_paths(a::ReceiverPairGeometry, b::ReceiverPairGeometry;
                           midpoint_radius_km::Float64=75.0,
                           azimuth_tolerance_deg::Float64=25.0,
                           distance_tolerance_fraction::Float64=0.35)
    mid_dist = _haversine_km(a.midpoint_lat, a.midpoint_lon, b.midpoint_lat, b.midpoint_lon)
    az_diff = _axial_angle_difference_deg(a.azimuth_deg, b.azimuth_deg)
    dist_rel = abs(a.distance_km - b.distance_km) / max((a.distance_km + b.distance_km) / 2.0, eps(Float64))
    return mid_dist <= midpoint_radius_km &&
           az_diff <= azimuth_tolerance_deg &&
           dist_rel <= distance_tolerance_fraction
end

# ╔═╡ 730cd90e-9afe-413a-ae31-fe624a919bb8
function _nonoverlapping_candidate_sets(consensus::SourceStateConsensusPick;
                                        max_mix_parts::Int=3,
                                        min_candidate_periods::Int=2)
    ncandidates = size(consensus.candidate_group_velocities, 2)
    valid_masks = [isfinite.(consensus.candidate_group_velocities[:, i]) .&
                   (consensus.candidate_group_velocities[:, i] .> 0.0)
                   for i in 1:ncandidates]
    valid_indices = [i for i in 1:ncandidates if count(valid_masks[i]) >= min_candidate_periods]
    mixes = Vector{Vector{Int}}()

    function extend!(current::Vector{Int}, start_pos::Int, used_mask::BitVector)
        !isempty(current) && push!(mixes, copy(current))
        length(current) >= max_mix_parts && return
        for pos in start_pos:length(valid_indices)
            idx = valid_indices[pos]
            any(used_mask .& valid_masks[idx]) && continue
            push!(current, idx)
            extend!(current, pos + 1, used_mask .| valid_masks[idx])
            pop!(current)
        end
    end

    extend!(Int[], 1, falses(length(consensus.periods)))
    return mixes
end

# ╔═╡ eab08182-daca-4206-882b-e4fb24ae17a2
function _build_mix_curve(consensus::SourceStateConsensusPick, candidate_indices::Vector{Int})
    nperiods = length(consensus.periods)
    velocities = fill(NaN, nperiods)
    confidence = fill(0.0, nperiods)
    support = zeros(Int, nperiods)

    for idx in candidate_indices
        vals = consensus.candidate_group_velocities[:, idx]
        mask = isfinite.(vals) .& (vals .> 0.0)
        velocities[mask] .= vals[mask]
        confidence[mask] .= consensus.candidate_confidence[mask, idx]
        support[mask] .= consensus.candidate_support[mask, idx]
    end

    return velocities, confidence, support
end

# ╔═╡ c6331917-8167-4dfa-9a50-6fe0888d16d5
function _neighbor_agreement(mix::TomographyCandidateMix,
                             all_mixes::Vector{TomographyCandidateMix},
                             pairs::Vector{PairConsensusForTomography};
                             midpoint_radius_km::Float64=75.0,
                             azimuth_tolerance_deg::Float64=25.0,
                             distance_tolerance_fraction::Float64=0.35,
                             velocity_tolerance_fraction::Float64=0.10)
    geom = pairs[mix.pair_index].geometry
    total = 0
    agree = 0

    for ip in eachindex(mix.group_velocities)
        v = mix.group_velocities[ip]
        isfinite(v) && v > 0.0 || continue
        total += 1
        period_agrees = false

        for other in all_mixes
            other.pair_index == mix.pair_index && continue
            similar_ray_paths(geom, pairs[other.pair_index].geometry;
                              midpoint_radius_km=midpoint_radius_km,
                              azimuth_tolerance_deg=azimuth_tolerance_deg,
                              distance_tolerance_fraction=distance_tolerance_fraction) || continue
            vo = other.group_velocities[ip]
            isfinite(vo) && vo > 0.0 || continue
            if _relative_difference(v, vo) <= velocity_tolerance_fraction
                period_agrees = true
                break
            end
        end

        period_agrees && (agree += 1)
    end

    total == 0 && return 0.0
    return agree / total
end

# ╔═╡ 3626d6ca-38f2-4b38-9398-9cd052f43950
"""
    tomography_candidate_mixes(pairs; kwargs...) -> Vector{TomographyCandidateMix}

Create and score tomography-ready candidate curves from per-pair consensus
results. Each output can be one consensus candidate or a non-overlapping mix of
candidate columns from the same receiver pair. Scores prefer period coverage,
confidence/support, and agreement with geometrically similar ray paths.
"""
function tomography_candidate_mixes(pairs::Vector{PairConsensusForTomography};
                                    max_mix_parts::Int=3,
                                    min_candidate_periods::Int=2,
                                    midpoint_radius_km::Float64=75.0,
                                    azimuth_tolerance_deg::Float64=25.0,
                                    distance_tolerance_fraction::Float64=0.35,
                                    velocity_tolerance_fraction::Float64=0.10,
                                    coverage_weight::Float64=1.0,
                                    confidence_weight::Float64=0.5,
                                    support_weight::Float64=0.25,
                                    neighbor_weight::Float64=1.0)
    mixes = TomographyCandidateMix[]

    for (pair_index, item) in enumerate(pairs)
        candidate_sets = _nonoverlapping_candidate_sets(item.consensus;
                                                        max_mix_parts=max_mix_parts,
                                                        min_candidate_periods=min_candidate_periods)
        for candidate_indices in candidate_sets
            velocities, confidence, support = _build_mix_curve(item.consensus, candidate_indices)
            coverage = count(v -> isfinite(v) && v > 0.0, velocities)
            coverage >= min_candidate_periods || continue
            mean_conf = _mean_finite([c for c in confidence if c > 0.0])
            isfinite(mean_conf) || (mean_conf = 0.0)
            label = "$(item.label) | candidates $(join(candidate_indices, "+"))"
            push!(mixes, TomographyCandidateMix(pair_index, label, candidate_indices,
                                                velocities, confidence, support,
                                                coverage, mean_conf, 0.0, 0.0))
        end
    end

    scored = TomographyCandidateMix[]
    max_coverage = maximum([m.coverage_count for m in mixes]; init=1)
    max_support = maximum([maximum(m.support; init=0) for m in mixes]; init=1)

    for mix in mixes
        neighbor = _neighbor_agreement(mix, mixes, pairs;
                                       midpoint_radius_km=midpoint_radius_km,
                                       azimuth_tolerance_deg=azimuth_tolerance_deg,
                                       distance_tolerance_fraction=distance_tolerance_fraction,
                                       velocity_tolerance_fraction=velocity_tolerance_fraction)
        support_score = _mean_finite([s / max_support for s in mix.support if s > 0])
        isfinite(support_score) || (support_score = 0.0)
        total = coverage_weight * (mix.coverage_count / max_coverage) +
                confidence_weight * mix.mean_confidence +
                support_weight * support_score +
                neighbor_weight * neighbor

        push!(scored, TomographyCandidateMix(mix.pair_index, mix.label, mix.candidate_indices,
                                             mix.group_velocities, mix.confidence, mix.support,
                                             mix.coverage_count, mix.mean_confidence, neighbor, total))
    end

    sort!(scored, by=m -> -m.total_score)
    return scored
end

# ╔═╡ a1000001-0000-0000-0000-000000000001
function _column_normalise(X)
    Xf = Float64.(X)
    return mapslices(x -> begin
        n = norm(x)
        n > 0 ? x ./ n : x
    end, Xf; dims=1)
end

# ╔═╡ a1000002-0000-0000-0000-000000000001
function _vector_normalise(x)
    xf = Float64.(vec(x))
    n = norm(xf)
    return n > 0 ? xf ./ n : xf
end

# ╔═╡ a1000003-0000-0000-0000-000000000001
function _ncc(a, b)
    a0 = Float64.(vec(a)) .- mean(a)
    b0 = Float64.(vec(b)) .- mean(b)
    return dot(a0, b0) / ((norm(a0) * norm(b0)) + 1e-8)
end

# ╔═╡ a1000004-0000-0000-0000-000000000001
_compact_number(x) = @sprintf("%g", Float64(x))

# ╔═╡ a1000005-0000-0000-0000-000000000001
function plot_source_state_waveforms(item; dt, velocity_range, period_min, period_max)
    isnothing(item) && return PlutoPlotly.plot(PlutoPlotly.scatter(x=[0], y=[0], text=["No run selected"]))
    cluster_avg_ac = _column_normalise(item.acausal)
    cluster_avg_c = _column_normalise(item.causal)
    nth = size(cluster_avg_ac, 1)
    t_neg = [-(nth - i + 1) * dt for i in 1:nth]
    t_pos = [i * dt for i in 1:nth]
    t_full = [t_neg; t_pos]

    global_avg_ac = _vector_normalise(item.global_avg_ac)
    global_avg_c = _vector_normalise(item.global_avg_c)
    global_full = [reverse(global_avg_ac); global_avg_c]
    global_ncc = _ncc(global_avg_ac, global_avg_c)

    combo_labels_local = item.combo_labels
    ncomb = size(cluster_avg_ac, 2)
    traces = AbstractTrace[]
    colors = begin
        nc = max(ncomb, 1)
        cs = ColorSchemes.rainbow
        [Colors.hex(get(cs, (i - 1) / max(1, nc - 1))) for i in 1:nc]
    end

    total_ac = sum(item.counts_ac)
    total_c = sum(item.counts_c)
    amp_peak = maximum(abs.(vcat(vec(cluster_avg_ac), vec(cluster_avg_c), global_full)))
    vertical_spacing = amp_peak * 2.5 + 1e-3

    for combo_idx in 1:ncomb
        c = colors[mod1(combo_idx, length(colors))]
        a = cluster_avg_ac[:, combo_idx]
        b = cluster_avg_c[:, combo_idx]
        full_k = [reverse(a); b]
        ncc = _ncc(a, b)
        pct_ac = 100 * item.counts_ac[combo_idx] / max(total_ac, 1)
        pct_c = 100 * item.counts_c[combo_idx] / max(total_c, 1)
        state_label = combo_idx <= length(combo_labels_local) ? combo_labels_local[combo_idx] : string(combo_idx)
        legend_label = "State $(state_label) (ac: $(round(pct_ac; digits=1))%, c: $(round(pct_c; digits=1))%, corr=$(round(ncc; digits=3)))"
        offset = (combo_idx - 1) * vertical_spacing
        push!(traces, PlutoPlotly.scatter(x=t_full, y=global_full .+ offset, mode="lines",
            name=combo_idx == 1 ? "Global mean (corr=$(round(global_ncc; digits=3)))" : "Global mean",
            showlegend=combo_idx == 1,
            line=attr(color="rgba(0,0,0,0.18)", width=3)))
        push!(traces, PlutoPlotly.scatter(x=t_full, y=full_k .+ offset, mode="lines",
            name=legend_label, line=attr(color=c, width=2)))
    end

    shapes = if isnothing(item.distance)
        []
    else
        vmin, vmax = velocity_range
        t_fast = item.distance / vmax
        t_slow = item.distance / vmin
        [attr(type="line", x0=t, x1=t, y0=0, y1=1, yref="paper",
              line=attr(color="rgba(0,0,0,0.25)", width=1, dash="dash"))
         for t in (-t_slow, -t_fast, t_fast, t_slow)]
    end

    distance_label = isnothing(item.distance) ? "distance unavailable" : "$(round(Int, item.distance))km"
    title = "Source State Average Waveforms ($(item.pair_label) seed=$(item.seed) $(distance_label) $(_compact_number(period_min))-$(_compact_number(period_max))s)"
    return PlutoPlotly.plot(traces, PlutoPlotly.Layout(
        title=attr(text=title, font=attr(size=18, family="Computer Modern, serif")),
        height=500 * max(1, cld(ncomb, 5)),
        width=900,
        xaxis=attr(title="Lag (s)", zeroline=true, zerolinecolor="rgba(0,0,0,0.3)"),
        yaxis=attr(title="Amplitude"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(x=0.5, xanchor="center", y=-0.2, orientation="h",
            font=attr(size=12, family="Computer Modern, serif")),
        shapes=shapes,
    ))
end

# ╔═╡ a1000006-0000-0000-0000-000000000001
function plot_cluster_histogram(counts_ac, counts_c; title="Cluster Usage", labels=nothing)
    K = length(counts_ac)
    total_ac = max(sum(counts_ac), 1)
    total_c  = max(sum(counts_c),  1)
    pct_ac = 100.0 .* Float64.(counts_ac) ./ total_ac
    pct_c  = 100.0 .* Float64.(counts_c)  ./ total_c
    xlabels = isnothing(labels) ? string.(1:K) : string.(labels)
    traces = [
        PlutoPlotly.bar(x=xlabels, y=pct_ac, name="Acausal",
            marker=attr(color="rgba(31,119,180,0.7)")),
        PlutoPlotly.bar(x=xlabels, y=pct_c,  name="Causal",
            marker=attr(color="rgba(214,39,40,0.7)")),
    ]
    layout = Layout(
        title=attr(text=title, font=attr(size=18)),
        barmode="group", height=400, width=700,
        xaxis=attr(title="Source state"),
        yaxis=attr(title="Usage (%)"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return PlutoPlotly.plot(traces, layout)
end

# ╔═╡ a1000007-0000-0000-0000-000000000001
function plot_state_ncc_heatmap(acausal::AbstractMatrix, causal::AbstractMatrix;
        labels=nothing, title="State-State Normalised Correlation")
    n = size(acausal, 2)
    xlabels = isnothing(labels) ? string.(1:n) : string.(labels)

    function ncc_matrix(A)
        C = Matrix{Float32}(undef, n, n)
        cols = [begin v = vec(Float64.(A[:, i])); v .- mean(v) end for i in 1:n]
        norms = [norm(c) + 1e-8 for c in cols]
        for i in 1:n, j in 1:n
            C[i, j] = dot(cols[i], cols[j]) / (norms[i] * norms[j])
        end
        return C
    end

    C_ac = ncc_matrix(acausal)
    C_c  = ncc_matrix(causal)
    trace_ac = PlutoPlotly.heatmap(
        z=C_ac, x=xlabels, y=xlabels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=attr(title="Corr", len=0.9, x=0.46),
        xaxis="x1", yaxis="y1",
    )
    trace_c = PlutoPlotly.heatmap(
        z=C_c, x=xlabels, y=xlabels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=attr(title="Corr", len=0.9, x=1.01),
        xaxis="x2", yaxis="y2",
    )
    sz = max(350, n * 40)
    layout = Layout(
        title=attr(text=title, font=attr(size=16)),
        grid=attr(rows=1, columns=2, pattern="independent"),
        annotations=[
            attr(text="Acausal", x=0.22, xref="paper", y=1.05, yref="paper",
                 showarrow=false, font=attr(size=14)),
            attr(text="Causal",  x=0.78, xref="paper", y=1.05, yref="paper",
                 showarrow=false, font=attr(size=14)),
        ],
        xaxis=attr(title="State", tickangle=-45),
        yaxis=attr(title="State"),
        xaxis2=attr(title="State", tickangle=-45),
        yaxis2=attr(title="State"),
        width=900, height=sz + 80,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=attr(t=80, b=80, l=80, r=80),
    )
    return PlutoPlotly.plot([trace_ac, trace_c], layout)
end

# ╔═╡ a1000008-0000-0000-0000-000000000001
function plot_top_tomography_mixes(mixes, pairs; n::Int=25,
        period_min::Real=1.0, period_max::Real=100.0)
    isempty(mixes) && return PlutoPlotly.plot(PlutoPlotly.scatter(x=[0], y=[0], text=["No mixes"]))
    top = mixes[1:min(n, length(mixes))]
    all_vels = Float64[]
    colors = [Colors.hex(get(ColorSchemes.viridis, (i - 1) / max(1, length(top) - 1))) for i in 1:length(top)]
    traces = [PlutoPlotly.scatter()]
    for (i, mix) in enumerate(top)
        valid = findall(v -> isfinite(v) && v > 0.0, mix.group_velocities)
        isempty(valid) && continue
        append!(all_vels, mix.group_velocities[valid])
        pair = pairs[mix.pair_index]
        push!(traces, PlutoPlotly.scatter(
            x=pair.consensus.periods[valid],
            y=mix.group_velocities[valid],
            mode="lines+markers",
            name=mix.label,
            line=attr(color=colors[i], width=2),
            marker=attr(size=6, color=colors[i]),
            customdata=[[mix.total_score, mix.neighbor_agreement, mix.coverage_count] for _ in valid],
            hovertemplate="%{fullData.name}<br>Period: %{x:.2f} s<br>v_g: %{y:.3f} km/s<br>Score: %{customdata[0]:.3f}<br>Neighbor agree: %{customdata[1]:.3f}<br>Periods: %{customdata[2]}<extra></extra>",
        ))
    end
    isempty(all_vels) && return PlutoPlotly.plot(PlutoPlotly.scatter(x=[0], y=[0], text=["No valid velocities"]))
    return PlutoPlotly.plot(traces, PlutoPlotly.Layout(
        title="Top trained-VQ-VAE tomography candidate mixes",
        xaxis=attr(title="Period (s)", type="linear", range=[period_min, period_max]),
        yaxis=attr(title="Group velocity (km/s)", range=[0.9 * minimum(all_vels), 1.1 * maximum(all_vels)]),
        width=1200, height=720,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=attr(x=1.02, y=1.0),
        margin=attr(l=80, r=180, t=70, b=70),
    ))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ColorSchemes = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
Peaks = "18e31ff7-3703-566c-8e60-38913d67486b"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[compat]
ColorSchemes = "~3.31.0"
Colors = "~0.13.1"
DSP = "~0.8.4"
FFTW = "~1.10.0"
Peaks = "~0.4"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.73"
StatsBase = "~0.34.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "9308f5dde31405d136237b9bf2bd75905a1f1439"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

    [deps.DSP.weakdeps]
    OffsetArrays = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Peaks]]
deps = ["Compat", "RecipesBase"]
git-tree-sha1 = "ca47b866754525ede84e5dec84a104c45f92afb6"
uuid = "18e31ff7-3703-566c-8e60-38913d67486b"
version = "0.4.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "28278bb0053da0fd73537be94afd1682cc5a0a83"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.21"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "8acd04abc9a636ef57004f4c2e6f3f6ed4611099"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.5"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "972089912ba299fba87671b025cd0da74f5f54f7"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.0"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═7ebf7c4e-353d-4cfd-8094-c07343589e4e
# ╠═a1b2c3d4-0001-0000-0000-000000000001
# ╠═bbcb5888-9968-49f5-97f0-db8e5c53c121
# ╠═de53b0d8-61aa-4b68-8fd3-a8d775ba8c95
# ╠═a210884b-23a6-4e03-9028-cd4949c0304d
# ╟─25162009-a8a6-4983-a89c-a6be9769f1f7
# ╟─161e84fc-db09-4a69-ab53-c73eaa4a1486
# ╠═e4d5e20b-2bcc-4a22-a2d1-aae7974b02cf
# ╠═8816d460-fa4d-4b46-803c-5e0c4b80fb9e
# ╠═2de6446f-d07c-4151-8e13-0719c3056826
# ╠═5ecefa6f-2a86-46bf-b91b-9ce4c0a32e08
# ╠═09d3b269-c8bd-425a-ad4a-1ecd29150507
# ╠═1585e739-23c2-4a23-8e80-6f132d503765
# ╟─9ce67356-5ff7-4a22-b244-75e35350f959
# ╠═b6d23222-ba2a-11f0-bcd7-8f110c4a4cd6
# ╠═cee98fdd-f0d9-4ea8-9256-2492a649a512
# ╟─72db9d05-2714-4e86-bf36-a0c8ee13c040
# ╠═7088299d-e267-42e3-9c87-1a238332f0f4
# ╠═fcf48627-0b80-4c6b-b204-f40da981aaa5
# ╠═d5ca213a-b085-4d65-88fa-d221bf5826e1
# ╠═4192fcef-4662-4944-b354-71e8e00285b1
# ╠═2c1a711e-261e-4e82-ade2-2fb6bc0e6b93
# ╠═e9bfb21b-f29d-410f-827f-e66d1117020f
# ╠═f71ee0c9-1e34-49c2-9385-4589c561f4e4
# ╠═aae31a73-fef6-48dc-a2b8-fb9b12c03099
# ╠═74f5430a-8d8a-41cf-b683-52cc29bb12c0
# ╠═0c8a6b9e-5e71-48d3-bff0-072ae085fd5a
# ╟─a9eed5d0-7811-4210-999c-ef28ef7769bb
# ╟─6cf5d72c-2380-4585-86ed-dfd2afda5bd0
# ╠═a7d1a9e8-c1dd-4fbe-a1c7-3b7d6a1fe3a1
# ╠═d779b115-911a-44f2-8cb9-6f1962a5d4a3
# ╠═df761225-7a00-4a7c-b56f-611243a75e97
# ╠═ee08ecac-5010-4a64-820e-43ee80823939
# ╠═3bab4eab-92da-4539-b3df-d5c637d1ce77
# ╠═c7a19591-be69-423d-a510-7ec0c0cedf27
# ╠═120b0f0e-2b75-447b-aeeb-caf9844e0290
# ╠═0f0ffc2f-3754-4b4a-9b69-27f9702c9541
# ╠═c322f226-d679-454a-9bcb-4f67f46b2bb2
# ╠═0e5573f5-5322-4784-8c53-97230373dc13
# ╠═78ba3876-d4d0-4eb8-8300-1c3cef28db58
# ╠═f0db8807-ddd7-47b8-a792-54b15fb07c7e
# ╟─ccfe0c61-2be2-4bf0-b01d-b3fa06bf2f05
# ╠═7a64a743-db7b-405c-a334-793d2b8a36db
# ╟─c2d1790c-cbac-447a-a234-dfb4d6d188e0
# ╠═875e8a94-2a80-4ad9-8cc6-c48adaab8e4f
# ╠═c169578d-3dd2-42f8-8e57-4146a8dca2cd
# ╠═9db0d269-d3ca-48fc-9583-91b5e11822d2
# ╠═1293cef0-186d-4a43-86f5-181372ef59e0
# ╠═7b8ea90d-a09d-4952-bd66-943808dc9d3b
# ╟─2ba1a355-d9fa-4268-b046-0b201191c11e
# ╠═44fad356-ff66-4bbc-a83b-5d8d996d95a5
# ╠═8dc0f4bc-00cb-11f1-9443-a59380edd23b
# ╠═8dc0fc46-00cb-11f1-9407-453086ae541f
# ╠═64882ed3-eea4-45a7-bf0f-6d0c1e14d5af
# ╠═6f3cb4dc-11ec-4e8d-a9f6-c4c2301cbcf1
# ╠═7fd65030-f541-4fd4-aea8-8ab6d9cc6d0e
# ╠═78a3288f-6b85-4ed1-a399-b4528799f95a
# ╠═871ae074-31c8-4b62-8a94-3aa018b1de87
# ╠═6d5ab2a2-8ce2-4b1d-96ea-0aa1c27e83fd
# ╠═76f5c114-3cdb-11f1-917c-2b05ca51409f
# ╠═82c85400-3cdb-11f1-9af8-1da7c901570c
# ╠═82c857c8-3cdb-11f1-8b14-3f44a70d8c9a
# ╠═82c86256-3cdb-11f1-befc-c7b40eeb6b06
# ╠═8cd3a49a-3cdb-11f1-bf54-4dd564d7b55a
# ╠═347f96a4-3cdc-11f1-9cee-ab8ff408ac98
# ╠═347f9a3c-3cdc-11f1-8f82-9d9b48d7858b
# ╠═347f9eb0-3cdc-11f1-a7ce-4d4b3258205b
# ╠═4c914cc4-3cdc-11f1-9254-6500ce058f12
# ╠═4c915e4e-3cdc-11f1-9235-33f20a13a658
# ╠═91b2f6bc-c0f9-4b84-a3d1-37ff6a4bf02d
# ╠═60f8c1a1-5c76-4cda-a257-84bc5bf8c791
# ╠═7f0eb497-46c0-4a43-95d5-39f6af0d783e
# ╠═5da0af71-cfbc-4598-bf6f-cf38f8fce6fe
# ╠═c0fb74a4-5f74-4451-a097-b0ddb4e79f2e
# ╠═4f47e6c2-0f17-4f71-a7da-08f651176632
# ╠═07f578f5-f269-45f1-a6e6-c90f4d0f5c2f
# ╠═58d6ecb6-2c42-4a53-9cbc-b72ebf48356c
# ╠═1a67f649-4fd7-4f63-bf4a-28d5cb66f52d
# ╠═88284a1d-00ff-4f69-ac6e-a39dbec70863
# ╠═f26e8f7a-80d8-4f99-ab64-8e13a870e145
# ╠═d10d4b32-4e4e-4198-b08d-084dffeb3694
# ╠═95f715de-a6aa-4ec5-a2ce-4f7bb241ce34
# ╠═c9597868-5ae3-4629-b96d-d25f9a7d316d
# ╠═a079a952-8252-4f56-a4c7-2a9ae58e0f11
# ╠═f2c83612-d161-4fef-a728-c2dcbfa48b1c
# ╠═1c36878c-f684-46a8-a2ab-4203a01d0169
# ╠═7f9cbf19-46b3-403b-90f8-a4591fd5ef4f
# ╠═f7f4e17f-0784-43db-9a6b-c2b07a963f94
# ╠═c17d7f26-3d82-44b8-860c-b12df0bf9293
# ╠═1a7d4cbb-a1ea-4319-a404-2293faf72069
# ╠═c7600895-0b53-477d-aaca-19925dec0d91
# ╠═07914f7d-70fb-477d-bb05-dfa2a68d03ff
# ╠═2e76ea52-ad8e-4f32-9b01-b88d7d1163f9
# ╠═bdccfc35-4436-4716-a868-bec5a18317d3
# ╠═0eed3963-d6e2-4c6d-938b-7e233811364c
# ╠═a669a82b-b147-45d8-bf4e-c2e4d97c1bd3
# ╠═6e91e0bb-ef16-48bc-a9c9-f5b2d691ee1a
# ╠═19ac99e3-f2b0-4d78-af7d-b688d548724b
# ╠═c50a86d5-7f3e-4d03-9583-0a33a238e227
# ╠═b2460f88-a89c-4402-a3ee-6f9190624932
# ╠═9da0699a-8146-44b7-8eef-89f3e7cf6882
# ╠═b67deee3-1244-4f49-81bd-05c86b0c9084
# ╠═40e5fc55-8f4e-43fa-8ac7-eaa9608a18c7
# ╠═f203a997-875b-42df-a0af-196d11e7c0af
# ╠═a05f6476-568b-4c82-9512-f74ae6d6c8f3
# ╠═730cd90e-9afe-413a-ae31-fe624a919bb8
# ╠═eab08182-daca-4206-882b-e4fb24ae17a2
# ╠═c6331917-8167-4dfa-9a50-6fe0888d16d5
# ╠═3626d6ca-38f2-4b38-9398-9cd052f43950
# ╠═a1000001-0000-0000-0000-000000000001
# ╠═a1000002-0000-0000-0000-000000000001
# ╠═a1000003-0000-0000-0000-000000000001
# ╠═a1000004-0000-0000-0000-000000000001
# ╠═a1000005-0000-0000-0000-000000000001
# ╠═a1000006-0000-0000-0000-000000000001
# ╠═a1000007-0000-0000-0000-000000000001
# ╠═a1000008-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
