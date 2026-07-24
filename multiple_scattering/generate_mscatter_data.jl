#!/usr/bin/env julia
# ─────────────────────────────────────────────────────────────────────────────
# Synthetic multichannel-blind-deconvolution data generator.
#
# Physics ported from ~/Notebooks/multiple_scattering_results_save1/
# multiple_scattering.jl. A point earthquake source radiates into a 2-D acoustic
# medium filled with randomly placed circular scatterers; the multiple-scattering
# wavefield is solved once per frequency (MultipleScattering.jl). The complex
# per-receiver frequency response IS the receiver Green's function g_r; the
# recorded seismogram is d_r = s * g_r (source convolved with that receiver's
# response), formed as a product in the frequency domain.
#
# This is the forward model for multichannel blind deconvolution: many receivers
# share ONE common source s but each has its OWN Green's function g_r. Traveltime
# moveout between receivers arises naturally from the receiver-line geometry
# (unequal source–receiver distances) — no jitter is injected.
#
# "Multiple sources" = multiple INDEPENDENT events: each source k gets its own
# random scatterer field, its own source-time function s_k, and produces its own
# (Nt × R) gather. Each event is written as a self-describing JLD2 file plus a
# companion PGFPlotsX PNG overview (scatterers + receivers + source, impulse
# responses, data) so the file can be understood at a glance.
#
# Parallelism: the expensive per-frequency solves are distributed with `pmap`
# (the primary axis). The source loop is kept serial because each event's
# frequency solves already saturate the worker pool. (If you ever want many
# cheap events instead, sources could become the pmap axis — not built here.)
#
# Run:  julia -p 8 generate_mscatter_data.jl
#       (without -p, the script adds workers itself via addprocs()).
# ─────────────────────────────────────────────────────────────────────────────

using Distributed

# If launched without `-p N`, spin up a local worker pool so the frequency
# `pmap` is actually parallel. With `-p N` julia already added the workers.
if nworkers() == 1
    addprocs()
end

# MultipleScattering must be available on every worker (the per-frequency solve
# runs remotely). Everything else that only the master touches stays plain.
@everywhere using MultipleScattering
@everywhere using LinearAlgebra: norm

using FFTW
using Random
using Distributions
using DSP: hilbert
using JLD2, FileIO
using Dates
using Statistics: std
using PGFPlotsX

# ── Configuration ────────────────────────────────────────────────────────────
Base.@kwdef struct MScatterConfig
    # Time / frequency grid
    Nt::Int = 512
    T::Float64 = 100.0            # total record length (s); dt = T/(Nt-1)
    # Medium
    num_particles::Int = 70
    radius::Float64 = 1.0
    max_width_factor::Float64 = 50.0   # box half-extent = max_width_factor*radius
    host_c::ComplexF64 = 1.0 + 0.001im # host sound speed (slight attenuation)
    host_rho::Float64 = 1.0
    part_c::Float64 = 0.7              # scatterer sound speed
    part_rho::Float64 = 0.7           # scatterer density
    res::Int = 256                     # wavefield grid resolution per solve
    # Receiver line (a vertical line at x = rec_x; spread in depth → moveout)
    n_receivers::Int = 100
    rec_x::Float64 = 50.0
    rec_zmin::Float64 = -50.0
    rec_zmax::Float64 = 50.0
    # Source-time function (earthquake STF)
    Mw::Float64 = 6.5
    num_pulses::Int = 4
    pulse_shape::String = "ricker"
    # Additive colored noise on the data (fraction, see build_noise)
    noise_amp::Float64 = 0.0
    # Dataset
    n_sources::Int = 3                 # number of independent events
    # Master seed for the run. Defaults to a fresh random draw so that EACH run
    # gets a different scatterer field (the medium changes between runs); pass an
    # explicit `seed=...` to reproduce a specific run. Within a run, event k uses
    # `seed + k` for its particles/source.
    seed::Int = rand(1:1_000_000)
    outdir::String = joinpath(@__DIR__, "data")
end

# ── Reused physics functions ────────────────────────────────────────────────
# (ported from the reference notebook; see cell line refs in comments)

"""
    generate_source_wavelet(tgrid; Mw, fs, num_pulses, shape, rng) -> (t, stf)

Synthetic earthquake source-time function: sum of `num_pulses` randomly placed
Gaussian or Ricker pulses, scaled by seismic moment. (Reference lines 235-261;
`rng` added for reproducibility, global `Distributions` calls threaded through it.)
"""
function generate_source_wavelet(tgrid; Mw=6.0, fs=100.0, num_pulses=3,
                                  shape="gaussian", rng::AbstractRNG=Random.default_rng())
    t = tgrid
    stf = zeros(length(t))
    M0 = 10.0^(1.5Mw + 9.1)     # seismic moment (Nm)
    amp_scale = M0 / 1e20        # normalize for numerics

    for _ in 1:num_pulses
        delay = rand(rng, Uniform(0.1 * last(tgrid), 0.2 * last(tgrid)))
        width = rand(rng, Uniform(0.5, 5.0))
        amp   = rand(rng, Uniform(0.5, 1.2)) * amp_scale / num_pulses
        center = delay
        if shape == "gaussian"
            stf .+= amp .* exp.(-((t .- center) .^ 2) ./ (2 * width^2))
        elseif shape == "ricker"
            f = 1 / width
            stf .+= amp .* (1 .- 2 * (pi * f * (t .- center)) .^ 2) .*
                    exp.(-(pi * f * (t .- center)) .^ 2)
        else
            error("Unsupported pulse shape: $shape")
        end
    end
    return t, stf
end

# Complex field at receiver (x_in, y_in): nearest grid point of a solve result.
# (Reference lines 203-207.) Defined @everywhere because get_recs runs on the
# master but is small; kept remote-safe in case it is ever mapped. A docstring
# cannot be attached to an `@everywhere` definition, hence this plain comment.
@everywhere function get_field_at_xy(x_in, y_in, result)
    query = [x_in, y_in]
    idx = argmin(norm.(result.x .- Ref(query)))
    return result.field[idx][1]
end

"""
    get_impulse_responses(rX, rZ, results, Nt) -> g   (Nt × R)

For each receiver `(rX[i], rZ[i])`, assemble the per-frequency complex response
`R` (the receiver Green's function) and return the time-domain impulse response
`g = irfft(conj(R), Nt)`.

The source convolution is intentionally NOT done here: the scattering physics
only produces the medium's impulse responses `g_r`. Any source `s` and the
observed data `d_r = s * g_r` are formed downstream (in the training notebook),
so one simulated medium is reusable for arbitrary sources.

The `conj(R)` convention matches the reference exactly; keep it so downstream
`d = s * g` is not time-reversed. (Reference lines 359-369.)
"""
function get_impulse_responses(rX, rZ, results, Nt)
    g = map(rX, rZ) do x, y
        R = map(results) do result
            get_field_at_xy(x, y, result)
        end
        R = vcat([0.0 + 0.0im], R)        # prepend zero-frequency component
        return irfft(conj(R), Nt)
    end
    return hcat(g...)
end

# ── One event ────────────────────────────────────────────────────────────────
"""
    simulate_event(cfg, k, tvec, ωvec, rX, rZ) -> NamedTuple

Run one independent event `k`: build its own random scatterer field, solve
per-frequency responses (parallel `pmap`), and form the impulse responses `g`.
The source and convolution are done downstream (in the training notebook), so
this returns only `g` + the geometry needed to save/plot the event.
"""
function simulate_event(cfg::MScatterConfig, k::Int, tvec, ωvec, rX, rZ)
    Nt = cfg.Nt
    nω = length(ωvec)

    max_width = cfg.max_width_factor * cfg.radius
    bottomleft = [-20.0, -max_width]
    topright = [max_width, max_width]
    shape = Box([bottomleft, topright])
    bounds = Box([bottomleft, topright])

    host_medium = Acoustic(cfg.host_rho, cfg.host_c, 2)
    particle_medium = Acoustic(cfg.part_rho, cfg.part_c, 2)
    particle_shape = [Circle(cfg.radius), Circle(cfg.radius + 1), Circle(cfg.radius + 2)]

    particles = random_particles(particle_medium, particle_shape;
                                 region_shape=shape, num_particles=cfg.num_particles,
                                 seed=cfg.seed + k)
    source_xy = [bottomleft[1], 1.0]
    source = point_source(host_medium, source_xy, 1.0)
    simulation = FrequencySimulation(particles, source)

    # Primary parallel axis: one multiple-scattering solve per frequency.
    # Only reference locals inside the closure (`res`, not `cfg.res`): capturing
    # `cfg` would serialize the master-only `MScatterConfig` type to workers.
    res = cfg.res
    results = pmap(ωvec[2:nω]) do ω
        run(simulation, bounds, [ω]; res=res)
    end

    # Impulse responses only — no source, no convolution (done in the notebook).
    g = get_impulse_responses(rX, rZ, results, Nt)

    # Scatterer geometry (for metadata + plotting) — reference lines 311-321.
    particle_origins = [p.shape.origin for p in simulation.particles]
    particle_radii = [p.shape.radius for p in simulation.particles]

    # A representative wavefield frame for the overview plot (a mid frequency).
    ω_idx_plot = clamp(60, 1, length(results))
    field_result = results[ω_idx_plot]
    Xg = unique(first.(field_result.x))
    Zg = unique(last.(field_result.x))
    field2d = abs.(reshape(first.(field_result.field), length(Xg), length(Zg)))

    return (; g, source_xy, particle_origins, particle_radii,
            Xg, Zg, field2d, ω_plot=ωvec[ω_idx_plot + 1])
end

# ── PGFPlotsX overview figure (publication style, ref lines 175-654) ─────────
PGFPlotsX.CUSTOM_PREAMBLE = [
    raw"\usetikzlibrary{backgrounds}",
    raw"\tikzset{every picture/.style={background rectangle/.style={fill=white}, show background rectangle}}",
    raw"\usepgfplotslibrary{fillbetween}",
]

"""Vertical-offset wiggle panel of a (Nt × R) gather (every 5th trace)."""
function wiggle_axis(title, tvec, M; dy=10, xwin=(0.1, 0.9))
    Irec = 1:5:size(M, 2)
    G = M ./ max.(std(M, dims=1), eps())
    nt = length(tvec)
    imin = max(1, round(Int, xwin[1] * nt)); imax = min(nt, round(Int, xwin[2] * nt))
    @pgf Axis(
        {
            xmajorgrids, ymajorgrids,
            title = title,
            width = "5cm", height = "10cm",
            enlargelimits = false,
            xmin = tvec[imin], xmax = tvec[imax],
            ymin = 0, ymax = (length(Irec) - 1) * dy,
            ytick = "\\empty",
            xlabel = "Time (s)",
            title_style = "{font=\\fontsize{15}{12}\\selectfont, at={(0.5,1)}}",
            ticklabel_style = "{font=\\fontsize{12}{10}\\selectfont}",
        },
        [@pgf PGFPlotsX.Plot({color = "blue", no_marks, "line width=0.6pt"},
                             Coordinates(tvec, dy * (i - 1) .+ G[:, j]))
         for (i, j) in enumerate(Irec)]...,
    )
end

"""Build the multi-panel PGFPlotsX overview and save it as `<basename>.png`."""
function save_overview_png(path, ev, tvec, cfg::MScatterConfig, k)
    # a) Wavefield with scatterers + receivers + source.
    delta = 2
    wave = @pgf Axis(
        {
            view = (0, 90),
            width = "7cm", height = "10cm",
            xlabel = "Distance", ylabel = "Depth",
            title = "a) Wavefield (\\omega=$(round(ev.ω_plot, digits=3)))",
            title_style = "{font=\\fontsize{15}{12}\\selectfont, at={(0.5,1)}}",
            ticklabel_style = "{font=\\fontsize{12}{10}\\selectfont}",
            "colormap/hot",
        },
        (@pgf Plot3({surf, shader = "flat"},
                    Coordinates(ev.Xg[1:delta:end], ev.Zg[1:delta:end],
                                ev.field2d[1:delta:end, 1:delta:end]))),
        # scatterers (white circles sized by radius)
        (@pgf PGFPlotsX.Plot({scatter, "only marks", "scatter src" = "explicit symbolic",
                              "scatter/classes" = {
                                  "1" = {mark = "o", "white", "mark size" = 2},
                                  "2" = {mark = "o", "white", "mark size" = 3},
                                  "3" = {mark = "o", "white", "mark size" = 4}}},
                             Table({meta = "label"},
                                   x = first.(ev.particle_origins),
                                   y = last.(ev.particle_origins),
                                   label = string.(Int.(round.(ev.particle_radii)))))),
        # receivers (red triangles)
        (@pgf PGFPlotsX.Plot({color = "red", "only marks", "mark size" = 3, mark = "triangle"},
                             Table([rx for rx in fill(cfg.rec_x, length(1:5:cfg.n_receivers))],
                                   collect(range(cfg.rec_zmin, cfg.rec_zmax, length=cfg.n_receivers))[1:5:end]))),
        # source (green star)
        (@pgf PGFPlotsX.Plot({color = "green!60!black", "only marks", "mark size" = 5, mark = "star"},
                             Coordinates([ev.source_xy[1]], [ev.source_xy[2]]))),
    )

    g_panel = wiggle_axis("b) Impulse responses \$g_r\$", tvec, ev.g)

    fig = @pgf GroupPlot(
        {group_style = {group_size = "2 by 1", "horizontal sep" = "1.4cm"}},
        wave, g_panel,
    )
    pgfsave(path, fig; dpi=250)
    return path
end

# ── Driver ───────────────────────────────────────────────────────────────────
function generate_dataset(cfg::MScatterConfig=MScatterConfig())
    mkpath(cfg.outdir)
    dt = cfg.T / (cfg.Nt - 1)
    tvec = range(0, cfg.T, length=cfg.Nt)
    ωvec = rfftfreq(cfg.Nt, inv(dt))
    rX = fill(cfg.rec_x, cfg.n_receivers)
    rZ = collect(range(cfg.rec_zmin, cfg.rec_zmax, length=cfg.n_receivers))

    git_commit = try
        readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
    catch
        "unknown"
    end
    config_dict = Dict(string(f) => getfield(cfg, f) for f in fieldnames(MScatterConfig))

    @info "Generating $(cfg.n_sources) event(s)" nworkers = nworkers() Nt = cfg.Nt outdir = cfg.outdir

    written = String[]
    for k in 1:cfg.n_sources
        @info "── Event $k / $(cfg.n_sources) ──"
        ev = simulate_event(cfg, k, tvec, ωvec, rX, rZ)

        # Include the run seed so concurrent background runs (which can share a
        # timestamp to the second) never collide/overwrite each other's files.
        base = "event_s$(cfg.seed)_k$(k)"
        jld2_path = joinpath(cfg.outdir, base * ".jld2")
        png_path = joinpath(cfg.outdir, base * ".png")

        save(jld2_path, Dict(
            "g" => ev.g,                       # per-receiver impulse response, Nt × R
            "tvec" => collect(tvec),
            "dt" => dt,
            "Nt" => cfg.Nt,
            "omega" => collect(ωvec),
            "rX" => rX, "rZ" => rZ, "R" => cfg.n_receivers,
            "source_xy" => ev.source_xy,      # source location used for the sim geometry
            "particle_origins" => ev.particle_origins,
            "particle_radii" => ev.particle_radii,
            "source_index" => k,
            "n_sources" => cfg.n_sources,
            "seed" => cfg.seed,
            "config" => config_dict,
            "created" => string(now()),
            "git_commit" => git_commit,
        ))
        @info "wrote $jld2_path"

        try
            save_overview_png(png_path, ev, tvec, cfg, k)
            @info "wrote $png_path"
        catch e
            @warn "PGFPlotsX PNG export failed; JLD2 still written" exception = (e, catch_backtrace())
        end
        push!(written, jld2_path)
    end
    @info "Done." files = written
    return written
end

# Run when executed as a script (not when `include`d for interactive use).
if abspath(PROGRAM_FILE) == @__FILE__
    generate_dataset()
end
