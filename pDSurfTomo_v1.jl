### A Pluto.jl notebook ###
# v0.20.27

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

# ╔═╡ 4ef5cf9c-0583-45d4-8f4e-87c79c56bc8a
begin
    using Dates
    using DelimitedFiles
    using Printf
    using Statistics

    using GMT
    using PlutoUI
end

# ╔═╡ aa000001-0000-0000-0000-000000000001
md"""
# pDSurfTomo GPU notebook

Provide a flat dispersion file (format: `period  lat1  lon1  lat2  lon2  velocity_km_s`),
set all inversion parameters below, and click **Run pDSurfTomo**.

The notebook writes `DSurfTomo.in`, `surfdata.dat`, `MOD`, and `ParallelConfig.in`
into a run directory under `DSurfTomo_runs/`, then runs `pDSurfTomo DSurfTomo.in`
and reuses the same GMT model/residual diagnostics as the baseline notebook.
"""

# ╔═╡ f62bc402-2a89-498b-8bed-020f40a447c5
begin
    project_root = @__DIR__
    pdsurftomo_root = joinpath(project_root, "External", "pDSurfTomo")
    runs_root = joinpath(project_root, "DSurfTomo_runs")
    repo_url = "https://github.com/zshang825/pDSurfTomo.git"
    repo_branch = "main"
    mkpath(runs_root)
end

# ╔═╡ aa000010-0000-0000-0000-000000000010
md"""
## Section 1 — Dispersion data input
"""

# ╔═╡ aa000011-0000-0000-0000-000000000011
md"""
**Flat dispersion file** (6 columns: `period lat1 lon1 lat2 lon2 velocity`):
"""

# ╔═╡ aa000012-0000-0000-0000-000000000012
@bind ui_dispersion_file confirm(TextField(80;
    default=""))

# ╔═╡ aa000013-0000-0000-0000-000000000013
dispersion_data = let
    path = strip(ui_dispersion_file)
    isfile(path) || error("File not found: $path")
    rows = readlines(path)
    parsed = NamedTuple[]
    for r in rows
        parts = split(strip(r))
        length(parts) == 6 || continue
        vals = tryparse.(Float64, parts)
        any(isnothing, vals) && continue
        push!(parsed, (;
            period=vals[1], lat1=vals[2], lon1=vals[3],
            lat2=vals[4], lon2=vals[5], velocity=vals[6]))
    end
    isempty(parsed) && error("No valid 6-column rows in $path")
    parsed
end

# ╔═╡ aa000014-0000-0000-0000-000000000014
data_summary = let
    periods  = sort(unique(r.period  for r in dispersion_data))
    all_lats = vcat([r.lat1 for r in dispersion_data], [r.lat2 for r in dispersion_data])
    all_lons = vcat([r.lon1 for r in dispersion_data], [r.lon2 for r in dispersion_data])
    n_pairs  = length(unique(Set([r.lat1, r.lon1, r.lat2, r.lon2]) for r in dispersion_data))
    (;
        n_rows   = length(dispersion_data),
        n_periods = length(periods),
        period_min = minimum(periods),
        period_max = maximum(periods),
        periods,
        lat_min = minimum(all_lats),
        lat_max = maximum(all_lats),
        lon_min = minimum(all_lons),
        lon_max = maximum(all_lons),
        n_pairs,
        vel_min = minimum(r.velocity for r in dispersion_data),
        vel_max = maximum(r.velocity for r in dispersion_data),
        n_stations = length(unique(vcat(
            [(r.lat1, r.lon1) for r in dispersion_data],
            [(r.lat2, r.lon2) for r in dispersion_data]))),
        n_sta1 = length(unique([(r.lat1, r.lon1) for r in dispersion_data])),
    )
end

# ╔═╡ aa000015-0000-0000-0000-000000000015
md"""
**Data summary:** $(data_summary.n_rows) measurements · $(data_summary.n_periods) periods
[$(round(data_summary.period_min; digits=2)) – $(round(data_summary.period_max; digits=2)) s] ·
$(data_summary.n_stations) stations ·
lat [$(round(data_summary.lat_min; digits=3)) – $(round(data_summary.lat_max; digits=3))] ·
lon [$(round(data_summary.lon_min; digits=3)) – $(round(data_summary.lon_max; digits=3))]
"""

# ╔═╡ aa000020-0000-0000-0000-000000000020
md"""
## Section 2 — Tomography parameters

All parameters are exposed here. Grid geometry is auto-suggested from the data
extent; override as needed.
"""

# ╔═╡ aa000021-0000-0000-0000-000000000021
md"""
### Grid geometry

The grid origin is the **upper-left** corner (max lat, min lon). Grid counts
$(nx × ny × nz)$ cover lat × lon × depth. Grid spacing is in **degrees** for
lat/lon and **km** for depth layers.

Auto-suggested spacing = ⅓ of smallest inter-station separation (DSurfTomo convention).
"""

# ╔═╡ aa000022-0000-0000-0000-000000000022
suggested_grid = let
    # Minimum inter-station distance in degrees (approx, ignoring Earth curvature)
    lats = unique(vcat([r.lat1 for r in dispersion_data], [r.lat2 for r in dispersion_data]))
    lons = unique(vcat([r.lon1 for r in dispersion_data], [r.lon2 for r in dispersion_data]))
    sta = unique(vcat([(r.lat1, r.lon1) for r in dispersion_data],
                      [(r.lat2, r.lon2) for r in dispersion_data]))
    min_dist = Inf
    for i in 1:length(sta), j in i+1:length(sta)
        d = sqrt((sta[i][1]-sta[j][1])^2 + (sta[i][2]-sta[j][2])^2)
        d < min_dist && (min_dist = d)
    end
    dg = ceil(min_dist / 3 * 1000) / 1000
    dg = max(dg, 0.05)  # at least 0.05°
    # Origin: 1 grid outside data extent (DSurfTomo convention)
    origin_lat = data_summary.lat_max + dg
    origin_lon = data_summary.lon_min - dg
    # DSurfTomo uses nvx = nx-2 internal nodes; grid spans (nvx-1)*dg = (nx-3)*dg degrees.
    # Need (nx-3)*dg >= lat_span + 2*dg (1 cell buffer each side), so nx >= lat_span/dg + 5.
    lat_span = data_summary.lat_max - data_summary.lat_min
    lon_span = data_summary.lon_max - data_summary.lon_min
    nx = ceil(Int, lat_span / dg) + 5
    ny = ceil(Int, lon_span / dg) + 5
    (; dg, origin_lat, origin_lon, nx, ny)
end

# ╔═╡ aa000023-0000-0000-0000-000000000023
@bind _grid_params PlutoUI.combine() do Child
    sg = suggested_grid
    md"""
    | Parameter | Value | Suggested |
    |:---|:---|:---|
    | Grid spacing dg (°) | $(Child("dg",      NumberField(0.01:0.01:1.0;    default=round(sg.dg; digits=3)))) | $(round(sg.dg; digits=3)) |
    | Origin latitude (N) | $(Child("orig_lat", NumberField(-90.0:0.001:90.0; default=round(sg.origin_lat; digits=3)))) | $(round(sg.origin_lat; digits=3)) |
    | Origin longitude (E)| $(Child("orig_lon", NumberField(-180.0:0.001:180.0; default=round(sg.origin_lon; digits=3)))) | $(round(sg.origin_lon; digits=3)) |
    | nx (lat nodes)      | $(Child("nx",       NumberField(5:1:200;           default=sg.nx))) | $(sg.nx) |
    | ny (lon nodes)      | $(Child("ny",       NumberField(5:1:200;           default=sg.ny))) | $(sg.ny) |
    | nz (depth layers)   | $(Child("nz",       NumberField(3:1:30;            default=9))) | 9 |
    """
end

# ╔═╡ aa000023-0000-0000-0000-000000000060
begin
	ui_dg        = _grid_params.dg
	ui_origin_lat = _grid_params.orig_lat
	ui_origin_lon = _grid_params.orig_lon
	ui_nx        = _grid_params.nx
	ui_ny        = _grid_params.ny
	ui_nz        = _grid_params.nz
end

# ╔═╡ aa000024-0000-0000-0000-000000000024
md"""
### Inversion parameters
"""

# ╔═╡ aa000025-0000-0000-0000-000000000025
@bind _inversion_params PlutoUI.combine() do Child
    md"""
    | Parameter | Value | Notes |
    |:---|:---|:---|
    | Minimum Vs (km/s) | $(Child("minvel", NumberField(0.1:0.1:5.0; default=1.5))) | lower model bound |
    | Maximum Vs (km/s) | $(Child("maxvel", NumberField(0.5:0.1:10.0; default=5.0))) | upper model bound |
    | Weight | $(Child("weight", NumberField(0.1:0.1:20.0; default=4.0))) | data weight |
    | Damping | $(Child("damp", NumberField(0.01:0.01:10.0; default=1.0))) | regularization |
    | Sub-layers | $(Child("sublayers", NumberField(2:1:5; default=3))) | depth-kernel layers |
    | Max iterations | $(Child("maxiter", NumberField(1:1:50; default=10))) | inversion iterations |
    | Sparsity fraction | $(Child("sparsity", NumberField(0.01:0.01:1.0; default=0.2))) | lower is faster |
    | Noise level (km/s) | $(Child("noiselevel", NumberField(0.001:0.001:0.5; default=0.02))) | expected data noise |
    | Outlier threshold (km/s) | $(Child("threshold", NumberField(0.1:0.1:10.0; default=3.0))) | residual down-weighting |
    | Depth layers (km) | $(Child("depth_layers", TextField(60; default="0, 1, 2, 4, 8, 15, 25, 40, 60"))) | comma-separated |
    | Wave type | $(Child("wavetype", Select(["2 — Rayleigh group velocity", "1 — Rayleigh phase velocity", "3 — Love group velocity", "4 — Love phase velocity"]; default="2 — Rayleigh group velocity"))) | matches dispersion rows |
    """
end

# ╔═╡ aa000026-0000-0000-0000-000000000026
begin
	ui_minvel = _inversion_params.minvel
	ui_maxvel = _inversion_params.maxvel
	ui_weight = _inversion_params.weight
	ui_damp = _inversion_params.damp
	ui_sublayers = _inversion_params.sublayers
	ui_maxiter = _inversion_params.maxiter
	ui_sparsity = _inversion_params.sparsity
	ui_noiselevel = _inversion_params.noiselevel
	ui_threshold = _inversion_params.threshold
	ui_depth_layers = _inversion_params.depth_layers
	ui_wavetype = _inversion_params.wavetype
	wavetype_int = parse(Int, first(split(ui_wavetype, " ")))
end

# ╔═╡ aa000028-0000-0000-0000-000000000028
depth_layers = let
    vals = tryparse.(Float64, strip.(split(ui_depth_layers, ",")))
    any(isnothing, vals) && error("Invalid depth layers — must be comma-separated numbers")
    sort(Float64.(vals))[1:min(ui_nz, length(vals))]
end

# ╔═╡ aa000029-0000-0000-0000-000000000029
md"""
### Initial model
"""

# ╔═╡ aa000029-0000-0000-0000-000000000050
suggested_ini = let
    vs_surface = round(data_summary.vel_min / 0.9; digits=2)
    vs_deep    = round(data_summary.vel_max / 0.9; digits=2)
    max_depth  = maximum(depth_layers)
    velgrad    = round(max((vs_deep - vs_surface) / max_depth, 0.01); digits=3)
    (; minvel=vs_surface, velgrad)
end

# ╔═╡ aa000029-0000-0000-0000-00000000002b
@bind _ini_params PlutoUI.combine() do Child
    si = suggested_ini
    md"""
    | Parameter | Value | Suggested |
    |:---|:---|:---|
    | Surface Vs (km/s at z=0) | $(Child("minvel",  NumberField(0.1:0.01:5.0;  default=si.minvel)))  | $(si.minvel)  |
    | Velocity gradient (km/s per km) | $(Child("velgrad", NumberField(0.0:0.001:1.0; default=si.velgrad))) | $(si.velgrad) |
    """
end

# ╔═╡ aa000029-0000-0000-0000-000000000061
begin
	ui_ini_minvel = _ini_params.minvel
	ui_ini_velgrad = _ini_params.velgrad
end

# ╔═╡ aa000032-0000-0000-0000-000000000032
md"""
### Run name

A subdirectory with this name will be created under `DSurfTomo_runs/`.
"""

# ╔═╡ aa000033-0000-0000-0000-000000000033
@bind ui_run_name TextField(40; default=let
    base = splitext(basename(strip(ui_dispersion_file)))[1]
    isempty(base) ? "my_run" : base
end)

# ╔═╡ aa000034-0000-0000-0000-000000000034
md"""
### pDSurfTomo parallel controls

GPU mode uses CuPy and does not silently fall back to CPU. If it fails, switch
solver mode or fix the CUDA/CuPy environment reported in the error output.
"""

# ╔═╡ aa000035-0000-0000-0000-000000000035
@bind _pdsurf_run_params PlutoUI.combine() do Child
    md"""
    | Parameter | Value | Notes |
    |:---|:---|:---|
    | Solver mode | $(Child("solver_mode", Select(["GPU CuPy — parallel, disba, cupy", "CPU optimized — parallel, disba, scipy", "Exact parallel — parallel, parallel, default", "Native baseline — default, default, default"]; default="GPU CuPy — parallel, disba, cupy"))) | default uses GPU/CuPy |
    | ThreadNum | $(Child("threads", NumberField(1:1:max(1, Sys.CPU_THREADS); default=max(1, Sys.CPU_THREADS ÷ 2)))) | pDSurfTomo worker threads |
    | CUDA_VISIBLE_DEVICES | $(Child("cuda_visible_devices", TextField(20; default=""))) | blank uses current visible GPU set |
    """
end

# ╔═╡ aa000036-0000-0000-0000-000000000036
begin
	ui_pdsurf_solver_mode = _pdsurf_run_params.solver_mode
	ui_pdsurf_threads = _pdsurf_run_params.threads
	ui_cuda_visible_devices = _pdsurf_run_params.cuda_visible_devices
end

# ╔═╡ aa000039-0000-0000-0000-000000000039
function pdsurf_solver_triplet(mode::AbstractString)
    startswith(mode, "GPU CuPy") && return "parallel, disba, cupy"
    startswith(mode, "CPU optimized") && return "parallel, disba, scipy"
    startswith(mode, "Exact parallel") && return "parallel, parallel, default"
    startswith(mode, "Native baseline") && return "default, default, default"
    error("Unknown pDSurfTomo solver mode: $mode")
end

# ╔═╡ bb000001-0000-0000-0000-000000000001
md"""
## Section 3 — Generate input files
"""

# ╔═╡ bb000002-0000-0000-0000-000000000002
run_dir = joinpath(runs_root, strip(ui_run_name))

# ╔═╡ bb000003-0000-0000-0000-000000000003
# DSurfTomo.in is generated entirely from the UI parameters — no hand-edited file needed.
generated_cfg = let
    # Compiled binary limits: surfdisp96.f NP=80, CalSurfG.f90 NP=60 — cap to 60.
    MAX_PERIODS = 60
    all_periods = data_summary.periods
    periods, period_warning = if length(all_periods) <= MAX_PERIODS
        all_periods, nothing
    else
        # Down-sample to MAX_PERIODS evenly spaced in log-period space
        idxs = round.(Int, range(1, length(all_periods); length=MAX_PERIODS))
        all_periods[idxs], length(all_periods)
    end
    (;
        datafile   = "surfdata.dat",
        nx         = ui_nx,
        ny         = ui_ny,
        nz         = ui_nz,
        goxd       = Float64(ui_origin_lat),
        gozd       = Float64(ui_origin_lon),
        dvxd       = Float64(ui_dg),
        dvzd       = Float64(ui_dg),
        nsrc       = data_summary.n_sta1 + 1,
        weight     = Float64(ui_weight),
        damp       = Float64(ui_damp),
        sublayers  = ui_sublayers,
        minvel     = Float64(ui_minvel),
        maxvel     = Float64(ui_maxvel),
        maxiter    = ui_maxiter,
        sparsity   = Float64(ui_sparsity),
        kmaxRc     = length(periods),
        tRc        = periods,
        noiselevel = Float64(ui_noiselevel),
        threshold  = Float64(ui_threshold),
        period_warning,
    )
end

# ╔═╡ bb000003-0000-0000-0000-00000000000a
isnothing(generated_cfg.period_warning) ? md"Periods: **$(generated_cfg.kmaxRc)** (all used)" :
    Markdown.MD(Markdown.Admonition("warning", "Period count capped",
        [md"DSurfTomo's `surfdisp96.f` has a hardcoded limit of **80 periods**. Your data has **$(generated_cfg.period_warning)** unique periods — down-sampled to **$(generated_cfg.kmaxRc)** evenly log-spaced periods. Measurements at excluded periods are dropped from `surfdata.dat`."]))

# ╔═╡ bb000004-0000-0000-0000-000000000004
function write_dsurftomo_in(path::AbstractString, c)
    open(path, "w") do io
        println(io, "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
        println(io, "c INPUT PARAMETERS")
        println(io, "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
        println(io, c.datafile)
        @printf(io, "%d %d %d\n", c.nx, c.ny, c.nz)
        @printf(io, "%-7.3f %7.3f\n", c.goxd, c.gozd)
        @printf(io, "%-7.3f %7.3f\n", c.dvxd, c.dvzd)
        println(io, c.nsrc)
        println(io, "$(c.weight) $(c.damp)")
        println(io, c.sublayers)
        println(io, "$(c.minvel) $(c.maxvel)")
        println(io, c.maxiter)
        println(io, c.sparsity)
        println(io, c.kmaxRc)
        println(io, join([@sprintf("%.7g", p) for p in c.tRc], " "))
        println(io, "0")
        println(io, "0")
        println(io, "0")
        println(io, "0")
        println(io, c.noiselevel)
        println(io, c.threshold)
    end
end

# ╔═╡ bb000005-0000-0000-0000-000000000005
function write_surfdata(path::AbstractString, data, periods)
    # Map each raw period to the nearest period in the (possibly capped) cfg list.
    # This ensures all measurements are used even when down-sampling was applied.
    nearest_idx(p) = argmin(abs(p - cp) for cp in periods)
    mapped = [(nearest_idx(r.period), r) for r in data]
    sort!(mapped, by=x -> (x[1], x[2].lat1, x[2].lon1))
    open(path, "w") do io
        cur_key = nothing
        wt = wavetype_int
        for (pidx, r) in mapped
            key = (pidx, r.lat1, r.lon1)
            if key != cur_key
                @printf(io, "#  %.6f %.6f %d %d 0\n", r.lat1, r.lon1, pidx, wt)
                cur_key = key
            end
            @printf(io, " %.6f %.6f %.6f\n", r.lat2, r.lon2, r.velocity)
        end
    end
end

# ╔═╡ bb000006-0000-0000-0000-000000000006
function write_mod(path::AbstractString, nx, ny, nz, dep1, minvel, velgrad)
    open(path, "w") do io
        for d in dep1
            @printf(io, "%5.1f", d)
        end
        print(io, "\n")
        for k in 1:nz
            for _ in 1:ny
                for _ in 1:nx
                    @printf(io, "%7.3f", minvel + dep1[k] * velgrad)
                end
                print(io, "\n")
            end
        end
    end
end

# ╔═╡ bb000006-0000-0000-0000-000000000016
function write_parallel_config(path::AbstractString; repo_dir::AbstractString,
                               thread_num::Integer, solver_mode::AbstractString)
    lsmr_script = abspath(joinpath(repo_dir, "bin", "lsmr_solver.py"))
    senk_script = abspath(joinpath(repo_dir, "bin", "senK_solver.py"))
    open(path, "w") do io
        println(io, "tt_senK_lsmr = $solver_mode")
        println(io, "ThreadNum = $thread_num")
        println(io, "InvResultDir = InvResult")
        println(io, "lsmr_script_path = $lsmr_script")
        println(io, "senK_script_path = $senk_script")
    end
    path
end

# ╔═╡ bb000007-0000-0000-0000-000000000007
prepare_status = let
    mkpath(run_dir)
    in_path       = joinpath(run_dir, "DSurfTomo.in")
    surf_path     = joinpath(run_dir, "surfdata.dat")
    mod_path      = joinpath(run_dir, "MOD")
    parallel_path = joinpath(run_dir, "ParallelConfig.in")

    write_dsurftomo_in(in_path, generated_cfg)

    write_surfdata(surf_path, dispersion_data, generated_cfg.tRc)
    n_src = count(startswith("#"), readlines(surf_path))

    dep1 = length(depth_layers) >= ui_nz ? depth_layers[1:ui_nz] :
           vcat(depth_layers, range(depth_layers[end]+5, step=10,
                                    length=ui_nz-length(depth_layers)))
    write_mod(mod_path, ui_nx, ui_ny, ui_nz, dep1, ui_ini_minvel, ui_ini_velgrad)
    solver_mode = pdsurf_solver_triplet(ui_pdsurf_solver_mode)
    write_parallel_config(parallel_path;
        repo_dir=pdsurftomo_root,
        thread_num=Int(ui_pdsurf_threads),
        solver_mode=solver_mode)

md"""
    **Files written to `$(run_dir)/`:**
    - `DSurfTomo.in` — grid $(ui_nx)×$(ui_ny)×$(ui_nz), spacing $(ui_dg)°,
      origin ($(round(ui_origin_lat;digits=3)) N, $(round(ui_origin_lon;digits=3)) E)
    - `surfdata.dat` — $(n_src) source blocks, $(length(dispersion_data)) measurements,
      $(length(data_summary.periods)) periods, wavetype=$(wavetype_int)
    - `MOD` — 1D gradient model, surface Vs=$(ui_ini_minvel) km/s,
      grad=$(ui_ini_velgrad) km/s per km
    - `ParallelConfig.in` — `$(solver_mode)`, ThreadNum=$(Int(ui_pdsurf_threads)),
      CUDA_VISIBLE_DEVICES=`$(strip(ui_cuda_visible_devices))`
"""
end

# ╔═╡ cc000001-0000-0000-0000-000000000001
md"""
## Section 4 — Run inversion
"""

# ╔═╡ 3f0a828f-df2c-4457-a42a-c9bfe1573767
md"""
### Julia process helpers
"""

# ╔═╡ 720c8fc5-a04e-48f3-a8cf-667c99afff4f
begin
    function run_command(cmd::Cmd; cwd::AbstractString=pwd())
        stdout_path, stdout_io = mktemp()
        stderr_path, stderr_io = mktemp()
        close(stdout_io)
        close(stderr_io)
        proc = run(pipeline(Cmd(cmd; dir=cwd, ignorestatus=true);
                            stdout=stdout_path, stderr=stderr_path))
        out = read(stdout_path, String)
        err = read(stderr_path, String)
        rm(stdout_path; force=true)
        rm(stderr_path; force=true)
        return (; rc=proc.exitcode, stdout=out, stderr=err,
                cmd=join(proc.cmd.exec, " "), cwd)
    end

    function assert_ok(result; context="")
        result.rc == 0 && return result
        msg = isempty(context) ? "Command failed" : context
        error("""
        $msg
        cwd: $(result.cwd)
        cmd: $(result.cmd)
        exit code: $(result.rc)
        stdout:
        $(result.stdout)
        stderr:
        $(result.stderr)
        """)
    end
end

# ╔═╡ 4b48e706-eef4-448a-91ab-5a82a6a9676c
begin
    function ensure_repo!(repo_dir; url=repo_url, branch=repo_branch, update=false)
        if isdir(joinpath(repo_dir, ".git"))
            update || return repo_dir
            assert_ok(run_command(`git fetch --depth=1 origin $branch`; cwd=repo_dir))
            assert_ok(run_command(`git checkout $branch`; cwd=repo_dir))
        else
            mkpath(dirname(repo_dir))
            assert_ok(run_command(`git clone --depth=1 --branch $branch $url $repo_dir`))
        end
        repo_dir
    end

    function ensure_uv!()
        uv = Sys.which("uv")
        isnothing(uv) && error("""
        `uv` was not found on PATH.
        Install it before syncing pDSurfTomo Python dependencies, for example:
            curl -LsSf https://astral.sh/uv/install.sh | sh
        Then restart Julia/Pluto so PATH includes `uv`.
        """)
        uv
    end

    function ensure_pdsurftomo_python!(repo_dir)
        uv = ensure_uv!()
        isdir(joinpath(repo_dir, ".venv")) && return uv
        # `uv sync` is idempotent and lets uv provision a Python >= 3.12 env
        # even when the system Python is older.
        assert_ok(run_command(`$uv sync`; cwd=repo_dir);
                  context="Could not sync pDSurfTomo Python dependencies with uv")
        uv
    end

    function build_pdsurftomo!(repo_dir)
        p_exe = joinpath(repo_dir, "bin", "pDSurfTomo")
        d_exe = joinpath(repo_dir, "bin", "DSurfTomo")
        if !(isfile(p_exe) && isfile(d_exe))
            assert_ok(run_command(`sh MyMake.sh`; cwd=joinpath(repo_dir, "src_DSurfTomo"));
                      context="Could not compile pDSurfTomo bundled DSurfTomo")
            assert_ok(run_command(`sh MyMake.sh`; cwd=joinpath(repo_dir, "src_pDSurfTomo"));
                      context="Could not compile pDSurfTomo")
        end
        required = (
            p_exe,
            joinpath(repo_dir, "bin", "lsmr_solver.py"),
            joinpath(repo_dir, "bin", "senK_solver.py"),
        )
        for path in required
            isfile(path) || error("Required pDSurfTomo file not found: $path")
        end
        p_exe
    end
end

# ╔═╡ e1dd47d1-11dc-4a1d-9e52-938edc24a4aa
pdsurftomo_setup = let
    ensure_repo!(pdsurftomo_root; update=false)
    uv_path = ensure_pdsurftomo_python!(pdsurftomo_root)
    pdsurftomo_exe = build_pdsurftomo!(pdsurftomo_root)
    venv_dir = joinpath(pdsurftomo_root, ".venv")
    python_exe = joinpath(venv_dir, "bin", "python")
    isfile(python_exe) ||
        error("pDSurfTomo uv environment is missing Python: $python_exe")
    (; pdsurftomo_exe, uv_path, venv_dir, python_exe)
end

# ╔═╡ ac7c9f25-3671-46cd-977d-49d6999d6d75
@bind run_button CounterButton("▶  Run pDSurfTomo")

# ╔═╡ 5b614467-5709-4d51-8eef-6c4599fd3125
pdsurftomo_result = let
    run_button  # re-run when button clicked
    prepare_status  # ensure files are written first
    pdsurftomo_setup
    input_path = joinpath(run_dir, "DSurfTomo.in")
    parallel_path = joinpath(run_dir, "ParallelConfig.in")
    isfile(joinpath(run_dir, "MOD")) ||
        error("MOD not found in $run_dir — check that Section 3 ran successfully")
    isfile(input_path) || error("DSurfTomo.in not found in $run_dir")
    isfile(parallel_path) || error("ParallelConfig.in not found in $run_dir")
    if startswith(ui_pdsurf_solver_mode, "GPU CuPy")
        nvidia_smi = Sys.which("nvidia-smi")
        isnothing(nvidia_smi) && error("GPU CuPy mode selected, but `nvidia-smi` was not found on PATH.")
        gpu_check = run_command(`$nvidia_smi -L`; cwd=run_dir)
        assert_ok(gpu_check; context="GPU CuPy mode selected, but NVIDIA GPUs are not visible")
    end
    venv_bin = joinpath(pdsurftomo_setup.venv_dir, "bin")
    env = copy(ENV)
    env["PATH"] = venv_bin * ":" * get(env, "PATH", "")
    env["VIRTUAL_ENV"] = pdsurftomo_setup.venv_dir
    env["PYTHONNOUSERSITE"] = "1"
    gpu_pin = strip(ui_cuda_visible_devices)
    isempty(gpu_pin) || (env["CUDA_VISIBLE_DEVICES"] = gpu_pin)
    cmd = setenv(`$(pdsurftomo_setup.pdsurftomo_exe) DSurfTomo.in`, env)
    result = run_command(cmd; cwd=run_dir)
    fail_context = isempty(gpu_pin) ? "pDSurfTomo inversion failed" :
        "pDSurfTomo inversion failed (CUDA_VISIBLE_DEVICES=$gpu_pin)"
    assert_ok(result; context=fail_context)
    md"**pDSurfTomo finished.** mode=`$(pdsurf_solver_triplet(ui_pdsurf_solver_mode))`, ThreadNum=$(Int(ui_pdsurf_threads)), stdout length: $(length(result.stdout)) chars"
end

# ╔═╡ dd000001-0000-0000-0000-000000000001
md"""
## Section 5 — Output model & GMT plots
"""

# ╔═╡ a4580fc1-3f2c-4cc5-898d-cab67dabdc71
output_paths = let
    pdsurftomo_result  # wait for inversion
    (;
        model    = joinpath(run_dir, "DSurfTomo.inMeasure.dat"),
        log      = joinpath(run_dir, "DSurfTomo.in.log"),
        res_first = joinpath(run_dir, "residualFirst.dat"),
        res_last  = joinpath(run_dir, "residualLast.dat"),
    )
end

# ╔═╡ 0f71fb37-3127-4d94-b53d-52716c508d01
model, vs_range = let
    path = output_paths.model
    isfile(path) || error("Output model not found: $path")
    data = readdlm(path, comments=true)
    size(data, 2) >= 4 || error("Expected ≥4 columns in model output")
    lon   = Float64.(data[:, 1])
    lat   = Float64.(data[:, 2])
    depth = Float64.(data[:, 3])
    vs    = Float64.(data[:, 4])
    (; lon, lat, depth, vs), extrema(vs)
end

# ╔═╡ 0f71fb37-0000-0000-0000-000000000002
depths = sort(unique(model.depth))

# ╔═╡ 21eeb13c-167e-4719-87b3-acda37ed1a3d
station_lonlat = let
    coords = unique(vcat(
        [(r.lon1, r.lat1) for r in dispersion_data],
        [(r.lon2, r.lat2) for r in dispersion_data],
    ))
    hcat(first.(coords), last.(coords))
end

# ╔═╡ 3c858401-4af6-476f-a9cf-d10a5e5a5fc2
md"""
### GMT plot parameters

Use `surface` to create a continuous GMT grid from the model samples. This is
usually the best option for removing black no-data pixels in the depth maps.
"""

# ╔═╡ d548ea4c-ffb1-42e8-aafa-d543bc9f04fd
@bind _gmt_plot_params PlutoUI.combine() do Child
    default_depths = string.(depths[1:min(4, length(depths))])
    md"""
    | Parameter | Value | Notes |
    |:---|:---|:---|
    | Display depth | $(Child("display_depth", Select(string.(depths); default=string(depths[1])))) | shown interactively below |
    | Depths to save | $(Child("selected_depths", MultiCheckBox(string.(depths); default=default_depths))) | batch slice export |
    | Interpolation | $(Child("interp_method", Select(["surface", "nearneighbor", "xyz2grd + grdfill"]; default="surface"))) | grid method |
    | Surface tension | $(Child("surface_tension", Slider(0.0:0.05:1.0; default=0.35, show_value=true))) | used by `surface` |
    | Search radius (cells) | $(Child("search_radius_cells", NumberField(1.0:0.5:20.0; default=4.0))) | used by `nearneighbor` |
    | Fill remaining holes | $(Child("fill_remaining_holes", CheckBox(default=true))) | apply `grdfill` |
    | Treat bounds as missing | $(Child("treat_bounds_as_missing", CheckBox(default=true))) | hide min/max clamp pixels |
    | Bound tolerance | $(Child("bound_tolerance", NumberField(0.0:0.0001:0.1; default=0.001))) | km/s tolerance |
    | Show stations | $(Child("show_stations", CheckBox(default=true))) | triangle markers |
    | Station size (pt) | $(Child("station_size_pt", NumberField(4:1:18; default=9))) | marker size |
    | Remote relief | $(Child("show_remote_relief", CheckBox(default=false))) | GMT earth relief background |
    | Relief resolution | $(Child("relief_resolution", Select(["01m", "30s", "15s"]; default="30s"))) | remote grid resolution |
    | Vs transparency (%) | $(Child("vs_transparency", NumberField(0:5:80; default=35))) | used with relief overlay |
    """
end

# ╔═╡ 917e2ac0-43b0-43f0-8e5e-ec254fc0355e
begin
    display_depth = _gmt_plot_params.display_depth
    selected_depths = _gmt_plot_params.selected_depths
    selected_depth_values = isempty(selected_depths) ?
        depths[1:min(4,length(depths))] : parse.(Float64, selected_depths)
    gmt_interp_method = _gmt_plot_params.interp_method
    gmt_surface_tension = _gmt_plot_params.surface_tension
    gmt_search_radius_cells = _gmt_plot_params.search_radius_cells
    gmt_fill_remaining_holes = _gmt_plot_params.fill_remaining_holes
    gmt_treat_bounds_as_missing = _gmt_plot_params.treat_bounds_as_missing
    gmt_bound_tolerance = _gmt_plot_params.bound_tolerance
    gmt_show_stations = _gmt_plot_params.show_stations
    gmt_station_size_pt = _gmt_plot_params.station_size_pt
    gmt_show_remote_relief = _gmt_plot_params.show_remote_relief
    gmt_relief_resolution = _gmt_plot_params.relief_resolution
    gmt_vs_transparency = _gmt_plot_params.vs_transparency
end

# ╔═╡ b996f724-cfe6-483e-a2c3-8c488bafef74
display_depth_value = parse(Float64, display_depth)

# ╔═╡ 73772f32-c78b-4b4b-b2a1-7b9587ec0f8a
begin
    function model_slice(m, depth::Real)
        mask = m.depth .== Float64(depth)
        hcat(m.lon[mask], m.lat[mask], m.vs[mask])
    end

    function gmt_slice_grid(m, depth::Real; vsr=vs_range, dv=Float64(ui_dg),
                             method=gmt_interp_method,
                             tension=Float64(gmt_surface_tension),
                             search_radius_cells=Float64(gmt_search_radius_cells),
                             fill_remaining=Bool(gmt_fill_remaining_holes),
                             treat_bounds_as_missing=Bool(gmt_treat_bounds_as_missing),
                             bound_tolerance=Float64(gmt_bound_tolerance),
                             outdir=joinpath(run_dir, "figures"))
        xyz_all = model_slice(m, depth)
        isempty(xyz_all) && error("No samples for depth $depth km")
        mkpath(outdir)
        xmin, xmax = extrema(xyz_all[:, 1])
        ymin, ymax = extrema(xyz_all[:, 2])
        # Snap region outward to exact multiples of dv so GMT is satisfied
        xmin_s = floor(xmin / dv) * dv
        xmax_s = ceil(xmax  / dv) * dv
        ymin_s = floor(ymin / dv) * dv
        ymax_s = ceil(ymax  / dv) * dv
        region = (xmin_s, xmax_s, ymin_s, ymax_s)
        xyz = if treat_bounds_as_missing
            valid = (xyz_all[:, 3] .> vsr[1] + bound_tolerance) .&
                    (xyz_all[:, 3] .< vsr[2] - bound_tolerance)
            count(valid) >= 4 ? xyz_all[valid, :] : xyz_all
        else
            xyz_all
        end
        grid = if method == "surface"
            surface(xyz; R=region, I=(dv, dv), T=tension,
                    Ll=vsr[1], Lu=vsr[2], preproc=true)
        elseif method == "nearneighbor"
            nearneighbor(xyz; R=region, I=(dv, dv),
                         S=search_radius_cells * dv, N=4)
        elseif method == "xyz2grd + grdfill"
            xyz2grd(xyz; R=region, I=(dv, dv))
        else
            error("Unknown GMT interpolation method: $method")
        end
        grid = fill_remaining ? grdfill(grid; A=:n) : grid
        cpt_pad = max(0.001, 0.001 * (vsr[2] - vsr[1]))
        C = makecpt(cmap=:seis,
                    range=(vsr[1] - cpt_pad, vsr[2] + cpt_pad, 0.05),
                    continuous=true)
        outpath = joinpath(outdir, @sprintf("vs_depth_%06.3f_km.png", depth))
        (; grid, C, outpath, depth, method, region)
    end

    function remote_relief(region; res=gmt_relief_resolution)
        gmtread(remotegrid("earth_relief", res=res), region=region)
    end

    function draw_gmt_slice!(s; title, show=true, savefig=s.outpath)
        overlay_stations = Bool(gmt_show_stations) && !isempty(station_lonlat)
        overlay_relief = Bool(gmt_show_remote_relief)
        if overlay_relief
            topo = remote_relief(s.region)
            topo_intensity = grdgradient(topo; A=315, N=:e)
            map_proj = "M15c"
            grdimage(topo; region=s.region, proj=map_proj, frame=:af, cmap=:geo,
                     shade=topo_intensity, title=title,
                     show=false, savefig=nothing)
            grdimage!(s.grid; region=s.region, proj=map_proj,
                      cmap=s.C, colorbar=true,
                      transparency=Int(gmt_vs_transparency),
                      show=!overlay_stations && show,
                      savefig=overlay_stations ? nothing : savefig)
        else
            grdimage(s.grid; proj=:Mercator, frame=:af, cmap=s.C,
                     title=title, colorbar=true,
                     show=!overlay_stations && show,
                     savefig=overlay_stations ? nothing : savefig)
        end
        if overlay_stations
            plot!(station_lonlat; marker=:triangle,
                  markersize="$(Int(gmt_station_size_pt))p",
                  markerfacecolor=:white,
                  markeredgecolor="0.9p,blue",
                  show=show, savefig=savefig)
        end
    end

    function save_gmt_slice!(m, depth::Real)
        s = gmt_slice_grid(m, depth)
        draw_gmt_slice!(s;
            title=@sprintf("Vs at %.3g km (%s)", s.depth, s.method),
            show=false, savefig=s.outpath)
        s.outpath
    end
end

# ╔═╡ ca0b7976-f269-4a5c-8115-69ecbc4ec0d6
let
    s = gmt_slice_grid(model, display_depth_value)
    draw_gmt_slice!(s;
        title=@sprintf("Vs at %.3g km (%s) - run: %s",
                       s.depth, s.method, ui_run_name),
        show=true, savefig=s.outpath)
end

# ╔═╡ b22a1f85-af12-4c06-b3f9-7f4f565a8aa4
[save_gmt_slice!(model, d) for d in selected_depth_values]

# ╔═╡ ee000001-0000-0000-0000-000000000001
md"""
## Section 6 — Residuals
"""

# ╔═╡ 06c62b44-ef4d-4488-b402-f8a8a74552d4
residual_summary = let
    function read_res(p)
        isfile(p) || return Float64[]
        data = readdlm(p, comments=true)
        size(data, 2) >= 3 || error("Expected at least 3 columns in residual file: $p")
        Float64.(data[:, 3] .- data[:, 2])
    end
    r1 = read_res(output_paths.res_first)
    r2 = read_res(output_paths.res_last)
    (;
        first_rms = isempty(r1) ? missing : sqrt(mean(r1 .^ 2)),
        last_rms  = isempty(r2) ? missing : sqrt(mean(r2 .^ 2)),
        n_first   = length(r1),
        n_last    = length(r2),
    )
end

# ╔═╡ 9ec510e5-05af-4de1-bd3e-30c5cd556598
residual_summary

# ╔═╡ f9c00101-52ef-437f-a5ea-1a15f6cc74b0
md"""
### Final-fit dispersion diagnostics

Observed curves are solid, final DSurfTomo estimates are dashed, and open
circles mark final-fit rows that DSurfTomo assigned zero weight.
"""

# ╔═╡ f9c00102-52ef-437f-a5ea-1a15f6cc74b0
begin
    function read_dsurftomo_surfdata_rows(path::AbstractString, periods)
        isfile(path) || error("Surfdata file not found: $path")
        isempty(periods) && error("No DSurfTomo periods available for diagnostics.")

        rows = NamedTuple[]
        lat1 = lon1 = NaN
        period_index = 0
        for (line_number, line) in enumerate(eachline(path))
            parts = split(strip(line))
            isempty(parts) && continue
            if parts[1] == "#"
                length(parts) >= 6 ||
                    error("Malformed surfdata header at $path:$line_number")
                lat1 = parse(Float64, parts[2])
                lon1 = parse(Float64, parts[3])
                period_index = parse(Int, parts[4])
                1 <= period_index <= length(periods) ||
                    error("Surfdata period index $period_index is outside 1:$(length(periods)) at $path:$line_number")
            else
                length(parts) >= 3 ||
                    error("Malformed surfdata measurement at $path:$line_number")
                isfinite(lat1) && isfinite(lon1) && period_index > 0 ||
                    error("Surfdata measurement before a valid header at $path:$line_number")
                lat2 = parse(Float64, parts[1])
                lon2 = parse(Float64, parts[2])
                input_velocity = parse(Float64, parts[3])
                period = Float64(periods[period_index])
                path_key = @sprintf("%.6f/%.6f -> %.6f/%.6f",
                                    lat1, lon1, lat2, lon2)
                push!(rows, (; path_key, period_index, period,
                              lat1, lon1, lat2, lon2, input_velocity))
            end
        end
        isempty(rows) && error("No measurements found in surfdata file: $path")
        rows
    end

    function final_fit_dispersion_rows(surfdata_path::AbstractString,
                                       residual_path::AbstractString,
                                       periods)
        surf_rows = read_dsurftomo_surfdata_rows(surfdata_path, periods)
        isfile(residual_path) || error("Final residual file not found: $residual_path")
        residuals = readdlm(residual_path, comments=true)
        size(residuals, 2) >= 6 ||
            error("Expected at least 6 columns in final residual file: $residual_path")
        size(residuals, 1) == length(surf_rows) ||
            error("Diagnostic row mismatch: surfdata has $(length(surf_rows)) measurements but $(basename(residual_path)) has $(size(residuals, 1)) rows.")

        rows = NamedTuple[]
        for (i, surf_row) in enumerate(surf_rows)
            interstation_distance_km = Float64(residuals[i, 1])
            estimated_time = Float64(residuals[i, 2])
            observed_time = Float64(residuals[i, 3])
            weight = Float64(residuals[i, 6])
            values = (surf_row.period, interstation_distance_km,
                      estimated_time, observed_time, weight)
            all(isfinite, values) ||
                error("Non-finite final-fit diagnostic values at measurement row $i.")
            interstation_distance_km > 0.0 ||
                error("Non-positive interstation distance at measurement row $i.")
            estimated_time > 0.0 && observed_time > 0.0 ||
                error("Non-positive travel time at measurement row $i.")

            observed_velocity = interstation_distance_km / observed_time
            estimated_velocity = interstation_distance_km / estimated_time
            residual_velocity = observed_velocity - estimated_velocity
            push!(rows, merge(surf_row,
                (; interstation_distance_km, observed_time, estimated_time,
                   observed_velocity, estimated_velocity, residual_velocity,
                   weight)))
        end
        rows
    end

    function grouped_fit_diagnostic_rows(rows)
        by_path = Dict{String,Vector{NamedTuple}}()
        for row in rows
            push!(get!(by_path, row.path_key, NamedTuple[]), row)
        end
        groups = collect(values(by_path))
        foreach(group -> sort!(group, by=row -> row.period), groups)
        sort!(groups, by=group -> mean(row.interstation_distance_km for row in group))
        groups
    end

    function fit_diagnostic_segments(groups, field::Symbol)
        segments = [hcat([row.period for row in group],
                         [getfield(row, field) for row in group])
                    for group in groups]
        mat2ds(segments)
    end

    function fit_diagnostic_cpt(rows)
        dmin, dmax = extrema(row.interstation_distance_km for row in rows)
        pad = dmin == dmax ? max(1.0, 0.05 * abs(dmin)) : 0.0
        lo = dmin - pad
        hi = dmax + pad
        inc = max((hi - lo) / 20.0, 0.1)
        makecpt(cmap=:turbo, range=(lo, hi, inc), continuous=true)
    end

    function fit_diagnostic_region(rows, fields; include_zero=false)
        periods = [row.period for row in rows]
        ys = Float64[]
        for field in fields, row in rows
            push!(ys, getfield(row, field))
        end
        include_zero && push!(ys, 0.0)
        xmin, xmax = extrema(periods)
        ymin, ymax = extrema(ys)
        xpad = xmax == xmin ? max(0.1, 0.05 * abs(xmin)) : 0.03 * (xmax - xmin)
        ypad = ymax == ymin ? max(0.05, 0.05 * abs(ymin)) : 0.08 * (ymax - ymin)
        (xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad)
    end
end

# ╔═╡ f9c00103-52ef-437f-a5ea-1a15f6cc74b0
dispersion_fit_diagnostics = final_fit_dispersion_rows(
    joinpath(run_dir, "surfdata.dat"),
    output_paths.res_last,
    generated_cfg.tRc)

# ╔═╡ f9c00104-52ef-437f-a5ea-1a15f6cc74b0
dispersion_fit_diagnostics_validation = let
    rows = dispersion_fit_diagnostics
    residual_definition_ok = all(isapprox(row.residual_velocity,
        row.observed_velocity - row.estimated_velocity;
        rtol=1e-12, atol=1e-12) for row in rows)
    (;
        n_rows = length(rows),
        n_paths = length(unique(row.path_key for row in rows)),
        n_zero_weight = count(row.weight <= 0.0 for row in rows),
        period_range_s = extrema(row.period for row in rows),
        distance_range_km = extrema(row.interstation_distance_km for row in rows),
        positive_finite_velocities = all(row.observed_velocity > 0.0 &&
            row.estimated_velocity > 0.0 &&
            isfinite(row.observed_velocity) &&
            isfinite(row.estimated_velocity) for row in rows),
        residual_definition_ok,
    )
end

# ╔═╡ f9c00105-52ef-437f-a5ea-1a15f6cc74b0
let
    rows = dispersion_fit_diagnostics
    groups = grouped_fit_diagnostic_rows(rows)
    distances = [mean(row.interstation_distance_km for row in group)
                 for group in groups]
    C = fit_diagnostic_cpt(rows)
    region = fit_diagnostic_region(rows, (:observed_velocity, :estimated_velocity))
    observed = fit_diagnostic_segments(groups, :observed_velocity)
    estimated = fit_diagnostic_segments(groups, :estimated_velocity)

    distance_lines = (data=distances, nofill=true)
    plot(observed; region, proj=:linear, frame=:af, cmap=C, Z=distance_lines,
         pen=(pen="0.7p", zlevels=true),
         xlabel="Period (s)", ylabel="Group velocity (km/s)",
         title="Observed (solid) and estimated (dashed) dispersions",
         show=false)
    plot!(estimated; cmap=C, Z=distance_lines, pen=(pen="0.7p", zlevels=true),
          linestyle=:Dash, show=false)

    zero_weight = [row for row in rows if row.weight <= 0.0]
    if !isempty(zero_weight)
        plot!(hcat([row.period for row in zero_weight],
                   [row.observed_velocity for row in zero_weight]);
              marker=:circle, markersize="2.5p", markerfacecolor=:white,
              markeredgecolor="0.35p,gray30", show=false)
        plot!(hcat([row.period for row in zero_weight],
                   [row.estimated_velocity for row in zero_weight]);
              marker=:circle, markersize="2.5p", markerfacecolor=:white,
              markeredgecolor="0.35p,gray30", show=false)
    end
    colorbar!(C=C, B="xaf+l\"Interstation distance (km)\"")
    showfig()
end

# ╔═╡ f9c00106-52ef-437f-a5ea-1a15f6cc74b0
let
    rows = dispersion_fit_diagnostics
    groups = grouped_fit_diagnostic_rows(rows)
    distances = [mean(row.interstation_distance_km for row in group)
                 for group in groups]
    C = fit_diagnostic_cpt(rows)
    region = fit_diagnostic_region(rows, (:residual_velocity,); include_zero=true)
    residuals = fit_diagnostic_segments(groups, :residual_velocity)

    distance_lines = (data=distances, nofill=true)
    plot(residuals; region, proj=:linear, frame=:af, cmap=C, Z=distance_lines,
         pen=(pen="0.7p", zlevels=true),
         xlabel="Period (s)", ylabel="Residual group velocity (km/s)",
         title="Final dispersion residuals: observed - estimated",
         show=false)
    plot!([region[1] 0.0; region[2] 0.0];
          pen="0.8p,gray,-", show=false)

    zero_weight = [row for row in rows if row.weight <= 0.0]
    if !isempty(zero_weight)
        plot!(hcat([row.period for row in zero_weight],
                   [row.residual_velocity for row in zero_weight]);
              marker=:circle, markersize="2.5p", markerfacecolor=:white,
              markeredgecolor="0.35p,gray30", show=false)
    end
    colorbar!(C=C, B="xaf+l\"Interstation distance (km)\"")
    showfig()
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
GMT = "5752ebe1-31b9-557e-87aa-f909b540aa54"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
GMT = "~1.41.0"
PlutoUI = "~0.7.80"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "4be5e62f4dd159d913bffadbf99e8216bc8f2edd"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arrow_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Lz4_jll", "Thrift_jll", "Zlib_jll", "Zstd_jll", "boost_jll", "brotli_jll", "snappy_jll"]
git-tree-sha1 = "55ecf3d16295c26e96d2f0b65386d1a8414e2283"
uuid = "8ce61222-c28f-5041-a97a-c2198fb817bf"
version = "19.0.1+0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Blosc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Lz4_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "535c80f1c0847a4c967ea945fca21becc9de1522"
uuid = "0b7ba130-8d10-5ba8-a3d6-c5182647fed9"
version = "1.21.7+0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8f05e9a2e7c2e3eb524102bb2926c5743c07fbe1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.0+0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6866aec60ef98e3164cd8d6855225684207e9dff"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.12+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.GDAL_jll]]
deps = ["Arrow_jll", "Artifacts", "Blosc_jll", "Expat_jll", "GEOS_jll", "HDF4_jll", "HDF5_jll", "JLLWrappers", "LERC_jll", "LibCURL_jll", "LibPQ_jll", "Libdl", "Libtiff_jll", "Lz4_jll", "NetCDF_jll", "OpenJpeg_jll", "PCRE2_jll", "PROJ_jll", "Qhull_jll", "SQLite_jll", "XML2_jll", "XZ_jll", "Zlib_jll", "Zstd_jll", "libgeotiff_jll", "libpng_jll", "libwebp_jll", "muparser_jll"]
git-tree-sha1 = "0e4385131431afe4cadb02f2e8b70156c23ac8f0"
uuid = "a7073274-a066-55f0-b90d-d619367d196c"
version = "303.1100.500+0"

[[deps.GEOS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fdaf62d2354bb398652ee612d487eb19d74468a6"
uuid = "d604d12d-fa86-5845-992e-78dc15976526"
version = "3.14.1+0"

[[deps.GMT]]
deps = ["Dates", "Downloads", "GDAL_jll", "GMT_jll", "Ghostscript_jll", "InteractiveUtils", "LASzip_jll", "Leptonica_jll", "LinearAlgebra", "PROJ_jll", "PrecompileTools", "Printf", "SparseArrays", "Statistics", "Tables"]
git-tree-sha1 = "b286f2f18b8ac6a50704032472ffe37d8b129af8"
uuid = "5752ebe1-31b9-557e-87aa-f909b540aa54"
version = "1.41.0"

    [deps.GMT.extensions]
    GMTDGTLidarExt = "HTTP"
    GMTDataFramesExt = "DataFrames"
    GMTExcelExt = "XLSX"
    GMTParkerFFTExt = "FFTW"

    [deps.GMT.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
    XLSX = "fdbf4ff8-1666-58a4-91e7-1b58723a45e0"

[[deps.GMT_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "FFTW_jll", "GDAL_jll", "Ghostscript_jll", "Glib_jll", "JLLWrappers", "LAPACK32_jll", "LLVMOpenMP_jll", "LibCURL_jll", "Libdl", "NetCDF_jll", "OpenBLAS32_jll", "PCRE_jll", "PROJ_jll"]
git-tree-sha1 = "a63357b5b46c5fd6f48d343b95b245cee4eb2317"
uuid = "b68b8c3f-ed99-5bef-9675-4739d9426b26"
version = "6.6.0+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.HDF4_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "ea9eff9cfef5f45b771096e5c2de3de0eab937c3"
uuid = "818ab7a1-5177-5f44-ba99-6e845030c6cb"
version = "4.3.2+0"

[[deps.HDF5_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "OpenSSL_jll", "TOML", "Zlib_jll", "libaec_jll"]
git-tree-sha1 = "e94f84da9af7ce9c6be049e9067e511e17ff89ec"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.14.6+0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XML2_jll", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "baaaebd42ed9ee1bd9173cfd56910e55a8622ee1"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.13.0+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.ICU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b3d8be712fbf9237935bde0ce9b5a736ae38fc34"
uuid = "a51ab1cf-af8e-5615-a023-bc2c838bba6b"
version = "76.2.0+0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.Kerberos_krb5_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0f2899fdadaab4b8f57db558ba21bdb4fb52f1f0"
uuid = "b39eb1a6-c29a-53d7-8c32-632cd16f18da"
version = "1.21.3+0"

[[deps.LAPACK32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "ff8dd29d35e5cdb26128a590487cad31b829cae3"
uuid = "17f450c3-bd24-55df-bb84-8c51b4b939e3"
version = "3.12.1+1"

[[deps.LASzip_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be79377cdff896d9e19f5c23795b05b056e8d7cd"
uuid = "8372b9c3-1e34-5cc3-bfab-1a98e101de11"
version = "3.4.4001+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.Leptonica_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "libpng_jll", "libwebp_jll"]
git-tree-sha1 = "bb37df9f514b5b0b8114d0f42c1ad9792dd29157"
uuid = "6a1430e4-294a-53a5-a485-ec66ef6b843c"
version = "1.87.0+0"

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

[[deps.LibPQ_jll]]
deps = ["Artifacts", "ICU_jll", "JLLWrappers", "Kerberos_krb5_jll", "Libdl", "OpenSSL_jll", "Zstd_jll"]
git-tree-sha1 = "9a92c141ca0c7df669b6a4dd2ef776d2bf1d61cb"
uuid = "08be9ffa-1c94-5ee5-a977-46a84ec9b350"
version = "16.13.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "70bd263e082a236c8c2661a474616d95ba59d2cf"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.19.0+0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "191686b1ac1ea9c89fc52e996ad15d1d241d1e33"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.10.1+0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "9341048b9f723f2ae2a72a5269ac2f15f80534dc"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.3.2+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "8e98d5d80b87403c311fd51e8455d4546ba7a5f8"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.12"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "675df097f8eeb28998b2cfe3b25655af73d5f7df"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.6+0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bc95bf4149bf535c09602e3acdf950d9b4376227"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+3"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NetCDF_jll]]
deps = ["Artifacts", "Blosc_jll", "Bzip2_jll", "HDF5_jll", "JLLWrappers", "LazyArtifacts", "LibCURL_jll", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "TOML", "XML2_jll", "Zlib_jll", "Zstd_jll", "libaec_jll", "libzip_jll"]
git-tree-sha1 = "d574803b6055116af212434460adf654ce98e345"
uuid = "7243133f-43d8-5620-bbf4-c2c921802cf3"
version = "401.900.300+0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "libblastrampoline_jll"]
git-tree-sha1 = "565175ce692c065e50ad32efbb61ba69b1586593"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.33+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "215a6666fee6d6b3a6e75f2cc22cb767e2dd393a"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.5+0"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "6d6c0ca4824268c1a7dca1f4721c535ac63d9074"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.11+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ccf0e9339e1f3e66e241ce01bbcbf57a0a9c15a1"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.45.0+0"

[[deps.PROJ_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "Libtiff_jll", "SQLite_jll"]
git-tree-sha1 = "fbfbc14815c5f5375abc971321aa5f468e715a38"
uuid = "58948b4f-47e0-5654-a9ad-f609743f8632"
version = "902.800.100+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "fbc875044d82c113a9dee6fc14e16cf01fd48872"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.80"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c69da20496799bbdd56c15ecf5d80a5e6cbcc904"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "10008.0.1004+0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SQLite_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll", "dlfcn_win32_jll"]
git-tree-sha1 = "0b5f220f90642566b65ba86549d1ee4118ab2579"
uuid = "76ed43ae-9a5d-5a62-8c75-30186b810ce8"
version = "3.51.2+0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

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

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Thrift_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "boost_jll"]
git-tree-sha1 = "4d16a4b4eab80099c19342b10d0bdb252c39bea6"
uuid = "e0b8ae26-5307-5830-91fd-398402328850"
version = "0.21.1+0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "80d3930c6347cfce7ccf96bd3bafdf079d9c0390"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.9+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.boost_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "25fb6ecbb784a45f8ea74584fa631a9e85393dd0"
uuid = "28df3c45-c428-5900-9ff8-a3135698ca75"
version = "1.87.0+0"

[[deps.brotli_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "46fda47f4215c957bc92fd5fbb5ad04fee1e3743"
uuid = "4611771a-a7d2-5e23-8d00-b1becdba1aae"
version = "1.2.0+0"

[[deps.dlfcn_win32_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e141d67ffe550eadfb5af1bdbdaf138031e4805f"
uuid = "c4b69c83-5512-53e3-94e6-de98773c479f"
version = "1.4.2+0"

[[deps.libaec_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1411bc34c180946d3cef591de1384012afa6edee"
uuid = "477f73a3-ac25-53e9-8cc3-50b2fa2566f0"
version = "1.1.6+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libgeotiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LibCURL_jll", "Libdl", "Libtiff_jll", "PROJ_jll", "Zlib_jll"]
git-tree-sha1 = "cbdbc9ae1127f81cb653a4f7545d89f8db2a17a7"
uuid = "06c338fa-64ff-565b-ac2f-249532af990e"
version = "100.702.400+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

[[deps.muparser_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "70ee0f42a44ef6e16298e5bfc8b6e311d08e49bb"
uuid = "888e69b1-873b-5047-a2fc-24c07cbe9dc8"
version = "2.3.5+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.snappy_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ca88363dd41d2547f52118287dd34dbbc14f3eb7"
uuid = "fe1e1685-f7be-5f59-ac9f-4ca204017dfd"
version = "1.2.3+0"
"""

# ╔═╡ Cell order:
# ╟─aa000001-0000-0000-0000-000000000001
# ╠═4ef5cf9c-0583-45d4-8f4e-87c79c56bc8a
# ╠═f62bc402-2a89-498b-8bed-020f40a447c5
# ╟─aa000010-0000-0000-0000-000000000010
# ╟─aa000011-0000-0000-0000-000000000011
# ╠═aa000012-0000-0000-0000-000000000012
# ╠═aa000013-0000-0000-0000-000000000013
# ╠═aa000014-0000-0000-0000-000000000014
# ╟─aa000015-0000-0000-0000-000000000015
# ╟─aa000020-0000-0000-0000-000000000020
# ╟─aa000021-0000-0000-0000-000000000021
# ╠═aa000022-0000-0000-0000-000000000022
# ╠═aa000023-0000-0000-0000-000000000023
# ╠═aa000023-0000-0000-0000-000000000060
# ╟─aa000024-0000-0000-0000-000000000024
# ╠═aa000025-0000-0000-0000-000000000025
# ╠═aa000026-0000-0000-0000-000000000026
# ╠═aa000028-0000-0000-0000-000000000028
# ╟─aa000029-0000-0000-0000-000000000029
# ╠═aa000029-0000-0000-0000-000000000050
# ╠═aa000029-0000-0000-0000-00000000002b
# ╠═aa000029-0000-0000-0000-000000000061
# ╟─aa000032-0000-0000-0000-000000000032
# ╠═aa000033-0000-0000-0000-000000000033
# ╟─aa000034-0000-0000-0000-000000000034
# ╠═aa000035-0000-0000-0000-000000000035
# ╠═aa000036-0000-0000-0000-000000000036
# ╠═aa000039-0000-0000-0000-000000000039
# ╟─bb000001-0000-0000-0000-000000000001
# ╠═bb000002-0000-0000-0000-000000000002
# ╠═bb000003-0000-0000-0000-000000000003
# ╠═bb000003-0000-0000-0000-00000000000a
# ╠═bb000004-0000-0000-0000-000000000004
# ╠═bb000005-0000-0000-0000-000000000005
# ╠═bb000006-0000-0000-0000-000000000006
# ╠═bb000006-0000-0000-0000-000000000016
# ╠═bb000007-0000-0000-0000-000000000007
# ╟─cc000001-0000-0000-0000-000000000001
# ╟─3f0a828f-df2c-4457-a42a-c9bfe1573767
# ╠═720c8fc5-a04e-48f3-a8cf-667c99afff4f
# ╠═4b48e706-eef4-448a-91ab-5a82a6a9676c
# ╠═e1dd47d1-11dc-4a1d-9e52-938edc24a4aa
# ╠═ac7c9f25-3671-46cd-977d-49d6999d6d75
# ╠═5b614467-5709-4d51-8eef-6c4599fd3125
# ╟─dd000001-0000-0000-0000-000000000001
# ╠═a4580fc1-3f2c-4cc5-898d-cab67dabdc71
# ╠═0f71fb37-3127-4d94-b53d-52716c508d01
# ╠═0f71fb37-0000-0000-0000-000000000002
# ╠═21eeb13c-167e-4719-87b3-acda37ed1a3d
# ╟─3c858401-4af6-476f-a9cf-d10a5e5a5fc2
# ╟─d548ea4c-ffb1-42e8-aafa-d543bc9f04fd
# ╠═ca0b7976-f269-4a5c-8115-69ecbc4ec0d6
# ╠═917e2ac0-43b0-43f0-8e5e-ec254fc0355e
# ╠═b996f724-cfe6-483e-a2c3-8c488bafef74
# ╠═73772f32-c78b-4b4b-b2a1-7b9587ec0f8a
# ╠═b22a1f85-af12-4c06-b3f9-7f4f565a8aa4
# ╟─ee000001-0000-0000-0000-000000000001
# ╠═06c62b44-ef4d-4488-b402-f8a8a74552d4
# ╠═9ec510e5-05af-4de1-bd3e-30c5cd556598
# ╟─f9c00101-52ef-437f-a5ea-1a15f6cc74b0
# ╠═f9c00102-52ef-437f-a5ea-1a15f6cc74b0
# ╠═f9c00103-52ef-437f-a5ea-1a15f6cc74b0
# ╠═f9c00104-52ef-437f-a5ea-1a15f6cc74b0
# ╠═f9c00105-52ef-437f-a5ea-1a15f6cc74b0
# ╠═f9c00106-52ef-437f-a5ea-1a15f6cc74b0
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
