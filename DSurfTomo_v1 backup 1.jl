### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto,
# this mock version gives bound variables a default value.
macro bind(def, element)
    return quote
        local iv = try
            Base.loaded_modules[
                Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")
            ].Bonds.initial_value
        catch
            _ -> missing
        end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 0a2a8a75-67dd-4e80-a9cf-55fb3adf0f01
begin
    using Pkg

    # First run only: uncomment this block if GMT.jl or PlutoUI.jl is not installed.
    # Pkg.add(["GMT", "PlutoUI"])
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

# ╔═╡ a2d6370a-9578-4b70-a0f8-d5f6f3e08d38
md"""
# DSurfTomo test notebook

This notebook clones and builds `HongjianFang/DSurfTomo` from the `stable`
branch, runs the bundled Taipei Basin demo, and plots output tomograms with
GMT.jl. It calls the compiled Fortran executable directly from Julia; PythonCall
is not required.

For a new run, change `run_name`, `input_file`, and `initial_model_file` below.
The DSurfTomo executable expects `MOD` in the working directory and writes files
such as `DSurfTomo.inMeasure.dat`, `DSurfTomo.in.log`, `residualFirst.dat`, and
`residualLast.dat`.
"""

# ╔═╡ f62bc402-2a89-498b-8bed-020f40a447c5
begin
    repo_url = "https://github.com/HongjianFang/DSurfTomo.git"
    repo_branch = "stable"

    project_root = @__DIR__
    dsurftomo_root = joinpath(project_root, "External", "DSurfTomo")
    runs_root = joinpath(project_root, "DSurfTomo_runs")

    demo_source_dir = joinpath(dsurftomo_root, "example_smoothing_clean")
    demo_run_dir = joinpath(runs_root, "demo_taipei")
end

# ╔═╡ d1cfa5ad-7f79-4e53-ad77-279ce4147372
md"""
## Run settings
"""

# ╔═╡ 89bfc697-86e3-4d45-9ddd-07c44d609816
begin
    run_name = "demo_taipei"
    run_dir = joinpath(runs_root, run_name)

    input_file = joinpath(run_dir, "DSurfTomo.in")
    initial_model_file = joinpath(run_dir, "MOD")
    true_model_file = joinpath(run_dir, "MOD.true")

    # Keep this true for the first run. Later set it false if you want to preserve
    # edited files in run_dir.
    refresh_demo_files = true
end

# ╔═╡ 3f0a828f-df2c-4457-a42a-c9bfe1573767
md"""
## Julia process helpers
"""

# ╔═╡ 720c8fc5-a04e-48f3-a8cf-667c99afff4f
begin
    function run_command(cmd::Cmd; cwd::AbstractString=pwd())
        stdout_path, stdout_io = mktemp()
        stderr_path, stderr_io = mktemp()
        close(stdout_io)
        close(stderr_io)

        proc = run(
            pipeline(Cmd(cmd; dir=cwd, ignorestatus=true);
                     stdout=stdout_path, stderr=stderr_path)
        )
        stdout = read(stdout_path, String)
        stderr = read(stderr_path, String)
        rm(stdout_path; force=true)
        rm(stderr_path; force=true)

        return (; rc=proc.exitcode, stdout, stderr,
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
    function ensure_repo!(repo_dir::AbstractString; url=repo_url, branch=repo_branch)
        if isdir(joinpath(repo_dir, ".git"))
            result = run_command(`git fetch --depth=1 origin $branch`; cwd=repo_dir)
            assert_ok(result; context="Could not update DSurfTomo repository")
            result = run_command(`git checkout $branch`; cwd=repo_dir)
            assert_ok(result; context="Could not checkout DSurfTomo branch")
        else
            mkpath(dirname(repo_dir))
            result = run_command(`git clone --depth=1 --branch $branch $url $repo_dir`)
            assert_ok(result; context="Could not clone DSurfTomo")
        end
        repo_dir
    end

    function build_dsurftomo!(repo_dir::AbstractString)
        exe = joinpath(repo_dir, "bin", "DSurfTomo")
        isfile(exe) && return exe

        src_exe = joinpath(repo_dir, "src", "DSurfTomo")
        result = run_command(`make`; cwd=joinpath(repo_dir, "src"))
        assert_ok(result; context="Could not compile DSurfTomo. Check that gfortran and make are available.")
        isfile(src_exe) || error("Build finished, but source executable was not found: $src_exe")

        mkpath(joinpath(repo_dir, "bin"))
        cp(src_exe, exe; force=true)
        isfile(exe) || error("Build finished, but executable was not found: $exe")
        exe
    end

    function copy_demo_run!(source_dir::AbstractString, target_dir::AbstractString; refresh=false)
        mkpath(dirname(target_dir))
        if refresh && isdir(target_dir)
            rm(target_dir; recursive=true, force=true)
        end
        if !isdir(target_dir)
            cp(source_dir, target_dir)
        end
        target_dir
    end
end

# ╔═╡ e1dd47d1-11dc-4a1d-9e52-938edc24a4aa
md"""
## Prepare DSurfTomo
"""

# ╔═╡ 1fe9ee2e-8d92-4ab3-bae1-e47fe195f2ee
begin
    ensure_repo!(dsurftomo_root)
    dsurftomo_exe = build_dsurftomo!(dsurftomo_root)
    copy_demo_run!(demo_source_dir, demo_run_dir; refresh=refresh_demo_files)

    (; dsurftomo_exe, run_dir)
end

# ╔═╡ 35d6a3d4-2ac7-4a77-aa77-4837ac21d165
md"""
## Inspect input
"""

# ╔═╡ d96eb30a-04da-48bc-ad04-196d89fa101c
begin
    function strip_comment(line::AbstractString)
        split(line, "c:"; limit=2)[1] |> strip
    end

    function read_dsurftomo_input(path::AbstractString)
        lines = readlines(path)
        payload = strip_comment.(lines[4:end])
        idx = 1
        datafile = payload[idx]; idx += 1
        nx, ny, nz = parse.(Int, split(payload[idx])); idx += 1
        goxd, gozd = parse.(Float64, split(payload[idx])); idx += 1
        dvxd, dvzd = parse.(Float64, split(payload[idx])); idx += 1
        nsrc = parse(Int, payload[idx]); idx += 1
        weight, damp = parse.(Float64, split(payload[idx])); idx += 1
        sublayers = parse(Int, payload[idx]); idx += 1
        minvel, maxvel = parse.(Float64, split(payload[idx])); idx += 1
        maxiter = parse(Int, payload[idx]); idx += 1
        sparsityfraction = parse(Float64, payload[idx]); idx += 1
        kmaxRc = parse(Int, payload[idx]); idx += 1
        tRc = kmaxRc > 0 ? parse.(Float64, split(payload[idx])) : Float64[]
        idx += kmaxRc > 0 ? 1 : 0
        kmaxRg = parse(Int, payload[idx]); idx += 1
        tRg = kmaxRg > 0 ? parse.(Float64, split(payload[idx])) : Float64[]
        idx += kmaxRg > 0 ? 1 : 0
        kmaxLc = parse(Int, payload[idx]); idx += 1
        tLc = kmaxLc > 0 ? parse.(Float64, split(payload[idx])) : Float64[]
        idx += kmaxLc > 0 ? 1 : 0
        kmaxLg = parse(Int, payload[idx]); idx += 1
        tLg = kmaxLg > 0 ? parse.(Float64, split(payload[idx])) : Float64[]
        idx += kmaxLg > 0 ? 1 : 0
        synthetic = parse(Int, payload[idx]); idx += 1
        noiselevel = parse(Float64, payload[idx]); idx += 1
        threshold = parse(Float64, payload[idx])
        return (; datafile, nx, ny, nz, goxd, gozd, dvxd, dvzd, nsrc,
                weight, damp, sublayers, minvel, maxvel, maxiter,
                sparsityfraction, kmaxRc, tRc, kmaxRg, tRg, kmaxLc, tLc,
                kmaxLg, tLg, synthetic, noiselevel, threshold)
    end

    cfg = read_dsurftomo_input(input_file)
end

# ╔═╡ 0157f8d7-2f99-42a6-b377-d3354f61a732
cfg

# ╔═╡ 58f160b8-23bd-4efa-9f95-16c558588e8c
md"""
## Run inversion
"""

# ╔═╡ ac7c9f25-3671-46cd-977d-49d6999d6d75
@bind run_button CounterButton("Run DSurfTomo")

# ╔═╡ 5b614467-5709-4d51-8eef-6c4599fd3125
begin
    function run_dsurftomo!(exe::AbstractString, input_path::AbstractString)
        workdir = dirname(input_path)
        input_name = basename(input_path)
        isfile(joinpath(workdir, "MOD")) ||
            error("DSurfTomo requires an initial model named MOD in $workdir")

        result = run_command(`$exe $input_name`; cwd=workdir)
        assert_ok(result; context="DSurfTomo inversion failed")
    end

    run_button
    dsurftomo_result = run_dsurftomo!(dsurftomo_exe, input_file)
end

# ╔═╡ a4580fc1-3f2c-4cc5-898d-cab67dabdc71
begin
    output_model = joinpath(run_dir, basename(input_file) * "Measure.dat")
    output_log = joinpath(run_dir, basename(input_file) * ".log")
    residual_first = joinpath(run_dir, "residualFirst.dat")
    residual_last = joinpath(run_dir, "residualLast.dat")

    (; output_model, output_log, residual_first, residual_last)
end

# ╔═╡ 7d02843b-019a-49f6-aaf3-44e47e27ff1c
md"""
## Read output model
"""

# ╔═╡ 0f71fb37-3127-4d94-b53d-52716c508d01
begin
    function read_xyzv(path::AbstractString)
        isfile(path) || error("Missing model file: $path")
        data = readdlm(path, comments=true)
        size(data, 2) >= 4 || error("Expected at least 4 columns in $path")
        lon = Float64.(data[:, 1])
        lat = Float64.(data[:, 2])
        depth = Float64.(data[:, 3])
        vs = Float64.(data[:, 4])
        (; lon, lat, depth, vs, matrix=hcat(lon, lat, depth, vs))
    end

    model = read_xyzv(output_model)
    depths = sort(unique(model.depth))
    lon_range = extrema(model.lon)
    lat_range = extrema(model.lat)
    vs_range = extrema(model.vs)
end

# ╔═╡ d548ea4c-ffb1-42e8-aafa-d543bc9f04fd
begin
    max_depth_buttons = min(4, length(depths))
    default_depths = depths[1:max_depth_buttons]
    @bind selected_depths confirm(MultiCheckBox(string.(depths); default=string.(default_depths)))
end

# ╔═╡ 917e2ac0-43b0-43f0-8e5e-ec254fc0355e
begin
    selected_depth_values = isempty(selected_depths) ? default_depths : parse.(Float64, selected_depths)
end

# ╔═╡ 02790c06-6f78-441b-9c42-d9576755f70c8
md"""
## GMT plots
"""

# ╔═╡ 73772f32-c78b-4b4b-b2a1-7b9587ec0f8a
begin
    function model_slice(model, depth::Real)
        mask = model.depth .== Float64(depth)
        hcat(model.lon[mask], model.lat[mask], model.vs[mask])
    end

    function gmt_region(x::AbstractVector, y::AbstractVector)
        xmin, xmax = extrema(x)
        ymin, ymax = extrema(y)
        return (xmin, xmax, ymin, ymax)
    end

    function gmt_slice_plot(model, depth::Real; cfg=cfg, cmap=:seis)
        xyz = model_slice(model, depth)
        isempty(xyz) && error("No samples for depth $depth")
        region = gmt_region(xyz[:, 1], xyz[:, 2])
        inc = (cfg.dvzd, cfg.dvxd)
        grid = xyz2grd(xyz; R=region, I=inc)
        C = makecpt(cmap=cmap, range=(vs_range[1], vs_range[2], 0.05), continuous=true)
        grdimage(grid; proj=:Mercator, frame=:af, cmap=C,
                 title=@sprintf("Vs at %.3g km", depth), colorbar=true)
    end
end

# ╔═╡ ca0b7976-f269-4a5c-8115-69ecbc4ec0d6
begin
    slice_figures = [gmt_slice_plot(model, d) for d in selected_depth_values]
end

# ╔═╡ 42627444-c2be-42f5-b0f6-01a47325d9fd
md"""
## Residuals
"""

# ╔═╡ 06c62b44-ef4d-4488-b402-f8a8a74552d4
begin
    function read_residual(path::AbstractString)
        isfile(path) || return Float64[]
        data = readdlm(path, comments=true)
        vec(Float64.(data[:, end]))
    end

    first_residual = read_residual(residual_first)
    last_residual = read_residual(residual_last)

    residual_summary = (;
        first_rms = isempty(first_residual) ? missing : sqrt(mean(first_residual .^ 2)),
        last_rms = isempty(last_residual) ? missing : sqrt(mean(last_residual .^ 2)),
        n_first = length(first_residual),
        n_last = length(last_residual),
    )
end

# ╔═╡ 9ec510e5-05af-4de1-bd3e-30c5cd556598
residual_summary

# ╔═╡ 69f5580b-a14d-4793-9c69-9289c58736c4
md"""
## Changing to your own data

1. Put your DSurfTomo input file, dispersion data file, and initial model in a
   new folder under `DSurfTomo_runs`.
2. Make sure the initial model is named `MOD`, or change `initial_model_file`
   and copy/link it to `MOD` before running.
3. Set `run_name` and `input_file` in the run settings cell, then press
   `Run DSurfTomo`.

DSurfTomo itself consumes the already-reformatted `surfdata.dat` format
referenced from the input file. If your data are raw station-pair dispersion
files like the upstream `TaipeiRawData/CDisp*.dat` files, convert them to that
format first, then point `DSurfTomo.in` at the converted file.
"""

# ╔═╡ Cell order:
# ╠═0a2a8a75-67dd-4e80-a9cf-55fb3adf0f01
# ╠═4ef5cf9c-0583-45d4-8f4e-87c79c56bc8a
# ╟─a2d6370a-9578-4b70-a0f8-d5f6f3e08d38
# ╠═f62bc402-2a89-498b-8bed-020f40a447c5
# ╟─d1cfa5ad-7f79-4e53-ad77-279ce4147372
# ╠═89bfc697-86e3-4d45-9ddd-07c44d609816
# ╟─3f0a828f-df2c-4457-a42a-c9bfe1573767
# ╠═720c8fc5-a04e-48f3-a8cf-667c99afff4f
# ╠═4b48e706-eef4-448a-91ab-5a82a6a9676c
# ╟─e1dd47d1-11dc-4a1d-9e52-938edc24a4aa
# ╠═1fe9ee2e-8d92-4ab3-bae1-e47fe195f2ee
# ╟─35d6a3d4-2ac7-4a77-aa77-4837ac21d165
# ╠═d96eb30a-04da-48bc-ad04-196d89fa101c
# ╠═0157f8d7-2f99-42a6-b377-d3354f61a732
# ╟─58f160b8-23bd-4efa-9f95-16c558588e8c
# ╠═ac7c9f25-3671-46cd-977d-49d6999d6d75
# ╠═5b614467-5709-4d51-8eef-6c4599fd3125
# ╠═a4580fc1-3f2c-4cc5-898d-cab67dabdc71
# ╟─7d02843b-019a-49f6-aaf3-44e47e27ff1c
# ╠═0f71fb37-3127-4d94-b53d-52716c508d01
# ╠═d548ea4c-ffb1-42e8-aafa-d543bc9f04fd
# ╠═917e2ac0-43b0-43f0-8e5e-ec254fc0355e
# ╟─02790c06-6f78-441b-9c42-d9576755f70c8
# ╠═73772f32-c78b-4b4b-b2a1-7b9587ec0f8a
# ╠═ca0b7976-f269-4a5c-8115-69ecbc4ec0d6
# ╟─42627444-c2be-42f5-b0f6-01a47325d9fd
# ╠═06c62b44-ef4d-4488-b402-f8a8a74552d4
# ╠═9ec510e5-05af-4de1-bd3e-30c5cd556598
# ╟─69f5580b-a14d-4793-9c69-9289c58736c4
