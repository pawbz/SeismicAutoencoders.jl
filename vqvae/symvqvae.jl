#!/usr/bin/env julia

# Include lightweight utility modules (NO Lux/Reactant dependencies)
# These load before early-exit checks for CLI parsing and inspect visualization
include("version.jl")
include("io_utils.jl")
include("whitening_utils.jl")

# Helper function for parsing pair specifications (shared by train and inspect)
function parse_pair_spec(pair_str::AbstractString)
    """
    Parse a single pair specification in any of these formats:
    - "STA1-STA2"  (dash separator)
    - "STA1_STA2"  (underscore separator)
    - "STA1,STA2"  (comma separator)

    Returns (sta1, sta2) as strings
    """
    pair_str = strip(pair_str)

    parts = if contains(pair_str, "-")
        split(pair_str, "-", limit=2)
    elseif contains(pair_str, "_")
        split(pair_str, "_", limit=2)
    elseif contains(pair_str, ",")
        split(pair_str, ",", limit=2)
    else
        error("Pair format error: '$pair_str'. Expected 'STA1-STA2', 'STA1_STA2', or 'STA1,STA2'")
    end

    length(parts) == 2 || error("Pair format error: '$pair_str'. Expected 'STA1-STA2', 'STA1_STA2', or 'STA1,STA2'")
    (String(strip(parts[1])), String(strip(parts[2])))
end

function save_dir_number_label(x::Real)
    xf = Float64(x)
    label = isinteger(xf) ? string(Int(xf)) : string(xf)
    replace(label, "." => "p", "-" => "m")
end

function default_train_save_root(data_dir::AbstractString, K_vec, period_min::Real, period_max::Real)
    period_label = "Tmin=$(save_dir_number_label(period_min))s_Tmax=$(save_dir_number_label(period_max))s"
    joinpath(data_dir, "SavedModels", "vqvae_$(VERSION)_K=$(K_vec)_$(period_label)")
end

function parse_period_range(periods::AbstractString)
    parts = strip.(split(periods, ","))
    length(parts) == 2 || error("--periods expects MIN,MAX, e.g. --periods 3,10")
    period_min, period_max = parse.(Float64, parts)
    period_min > 0 && period_max > 0 || error("--periods values must be positive")
    period_min < period_max || error("--periods MIN must be less than MAX")
    period_min, period_max
end

# Check for help/info flags BEFORE loading expensive packages
if isempty(ARGS) || ARGS[1] in ("--help", "-h")
    println("""
$(version_string()) — SymVQVAE training and inspection CLI.

Usage:
  symvqvae train [pairs] [options]      Train models
  symvqvae inspect PAIR [options]        Inspect a pair
  symvqvae list-pairs [--data-dir DIR]   List available station pairs
  symvqvae --help                        Show help

Commands:

  train [pairs] [options]
    Train SymVQVAE models on station pairs.

    Arguments:
      pairs    Comma-separated pairs e.g. "AP-BK,AP-CL" (default: all)

    Options:
      --data-dir DIR                JLD2 files directory (default: pwd)
      --save-dir DIR                Output directory
      --nepoch INT                  Training epochs (default: 100)
      --batchsize INT               Minibatch size (default: 4096)
      --Nmax INT                    Compiled encoder inference width (default: 25000)
      --lr FLOAT                    Learning rate (default: 0.001)
      --seeds LIST                  Seeds per model (default: "1234,1235")
      --periods MIN,MAX             Period range (default: 3,10s)
      --dt FLOAT                    Sample interval (default: 1.0s)
      --K LIST                      Codebook sizes (default: "5,3")
      --d INT                       Latent dimension (default: 40)
      --n-filters INT               Encoder filters (default: 32)
      --ratios LIST                 Stride ratios (default: "2,5")
      --n-residual-layers INT       Residual blocks (default: 3)
      --entropy-weight FLOAT        Entropy weight (default: 0.1)
      --whitening-kernel-length INT FIR taps (default: 128)
      --autodiff-backend STR        "zygote", "enzyme", "auto" (default: "auto")
      --verbose, -v                 Print per-epoch metrics

  inspect PAIR [options]
    Inspect station pair: print statistics and metadata.

    Arguments:
      PAIR    Station pair e.g. "AP-BK" or "AP_BK"

    Options:
      --data-dir DIR                JLD2 files directory (default: pwd)
      --periods MIN,MAX             Period range (default: 3,10s)
      --dt FLOAT                    Sample interval (default: 1.0s)
      --whitening-kernel-length INT FIR taps for whitening (default: 128)

  list-pairs [options]
    List all station pairs found in the data directory. Fast — no GPU/model loading.

    Options:
      --data-dir DIR                JLD2 files directory (default: pwd)

  --help, -h
    Show this help message
""")
    exit(0)
end

# Handle list-pairs command — pure file scan, no heavy dependencies
function cmd_list_pairs(args::Vector{String})
    data_dir = pwd()
    i = 1
    while i <= length(args)
        if args[i] == "--data-dir" && i < length(args)
            i += 1
            data_dir = args[i]
        end
        i += 1
    end
    pairs = list_station_pairs(data_dir)
    if isempty(pairs)
        println("No station pairs found in $(data_dir)")
    else
        println("$(length(pairs)) pair(s) in $(data_dir):")
        for p in pairs; println("  $(p[1])-$(p[2])"); end
    end
end

if !isempty(ARGS) && ARGS[1] == "list-pairs"
    cmd_list_pairs(ARGS[2:end])
    exit(0)
end

# Handle inspect command — needs DSP, JLD2, and UnicodePlots for visualization
if !isempty(ARGS) && ARGS[1] == "inspect"
    using DSP, Statistics
    import JLD2, UnicodePlots

    function cmd_inspect(args::Vector{String})
        if isempty(args)
            error("inspect: PAIR argument required")
        end

        if args[1] in ("--help", "-h")
            println("""
inspect PAIR [options]
  Inspect a station pair: print metadata, unified waveform, and PSD comparison.

  Arguments:
    PAIR    Station pair e.g. "AP-BK" or "AP_BK"

  Options:
    --data-dir DIR                JLD2 files directory (default: pwd)
    --periods MIN,MAX             Period range (default: 3,10s)
    --dt FLOAT                    Sample interval (default: 1.0s)
    --whitening-kernel-length INT FIR taps for whitening (default: 128)
""")
            return
        end

        pair = args[1]
        data_dir = pwd()
        period_min = 3.0
        period_max = 10.0
        dt = 1.0
        whitening_kernel_length = 128

        i = 2
        while i <= length(args)
            if args[i] == "--data-dir"
                i += 1
                i <= length(args) && (data_dir = args[i])
            elseif args[i] == "--periods"
                i += 1
                i <= length(args) && ((period_min, period_max) = parse_period_range(args[i]))
            elseif args[i] == "--dt"
                i += 1
                i <= length(args) && (dt = parse(Float64, args[i]))
            elseif args[i] == "--whitening-kernel-length"
                i += 1
                i <= length(args) && (whitening_kernel_length = parse(Int, args[i]))
            end
            i += 1
        end

        all_pairs = list_station_pairs(data_dir)
        if isempty(all_pairs)
            jld2_files = filter(f -> endswith(f, ".jld2"), readdir(data_dir))
            if isempty(jld2_files)
                error("No .jld2 files found in $(data_dir)")
            else
                error("No station pair files matching pattern STATION_STATION.jld2 found in $(data_dir). Found: $(join(jld2_files, ", "))")
            end
        end

        # Parse the pair specification (handles "-", "_", and "," separators)
        pair_parsed = parse_pair_spec(pair)
        pair_normalized = "$(pair_parsed[1])_$(pair_parsed[2])"
        matched_pair = filter(p -> "$(p[1])_$(p[2])" == pair_normalized, all_pairs)

        if isempty(matched_pair)
            error("Pair $pair not found. Available pairs: $(join([join(p, "-") for p in all_pairs], ", "))")
        end

        pair_obj = matched_pair[1]
        println("Inspecting pair: $(pair_obj[1])-$(pair_obj[2])")

        # Print inspection parameters for user verification
        println("\n" * "="^80)
        println("Inspection Parameters:")
        println("="^80)
        println("Data directory:              $data_dir")
        println("Period range:               $period_min — $period_max s")
        println("Sample interval (dt):       $dt s")
        println("Whitening kernel length:    $whitening_kernel_length")
        println("="^80 * "\n")

        all_files = readdir(data_dir, join=true)
        jld2_files = filter(f -> endswith(f, ".jld2") && startswith(basename(f), "$(pair_obj[1])_$(pair_obj[2])"), all_files)

        if isempty(jld2_files)
            error("No JLD2 file found for pair $(pair_obj[1])-$(pair_obj[2])")
        end

        jld2_file = jld2_files[1]
        println("File: $(basename(jld2_file))")

        jld2_data = JLD2.load(jld2_file)

        # Handle supported JLD2 schemas:
        # - correlations, dist
        # - D, Distances
        correlations = jld2_correlations(jld2_data)
        nt = size(correlations, 1)
        n_waveforms = size(correlations, 2)
        println("  nt: $nt")
        println("  Number of waveforms: $n_waveforms")
        println("  Data shape: $(size(correlations))")
        println("  Data dtype: $(eltype(correlations))")
        println("  Data range: [$(minimum(correlations)), $(maximum(correlations))]")

        dist_km = jld2_distance(jld2_data)
        if !isnothing(dist_km)
            println("  Interstation distance: $dist_km km")
        end

        if haskey(jld2_data, "latitudes") && haskey(jld2_data, "longitudes")
            lats = jld2_data["latitudes"]
            lons = jld2_data["longitudes"]
            if length(lats) >= 2 && length(lons) >= 2
                println("  Station $(pair_obj[1]): lat=$(lats[1]), lon=$(lons[1])")
                println("  Station $(pair_obj[2]): lat=$(lats[2]), lon=$(lons[2])")
            end
        end

        # Split acausal (negative lags) and causal (positive lags) sides
        mid = div(nt, 2) + 1
        acausal = correlations[1:mid-1, :]
        causal = correlations[mid:nt, :]

        mean_acausal = vec(mean(acausal, dims=2))
        mean_causal = vec(mean(causal, dims=2))

        # Plot 1: Unified waveform plot on single lag axis
        n_acausal = size(acausal, 1)
        n_causal = size(causal, 1)
        # Unified lag axis: acausal lags from -n_acausal to -1, causal from 0 to n_causal-1
        lags_unified = collect(-(n_acausal):(n_causal-1)) .* dt

        # Combine acausal and causal into single vector (concatenate on same axis)
        # Acausal occupies indices 1:n_acausal, causal occupies indices n_acausal+1:n_acausal+n_causal
        waveform_unified = vcat(mean_acausal, mean_causal)

        plt_wave = UnicodePlots.lineplot(
            lags_unified, waveform_unified;
            xlabel="lag (s)", ylabel="amplitude",
            title="$(pair_obj[1])-$(pair_obj[2])  global average waveform (unified lag axis)",
            width=80, height=12,
        )
        println(plt_wave)

        # Plot 2: PSD comparison (raw vs whitened) with period-range filtering
        # Compute raw PSD for acausal and causal separately
        psd_acausal_raw = DSP.periodogram(Float64.(mean_acausal); fs=inv(dt))
        psd_causal_raw = DSP.periodogram(Float64.(mean_causal); fs=inv(dt))

        freqs_acausal = DSP.freq(psd_acausal_raw)
        valid_freqs_ac = freqs_acausal .> 0
        periods_acausal = inv.(freqs_acausal[valid_freqs_ac])
        pow_acausal_raw = 10 .* log10.(max.(DSP.power(psd_acausal_raw)[valid_freqs_ac], 1e-30))

        freqs_causal = DSP.freq(psd_causal_raw)
        valid_freqs_c = freqs_causal .> 0
        periods_causal = inv.(freqs_causal[valid_freqs_c])
        pow_causal_raw = 10 .* log10.(max.(DSP.power(psd_causal_raw)[valid_freqs_c], 1e-30))

        # Combine periods and power (average across acausal and causal)
        # Note: periods may have different lengths; use acausal for plotting (longer)
        pow_raw_combined = (pow_acausal_raw .+ pow_causal_raw[1:length(pow_acausal_raw)]) ./ 2

        # Filter to period range
        period_mask_raw = (periods_acausal .>= period_min) .& (periods_acausal .<= period_max)
        periods_filtered = periods_acausal[period_mask_raw]
        pow_raw_filtered = pow_raw_combined[period_mask_raw]

        # Compute whitened PSD
        # Apply whitening to both acausal and causal separately
        acausal_f32 = Float32.(acausal)
        causal_f32 = Float32.(causal)

        fir_ac = compute_whitening_fir(acausal_f32; kernel_length=whitening_kernel_length)
        fir_c = compute_whitening_fir(causal_f32; kernel_length=whitening_kernel_length)

        acausal_whitened = apply_whitening_fir(acausal_f32, fir_ac)
        causal_whitened = apply_whitening_fir(causal_f32, fir_c)

        mean_acausal_whitened = vec(mean(acausal_whitened, dims=2))
        mean_causal_whitened = vec(mean(causal_whitened, dims=2))

        psd_acausal_whitened = DSP.periodogram(Float64.(mean_acausal_whitened); fs=inv(dt))
        psd_causal_whitened = DSP.periodogram(Float64.(mean_causal_whitened); fs=inv(dt))

        pow_acausal_wh = 10 .* log10.(max.(DSP.power(psd_acausal_whitened)[valid_freqs_ac], 1e-30))
        pow_causal_wh = 10 .* log10.(max.(DSP.power(psd_causal_whitened)[valid_freqs_c], 1e-30))
        pow_whitened_combined = (pow_acausal_wh .+ pow_causal_wh[1:length(pow_acausal_wh)]) ./ 2

        pow_whitened_filtered = pow_whitened_combined[period_mask_raw]

        # Plot raw PSD
        plt_psd = UnicodePlots.lineplot(
            periods_filtered, pow_raw_filtered;
            name="raw", xlabel="period (s)", ylabel="PSD (dB)",
            title="$(pair_obj[1])-$(pair_obj[2])  PSD: raw vs whitened",
            width=80, height=12,
        )
        # Overlay whitened PSD
        UnicodePlots.lineplot!(plt_psd, periods_filtered, pow_whitened_filtered; name="whitened")
        println(plt_psd)
    end

    cmd_inspect(ARGS[2:end])
    exit(0)
end

function append_xla_flags!(flags::Vector{String})
    current = split(get(ENV, "XLA_FLAGS", ""))
    for flag in flags
        flag in current || push!(current, flag)
    end
    ENV["XLA_FLAGS"] = join(current, " ")
end

append_xla_flags!([
    "--xla_gpu_enable_cublaslt=true",
])

using Lux, Reactant, DSP, Statistics
import JLD2

function include_vqvae_architecture_for_cli()
    arch_path = joinpath(@__DIR__, "SymVQVAE_architecture.jl")
    src = read(arch_path, String)
    src = replace(src, "gpu_device(force=true)" => "nothing # skipped CLI include-time GPU probe")
    return include_string(Main, src, arch_path * " (CLI patched: no include-time GPU probe)")
end

include_vqvae_architecture_for_cli()


# Test helper: train on synthetic random waveform data using real training pipeline
function train_selected_pairs_synthetic(compiled_model;
    seeds, training_para, save_root, nt, n_synthetic,
    period_min, period_max, dt, bp_filter,
    per_waveform_whitening_kernel_length, device)

    @info "Test mode: creating synthetic data and running full training pipeline" nt n_synthetic nepoch=training_para.nepoch

    # Create synthetic waveform data
    n_acausal = div(n_synthetic, 2)
    n_causal = n_synthetic - n_acausal
    D_acausal_raw = Float32.(randn(nt, n_acausal))
    D_causal_raw = Float32.(randn(nt, n_causal))

    pair = ("TEST", "PAIR")

    # Build data bundle matching real format
    data_bundle = (;
        pair=pair,
        distance=100.0,
        D1fac=D_acausal_raw,
        D1fc=D_causal_raw,
        headers=nothing,
    )

    # Create a minimal vqvae_parameters tuple (use same as passed in)
    # Extract from compiled_model which has para field
    para = compiled_model.para

    # Use the real training pipeline with synthetic data
    # This ensures we test the same code path as production training
    @info "Running real training pipeline on synthetic data (1 epoch to test)..."

    # We'll call the core training function directly with synthetic data
    # This exercises all code paths including line 1684 fix
    for seed in seeds
        @info "Test: calling update() on synthetic pair" pair seed

        try
            # Create training data matching real format
            train_x_cpu = hcat(data_bundle.D1fac, data_bundle.D1fc)
            test_x_cpu = hcat(data_bundle.D1fac[:, 1:min(50, div(size(data_bundle.D1fac, 2), 2))],
                             data_bundle.D1fc[:, 1:min(50, div(size(data_bundle.D1fc, 2), 2))])

            @info "Synthetic data ready" train_size=size(train_x_cpu) test_size=size(test_x_cpu) batchsize=training_para.batchsize

            # Reset model for this seed
            ps, st = reset_vqvae(compiled_model.model; seed, device)
            loss_history = fresh_loss_history()

            # Call update() - this tests the entire training path including line 1684 fix
            # If there's a shape mismatch, it will fail here with RuntimeProgramInputMismatch
            ps, st, loss_history = update(
                compiled_model.model, ps, st, loss_history,
                train_x_cpu, test_x_cpu,
                para, training_para;
                device=device,
                compiled=compiled_model.compiled,
                cdev=default_cdev(),
                n_compiled=compiled_model.n_compiled_encoder,
            )

            final_loss = isempty(loss_history.train_objective) ? missing : last(loss_history.train_objective)
            @info "Test: update() succeeded" seed final_loss train_batches=div(size(train_x_cpu,2), training_para.batchsize)

        catch e
            @error "TEST FAILURE: update() crashed" exception=(e, catch_backtrace())
            @error "If this is RuntimeProgramInputMismatch, the line 1684 fix didn't work"
            rethrow(e)
        end
    end

    @info "Test mode PASSED - all code paths executed successfully"
end

function cmd_train(args::Vector{String})
    if !isempty(args) && args[1] in ("--help", "-h")
        println("""
train [pairs] [options]
  Train SymVQVAE models on station pairs.

  Arguments:
    pairs    Comma-separated pairs e.g. "AP-BK,AP-CL" (default: all)

  Options:
    --data-dir DIR                JLD2 files directory (default: pwd)
    --save-dir DIR                Output directory
    --nepoch INT                  Training epochs (default: 100)
    --batchsize INT               Minibatch size (default: 4096)
    --Nmax INT                    Compiled encoder inference width (default: 25000)
    --lr FLOAT                    Learning rate (default: 0.001)
    --seeds LIST                  Seeds per model (default: "1234,1235")
    --periods MIN,MAX             Period range (default: 3,10s)
    --dt FLOAT                    Sample interval (default: 1.0s)
    --K LIST                      Codebook sizes (default: "5,3")
    --d INT                       Latent dimension (default: 40)
    --n-filters INT               Encoder filters (default: 32)
    --ratios LIST                 Stride ratios (default: "2,5")
    --n-residual-layers INT       Residual blocks (default: 3)
    --entropy-weight FLOAT        Entropy weight (default: 0.1)
    --whitening-kernel-length INT FIR taps (default: 128)
    --autodiff-backend STR        "zygote", "enzyme", "auto" (default: "auto")
    --verbose, -v                 Print per-epoch metrics
""")
        return
    end

    pairs = "all"
    data_dir = pwd()
    save_dir = ""
    seeds = "1234,1235"
    nepoch = 100
    batchsize = 4096
    Nmax = 25_000
    lr = 0.001
    period_min = 3.0
    period_max = 10.0
    dt = 1.0
    K = "5,3"
    d = 40
    n_filters = 32
    ratios = "2,5"
    n_residual_layers = 3
    entropy_weight = 0.1
    whitening_kernel_length = 128
    autodiff_backend = "auto"
    verbose = false

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--data-dir"
            i += 1
            i <= length(args) && (data_dir = args[i])
        elseif a == "--save-dir"
            i += 1
            i <= length(args) && (save_dir = args[i])
        elseif a == "--seeds"
            i += 1
            i <= length(args) && (seeds = args[i])
        elseif a == "--nepoch"
            i += 1
            i <= length(args) && (nepoch = parse(Int, args[i]))
        elseif a == "--batchsize"
            i += 1
            i <= length(args) && (batchsize = parse(Int, args[i]))
        elseif a == "--Nmax"
            i += 1
            i <= length(args) && (Nmax = parse(Int, args[i]))
        elseif a == "--lr"
            i += 1
            i <= length(args) && (lr = parse(Float64, args[i]))
        elseif a == "--periods"
            i += 1
            i <= length(args) && ((period_min, period_max) = parse_period_range(args[i]))
        elseif a == "--dt"
            i += 1
            i <= length(args) && (dt = parse(Float64, args[i]))
        elseif a == "--K"
            i += 1
            i <= length(args) && (K = args[i])
        elseif a == "--d"
            i += 1
            i <= length(args) && (d = parse(Int, args[i]))
        elseif a == "--n-filters"
            i += 1
            i <= length(args) && (n_filters = parse(Int, args[i]))
        elseif a == "--ratios"
            i += 1
            i <= length(args) && (ratios = args[i])
        elseif a == "--n-residual-layers"
            i += 1
            i <= length(args) && (n_residual_layers = parse(Int, args[i]))
        elseif a == "--entropy-weight"
            i += 1
            i <= length(args) && (entropy_weight = parse(Float64, args[i]))
        elseif a == "--whitening-kernel-length"
            i += 1
            i <= length(args) && (whitening_kernel_length = parse(Int, args[i]))
        elseif a == "--autodiff-backend"
            i += 1
            i <= length(args) && (autodiff_backend = args[i])
        elseif a == "--verbose" || a == "-v"
            verbose = true
        elseif a == "-f" || a == "--foreground"
            # Foreground flag is handled by the bash wrapper, silently ignore here
            nothing
        elseif a == "--test-mode"
            # Override parameters for quick testing with small synthetic data
            nepoch = 2
            batchsize = 32  # Much smaller than real (4096)
            Nmax = max(Nmax, 512)
            pairs = "TEST"  # Special marker for synthetic data
        elseif !startswith(a, "-")
            pairs = a
        end
        i += 1
    end

    seeds_vec  = parse.(Int, strip.(split(seeds, ",")))
    K_vec      = parse.(Int, strip.(split(K, ",")))
    ratios_vec = parse.(Int, strip.(split(ratios, ",")))
    default_save_root = default_train_save_root(data_dir, K_vec, period_min, period_max)

    # Print all parameters for user verification
    println("\n" * "="^80)
    println("$(version_string()) — Training Configuration:")
    println("="^80)
    println("Pair(s):                     $(pairs == "all" ? "all" : pairs)")
    println("Data directory:              $data_dir")
    println("Save directory:              $(isempty(save_dir) ? default_save_root : save_dir)")
    println("Number of epochs:            $nepoch")
    println("Batch size:                  $batchsize")
    println("Encoder compile Nmax:        $Nmax")
    println("Learning rate:               $lr")
    println("Seeds:                       $seeds")
    println("Period range (bandpass):     $period_min — $period_max s")
    println("Sample interval (dt):        $dt s")
    println("Codebook sizes (K):          $K")
    println("Latent dimension (d):        $d")
    println("Encoder filters:             $n_filters")
    println("Stride ratios:               $ratios")
    println("Residual layers:             $n_residual_layers")
    println("Entropy weight:              $entropy_weight")
    println("Whitening kernel length:     $whitening_kernel_length")
    println("Autodiff backend:            $autodiff_backend")
    println("Verbose output:              $(verbose ? "yes" : "no")")
    println("="^80 * "\n")

    # Handle test mode: generate synthetic data
    if pairs == "TEST"
        selected_pairs = [("TEST", "PAIR")]
        test_mode = true
    else
        test_mode = false
        all_pairs  = list_station_pairs(data_dir)
        isempty(all_pairs) && error("No station pairs found in $(data_dir). Check --data-dir.")

        selected_pairs = if pairs == "all"
            all_pairs
        else
        # Parse pair specifications: supports formats like:
        # - "AP-BK" or "AP_BK" (single pair)
        # - "AP-BK,SM17-SM42" (multiple pairs with - separator)
        # - "AP_BK,SM17_SM42" (multiple pairs with _ separator)
        # - "AP,BK" (single pair with , separator)
        # - "AP-BK SM17-SM42" (space-separated pairs)

        parsed = []

        # Check if format is "STA1,STA2" (single pair with comma separator, no dashes/underscores)
        if count(',', pairs) == 1 && !contains(pairs, '-') && !contains(pairs, '_')
            push!(parsed, parse_pair_spec(pairs))
        else
            # Multiple pairs separated by commas or spaces
            pair_strs = if contains(pairs, ',')
                split(pairs, ",")
            else
                split(pairs)
            end

            for pr in pair_strs
                isempty(strip(pr)) && continue
                push!(parsed, parse_pair_spec(pr))
            end
        end

        parsed
    end
    end  # Close outer if-else for test mode vs normal mode

    vqvae_parameters = (;
        d, K=K_vec, n_filters, ratios=ratios_vec,
        n_residual_layers,
        entropy_weight=Float32(entropy_weight),
        beta_commit=0.25f0, ema_decay=0.99f0,
        dilation_base=2, residual_kernel_size=3,
        enc_kernel_size=7, dec_kernel_size=7,
        dead_threshold=50,
    )
    training_para = VQVAE_Training_Para(;
        batchsize,
        nepoch,
        initial_learning_rate=lr,
        autodiff_backend=Symbol(autodiff_backend),
        Mnn_schedule=[(1, 128), (5, 256), (26, 256)],
        warmup_epochs=0,
        verbose,
        knn_search_chunk_size_fraction=0.25,
        index_refresh_every=2,
    )

    bp_filter = let
        rt = DSP.Bandpass(inv(period_max), inv(period_min))
        DSP.digitalfilter(rt, DSP.Butterworth(2); fs=inv(dt))
    end

    save_root = isempty(save_dir) ? default_save_root : save_dir

    device = default_xdev(; force=true)

    # For test mode, create synthetic data with small nt
    if test_mode
        @info "TEST MODE: Using synthetic data with small sizes"
        nt = 100  # Very small for fast compilation
    else
        @info "Loading first pair to determine nt for XLA compilation..."
        first_pair_data = load_pairs_data([selected_pairs[1]];
            filepath=data_dir, dt, period_min, period_max)
        nt = size(first_pair_data[1].data.D_train, 1)
    end

    @info "Compiling Reactant XLA graph (once for this session)..." nt K=K_vec batchsize Nmax
    compiled_model = compile_model(nt;
        vqvae_parameters, training_para, seed=seeds_vec[1], device, Nmax)

    if test_mode
        n_synthetic = 250  # NOT a multiple of batchsize to test edge case
        @info "Generating synthetic test data..." nt n_synthetic batchsize nepoch
        train_selected_pairs_synthetic(compiled_model;
            seeds=seeds_vec,
            training_para,
            save_root,
            nt, n_synthetic,
            period_min, period_max, dt,
            bp_filter,
            per_waveform_whitening_kernel_length=whitening_kernel_length,
            device,
        )
    else
        train_selected_pairs_lazy(selected_pairs, compiled_model;
            seeds=seeds_vec,
            training_para,
            save_root,
            filepath=data_dir,
            dt, period_min, period_max,
            bp_filter,
            per_waveform_whitening_kernel_length=whitening_kernel_length,
            device,
        )
    end
end

function main(args::Vector{String})
    if isempty(args) || args[1] in ("--help", "-h")
        show_help()
        exit(0)
    end

    cmd = args[1]
    cmd_args = args[2:end]

    if cmd == "train"
        cmd_train(cmd_args)
    elseif cmd == "inspect"
        cmd_inspect(cmd_args)
    else
        error("Unknown command: $cmd. Use 'symvqvae --help' for usage.")
    end
end

main(ARGS)
