#!/usr/bin/env julia

# Include lightweight utility modules (NO Lux/Reactant dependencies)
# These load before early-exit checks for CLI parsing and inspect visualization
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

# Check for help/info flags BEFORE loading expensive packages
if isempty(ARGS) || ARGS[1] in ("--help", "-h")
    println("""
symvqvae — Symmetric VQ-VAE training and inspection CLI.

Usage:
  symvqvae train [pairs] [options]      Train models
  symvqvae inspect PAIR [options]        Inspect a pair
  symvqvae --help                        Show help

Commands:

  train [pairs] [options]
    Train VQ-VAE models on station pairs.

    Arguments:
      pairs    Comma-separated pairs e.g. "AP-BK,AP-CL" (default: all)

    Options:
      --data-dir DIR                JLD2 files directory (default: pwd)
      --save-dir DIR                Output directory
      --nepoch INT                  Training epochs (default: 100)
      --batchsize INT               Minibatch size (default: 4096)
      --lr FLOAT                    Learning rate (default: 0.001)
      --seeds LIST                  Seeds per model (default: "1234,1235")
      --nwindows INT                Waveforms per pair (default: 20000)
      --period-min FLOAT            Min period (default: 3.0s)
      --period-max FLOAT            Max period (default: 10.0s)
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
      --period-min FLOAT            Min period (default: 3.0s)
      --period-max FLOAT            Max period (default: 10.0s)
      --dt FLOAT                    Sample interval (default: 1.0s)
      --whitening-kernel-length INT FIR taps for whitening (default: 128)

  --help, -h
    Show this help message
""")
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
    --period-min FLOAT            Min period (default: 3.0s)
    --period-max FLOAT            Max period (default: 10.0s)
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
            elseif args[i] == "--period-min"
                i += 1
                i <= length(args) && (period_min = parse(Float64, args[i]))
            elseif args[i] == "--period-max"
                i += 1
                i <= length(args) && (period_max = parse(Float64, args[i]))
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

        # Handle Sanket's data format (correlations, dist, latitudes, longitudes, headers)
        if haskey(jld2_data, "correlations")
            correlations = jld2_data["correlations"]
            nt = size(correlations, 1)
            n_waveforms = size(correlations, 2)
            println("  nt: $nt")
            println("  Number of waveforms: $n_waveforms")
            println("  Data shape: $(size(correlations))")
            println("  Data dtype: $(eltype(correlations))")
            println("  Data range: [$(minimum(correlations)), $(maximum(correlations))]")

            if haskey(jld2_data, "dist")
                dist_km = jld2_data["dist"]  # distance is always in km
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
        else
            # Fallback for other JLD2 formats
            println("Available keys in file: $(join(keys(jld2_data), ", "))")
            error("Unrecognized JLD2 data format")
        end
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
    "--xla_gpu_autotune_level=0",
])

using Lux, Reactant, DSP, Statistics
import JLD2

function include_vqvae_architecture_for_cli()
    arch_path = joinpath(@__DIR__, "VQVAE_architecture_v9.jl")
    src = read(arch_path, String)
    src = replace(src, "gpu_device(force=true)" => "nothing # skipped CLI include-time GPU probe")
    return include_string(Main, src, arch_path * " (CLI patched: no include-time GPU probe)")
end

include_vqvae_architecture_for_cli()


function cmd_train(args::Vector{String})
    if !isempty(args) && args[1] in ("--help", "-h")
        println("""
train [pairs] [options]
  Train VQ-VAE models on station pairs.

  Arguments:
    pairs    Comma-separated pairs e.g. "AP-BK,AP-CL" (default: all)

  Options:
    --data-dir DIR                JLD2 files directory (default: pwd)
    --save-dir DIR                Output directory
    --nepoch INT                  Training epochs (default: 100)
    --batchsize INT               Minibatch size (default: 4096)
    --lr FLOAT                    Learning rate (default: 0.001)
    --seeds LIST                  Seeds per model (default: "1234,1235")
    --nwindows INT                Waveforms per pair (default: 20000)
    --period-min FLOAT            Min period (default: 3.0s)
    --period-max FLOAT            Max period (default: 10.0s)
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
    lr = 0.001
    nwindows = 20000
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
        elseif a == "--lr"
            i += 1
            i <= length(args) && (lr = parse(Float64, args[i]))
        elseif a == "--nwindows"
            i += 1
            i <= length(args) && (nwindows = parse(Int, args[i]))
        elseif a == "--period-min"
            i += 1
            i <= length(args) && (period_min = parse(Float64, args[i]))
        elseif a == "--period-max"
            i += 1
            i <= length(args) && (period_max = parse(Float64, args[i]))
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
        elseif !startswith(a, "-")
            pairs = a
        end
        i += 1
    end

    # Print all parameters for user verification
    println("\n" * "="^80)
    println("Training Configuration:")
    println("="^80)
    println("Pair(s):                     $(pairs == "all" ? "all" : pairs)")
    println("Data directory:              $data_dir")
    println("Save directory:              $(isempty(save_dir) ? "$(data_dir)/SavedModels/vqvae_v9_..." : save_dir)")
    println("Number of epochs:            $nepoch")
    println("Batch size:                  $batchsize")
    println("Learning rate:               $lr")
    println("Seeds:                       $seeds")
    println("Number of waveforms/pair:    $nwindows")
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

    seeds_vec  = parse.(Int, strip.(split(seeds, ",")))
    K_vec      = parse.(Int, strip.(split(K, ",")))
    ratios_vec = parse.(Int, strip.(split(ratios, ",")))

    vqvae_parameters = (;
        d, K=K_vec, n_filters, ratios=ratios_vec,
        n_residual_layers,
        entropy_weight=Float32(entropy_weight),
        beta_commit=0.25f0, ema_decay=0.99f0,
        dilation_base=2, residual_kernel_size=3,
        enc_kernel_size=7, dec_kernel_size=7,
        use_bn=false, dead_threshold=50,
        codebook_exclusivity_weight=0.0f0,
        reconstruction_loss=:l1,
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

    save_root = isempty(save_dir) ?
        joinpath(data_dir, "SavedModels", "vqvae_v9_K=$(K_vec)") : save_dir

    device = default_xdev(; force=true)

    @info "Loading first pair to determine nt for XLA compilation..."
    first_pair_data = load_pairs_data([selected_pairs[1]];
        filepath=data_dir, dt, period_min, period_max, n_max=nwindows)
    nt      = size(first_pair_data[1].data.D_train, 1)
    n_train = nwindows

    @info "Compiling Reactant XLA graph (once for this session)..." nt n_train K=K_vec batchsize
    compiled_model = compile_model(nt, n_train;
        vqvae_parameters, training_para, seed=seeds_vec[1], device)

    train_selected_pairs_lazy(selected_pairs, compiled_model;
        seeds=seeds_vec,
        training_para,
        save_root,
        filepath=data_dir,
        dt, period_min, period_max,
        n_max=nwindows,
        bp_filter,
        per_waveform_whitening_kernel_length=whitening_kernel_length,
        device,
    )
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
