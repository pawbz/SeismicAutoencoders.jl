#!/usr/bin/env julia

function usage()
    println("""
Usage: train_vqvae [pairs] [options]

Arguments:
  pairs                  Comma-separated station pairs e.g. "AP_BK,AP_CL"
                         Use "all" or omit to train all discovered pairs (default: all)

Info Options:
  --help, -h             Show this message
  --list-pairs, -l       Print discovered pairs and exit
  --sample-pair          Sample random pair and print statistics, exit

Training Options:
  --data-dir DIR         JLD2 correlation files directory (default: pwd)
  --save-dir DIR         Output root directory (default: data-dir/SavedModels/...)
  --seeds LIST           Comma-separated seeds, trains separate model per seed (default: "1234,1235")
  --nepoch INT           Training epochs (default: 100)
  --batchsize INT        Minibatch size (default: 4096)
  --lr FLOAT             Learning rate (default: 0.001)
  --verbose, -v          Print per-epoch metrics

Data Processing Options:
  --period-min FLOAT     Minimum period in seconds (default: 10.0)
  --period-max FLOAT     Maximum period in seconds (default: 75.0)
  --dt FLOAT             Sample interval in seconds (default: 1.0)
  --nwindows INT         Waveforms/windows per pair and XLA compile shape (default: 20000)
  --whitening-kernel-length INT  FIR tap count (default: 128)

Model Architecture Options:
  --K LIST               Codebook sizes e.g. "5,3" (default: "5,3")
  --d INT                Latent dimension (default: 40)
  --n-filters INT        Encoder filter count (default: 32)
  --ratios LIST          Encoder stride ratios e.g. "2,5" (default: "2,5")
  --n-residual-layers INT  Residual blocks per stage (default: 3)
  --entropy-weight FLOAT   Entropy regularization weight (default: 0.1)
  --autodiff-backend STR "zygote", "enzyme", or "auto" (default: "auto")

Compilation Test Options (exit after test):
  --dummy-compile-test      Compile tiny synthetic model
  --dummy-forward-test      Compile tiny synthetic forward pass
  --dummy-loss-test         Compile tiny synthetic loss pass
  --dummy-grad-test         Compile tiny synthetic gradient pass
  --dummy-grad-encoder-test Compile gradient through encoder/head only
  --dummy-grad-decoder-test Compile gradient through decoders only
  --dummy-grad-recon-test   Compile gradient for reconstruction loss only
  --dummy-grad-commit-test  Compile gradient for commitment loss only
  --dummy-grad-then-apply-test Compile gradient, then apply optimizer separately
  --dummy-train-step-test   Compile tiny synthetic gradient+optimizer step
""")
end

function list_station_pairs(filepath::String)
    files = readdir(filepath)
    pairs = Set{Tuple{String,String}}()
    for f in files
        m = match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)", basename(f))
        m === nothing && continue
        push!(pairs, (m.captures[1], m.captures[2]))
    end
    return sort!(collect(pairs), by=x -> (x[1], x[2]))
end

function quick_parse_args(args)
    data_dir = pwd()
    list_pairs = false
    sample_pair = false
    period_min = 10.0
    period_max = 75.0
    dt = 1.0
    pairs_arg = "all"
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--help" || a == "-h"
            return :help
        elseif a == "--list-pairs" || a == "-l"
            list_pairs = true
        elseif a == "--sample-pair"
            sample_pair = true
        elseif a == "--data-dir"
            i += 1
            i <= length(args) && (data_dir = args[i])
        elseif a == "--period-min"
            i += 1
            i <= length(args) && (period_min = parse(Float64, args[i]))
        elseif a == "--period-max"
            i += 1
            i <= length(args) && (period_max = parse(Float64, args[i]))
        elseif a == "--dt"
            i += 1
            i <= length(args) && (dt = parse(Float64, args[i]))
        elseif !startswith(a, "-")
            pairs_arg = a
        end
        i += 1
    end
    list_pairs && return (:list_pairs, data_dir)
    sample_pair && return (:sample_pair, data_dir, period_min, period_max, dt, pairs_arg)
    return nothing
end

result = quick_parse_args(ARGS)
if result === :help
    usage()
    exit(0)
elseif result isa Tuple && result[1] === :list_pairs
    all_pairs = list_station_pairs(result[2])
    if isempty(all_pairs)
        println("No pairs found in $(result[2])")
    else
        println("Available pairs in $(result[2]):")
        for p in all_pairs; println("  $(p[1])-$(p[2])"); end
    end
    exit(0)
elseif result isa Tuple && result[1] === :sample_pair
    # sample-pair needs architecture loaded, handle it after DSP import
    sample_pair_data = (result[2], result[3], result[4], result[5], result[6])
else
    sample_pair_data = nothing
end

function append_xla_flags!(flags::Vector{String})
    current = split(get(ENV, "XLA_FLAGS", ""))
    for flag in flags
        flag in current || push!(current, flag)
    end
    ENV["XLA_FLAGS"] = join(current, " ")
    return ENV["XLA_FLAGS"]
end

append_xla_flags!([
    "--xla_gpu_enable_cublaslt=true",
    "--xla_gpu_autotune_level=0",
])

using Lux, Reactant

function include_vqvae_architecture_for_cli()
    arch_path = joinpath(@__DIR__, "VQVAE_architecture_v9.jl")
    src = read(arch_path, String)
    src = replace(src, "gpu_device(force=true)" => "nothing # skipped CLI include-time GPU probe")
    return include_string(Main, src, arch_path * " (CLI patched: no include-time GPU probe)")
end

include_vqvae_architecture_for_cli()

using DSP
using UnicodePlots

if sample_pair_data !== nothing
    data_dir, period_min, period_max, dt, pairs_arg = sample_pair_data
    candidate_pairs = if pairs_arg == "all"
        all_p = list_station_pairs(data_dir)
        isempty(all_p) && error("No pairs found in $(data_dir)")
        all_p
    else
        [(String(parts[1]), String(parts[2])) for parts in
            [split(pr, "-", limit=2) for pr in split(pairs_arg, ",")]]
    end
    pair = candidate_pairs[rand(1:length(candidate_pairs))]

    println("Sampled pair: $(pair[1])-$(pair[2])")

    bp_filter = let
        rt = DSP.Bandpass(inv(period_max), inv(period_min))
        DSP.digitalfilter(rt, DSP.Butterworth(2); fs=inv(dt))
    end

    pair_data = load_pairs_data([pair]; filepath=data_dir, dt, period_min, period_max, n_max=100000)

    if !isempty(pair_data) && hasfield(typeof(pair_data[1]), :data)
        pair_obj = pair_data[1]
        D = pair_obj.data.D_train
        nt = size(D, 1)
        n_waveforms = size(D, 2)

        files = filter(f -> startswith(basename(f), "$(pair[1])_$(pair[2])"), readdir(data_dir, join=true))
        if !isempty(files)
            println("File: $(basename(files[1]))")
        end

        println("  nt: $nt")
        println("  Number of waveforms: $n_waveforms")
        println("  Data shape: $(size(D))")
        println("  Data dtype: $(eltype(D))")
        println("  Data range: [$(minimum(D)), $(maximum(D))]")

        if hasfield(typeof(pair_obj), :distance)
            dist_km = pair_obj.distance / 1000
            println("  Interstation distance: $(round(dist_km, digits=2)) km")
        end

        if hasfield(typeof(pair_obj), :station1) && hasfield(typeof(pair_obj), :station2)
            s1 = pair_obj.station1
            s2 = pair_obj.station2
            if hasfield(typeof(s1), :lat) && hasfield(typeof(s1), :lon)
                println("  Station $(pair[1]): lat=$(s1.lat), lon=$(s1.lon)")
                println("  Station $(pair[2]): lat=$(s2.lat), lon=$(s2.lon)")
            end
        end

        # Unicode plot 1: mean causal and acausal waveforms
        if hasfield(typeof(pair_obj.data), :D_ac_all) && hasfield(typeof(pair_obj.data), :D_c_all)
            D_ac = pair_obj.data.D_ac_all
            D_c  = pair_obj.data.D_c_all
            lags_s  = collect((1:nt) .* dt)
            mean_ac = vec(mean(D_ac, dims=2))
            mean_c  = vec(mean(D_c,  dims=2))
            plt_wave = UnicodePlots.lineplot(
                lags_s, mean_ac;
                name="acausal", xlabel="lag (s)", ylabel="amplitude",
                title="$(pair[1])-$(pair[2])  mean waveforms  [$(size(D_ac,2)) acausal, $(size(D_c,2)) causal]",
                width=80, height=12,
            )
            UnicodePlots.lineplot!(plt_wave, lags_s, mean_c; name="causal")
            println(plt_wave)

            # Unicode plot 2: mean PSD (period axis, seismology convention)
            psd_ac = DSP.periodogram(Float64.(mean_ac); fs=inv(dt))
            psd_c  = DSP.periodogram(Float64.(mean_c);  fs=inv(dt))
            freqs  = DSP.freq(psd_ac)
            valid  = freqs .> 0
            periods = inv.(freqs[valid])
            pow_ac  = 10 .* log10.(max.(DSP.power(psd_ac)[valid], 1e-30))
            pow_c   = 10 .* log10.(max.(DSP.power(psd_c)[valid],  1e-30))
            plt_psd = UnicodePlots.lineplot(
                periods, pow_ac;
                name="acausal", xlabel="period (s)", ylabel="PSD (dB)",
                title="$(pair[1])-$(pair[2])  mean PSD",
                width=80, height=12,
                xscale=:log10,
            )
            UnicodePlots.lineplot!(plt_psd, periods, pow_c; name="causal")
            println(plt_psd)
        end
    else
        error("Failed to load pair data")
    end
    exit(0)
end
    opts = Dict{String,Any}(
        "pairs"                    => "all",
        "data-dir"                 => pwd(),
        "dummy-compile-test"       => false,
        "dummy-forward-test"       => false,
        "dummy-loss-test"          => false,
        "dummy-grad-test"          => false,
        "dummy-grad-encoder-test"  => false,
        "dummy-grad-decoder-test"  => false,
        "dummy-grad-recon-test"    => false,
        "dummy-grad-commit-test"   => false,
        "dummy-grad-then-apply-test" => false,
        "dummy-train-step-test"    => false,
        "save-dir"                 => "",
        "seeds"                    => "1234,1235",
        "period-min"               => 10.0,
        "period-max"               => 75.0,
        "dt"                       => 1.0,
        "nwindows"                 => 20000,
        "K"                        => "5,3",
        "d"                        => 40,
        "n-filters"                => 32,
        "ratios"                   => "2,5",
        "n-residual-layers"        => 3,
        "entropy-weight"           => 0.1,
        "batchsize"                => 4096,
        "nepoch"                   => 100,
        "lr"                       => 0.001,
        "whitening-kernel-length"  => 128,
        "autodiff-backend"         => "auto",
        "verbose"                  => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--dummy-compile-test"
            opts["dummy-compile-test"] = true
        elseif a == "--dummy-forward-test"
            opts["dummy-forward-test"] = true
        elseif a == "--dummy-loss-test"
            opts["dummy-loss-test"] = true
        elseif a == "--dummy-grad-test"
            opts["dummy-grad-test"] = true
        elseif a == "--dummy-grad-encoder-test"
            opts["dummy-grad-encoder-test"] = true
        elseif a == "--dummy-grad-decoder-test"
            opts["dummy-grad-decoder-test"] = true
        elseif a == "--dummy-grad-recon-test"
            opts["dummy-grad-recon-test"] = true
        elseif a == "--dummy-grad-commit-test"
            opts["dummy-grad-commit-test"] = true
        elseif a == "--dummy-grad-then-apply-test"
            opts["dummy-grad-then-apply-test"] = true
        elseif a == "--dummy-train-step-test"
            opts["dummy-train-step-test"] = true
        elseif a == "--verbose" || a == "-v"
            opts["verbose"] = true
        elseif startswith(a, "--")
            key = a[3:end]
            i += 1
            i > length(args) && error("Missing value for $a")
            val = args[i]
            if key in ("n-max", "compile-n-train")
                key = "nwindows"
            end
            if key in ("period-min","period-max","dt","entropy-weight","lr")
                opts[key] = parse(Float64, val)
            elseif key in ("nwindows","d","n-filters","n-residual-layers","batchsize","nepoch","whitening-kernel-length")
                opts[key] = parse(Int, val)
            else
                opts[key] = val
            end
        elseif !startswith(a, "-")
            opts["pairs"] = a
        else
            error("Unknown option: $a")
        end
        i += 1
    end
    return opts
end

function dummy_parameters()
    vqvae_parameters = (;
        d=8, K=[2, 2], n_filters=4, ratios=[2, 5],
        n_residual_layers=1,
        entropy_weight=0.1f0,
        beta_commit=0.25f0, ema_decay=0.99f0,
        dilation_base=2, residual_kernel_size=3,
        enc_kernel_size=7, dec_kernel_size=7,
        use_bn=false, dead_threshold=50,
        codebook_exclusivity_weight=0.0f0,
        reconstruction_loss=:l1,
    )
    training_para = VQVAE_Training_Para(;
        batchsize=8,
        nepoch=1,
        initial_learning_rate=1f-3,
        autodiff_backend=:enzyme,
        Mnn_schedule=[(1, 8)],
        warmup_epochs=0,
        verbose=false,
        knn_search_chunk_size_fraction=0.25,
        index_refresh_every=2,
    )
    return vqvae_parameters, training_para
end

function dummy_problem()
    vqvae_parameters, training_para = dummy_parameters()
    nt = 100
    n_train = 32
    return (; vqvae_parameters, training_para, nt, n_train, seed=1234)
end

function make_dummy_training_objects()
    problem = dummy_problem()
    ensure_reactant_xla_flags!()
    device = default_xdev(; force=true)
    cdev = default_cdev()
    rng = Xoshiro(problem.seed)
    para = VQVAE_Para(; merge(problem.vqvae_parameters, (; nt=problem.nt, seed=problem.seed))...)
    model, ps, st, _ = get_vqvae(para; rng, device)
    train_x_cpu = randn(rng, Float32, problem.nt, problem.n_train)
    dummy_batch_cpu = train_x_cpu[:, 1:problem.training_para.batchsize]
    dummy_target_cpu = Float32.(MLUtils.normalise(dummy_batch_cpu; dims=1))

    ps_cpu = cdev(ps)
    st_cpu = Lux.testmode(cdev(st))
    lat, _ = encoder_latents(model, dummy_batch_cpu, ps_cpu, st_cpu)
    payload_cpu, rvq_cpu = prepare_split_payload(lat.z_e1, lat.z_e2, st_cpu.rvq, model.K;
        ema_decay=para.ema_decay,
        epsilon=para.epsilon,
        dead_threshold=para.dead_threshold,
        training=true)
    st_dev = merge(st, (; rvq=device(rvq_cpu)))
    batch_dev = (;
        x=device(Float32.(dummy_batch_cpu)),
        target=device(dummy_target_cpu),
        vq_payload=device(payload_cpu),
    )
    loss_fn = VQVAELoss(para)
    opt = Optimisers.AdamW(; eta=Float64(problem.training_para.initial_learning_rate),
        lambda=Float64(problem.training_para.weight_decay))
    train_state = Training.TrainState(model, ps, Lux.trainmode(st_dev), opt)
    ad_backend = training_backend(problem.training_para, device)
    return merge(problem, (; device, cdev, para, model, ps, st=st_dev, batch_dev,
        loss_fn, train_state, ad_backend, train_x_cpu))
end

function dummy_forward_objective(model, ps, st, batch, para)
    result, st_new = forward_with_precomputed_vq(
        model, batch.x, ps, st, batch.vq_payload; beta_commit=para.beta_commit)
    return result.xhat, st_new
end

function dummy_loss_objective(loss_fn, model, ps, st, batch)
    return loss_fn(model, ps, st, batch)
end

function dummy_encoder_norm_objective(model, ps, st, batch)
    lat, st_lat = encoder_latents(model, batch.x, ps, st)
    st_new = merge(st, (; encoder=st_lat.encoder, head1=st_lat.head1, head2=st_lat.head2))
    loss = mean(abs2, lat.z_e)
    return loss, st_new, (; recon_loss=loss, commit_loss=zero(loss))
end

function dummy_decoder_recon_objective(model, ps, st, batch)
    z_q1 = EnzymeCore.ignore_derivatives(batch.vq_payload.z_q_stages[1])
    z_q2 = EnzymeCore.ignore_derivatives(batch.vq_payload.z_q_stages[2])
    x1hat, st_dec1 = model.decoder1(z_q1, ps.decoder1, st.decoder1)
    x2hat, st_dec2 = model.decoder2(z_q2, ps.decoder2, st.decoder2)
    xhat = x1hat .+ x2hat
    loss = mse_loss(xhat, batch.target)
    st_new = merge(st, (; decoder1=st_dec1, decoder2=st_dec2))
    return loss, st_new, (; recon_loss=loss, commit_loss=zero(loss))
end

function dummy_recon_only_objective(model, ps, st, batch)
    result, st_new = forward_with_precomputed_vq(
        model, batch.x, ps, st, batch.vq_payload; beta_commit=model.beta_commit)
    loss = mse_loss(result.xhat, batch.target)
    return loss, st_new, (; recon_loss=loss, commit_loss=result.commit_loss)
end

function dummy_commit_only_objective(model, ps, st, batch)
    result, st_new = forward_with_precomputed_vq(
        model, batch.x, ps, st, batch.vq_payload; beta_commit=model.beta_commit)
    loss = result.commit_loss
    return loss, st_new, (; recon_loss=mse_loss(result.xhat, batch.target), commit_loss=loss)
end

function run_dummy_forward_test()
    ctx = make_dummy_training_objects()
    @info "Running dummy forward compile test" nt=ctx.nt n_train=ctx.n_train batchsize=ctx.training_para.batchsize
    compiled_forward = @compile dummy_forward_objective(
        ctx.model, ctx.ps, Lux.trainmode(ctx.st), ctx.batch_dev, ctx.para)
    @info "Dummy forward compile OK; executing compiled forward"
    xhat, _ = compiled_forward(ctx.model, ctx.ps, Lux.trainmode(ctx.st), ctx.batch_dev, ctx.para)
    println("dummy forward OK: xhat size=$(size(ctx.cdev(xhat)))")
    return compiled_forward
end

function run_dummy_loss_test()
    ctx = make_dummy_training_objects()
    @info "Running dummy loss compile test" nt=ctx.nt n_train=ctx.n_train batchsize=ctx.training_para.batchsize
    compiled_loss = @compile dummy_loss_objective(
        ctx.loss_fn, ctx.model, ctx.ps, Lux.trainmode(ctx.st), ctx.batch_dev)
    @info "Dummy loss compile OK; executing compiled loss"
    loss, _, stats = compiled_loss(ctx.loss_fn, ctx.model, ctx.ps, Lux.trainmode(ctx.st), ctx.batch_dev)
    metrics = ctx.cdev((; loss, recon_loss=stats.recon_loss, commit_loss=stats.commit_loss))
    println("dummy loss OK: loss=$(Float32(metrics.loss)), recon=$(Float32(metrics.recon_loss)), commit=$(Float32(metrics.commit_loss))")
    return compiled_loss
end

function run_dummy_grad_test()
    ctx = make_dummy_training_objects()
    @info "Running dummy grad compile test" nt=ctx.nt n_train=ctx.n_train batchsize=ctx.training_para.batchsize
    _, loss, stats, train_state = Training.compute_gradients(
        ctx.ad_backend, ctx.loss_fn, ctx.batch_dev, ctx.train_state)
    metrics = ctx.cdev((; loss, recon_loss=stats.recon_loss, commit_loss=stats.commit_loss))
    println("dummy grad OK: loss=$(Float32(metrics.loss)), recon=$(Float32(metrics.recon_loss)), commit=$(Float32(metrics.commit_loss))")
    return train_state.cache
end

function run_dummy_grad_objective_test(label::String, objective)
    ctx = make_dummy_training_objects()
    @info "Running dummy $(label) gradient test" nt=ctx.nt n_train=ctx.n_train batchsize=ctx.training_para.batchsize
    _, loss, stats, train_state = Training.compute_gradients(
        ctx.ad_backend, objective, ctx.batch_dev, ctx.train_state)
    metrics = ctx.cdev((; loss, recon_loss=stats.recon_loss, commit_loss=stats.commit_loss))
    println("dummy $(label) grad OK: loss=$(Float32(metrics.loss)), recon=$(Float32(metrics.recon_loss)), commit=$(Float32(metrics.commit_loss))")
    return train_state.cache
end

run_dummy_grad_encoder_test() = run_dummy_grad_objective_test("encoder", dummy_encoder_norm_objective)
run_dummy_grad_decoder_test() = run_dummy_grad_objective_test("decoder", dummy_decoder_recon_objective)
run_dummy_grad_recon_test() = run_dummy_grad_objective_test("recon-only", dummy_recon_only_objective)
run_dummy_grad_commit_test() = run_dummy_grad_objective_test("commit-only", dummy_commit_only_objective)

function run_dummy_grad_then_apply_test()
    ctx = make_dummy_training_objects()
    @info "Running dummy grad-then-apply test" nt=ctx.nt n_train=ctx.n_train batchsize=ctx.training_para.batchsize
    grads, loss, stats, train_state = Training.compute_gradients(
        ctx.ad_backend, ctx.loss_fn, ctx.batch_dev, ctx.train_state)
    metrics = ctx.cdev((; loss, recon_loss=stats.recon_loss, commit_loss=stats.commit_loss))
    @info "Dummy gradient OK; applying optimizer separately" loss=Float32(metrics.loss)
    train_state = Training.apply_gradients!(train_state, grads)
    println("dummy grad then apply OK: loss=$(Float32(metrics.loss)), recon=$(Float32(metrics.recon_loss)), commit=$(Float32(metrics.commit_loss))")
    return train_state.cache
end

function run_dummy_train_step_test()
    ctx = make_dummy_training_objects()
    @info "Running dummy train-step compile test" nt=ctx.nt n_train=ctx.n_train batchsize=ctx.training_para.batchsize
    _, loss, stats, train_state = Training.single_train_step!(
        ctx.ad_backend, ctx.loss_fn, ctx.batch_dev, ctx.train_state; return_gradients=Val(false))
    metrics = ctx.cdev((; loss, recon_loss=stats.recon_loss, commit_loss=stats.commit_loss))
    println("dummy train step OK: loss=$(Float32(metrics.loss)), recon=$(Float32(metrics.recon_loss)), commit=$(Float32(metrics.commit_loss))")
    return train_state.cache
end

function run_dummy_compile_test()
    problem = dummy_problem()
    @info "Running dummy compile test" nt=problem.nt n_train=problem.n_train batchsize=problem.training_para.batchsize
    compiled_model = compile_model(problem.nt, problem.n_train;
        vqvae_parameters=problem.vqvae_parameters,
        training_para=problem.training_para,
        seed=problem.seed,
        device=default_xdev(; force=true))
    @info "Dummy compile test complete" n_train=compiled_model.n_train
    println("dummy compile_model OK: n_train=$(compiled_model.n_train)")
    return compiled_model
end

function main(args)
    opts = parse_args(args)

    if opts["dummy-compile-test"]
        run_dummy_compile_test()
        return
    elseif opts["dummy-forward-test"]
        run_dummy_forward_test()
        return
    elseif opts["dummy-loss-test"]
        run_dummy_loss_test()
        return
    elseif opts["dummy-grad-test"]
        run_dummy_grad_test()
        return
    elseif opts["dummy-grad-encoder-test"]
        run_dummy_grad_encoder_test()
        return
    elseif opts["dummy-grad-decoder-test"]
        run_dummy_grad_decoder_test()
        return
    elseif opts["dummy-grad-recon-test"]
        run_dummy_grad_recon_test()
        return
    elseif opts["dummy-grad-commit-test"]
        run_dummy_grad_commit_test()
        return
    elseif opts["dummy-grad-then-apply-test"]
        run_dummy_grad_then_apply_test()
        return
    elseif opts["dummy-train-step-test"]
        run_dummy_train_step_test()
        return
    end

    data_dir   = opts["data-dir"]
    all_pairs  = list_station_pairs(data_dir)

    isempty(all_pairs) && error("No station pairs found in $(data_dir). Check --data-dir.")

    selected_pairs = if opts["pairs"] == "all"
        all_pairs
    else
        [(String(parts[1]), String(parts[2])) for parts in
            [split(pr, "-", limit=2) for pr in split(opts["pairs"], ",")]]
    end

    seeds_vec  = parse.(Int, strip.(split(opts["seeds"], ",")))
    nwindows = opts["nwindows"]
    nwindows > 0 || error("--nwindows must be positive.")
    nwindows >= opts["batchsize"] ||
        error("--nwindows ($(nwindows)) must be >= --batchsize ($(opts["batchsize"])).")
    n_max_val  = nwindows
    K_vec      = parse.(Int, strip.(split(opts["K"], ",")))
    ratios_vec = parse.(Int, strip.(split(opts["ratios"], ",")))

    period_min = opts["period-min"]
    period_max = opts["period-max"]
    dt         = opts["dt"]

    vqvae_parameters = (;
        d=opts["d"], K=K_vec, n_filters=opts["n-filters"], ratios=ratios_vec,
        n_residual_layers=opts["n-residual-layers"],
        entropy_weight=Float32(opts["entropy-weight"]),
        beta_commit=0.25f0, ema_decay=0.99f0,
        dilation_base=2, residual_kernel_size=3,
        enc_kernel_size=7, dec_kernel_size=7,
        use_bn=false, dead_threshold=50,
        codebook_exclusivity_weight=0.0f0,
        reconstruction_loss=:l1,
    )
    training_para = VQVAE_Training_Para(;
        batchsize=opts["batchsize"],
        nepoch=opts["nepoch"],
        initial_learning_rate=opts["lr"],
        autodiff_backend=Symbol(opts["autodiff-backend"]),
        Mnn_schedule=[(1, 128), (5, 256), (26, 256)],
        warmup_epochs=0,
        verbose=opts["verbose"],
        knn_search_chunk_size_fraction=0.25,
        index_refresh_every=2,
    )

    bp_filter = let
        rt = DSP.Bandpass(inv(period_max), inv(period_min))
        DSP.digitalfilter(rt, DSP.Butterworth(2); fs=inv(dt))
    end

    save_root = isempty(opts["save-dir"]) ?
        joinpath(data_dir, "SavedModels", "vqvae_v9_K=$(K_vec)") : opts["save-dir"]

    device = default_xdev(; force=true)

    @info "Loading first pair to determine nt for XLA compilation..."
    first_pair_data = load_pairs_data([selected_pairs[1]];
        filepath=data_dir, dt, period_min, period_max, n_max=n_max_val)
    nt      = size(first_pair_data[1].data.D_train, 1)
    n_train = nwindows

    @info "Compiling Reactant XLA graph (once for this session)..." nt n_train K=K_vec batchsize=opts["batchsize"]
    compiled_model = compile_model(nt, n_train;
        vqvae_parameters, training_para, seed=seeds_vec[1], device)

    train_selected_pairs_lazy(selected_pairs, compiled_model;
        seeds=seeds_vec,
        training_para,
        save_root,
        filepath=data_dir,
        dt, period_min, period_max,
        n_max=n_max_val,
        bp_filter,
        per_waveform_whitening_kernel_length=opts["whitening-kernel-length"],
        device,
    )
end

main(ARGS)
