using LinearAlgebra
using MLDataDevices
using Random
using Reactant

function cpu_ensemble_targets(X::AbstractMatrix{Float32}, nbrs::AbstractMatrix{<:Integer})
    T, _ = size(X)
    Mnn, B = size(nbrs)
    target = Matrix{Float32}(undef, T, B)
    @inbounds for b in 1:B
        col = view(target, :, b)
        fill!(col, 0f0)
        for k in 1:Mnn
            col .+= view(X, :, Int(nbrs[k, b]))
        end
        col ./= Float32(Mnn)
    end
    return target
end

struct EnsembleTargetBuilder
    Mnn::Int
end

function (builder::EnsembleTargetBuilder)(X, nbrs)
    B = size(nbrs, 2)
    target = X[:, nbrs[1, :]]
    for k in 2:builder.Mnn
        target = target .+ X[:, nbrs[k, :]]
    end
    return target ./ Float32(builder.Mnn)
end

ensemble_target_inference(builder::EnsembleTargetBuilder, X, nbrs) = builder(X, nbrs)

struct BatchGatherer end

function (g::BatchGatherer)(X, ids)
    return X[:, ids]
end

batch_gather_inference(g::BatchGatherer, X, ids) = g(X, ids)

function main()
    T = parse(Int, get(ENV, "T", "500"))
    N = parse(Int, get(ENV, "N", "4096"))
    B = parse(Int, get(ENV, "B", "512"))
    Mnn = parse(Int, get(ENV, "MNN", "10"))
    seed = parse(Int, get(ENV, "SEED", "1234"))

    rng = Xoshiro(seed)
    X = randn(rng, Float32, T, N)
    nbrs = rand(rng, Int32(1):Int32(N), Mnn, B)
    builder = EnsembleTargetBuilder(Mnn)

    cpu_time = @elapsed target_cpu = cpu_ensemble_targets(X, nbrs)
    println("cpu_time_s=", round(cpu_time; digits=4))

    dev = reactant_device(; force=true)
    X_dev = dev(X)
    nbrs_dev = dev(nbrs)

    eager_time = @elapsed target_dev = ensemble_target_inference(builder, X_dev, nbrs_dev)
    target_eager = Array(cpu_device()(target_dev))
    eager_err = maximum(abs.(target_eager .- target_cpu))
    println("reactant_eager_time_s=", round(eager_time; digits=4), " maxerr=", eager_err)

    compiled = try
        compile_time = @elapsed compiled_fn = @compile ensemble_target_inference(builder, X_dev, nbrs_dev)
        println("compile_ok=true compile_time_s=", round(compile_time; digits=4))
        compiled_fn
    catch err
        println("compile_ok=false")
        showerror(stdout, err)
        println()
        return
    end

    run_time = @elapsed target_compiled_dev = compiled(builder, X_dev, nbrs_dev)
    target_compiled = Array(cpu_device()(target_compiled_dev))
    compiled_err = maximum(abs.(target_compiled .- target_cpu))
    println("compiled_run_time_s=", round(run_time; digits=4), " maxerr=", compiled_err)

    nbatches = parse(Int, get(ENV, "NBATCHES", "32"))
    batch_ids = [rand(rng, Int32(1):Int32(N), B) for _ in 1:nbatches]

    transfer_time = @elapsed begin
        transfer_checksum = 0f0
        for ids in batch_ids
            xb_dev = dev(X[:, ids])
            xb = Array(cpu_device()(xb_dev))
            transfer_checksum += xb[1, 1]
        end
        println("cpu_slice_transfer_checksum=", transfer_checksum)
    end
    println("cpu_slice_transfer_loop_time_s=", round(transfer_time; digits=4))

    gatherer = BatchGatherer()
    sample_ids_dev = dev(batch_ids[1])
    gather_compiled = try
        gather_compile_time = @elapsed gather_fn =
            @compile batch_gather_inference(gatherer, X_dev, sample_ids_dev)
        println("batch_gather_compile_ok=true compile_time_s=", round(gather_compile_time; digits=4))
        gather_fn
    catch err
        println("batch_gather_compile_ok=false")
        showerror(stdout, err)
        println()
        return
    end

    device_gather_time = @elapsed begin
        gather_checksum = 0f0
        for ids in batch_ids
            xb_dev = gather_compiled(gatherer, X_dev, dev(ids))
            xb = Array(cpu_device()(xb_dev))
            gather_checksum += xb[1, 1]
        end
        println("device_gather_checksum=", gather_checksum)
    end
    println("device_gather_loop_time_s=", round(device_gather_time; digits=4))
end

main()
