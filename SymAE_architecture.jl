### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ cc11647d-1c56-4ceb-9677-703aca03c9f4
using Functors

# ╔═╡ d73472ff-9e09-45b0-8811-b7dd8d820358
using CUDA,
    Enzyme,
    Flux,
    MLUtils,
    Statistics,
    PlutoUI,
    LinearAlgebra,
    ProgressLogging,
    Optimisers,
    FFTW,
    DSP,
    PlutoPlotly,
    BlackBoxOptim,
    Metaheuristics,
    Random,
    ParameterSchedulers,
    Metalhead

# ╔═╡ 76dbf599-a9b3-459f-992b-16ab2f7b74f1
using PlutoLinks, PlutoHooks

# ╔═╡ 4a95997e-5c12-4658-9b8e-a5065328e1c1
using BenchmarkTools

# ╔═╡ 97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
TableOfContents(include_definitions=true)

# ╔═╡ 26fb86d5-c844-469a-aef5-ed3c2a9ba949
xpu = gpu

# ╔═╡ 631e5584-c4c3-4320-9a2d-477b7945c684
Base.@kwdef struct GroupConditioning
    mode::Symbol = :none
    real1 = nothing
    real2 = nothing
    labels = nothing
    nlabels::Int = 0
end

# ╔═╡ 96df0a0a-fd55-46b1-95af-067d02558380
has_conditioning(c::Union{Nothing,GroupConditioning}) = (c !== nothing) && (c.mode != :none)

# ╔═╡ f49d0e70-57e6-44a0-8e46-4d801c4b3f85
function _vec_condition_input(x)
    if (x isa Number)
        return Float32[x]
    elseif (x isa AbstractVector)
        return Float32.(collect(x))
    elseif (x isa AbstractMatrix)
        if (size(x, 2) == 1)
            return Float32.(vec(x))
        else
            return Float32.(vec(mean(x, dims=2)))
        end
    else
        error("Unsupported condition input type: $(typeof(x)). Use Number, Vector, or Matrix.")
    end
end

# ╔═╡ 065f859c-e7cf-4b42-bcc8-dfbf63104d26
function _label_to_condition(label, nlabels::Int)
    @assert nlabels > 1 "For label conditioning, nlabels must be > 1"
    ilabel = Int(label)
    @assert 1 <= ilabel <= nlabels "Label out of range. Got $ilabel, expected 1:$nlabels"
    return Float32.(Flux.onehot(ilabel, 1:nlabels))
end

# ╔═╡ 17f60265-fac3-411c-b1dd-49b284d23597
function get_group_condition_vector(conditioning::GroupConditioning, group_index::Int)
    if (conditioning.mode == :none)
        return Float32[]
    elseif (conditioning.mode == :real1)
        @assert conditioning.real1 !== nothing "conditioning.real1 must be set for :real1 mode"
        return _vec_condition_input(conditioning.real1[group_index])
    elseif (conditioning.mode == :real2)
        @assert conditioning.real1 !== nothing "conditioning.real1 must be set for :real2 mode"
        @assert conditioning.real2 !== nothing "conditioning.real2 must be set for :real2 mode"
        return vcat(
            _vec_condition_input(conditioning.real1[group_index]),
            _vec_condition_input(conditioning.real2[group_index]),
        )
    elseif (conditioning.mode == :label)
        @assert conditioning.labels !== nothing "conditioning.labels must be set for :label mode"
        return _label_to_condition(conditioning.labels[group_index], conditioning.nlabels)
    else
        error("Unsupported conditioning mode: $(conditioning.mode). Supported: :none, :real1, :real2, :label")
    end
end

# ╔═╡ 10429267-5808-4840-8678-f9dbf5b453c5
# fc228dea-21fc-4fcd-82a9-7ac3bc7ee722
"""
**Group-Based Batch View Generator**

Creates BatchView objects for each waveform group after shuffling instances and organizing them 
for sampling `ntau` waveforms at a time.

## Purpose in Group Training:
- Prepares each group for efficient random sampling during training
- Maintains group structure while enabling stochastic waveform selection
- Essential for coherent information learning within groups

## Arguments:
- `dvec`: Vector of waveform groups [group1, group2, ..., groupN]  
- `ntau`: Number of waveforms to sample per group (default: 20)

## Returns:
- Vector of BatchView objects, one per input group
- Each BatchView enables efficient sampling of `ntau` waveforms from its group

## Usage in Training Pipeline:
```julia
# Prepare data for group-based training
waveform_groups = [recordings_site1, recordings_site2, recordings_site3]
batch_views = get_batchviews(waveform_groups, ntau=20)
training_samples = get_sample(batch_views, batchsize=32)
```
"""
function get_batchviews(dvec, ntau=20)
    X = map(dvec) do d
        BatchView(shuffleobs(ObsView(d)), batchsize=ntau, partial=false)
    end
    return X
end

# ╔═╡ dc7e0a28-2739-44c2-9a44-c66079aaae17
DG = @ingredients("/mnt/NAS/EQData/SeismicAutoencoders/data_generators.jl")

# ╔═╡ 6affb3b3-9dc4-4bbc-a582-495fc1783a7a
activation = x -> leakyrelu(x, 0.1)
# activation = gelu

# ╔═╡ 5847ea08-43ca-4c6d-a694-0017d7396f60
# function get_conv_decoder(nt, pq)
#     @assert nt % 16 == 0 "Output length nt must be divisible by 16 for DCGAN generator."
#     latent_length = div(nt, 16)
# 	nc0 = 256
#     return Conv1DChain(Chain(
#         Dense(pq, latent_length * nc0, activation),
# 		x->reshape(x, latent_length, nc0, :),
#         ConvTranspose((4,), nc0 => div(nc0, 2); stride=2, pad=1),
# 		BatchNorm(div(nc0, 2)),
# 		activation,
# 		ConvTranspose((4,), div(nc0, 2) => div(nc0, 4); stride=2, pad=1),
# 		BatchNorm(div(nc0, 4)),
# 		activation,
# 		ConvTranspose((4,), div(nc0, 4) => div(nc0, 8); stride=2, pad=1),
# 		BatchNorm(div(nc0, 8)),
# 		activation,
# 		ConvTranspose((4,), div(nc0, 8) => 1; stride=2, pad=1),
#     ))
# end

# ╔═╡ 4fb77fae-61bb-4484-b733-82e1d1002371
begin
end

# ╔═╡ bf31f347-bc9a-4bf8-a086-99dba2f6fea0
begin
    Base.@kwdef struct ConvAE
        # Encoder parameters
        enc_kernels::Vector{Int} = [64, 32, 16, 4]
        enc_filters::Vector{Int} = [8, 16, 32, 64]
        enc_strides::Vector{Int} = [2, 2, 2, 2]
        use_bn::Bool = false

        # Decoder parameters
        dec_kernels::Vector{Int} = [8, 16, 32]
        dec_filters::Vector{Int} = [64, 48, 16, 1]
        dec_upstrides::Vector{Int} = [2, 2, 1]
    end
end

# ╔═╡ 91a25156-e121-4d53-a5a1-422f1230d235
Base.@kwdef struct SymAE_Para
    nt::Int
    p::Int
    k::Int = 1
    q::Int
    network_type = ConvAE()
    transformer::Symbol = :null
	transformer_k::Int = 1
    condition_dim::Int = 0
    condition_embed_dim::Int = 0
    seed = nothing
end

# ╔═╡ 0fbf59a9-74bc-479f-879c-3f72f7c76489
begin
    struct DenseAE
        flat_flag::Bool # linear decrease the width of dense layers, or not?
        nt_hidden::Int # used in the case of flat flag
        nlayers::Int # depth
    end
    function DenseAE()
        return DenseAE(false, 256, 4)
    end
end

# ╔═╡ 06a8d9e2-495c-4a25-8c23-527ba1b8e089
begin
    # DenseConvAE and ViTConvAE removed
end

# ╔═╡ b65ae9dc-dd50-4007-9894-cadef28e0552
function accumulate_Gaussians(μ, loginvvar)
    invvar = exp.(loginvvar)
    invvarG = sum(invvar, dims=ndims(μ) - 1)
    cμ = sum(μ .* invvar, dims=ndims(μ) - 1) ./ invvarG
    logσ = @. -0.5f0 * log(invvarG)
    return (; μ=cμ, logσ)
end

# ╔═╡ 8c54f11c-51b2-4500-9923-d3d38aa91e9b
"""
Weighted averaging without probabilty thresold
"""
function get_cluster_averages(model, D_to_get_prob; D_to_avg=D_to_get_prob)
    A = cat(map(D_to_get_prob, D_to_avg) do d, da
            λp = (softmax(model(d, Val(:coherent)).λlogits))
            cat(map(eachslice(λp, dims=1)) do l
                    mean(reshape(l, 1, :) .* da, dims=2)
                end..., dims=2)
        end..., dims=3)
    return A
end

# ╔═╡ 76c4f167-11b5-48a3-b3b3-4fe8fd9646c1
"""
Coherent code is distributed across all the wavefield instances
"""
function sample_q_decode(cμ, clogσ, nμ, nlogσ, nnoise, cnoise, decoder, temperature; cond_embedding=nothing)
    if (cnoise)
        cx = cμ + xpu(randn(Float32, size(clogσ))) .* exp.(clogσ)
    else
        cx = cμ
    end
    cx = dropdims(cx, dims=ndims(cμ) - 1)
    cx = Flux.stack(fill(cx, size(nμ, ndims(nμ) - 1)), dims=ndims(nμ) - 1)

    if (nnoise)
        nx = nμ + xpu(randn(Float32, size(nlogσ))) .* exp.(nlogσ)
    else
        nx = nμ
    end

    latent = cat(cx, nx, dims=1)
    xhat, xhat_logvar = decoder(latent, cond_embedding)
    return xhat, xhat_logvar
end

# ╔═╡ d74b7838-98c4-4356-8a0d-1a2388369788
# begin
#     # NOT USED
#     struct Model_Dropout{T1,T2,T3}
#         sencb::T1
#         nencb::T2
#         decb::T3
#     end
#     function (m::Model_Dropout)(x, nnoise, cnoise)
#         cx, cμ, clogσ = m.sencb(x)

#         nμ, nlogσ = m.nencb(x)

#         if (nnoise)
#             nx = dropout(nμ, 0.8)
#         else
#             nx = nμ
#         end
#         cx = dropdims(cx, dims=ndims(cμ) - 1)
#         cx = Flux.stack(fill(cx, size(nx, ndims(nx) - 1)), dims=ndims(nx) - 1)
#         xhat, xhat_logvar = m.decb(cat(cx, nx, dims=1))

#         return (;
#             Z=(; N=(; μ=nμ, logσ=nlogσ), C=(; μ=cμ, logσ=clogσ)),
#             X=(; xhat, xhat_logvar),
#         )
#     end
#     Flux.@layer Model_Dropout trainable = (sencb, nencb, decb)
# end

# ╔═╡ e80dd767-72ef-410b-b7a2-38e0545a5df3
function get_dense_transformer(nt; nt_out=1, nt_hidden=nt)
    transformer =
        Chain(
            Dense(nt, nt_hidden, activation),
            Dense(nt_hidden, nt_hidden, activation),
            Dense(nt_hidden, nt_out, init=zeros),
        ) |> xpu
    return transformer
end

# ╔═╡ 1f139691-e19f-41e5-8113-3cc00e8fe2b8
#===
        code for full spatial transformer (commented for now, as only Fourier shifts is used)
              #         shifts = cat(
              #             cat(
              #                 xpu(ones(Float32, 1, 1, 1, n2 * n3)),
              #                 reshape(shifts1, 1, 1, 1, n2 * n3),
              #                 dims = 2,
              #             ),
              #             xpu(zeros(Float32, 1, 2, 1, n2 * n3)),
              #             dims = 1,
              #         )

              #         inv_shifts = cat(
              #             cat(
              #                 xpu(ones(Float32, 1, 1, 1, n2 * n3)),
              #                 -1.0f0 * reshape(shifts1, 1, 1, 1, n2 * n3),
              #                 dims = 2,
              #             ),
              #             xpu(zeros(Float32, 1, 2, 1, n2 * n3)),
              #             dims = 1,
              #         )
              # return (; shifts, inv_shifts, shiftsμ = sμ)
        ===#

# ╔═╡ 825dda0d-6472-405c-b149-5c4d2202963f
# """
# code for spatial transformer (commented for now, as only Fourier shifts is used)
# shift traces with localization net
# - input_traces have size (nt, nr)
# - localization_net returns the time shifts (nr)
# - uses global variable sampling_grid
# """
# function shift_traces_grid_sample(input_traces, shifts, sampling_grid)
#     S = Flux.stack(fill(sampling_grid, size(shifts, 4)), dims=4)
#     grids = batched_mul(shifts, S)
#     input_traces1 =
#         reshape(input_traces, size(input_traces, 1), 1, 1, prod(size(input_traces)[2:end]))
#     output_traces = grid_sample(input_traces1, grids; padding_mode=:zeros)
#     return reshape(output_traces, size(input_traces))
# end

# ╔═╡ 84d56fa3-50de-48ad-8e07-dfaecc1cfdf3
function fouriertransform1D(𝐱::AbstractArray)
    return fft(𝐱, 1) # perform fft along first dimension
end

# ╔═╡ 86a120ae-8865-4e99-a028-f567f3c1bbad
function inversefouriertransform1D(𝐱_fft::AbstractArray)
    return real(ifft(𝐱_fft, 1)) # perform ifft along first dimension
end

# ╔═╡ a48d23b6-86d6-4232-a1b9-300e65b264ff
begin
    """
    Select only μ and logσ from coherent codes
    Depending on kopt, select respective chunk of the coherent code
    """
    function batch_coherent_codes(vec::Vector{<:NamedTuple})
        merged = (; μ=(getfield.(vec, :μ)..., dims=3), logσ=cat(getfield.(vec, :logσ)..., dims=3))
        return merged
    end
    function batch_coherent_codes(v::NamedTuple)
        return (; μ=getfield(v, :μ), logσ=getfield(v, :logσ))
    end
    function apply_λlogits_cond(vec::Vector{<:NamedTuple}, λlogits_cond)
        merged = batch_coherent_codes(vec)
        # select kopt chunk
        return map(merged) do m
            m_chunks = chunk(m, size(λlogits_cond, 1), dims=1)
            λ_chunks = chunk(λlogits_cond, size(λlogits_cond, 1), dims=1)
            cx = sum(map(m_chunks, λ_chunks) do c, l
                c .* l
            end)
            return cx
        end
    end
    function apply_λlogits_cond(C::NamedTuple, λlogits_cond)
        merged = (; μ=C.μ, logσ=C.logσ)
        # select kopt chunk
        return map(merged) do m
            m_chunks = chunk(m, size(λlogits_cond, 1), dims=1)
            λ_chunks = chunk(λlogits_cond, size(λlogits_cond, 1), dims=1)
            cx = sum(map(m_chunks, λ_chunks) do c, l
                c .* l
            end)
            return cx
        end
    end
end

# ╔═╡ a85b6958-d894-45cc-86c6-933fba757e1c
# function loss_kl_Qc_accumulated(nuisance_codes, optimal_nuisance_model, C, alpha)
#     copyto!(optimal_nuisance_model.nuisance_codes, nuisance_codes)
#     return loss_kl_Qc_accumulated(optimal_nuisance_model, C, alpha)
# end

# ╔═╡ 74c8e79f-b1c8-484b-86bf-65c2a2831b53
# """
# """
# function optimize_nuisance_code_black_box_optim(d, para, model; MaxTime=10.0)
#     coherent_codes = model(d, Val(:coherent))
#     optimal_nuisance_model = get_Model_Optimal_Nuisance(para, model, init=randobs(d))
#     loss(x) = Float64(
#         loss_kl_Qc_accumulated(
#             x,
#             optimal_nuisance_model,
#             coherent_codes
#         )[1],
#     )
#     result = bboptimize(
#         loss;
#         NumDimensions=para.q,
#         SearchRange=(-10.0, 10.0),
#         MaxTime=MaxTime,
#         MaxSteps=-1,
#         TraceMode=:silent,
#         #inDeltaFitnessTolerance = 0.01
#     )
#     return optimal_nuisance_model(
#         best_candidate(result),
#         coherent_codes.cμ,
#         coherent_codes.clogσ,
#     )[1],
#     result
# end

# ╔═╡ 073c77d1-7598-45d3-858d-9222f2e4c590
# """
# """
# function optimize_nuisance_code_metaheuristics(d, para, model; MaxTime=10)
#     coherent_codes = model([d], Val())[1]

#     _, _, ideal_seismogram_ids = kl_Qc_accumulated(para, model, [d], 255, 1)
#     ideal_seismograms = mapreduce(hcat, ideal_seismogram_ids) do ideal_seismogram_id
#         d[:, ideal_seismogram_id]
#     end

#     optimal_nuisance_model =
#         get_Model_Optimal_Nuisance(para, model, init=ideal_seismograms)
#     function loss(X)
#         if (ndims(X) == 2)
#             x = permutedims(X, (2, 1))
#         elseif (ndims(X) == 3)
#             x = permutedims(X, (2, 1, 3))
#         else
#             x = X
#         end
#         L = loss_kl_Qc_accumulated(
#             x,
#             optimal_nuisance_model,
#             coherent_codes
#         )[2]
#         if (ndims(X) == 1)
#             return Float64(L[1])
#         else
#             return Float64.(L[1:size(X, 1)])
#         end
#     end
#     options = Options(
#         f_calls_limit=0,
#         time_limit=MaxTime,
#         parallel_evaluation=true,
#         iterations=1000,
#     )
#     bounds = boxconstraints(lb=-Inf * ones(para.q), ub=Inf * ones(para.q))
#     algo = GA(
#         options=options,
#         mutation=Metaheuristics.PolynomialMutation(; bounds),
#         crossover=SBX(; bounds),
#         environmental_selection=GenerationalReplacement(),
#     )
#     # algo = ECA(options = options)

#     initial_nuisances =
#         Float64.(
#             cpu(
#                 dropdims(
#                     permutedims(optimal_nuisance_model.nuisance_codes, (2, 1, 3)),
#                     dims=3,
#                 ),
#             )
#         )
#     set_user_solutions!(algo, initial_nuisances[1:100, :], loss)
#     result = optimize(loss, bounds, algo)

#     return optimal_nuisance_model(
#         minimizer(result),
#         coherent_codes.cμ,
#         coherent_codes.clogσ,
#     )[1],
#     result
# end

# ╔═╡ 4ec8578c-a202-4a3d-9425-60bf623a3a02
"""
    # x: point where log-pdf is evaluated (vector)
    # mean: mean vector of the Gaussian
    # log_std: log of standard deviations (vector)
"""
function neg_logpdf_gaussian(x, mean, log_std)
    log_det = sum(log_std, dims=1)
    squared_term = sum(((x .- mean) ./ exp.(log_std)) .^ 2, dims=1)

    neg_log_pdf = @. log_det + 0.5 * squared_term
    return neg_log_pdf
end

# ╔═╡ 41b5d143-6a0f-4f4c-8e62-8483d9e0d5f6
"""
KL divergence between two categorical distributions
# Arguments
- `logits`: logits unnormalized
- `prior`: prior 
"""
function kl_divergence_categorial_distributions(logits, prior)
    λ = Flux.softmax(logits; dims=1)
    λ = clamp.(λ, 0.0001f0, 1.0f0) # avoid log(0) in posterior
    prior = clamp.(prior, 0.0001f0, 1.0f0)  # avoid log(0) in posterior
    kl = @. (λ * (log(λ) - log(prior)))
    return sum(kl)
end

# ╔═╡ e57556e8-b287-4bc5-9ea1-4f68948eccf2
"""
    kl_divergence_multivariate_gaussians(mean1, log_std1, mean2, log_std2)

Compute the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions.

# Arguments
- `mean1`: The mean of the first Gaussian distribution.
- `log_std1`: The log standard deviation of the first Gaussian distribution.
- `mean2`: The mean of the second Gaussian distribution.
- `log_std2`: The log standard deviation of the second Gaussian distribution.

# Returns
The KL divergence between the two Gaussian distributions.
"""
function kl_divergence_multivariate_gaussians(mean1, log_std1, mean2, log_std2)
    kl1 = @. (exp(2.0f0 * log_std1) / exp(2.0f0 * log_std2) + abs2(mean1 - mean2) / exp(2.0f0 * log_std2) - 1.0f0 - 2.0f0 * log_std1 + 2.0f0 * log_std2)
    kl = 0.5f0 .* sum(kl1, dims=1)
    return kl
end

# ╔═╡ e46630eb-5e29-4c5e-b792-0f7ca0af0ff0
"""
    hellinger_distance_multivariate_gaussians(mean1, log_sigma1, mean2, log_sigma2)
The Hellinger distance between two multivariate Gaussian distributions
"""
function hellinger_distance_multivariate_gaussians(meanX, logσ1, meanY, logσ2)
    σ1 = exp.(2.0f0 * logσ1)
    σ2 = exp.(2.0f0 * logσ2)

    detX = prod(σ1)
    detY = prod(σ2)
    covXY = (σ1 .+ σ2) / 2.0f0
    detXY = prod(covXY)

    # Compute the exponent term
    diff = meanX .- meanY
    covXY_inverted = 1 ./ covXY  # Since covXY is diagonal, inversion is element-wise
    exponent_term = exp(-0.125 * sum((diff .^ 2) .* covXY_inverted))

    # Compute the Hellinger distance
    dist = 1.0 - (fourthroot(detX) * fourthroot(detY) / sqrt(detXY)) * exponent_term

    return dist
end

# ╔═╡ 361ead1b-3fe3-4ae2-b018-2c6904d0f889
"""
    kl_divergence_multivariate_gaussian(mean1, log_std1)

Compute the Kullback-Leibler (KL) divergence between two multivariate Gaussian distributions.

# Arguments
- `mean1`: The mean of the first Gaussian distribution.
- `log_std1`: The log standard deviation of the first Gaussian distribution.

# Returns
The KL divergence between the two Gaussian distributions.
"""
function kl_divergence_multivariate_gaussian(mean1, log_std1)
    return 0.5f0 * sum(@. (exp(2.0f0 * log_std1) + abs2(mean1) - 1.0f0 - 2.0f0 * log_std1))
end

# ╔═╡ d966211f-012e-45f2-b9ae-abf599666edf
begin
    function get_kl(μ, logσ, beta)
        return beta * kl_divergence_multivariate_gaussian(μ, logσ)
    end
    function get_kl(μ, logσ, beta, betadummy)
        return get_kl(μ, logσ, beta)
    end
end

# ╔═╡ 6556c7c7-9934-49be-8f8a-423a0d16e57e
function kl_divergence_multivariate_gaussians(mean1, log_std1, mean2)
    # std2 = 1.0  →  log_std2 = 0.0
    # exp(2*log_std2) = 1

    # σ1^2
    var1 = @. exp(2f0 * log_std1)

    # (μ1 − μ2)^2
    diff2 = @. abs2(mean1 - mean2)

    # KL per dimension
    kl1 = @. var1 + diff2 - 1f0 - 2f0*log_std1

    # Sum over dimensions, keep batch dimension intact
    return 0.5f0 .* sum(kl1)
end

# ╔═╡ afb6c675-0ef8-445c-a204-795e300c8589
"""
Return KL divergence between accumulated coherent information and coherent information in the instances generated using optimal nuisance model
"""
function loss_kl_Qc_accumulated(optimal_nuisance_model, coherent_codes::Vector{T}, alpha; p=1, Bref=1f0, Cref=1f0) where {T}

    virtualdata, cμ_all, clogσ_all, = optimal_nuisance_model(coherent_codes)
    coherent_codes_batched = apply_λlogits_cond(batch_coherent_codes(coherent_codes), optimal_nuisance_model.λlogits_cond)
    C_all_logits = apply_λlogits_cond((; μ=cμ_all, logσ=clogσ_all), optimal_nuisance_model.λlogits_cond)

    cμ_in = coherent_codes_batched.μ
    cμ_in = dropdims(cμ_in, dims=ndims(cμ_in) - 1)
    cμ_in = Flux.stack(fill(cμ_in, size(cμ_all, ndims(cμ_all) - 1)), dims=ndims(cμ_all) - 1)

    clogσ_in = coherent_codes_batched.logσ
    clogσ_in = dropdims(clogσ_in, dims=ndims(clogσ_in) - 1)
    clogσ_in = Flux.stack(fill(clogσ_in, size(clogσ_all, ndims(clogσ_all) - 1)), dims=ndims(clogσ_all) - 1)

    B = kl_divergence_multivariate_gaussians(cμ_in, clogσ_in, C_all_logits.μ, C_all_logits.logσ)
    # B = neg_logpdf_gaussian(cμ_in, cμ_all, clogσ_all)
    # virtualdata1 = normalise(virtualdata, dims=1)
    virtualdata1 = virtualdata
    return (; loss=sum(B) / Bref + alpha * norm(virtualdata1, p) / Cref, kl=B, L=norm(virtualdata1, p), virtualdata)
end

# ╔═╡ 6e7c70c8-cffe-4b30-85b4-5942fbb60da8
"""
    js_divergence_multivariate_gaussians(mean1, log_std1, mean2, log_std2)

Compute the Jensen-Shannon (JS) divergence between two multivariate Gaussian distributions.

# Arguments
- `mean1`: The mean of the first Gaussian distribution.
- `log_std1`: The log standard deviation of the first Gaussian distribution.
- `mean2`: The mean of the second Gaussian distribution.
- `log_std2`: The log standard deviation of the second Gaussian distribution.

# Returns
The JS divergence between the two Gaussian distributions.
"""
function js_divergence_multivariate_gaussians(mean1, log_std1, mean2, log_std2)
    js =
        0.5 * (
            kl_divergence_multivariate_gaussians(mean1, log_std1, mean2, log_std2) +
            kl_divergence_multivariate_gaussians(mean2, log_std2, mean1, log_std1)
        )
    return js
end

# ╔═╡ 4e527961-5734-4294-9f3c-e2aa8314eef6
"""
Compute the Bhattacharyya distance between two Gaussian distributions.

Parameters:
- mean1: The mean of the first Gaussian distribution.
- log_std1: The log standard deviation of the first Gaussian distribution.
- mean2: The mean of the second Gaussian distribution.
- log_std2: The log standard deviation of the second Gaussian distribution.

Returns:
- bd: The Bhattacharyya distance between the two Gaussian distributions.
"""
function bhattacharyya_distance(cμ1, clogσ1, cμ2, clogσ2)
    # Bhattacharya Distance
    cvar1 = @. exp(2.0f0 * clogσ1)
    cvar2 = @. exp(2.0f0 * clogσ2)
    Dvar = 0.5f0 * (sum(@. 2.0f0 * clogσ1) + sum(@. 2.0f0 * clogσ2))
    cvarmean = @. 0.5f0 * (cvar1 + cvar2)
    Nvar = sum(@. log.(cvarmean))
    return sum(@. abs2(cμ1 - cμ2) / cvarmean) / 8.0f0 + 0.5f0 * (Nvar - Dvar)

    # # Convert log standard deviations to standard deviations
    # std1 = exp.(log_std1)
    # std2 = exp.(log_std2)

    # # Compute the Bhattacharyya distance
    # bd = 0.5 * sum(@. ((mean1 - mean2)^2 / (std1^2 + std2^2) + log((std1 * std2) / sqrt(std1^2 + std2^2))))
    # return bd
end

# ╔═╡ c5d5be4d-882e-4c34-ae8f-1a40aa4cf215
# ╠═╡ show_logs = false
# ╠═╡ disabled = true
#=╠═╡
begin
    Xv = xpu([randn(100, 10), randn(100, 10)])
    ref = xpu(randn(100))
    ref2 = xpu(randn(100, 2))
    ref3 = xpu(randn(100, 10, 2))

end
  ╠═╡ =#

# ╔═╡ b9b5f42d-4ec1-43b5-9484-d3ec47dea61a
"""
    average_leave_one_out_correlation(X::AbstractMatrix)

Compute the average Pearson correlation between each column of `X` and the mean of the remaining columns.

# Arguments
- `X`: A matrix of shape (m, n), where columns represent features or signals.

# Returns
- The average Pearson correlation coefficient.
"""
function average_leave_one_out_correlation(X::AbstractMatrix)
    n = size(X, 2)
    colsum = sum(X, dims=2)
    total_corr = 0.0

    for (i, xi) in enumerate(eachcol(X))
        mean_rest = (colsum .- xi) ./ (n - 1)

        a = vec(xi)
        b = vec(mean_rest)

        ma, mb = mean(a), mean(b)
        numerator = sum((a .- ma) .* (b .- mb))
        denominator = sqrt(sum((a .- ma) .^ 2) * sum((b .- mb) .^ 2))
        total_corr += numerator / denominator
    end

    return total_corr / n
end

# ╔═╡ ae75eee7-ed34-4a2a-8aa3-08a06f504d36
function average_leave_one_out_correlation(X::Vector)
    return map(X) do x
        average_leave_one_out_correlation(x)
    end
end

# ╔═╡ eafe181e-19e9-409e-ad1d-ce859cf0e672
Base.@kwdef struct Training_Para
    ntau::Int = 20
    beta = (; N=[1f0], C=[1f0, 1f0])
    gamma::Float32 = 1f2
    temperature::Float32 = 1f0
    batchsize::Int = 32
    nsteps::Int = 100
    nprint::Int = 1
    nepoch::Int = 10
    initial_learning_rate::Float64 = 0.001
end

# ╔═╡ 0f0fe0e8-a239-474b-b9af-54507cb968aa
"""
use virtual data where source is used from x and nuisance from xaug
"""
function generate_virtual_data(x, xaug, nnoise, cnoise, model)
    output = model(x, xaug, nothing, nnoise, cnoise, 0f0)
    return output.X.xhat
end

# ╔═╡ 7531c2eb-5505-4ad2-9f79-111bed3bef74
"""
    redatum(d1, d2, model, nnoise=false, cnoise=false)

Redatum two input data `d1` and `d2` using the given `model`. The `model` should be a function that takes input data and returns the redatumed output.

## Arguments
- `d1`: The first input data.
- `d2`: The second input data.
- `model`: symae model
- `nnoise`: A boolean indicating whether to sample Q(n|x) or use its mean
- `cnoise`: A boolean indicating whether to sample Q(c|x) or use its mean

## Returns
A named tuple containing the redatumed data `d1hat`, `d2hat`, `d12hat`, and `d21hat`.
"""
function redatum(d1, d2, model, nnoise=false, cnoise=false; condition=nothing, temperature=0f0)
    d1hat = model(d1, condition, nnoise, cnoise, temperature).X.xhat
    d2hat = model(d2, condition, nnoise, cnoise, temperature).X.xhat
    d12hat = model(d1, d2, condition, nnoise, cnoise, temperature).X.xhat
    d21hat = model(d2, d1, condition, nnoise, cnoise, temperature).X.xhat
    return map(cpu, (; d1=d1, d2=d2, d1hat, d2hat, d12hat, d21hat))
end

# ╔═╡ c9f4b3e6-6691-4e7f-86c7-dcacdde924a5
"""
Plot training and test loss history with publication-quality styling.

Arguments:
- loss_history: struct containing loss arrays
- title: main plot title (default: "")

Returns: PlutoPlotly plot with MSE and other losses in subplots
"""
function plot_loss_history(loss_history; title="")
    # Publication-quality color palette
    colors_mse = ["#E74C3C", "#3498DB"]
    colors_other = ["#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C", "#E67E22", "#95A5A6"]

    # MSE traces
    trace_mse = [
        PlutoPlotly.scatter(
            y=cpu(getfield(loss_history, label)),
            mode="lines",
            name=string(label),
            line=attr(color=colors_mse[i], width=2.5)
        ) for (i, label) in enumerate([:train_mse, :test_mse])
    ]

    # Other loss traces
    other_labels = [:train_neg_llh, :test_neg_llh, :train_kl, :test_kl, :train_neg_elbo, :test_neg_elbo]
    trace_other = [
        PlutoPlotly.scatter(
            y=cpu(getfield(loss_history, label)),
            mode="lines",
            name=string(label),
            line=attr(color=colors_other[i], width=2.5)
        ) for (i, label) in enumerate(other_labels)
    ]

    # Common layout settings
    common_layout = attr(
        width=600,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showgrid=true,
        gridcolor="rgba(128,128,128,0.2)",
        gridwidth=1,
        showline=true,
        linewidth=1.5,
        linecolor="black",
        mirror=true
    )

    # MSE subplot layout (log scale)
    layout_mse = Layout(
        title=attr(text="MSE Loss", font=attr(size=23, family="Computer Modern, Latin Modern Math, serif")),
        height=600,
        width=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            font=attr(size=18, family="Computer Modern, Latin Modern Math, serif"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        xaxis=merge(common_layout, attr(
            title=attr(text="Epoch", font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
            tickfont=attr(size=18, family="Computer Modern, Latin Modern Math, serif")
        )),
        yaxis=merge(common_layout, attr(
            title=attr(text="MSE", font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
            type="log",
            tickfont=attr(size=18, family="Computer Modern, Latin Modern Math, serif")
        )),
        margin=attr(l=80, r=40, t=60, b=70)
    )

    # Other losses subplot layout (linear scale)
    layout_other = Layout(
        title=attr(text="Other Losses", font=attr(size=23, family="Computer Modern, Latin Modern Math, serif")),
        height=600,
        width=400,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=attr(
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            font=attr(size=18, family="Computer Modern, Latin Modern Math, serif"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        xaxis=merge(common_layout, attr(
            title=attr(text="Epoch", font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
            tickfont=attr(size=18, family="Computer Modern, Latin Modern Math, serif")
        )),
        yaxis=merge(common_layout, attr(
            title=attr(text="Loss", font=attr(size=20, family="Computer Modern, Latin Modern Math, serif")),
            tickfont=attr(size=18, family="Computer Modern, Latin Modern Math, serif")
        )),
        margin=attr(l=80, r=40, t=60, b=70)
    )

    # Create subplots side by side
    p1 = PlutoPlotly.plot(trace_mse, layout_mse)
    p2 = PlutoPlotly.plot(trace_other, layout_other)

    # Combine plots
    p = [p1; p2]

    # Add overall title if provided
    if !isempty(title)
        relayout!(p, title_text=title,
            title_font=attr(size=27, family="Computer Modern, Latin Modern Math, serif"))
    end

    return p
end

# ╔═╡ 1a8619c6-4c19-4f55-a968-d18a4f1016e7
function taper(x)
    w = (cat(tukey(size(x, 1), 0.1), dims=ndims(x)))
    return w .* x
end

# ╔═╡ 80f77b52-84e0-4664-8aa0-3d79fded40de
"""
Instead of cat(x, dims=3)
"""
add_dim3_reshape(::Nothing) = nothing

# ╔═╡ 6ba143e2-50df-441a-8f38-3ea8d9edd4d8
function add_dim3_reshape(x)
	x === nothing && return nothing
	nd = ndims(x)
    if nd == 3
        return x
    elseif nd == 2
        return reshape(x, size(x,1), size(x,2), 1)
    elseif nd == 1
        return reshape(x, size(x,1), 1, 1)
    else
        error("Input has more than 3 dimensions")
    end
end

# ╔═╡ 96aebf7d-2112-4a4d-9993-6f53f40ffca5
function generate_dense_chain(
    nt,
    p,
    nlayers,
    output_activation,
    flat_flag=false,
    ;
    nt_hidden=nt,
)
    if (flat_flag)
        lp = fill(nt_hidden, nlayers + 1)
    else
        lp = floor.(Int, LinRange(nt, p, nlayers + 1))
    end
    layers = []

    # make input 3D
    push!(layers, x -> add_dim3_reshape(x))

    # Add input layer
    push!(layers, Dense(nt, lp[2], activation))

    # Add hidden layers
    for il = 2:nlayers-1
        push!(layers, Dense(lp[il], lp[il+1], activation))
    end

    # Add output layer
    push!(layers, Dense(lp[nlayers], p, output_activation))

    return layers
end

# ╔═╡ 0dd8d418-a664-4ecd-ac90-bab97861f9f0
function get_symae(t::DenseAE, nt, p, q, k, condition_embed_dim=0)
    P = p * k
    nt_enc = nt + condition_embed_dim
    senc =
        Chain(
            generate_dense_chain(
                nt_enc,
                P,
                t.nlayers,
                activation,
                t.flat_flag,
                nt_hidden=t.nt_hidden,
            )...,
        ) |> xpu

    # instance means
    senc_μ = Chain(Dense(P, P)) |> xpu

    # inverse variances
    senc_loginvvar = Chain(Dense(P, P)) |> xpu
    senc_λlogits = xpu(Chain(Dense(P + condition_embed_dim, P, activation), Dense(P, 2 * P, activation), Dense(2 * P, 2 * P, activation), Dense(2 * P, k))) # only used when k>1

    nenc =
        Chain(
            generate_dense_chain(
                nt_enc,
                q,
                t.nlayers,
                activation,
                t.flat_flag,
                nt_hidden=t.nt_hidden,
            )...,
        ) |> xpu

    # produce logσ for the nuisance encoder
    nenc_μ = Chain(Dense(q, q)) |> xpu
    nenc_logσ = Chain(Dense(q, q)) |> xpu

    dec =
        Chain(
            generate_dense_chain(
                p + q + condition_embed_dim,
                nt,
                t.nlayers,
                identity,
                t.flat_flag,
                nt_hidden=t.nt_hidden,
            )...,
        ) |> xpu

    dec_logvar = xpu(cat(1.0f0, dims=3))

    return (; senc, senc_μ, senc_loginvvar, senc_λlogits, nenc, nenc_μ, nenc_logσ, dec, dec_logvar)
end

# ╔═╡ 32baae21-b5fa-4391-9066-23436a5a2b1d
begin
    struct Conv1DChain
        chain::Chain
    end
    Flux.@layer Conv1DChain trainable = (chain)
    (m::Conv1DChain)(::Nothing) = nothing
    function (m::Conv1DChain)(x)
        x = add_dim3_reshape(x)
        n1, n2, n3 = size(x)
        X = reshape(x, :, 1, n2 * n3)
        X = m.chain(X)
        X = reshape(X, :, n2, n3)
        return X
    end
end

# ╔═╡ 237ad98e-75db-41fc-b378-b895facdd8d9
begin
    struct BroadcastDec{T1,T2}
        chain::T1
        logvar::T2
    end
    Flux.@layer BroadcastDec trainable = (chain, logvar)
    function (m::BroadcastDec)(x, cond_embedding=nothing)
        Xμ = _maybe_forward(m.chain, x, cond_embedding)
        return Xμ, m.logvar
    end

    struct NoConditionProjector end
    Flux.@layer NoConditionProjector
    (m::NoConditionProjector)(::Any) = nothing

    function _get_condition_embedding(cond_proj, condition_dim::Int, x, condition)
        if (condition_dim <= 0) || (x === nothing)
            return nothing
        end
        batch_size = size(x, ndims(x))
        cond = if (condition === nothing)
            zeros(Float32, condition_dim, batch_size)
        elseif (ndims(condition) == 1)
            reshape(Float32.(condition), :, 1)
        else
            Float32.(condition)
        end
        if (size(cond, 2) == 1) && (batch_size > 1)
            cond = reshape(Flux.stack(fill(cond, batch_size), dims=2), size(cond, 1), batch_size)
        end
        @assert size(cond, 1) == condition_dim "Condition feature size mismatch. Expected $condition_dim, got $(size(cond, 1))."
        @assert size(cond, 2) == batch_size "Condition batch size mismatch. Expected $batch_size, got $(size(cond, 2))."
        return cond_proj(xpu(cond))
    end

    function _expand_condition_for_decoder(cond_embedding, target_shape)
        if (cond_embedding === nothing)
            return nothing
        end
        ntau = size(target_shape, ndims(target_shape) - 1)
        return Flux.stack(fill(cond_embedding, ntau), dims=2)
    end

    function _expand_condition_for_encoder(cond_embedding, x)
        if (cond_embedding === nothing)
            return nothing
        end
        x3 = add_dim3_reshape(x)
        ntau = size(x3, 2)
        return Flux.stack(fill(cond_embedding, ntau), dims=2)
    end

    function _augment_encoder_input(x, cond_embedding)
        x3 = add_dim3_reshape(x)
        cenc = _expand_condition_for_encoder(cond_embedding, x3)
        if (cenc === nothing)
            return x3
        end
        return cat(x3, cenc, dims=1)
    end
end

# ╔═╡ a5302fa2-4f67-4ed6-96ce-dda78a160ffe
begin
    abstract type AbstractParallel end

    _maybe_forward(layer::AbstractParallel, x, ys...) = layer(x, ys...)
    _maybe_forward(layer::Parallel, x, ys...) = layer(x, ys...)
    _maybe_forward(layer, x, ys...) = layer(x)

    struct ConditionalChain{T<:Union{Tuple,NamedTuple}} <: AbstractParallel
        layers::T
    end
    Flux.@layer ConditionalChain

    ConditionalChain(xs...) = ConditionalChain(xs)
    function ConditionalChain(; kw...)
        :layers in keys(kw) && throw(ArgumentError("a ConditionalChain cannot have a named layer called `layers`"))
        isempty(kw) && return ConditionalChain(())
        ConditionalChain(values(kw))
    end

    Flux.@forward ConditionalChain.layers Base.getindex, Base.length, Base.first, Base.last,
    Base.iterate, Base.lastindex, Base.keys, Base.firstindex

    Base.getindex(c::ConditionalChain, i::AbstractArray) = ConditionalChain(c.layers[i]...)

    function (c::ConditionalChain)(x, ys...)
        for layer in c.layers
            x = _maybe_forward(layer, x, ys...)
        end
        return x
    end

    struct ConditionConcat <: AbstractParallel end
    Flux.@layer ConditionConcat
    (m::ConditionConcat)(x, cond_embedding=nothing) = _augment_encoder_input(x, cond_embedding)

end

# ╔═╡ 461f0505-2230-4b84-b6c6-1a9730808437
md"""# Symmetric Variational Autoencoders"""

# ╔═╡ 3983e7d0-9ad0-11f0-0a96-7d2d98772fd2
md"""
## SymVAE Overview: Group-Based Training for Coherent Information Extraction

### Core Training Philosophy
SymVAE employs a sophisticated group-based training strategy designed to achieve two fundamental objectives when working with collections of seismic waveforms:

#### **Objective 1: Coherent Information Extraction Within Groups**
The primary goal is to extract coherent information that represents common source characteristics shared across multiple waveforms within each group. This coherent information captures:
- **Source signature**: The fundamental waveform characteristics originating from the seismic source
- **Geological structure**: Common subsurface properties that affect all waveforms in the group
- **Acquisition geometry**: Shared recording configuration effects

#### **Objective 2: Subgroup Identification and Categorization**
The secondary goal is to automatically identify and categorize distinct subgroups within each provided group based on coherent information similarity. This enables:
- **Automatic clustering**: Discovery of natural waveform families within larger datasets
- **Quality control**: Identification of outliers or anomalous waveforms
- **Geological interpretation**: Recognition of different propagation regimes or source types

### Training Methodology

#### **Multi-Group Training Strategy**
```
Group 1: [w₁₁, w₁₂, ..., w₁ₙ] → Coherent Code C₁ + Nuisance Codes {N₁₁, N₁₂, ..., N₁ₙ}
Group 2: [w₂₁, w₂₂, ..., w₂ₘ] → Coherent Code C₂ + Nuisance Codes {N₂₁, N₂₂, ..., N₂ₘ}
   ⋮
Group G: [wG₁, wG₂, ..., wGₖ] → Coherent Code CG + Nuisance Codes {NG₁, NG₂, ..., NGₖ}
```

#### **Key Training Principles**

1. **Batch Composition Strategy**
   - Each training batch contains `ntau` waveforms sampled from the same group/state
   - Multiple groups are processed in parallel during training
   - Critical requirement: `batchsize > 1` to enable proper disentanglement

2. **Information Decomposition**
   ```julia
   # For each group of waveforms
   coherent_info = shared_across_all_waveforms_in_group
   nuisance_info = instance_specific_propagation_effects
   reconstruction = decoder(coherent_info ⊕ nuisance_info)
   ```

3. **Variational Learning Framework**
   - **Encoder Networks**: Learn probabilistic representations q(c|x) and q(n|x)
   - **Decoder Network**: Reconstructs waveforms from concatenated codes
   - **KL Regularization**: Ensures proper statistical behavior of learned codes

#### **Coherent Information Extraction Process**

1. **Training Phase**
   ```julia
   # Train on multiple groups simultaneously
   for group in waveform_groups
       coherent_code = extract_common_features(group)
       nuisance_codes = extract_individual_variations(group)
       loss += reconstruction_loss + kl_coherent + kl_nuisance
   end
   ```

2. **Coherent Extraction Phase**
   ```julia
   # Post-training coherent information recovery
   optimal_nuisance = optimize_nuisance_codes(target_group)
   coherent_signal = decode(coherent_code, optimal_nuisance)
   ```

#### **Subgroup Identification Mechanism**

When `k > 1` (multi-category coherent codes):

1. **Categorical Coherent Encoding**
   - Coherent code is divided into `k` categories
   - Gumbel-Softmax selection determines active category per waveform
   - Each category represents a distinct subgroup type

2. **Automatic Subgroup Discovery**
   ```julia
   # For each waveform, determine most likely subgroup
   λ_probabilities = softmax(coherent_logits)
   subgroup_assignment = argmax(λ_probabilities)
   ```

3. **Subgroup-Specific Processing**
   ```julia
   # Filter waveforms by subgroup probability
   subgroup_waveforms = filter_mode(waveforms, model, k_target, p=0.9)
   ```

#### **Practical Implementation**

##### **Data Preparation**
```julia
# Organize waveforms into groups (states)
waveform_groups = [group1, group2, ..., groupN]

# Create batch iterators for training
data_iterators = get_data_iterator(waveform_groups, 
                                  ntau=20,      # waveforms per group sample
                                  batchsize=32) # groups per training batch
```

##### **Model Configuration**
```julia
# Configure for multi-category coherent codes
para = SymAE_Para(
    nt = 512,           # time samples per waveform
    p = 32,             # coherent code dimension per category  
    k = 3,              # number of coherent categories (subgroups)
    q = 16,             # nuisance code dimension
    network_type = ConvAE()
)
```

##### **Training Execution**
```julia
# Train with group-aware loss function
model, loss_history = get_symae(para)
update(model, loss_history, train_groups, test_groups, training_para)
```

##### **Coherent Information Extraction**
```julia
# Extract coherent signals from trained model
coherent_signals, _, _, _ = get_coherent_information(
    para, model, target_groups,
    N=10,                    # number of optimized nuisance codes
    nepochs=100,            # optimization iterations
    alpha=0.1,              # regularization strength
    kopt=1                  # target coherent category
)
```

### Mathematical Foundation

The SymVAE loss function balances multiple objectives:

```
L = Σᵢ[-log p(xᵢ|cᵢ,nᵢ)] + βc·KL(q(c|X)||p(c)) + βn·Σᵢ KL(q(nᵢ|xᵢ)||p(nᵢ)) + βλ·KL(q(λ|X)||p(λ))
```

Where:
- **Reconstruction term**: Ensures faithful waveform reproduction
- **Coherent KL term**: Regularizes shared information extraction  
- **Nuisance KL terms**: Regularizes instance-specific variations
- **Categorical KL term**: Controls subgroup assignment sharpness

### Expected Outcomes

#### **Successful Coherent Extraction**
- Recovered signals show high correlation with true source signatures
- Nuisance codes capture propagation path variations
- Coherent codes cluster meaningfully in latent space

#### **Effective Subgroup Identification**  
- Clear separation of waveform categories in coherent space
- Automatic discovery of geological or acquisition-related groupings
- Robust classification of new waveforms into existing subgroups

This framework enables both **unsupervised feature learning** and **automatic clustering** in a single unified model, making it particularly powerful for seismic data analysis where both source characteristics and propagation effects must be carefully separated and understood.
"""

# ╔═╡ dc2cd512-9ad0-11f0-1dca-71ffa60b282a
md"""
## Complete SymVAE Workflow Summary

### Two-Phase Process for Group-Based Coherent Information Extraction

The SymVAE framework implements a comprehensive two-phase approach to achieve the dual objectives of coherent information extraction and subgroup identification:

---

#### **Phase 1: Training on Group Collections**

##### **Step 1.1: Data Organization**
```julia
# Organize waveforms into meaningful groups
waveform_groups = [
    site1_recordings,    # Group 1: All recordings from site 1
    site2_recordings,    # Group 2: All recordings from site 2  
    site3_recordings,    # Group 3: All recordings from site 3
    # ... additional groups
]

# Each group should contain waveforms sharing coherent characteristics
# but with different propagation/noise effects (nuisance variations)
```

##### **Step 1.2: Model Configuration**
```julia
# Configure architecture for your specific problem
para = SymAE_Para(
    nt = 512,                    # Time samples per waveform
    p = 32,                      # Coherent code dimension per category
    k = 3,                       # Number of coherent categories (subgroups)
    q = 16,                      # Nuisance code dimension
    network_type = ConvAE(),     # Convolutional architecture
    transformer = :spatial       # Enable time-shift correction
)
```

##### **Step 1.3: Group-Based Training**
```julia
# Initialize model
model, loss_history = get_symae(para)

# Create group-aware data iterators
train_data = get_data_iterator(
    train_groups,
    ntau=20,        # 20 waveforms per group sample
    batchsize=64,   # 64 group samples per training batch  
    nsteps=1000     # 1000 training steps per epoch
)

# Train with variational objective
training_para = Training_Para(
    nepoch=100,
    beta=(N=[1f0], C=[1f0, 0.5f0]),  # KL weights: [nuisance], [coherent, categorical]
    gamma=1f2,                        # Spatial regularization
    temperature=1f0                   # Gumbel-softmax temperature
)

update(model, loss_history, train_data, test_data, training_para)
```

**Training Objectives During Phase 1:**
- **Coherent Learning**: Extract common features across `ntau` instances within each group
- **Nuisance Learning**: Model instance-specific variations within each group  
- **Categorical Learning**: Discover `k` distinct coherent categories across all groups
- **Spatial Learning**: Learn time-shift corrections for better alignment

---

#### **Phase 2: Coherent Information Extraction and Subgroup Identification**

##### **Step 2.1: Subgroup Identification (Objective 2)**
```julia
# Automatically identify subgroups within each provided group
# Model assigns each waveform to one of k coherent categories

for (i, group) in enumerate(target_groups)
    # Get category probabilities for each waveform
    coherent_codes = model(group, Val(:coherent))
    
    if haskey(coherent_codes, :λlogits)
        # Multi-category model: compute subgroup probabilities
        λ_probs = softmax(coherent_codes.λlogits)
        
        # Identify dominant subgroups
        for k_category = 1:para.k
            high_prob_indices = findall(λ_probs[k_category, :, 1] .> 0.8)
            println("Group $i, Category $k_category: $(length(high_prob_indices)) waveforms")
        end
    end
end
```

##### **Step 2.2: Coherent Information Extraction (Objective 1)**
```julia
# Extract pure coherent signals for each target group
coherent_signals, references, optimization_history, transforms = get_coherent_information(
    para, model, target_groups,
    N=15,                        # Number of optimized nuisance codes
    nepochs=200,                # Optimization iterations
    alpha=0.1,                  # Regularization strength  
    kopt=1,                     # Target coherent category
    filter_probability=0.9      # Subgroup membership threshold
)

# Results:
# coherent_signals[i] = extracted source signature for group i
# Each signal has propagation effects and noise removed
# Represents the "pure" coherent information shared across the group
```

##### **Step 2.3: Quality Assessment**
```julia
# Evaluate extraction quality
for (i, coherent_signal) in enumerate(coherent_signals)
    # Measure coherence within extracted signals
    correlation = average_leave_one_out_correlation(coherent_signal)
    println("Group $i coherence: $correlation")
    
    # Compare with original group variability
    original_correlation = average_leave_one_out_correlation(target_groups[i])
    improvement = correlation - original_correlation
    println("Group $i improvement: $improvement")
end
```

---

### **Key Success Indicators**

#### **For Coherent Information Extraction (Objective 1):**
✅ **High Intra-Group Coherence**: Extracted signals show strong correlation within each group  
✅ **Preserved Signal Content**: Important waveform characteristics are maintained  
✅ **Noise Reduction**: Instance-specific variations are minimized  
✅ **Geological Interpretability**: Extracted signals relate to known source/structure properties

#### **For Subgroup Identification (Objective 2):**
✅ **Clear Category Separation**: λ-probabilities show distinct peaks for different categories  
✅ **Meaningful Groupings**: Discovered subgroups correspond to geological/acquisition differences  
✅ **Robust Classification**: New waveforms consistently assign to appropriate subgroups  
✅ **Balanced Categories**: Each subgroup contains sufficient instances for reliable statistics

---

### **Practical Applications**

#### **Seismic Processing:**
- **Source Signature Extraction**: Remove propagation effects to recover pure source wavelets
- **Multiple Attenuation**: Separate primary reflections from multiples and noise
- **Reservoir Characterization**: Extract formation-specific signatures from seismic data

#### **Quality Control:**
- **Outlier Detection**: Identify waveforms that don't fit any coherent category
- **Data Validation**: Assess consistency within acquired datasets  
- **Processing Optimization**: Guide parameter selection for subsequent processing steps

#### **Interpretation:**
- **Geological Clustering**: Group waveforms by subsurface properties
- **Facies Analysis**: Identify distinct rock types from seismic response
- **Structural Interpretation**: Separate signals from different geological structures

This workflow provides a systematic approach to both **unsupervised feature learning** (coherent extraction) and **automatic clustering** (subgroup identification) in a unified framework, making it particularly powerful for complex seismic data analysis scenarios.
"""

# ╔═╡ a91e28fb-e769-418d-953f-0e0bb366d853
md"""
## Parameters
- `nt`: number of samples in time (length of each waveform)
- `ntau`: number of waveforms used per step (as large as your GPU memory supports)
- `p`: length of coherent code per each coherent code category
- `k`: number of coherent information categories
- `q`: length of nuisance code for each waveform
- `beta`: custom weight of the KL terms in this order [[nuisance], [coherent, lambda]]
- `batchsize`: because of stochastic gradient descent
- `nsteps`: how long you want a single epoch to be?
- `network_type`:
  - `ConvAE()` standard choice
  - `DenseAE()` don't use unless you want to quickly test something
- `transformer`:
  - `:spatial` each waveform is time shifted prior to encoding
  - `:spatial_state` each state is time shifted prior to encoding
  - `:null` 
- `spatial_transformer_gamma`::Float32 = 0.5f0
"""

# ╔═╡ 8fefefd8-b63a-4be7-92e3-8ed7c3f88865
function infer_condition_dim(conditioning::Union{Nothing,GroupConditioning})
    if !has_conditioning(conditioning)
        return 0
    end
    return length(get_group_condition_vector(conditioning, 1))
end

# ╔═╡ 29d41554-3a0a-4972-a9e5-54c998429acd
md"""## Data Generator for Group-Based Training

### Group-Based Data Organization

The SymVAE training process relies on organizing waveforms into **groups** (also called "states") where each group contains multiple waveform instances that should share common coherent information. The data generators implement sophisticated sampling strategies to ensure effective learning of both coherent and nuisance representations.

### Key Concepts:
- **`dvec`**: Vector of data groups, where `dvec[i]` contains all waveforms for group i
- **`ntau`**: Number of waveforms sampled from each group per training step
- **Group sampling**: Ensures coherent information is learned from multiple instances
- **Cross-group batching**: Enables comparison and contrast learning between different groups

### Training Data Flow:
1. **Group Formation**: Waveforms with shared source characteristics are grouped together
2. **Within-Group Sampling**: `ntau` waveforms randomly selected from each group
3. **Batch Assembly**: Multiple group samples combined into training batches
4. **Coherent Learning**: Model learns to extract common features across `ntau` instances
5. **Nuisance Learning**: Model learns instance-specific variations within each group
"""

# ╔═╡ ce690827-fa3f-48bc-bc09-1df5ee15f683
md"## Architecture"

# ╔═╡ 2df712f5-6969-4ceb-98c5-82415718b740
begin
	
	    # ── FiLMWithMLP: Non-linear condition projection via MLP ────────────────
	    # Two-layer MLP: cond_dim → hidden_dim (nonlinear) → 2*nchannels (linear)
	    # Enables richer condition-to-parameter mapping while maintaining near-identity init
	    struct FiLMWithMLP{T1,T2} <: AbstractParallel
	        hidden_layer::T1  # Dense(cond_dim → hidden_dim, activation)
	        output_layer::T2  # Dense(hidden_dim → 2*nchannels, no activation)
	    end
	    Flux.@layer FiLMWithMLP

	    function FiLMWithMLP(cond_dim::Int, nchannels::Int; hidden_dim::Int=max(cond_dim, nchannels))
	        # Hidden layer: cond_dim → hidden_dim with nonlinearity
	        hidden = Dense(cond_dim, hidden_dim, activation)
	        
	        # Output layer: hidden_dim → 2*nchannels, near-identity init
	        W_out = zeros(Float32, 2 * nchannels, hidden_dim)
	        b_out = Float32[ones(nchannels); zeros(nchannels)]  # γ-bias=1, β-bias=0
	        output = Dense(W_out, b_out)
	        
	        return FiLMWithMLP(hidden, output)
	    end

	    function (m::FiLMWithMLP)(x, cond_embedding=nothing)
	        cond_embedding === nothing && return x
	        # x has shape: (nt, nc, batch)
	        nc = size(x, 2)
	        h = m.hidden_layer(cond_embedding)   # (hidden_dim, batch)
	        γβ = reshape(m.output_layer(h), nc, 2, :)  # (nc, 2, batch)
	        γ_raw, β_raw = chunk(γβ, 2, dims=2)
	        γ = reshape(γ_raw, 1, nc, :)                    # (1, nc, batch)
	        β = reshape(β_raw, 1, nc, :)                    # (1, nc, batch)
	        return γ .* x .+ β
	    end
end

# ╔═╡ d86a890f-942f-4e3a-8405-709507450903
begin
	    # FiLM-conditioned decoder chain:
	    # 1) concat condition then latent Dense projection, 2) ConvTranspose stages each followed by FiLM.
	    struct FiLMConvDecoder1DChain{T1,S,F,A} <: AbstractParallel
	        proj::T1
	        stages::S
	        films::F
	        apply_activation::A
	        bottleneck_len::Int
	        bottleneck_channels::Int
	    end
	    Flux.@layer FiLMConvDecoder1DChain
	
	    function (m::FiLMConvDecoder1DChain)(x, cond_embedding=nothing)
	        x = add_dim3_reshape(x)
	        n1, n2, n3 = size(x)
	        Z = reshape(x, n1, n2 * n3)
            cond_exp = if cond_embedding === nothing
                nothing
            elseif size(cond_embedding, 2) == n2 * n3
                cond_embedding
            elseif size(cond_embedding, 2) == n3
                reshape(Flux.stack(fill(cond_embedding, n2), dims=2), size(cond_embedding, 1), n2 * n3)
            else
                throw(DimensionMismatch("Condition batch mismatch in FiLMConvDecoder1DChain: cond has $(size(cond_embedding, 2)) columns, expected $n3 or $(n2 * n3)."))
            end
	        @assert cond_exp !== nothing "FiLMConvDecoder1DChain requires conditioning when cond_dim > 0"
	
	        H = m.proj(cat(Z, cond_exp, dims=1))
	        X = reshape(H, m.bottleneck_len, m.bottleneck_channels, :)
	
	        for (stage, film, do_activation) in zip(m.stages, m.films, m.apply_activation)
	            X = stage(X)
	            X = film(X, cond_exp)
	            if do_activation
	                X = activation(X)
	            end
	        end
	
	        return reshape(X, :, n2, n3)
	    end
end

# ╔═╡ 5a047afc-ec40-4541-ad50-0518beba3f2e
begin
	    # ── FiLMConv1DChain ───────────────────────────────────────────────────────
	    # Mirrors Conv1DChain but accepts an optional conditioning vector and applies
	    # a FiLM layer after every convolutional stage.
	    struct FiLMConv1DChain{S,F,A,T} <: AbstractParallel
	        stages::S   # Tuple of per-stage Chain (Conv + optional BN)
	        films::F    # Tuple of FiLM layers, one per stage
	        apply_activation::A
	        tail::T     # Chain: flatten [→ Dense]
	    end
	    Flux.@layer FiLMConv1DChain
	
	    function (m::FiLMConv1DChain)(x, cond_embedding=nothing)
	        x  = add_dim3_reshape(x)
	        n1, n2, n3 = size(x)
	        X  = reshape(x, n1, 1, n2 * n3)
	        # Expand cond from (d, batch) → (d, ntau*batch) matching the reshape above
            cond_exp = if cond_embedding === nothing
                nothing
            elseif size(cond_embedding, 2) == n2 * n3
                cond_embedding
            elseif size(cond_embedding, 2) == n3
                reshape(Flux.stack(fill(cond_embedding, n2), dims=2), size(cond_embedding, 1), n2 * n3)
            else
                throw(DimensionMismatch("Condition batch mismatch in FiLMConv1DChain: cond has $(size(cond_embedding, 2)) columns, expected $n3 or $(n2 * n3)."))
            end
	        for (stage, film, do_activation) in zip(m.stages, m.films, m.apply_activation)
	            X = stage(X)
	            X = film(X, cond_exp)
	            if do_activation
	                X = activation(X)
	            end
	        end
	        X = m.tail(X)
	        return reshape(X, :, n2, n3)
	    end
end

# ╔═╡ ae96f920-5828-4c5f-b69f-48d8c4fee378
md"## Dense Networks"

# ╔═╡ ea372a8f-212f-425d-947c-b57bba6b5574
md"## Conv & ViT Networks"

# ╔═╡ 89599b3f-8c20-46c5-8f5c-ccbb71b26b36
function get_conv_encoder(nt, p=nothing; kernels=[32, 16, 8, 4], filters=[8, 16, 32, 64], strides=[2, 2, 2, 2], use_bn::Bool=true, cond_dim::Int=0)
    @assert length(kernels) == length(filters) "kernels and filters must align"

    if cond_dim > 0
        # ── FiLM path: one stage per conv layer, one FiLM per stage ──────────
        stages     = Any[]
        film_layers = Any[]
        stage_activation = Bool[]
        nin = 1
        for (i, k) in enumerate(kernels)
            nout = filters[i]
            s    = i <= length(strides) ? strides[i] : 1
            stage_layers = Any[Conv((k,), nin => nout; pad=SamePad(), stride=s)]
            if use_bn && i < length(kernels)
                push!(stage_layers, BatchNorm(nout))
            end
            push!(stages,      Chain(stage_layers...))
            push!(film_layers, FiLMWithMLP(cond_dim, nout))
            push!(stage_activation, true)
            nin = nout
        end
        trunk      = Chain([l for ch in stages for l in ch.layers]...)
        outsize    = Flux.outputsize(trunk, (nt, 1); padbatch=true)
        flat_len   = prod(outsize)
        tail_layers = Any[Flux.flatten]
        output_length = flat_len
        if p !== nothing
            push!(tail_layers, Dense(flat_len => p, activation))
            output_length = p
        end
        return FiLMConv1DChain(Tuple(stages), Tuple(film_layers), Tuple(stage_activation), Chain(tail_layers...)), output_length
    else
        # ── Original path: flat Conv1DChain, condition concatenated at input ──
        layers = Any[]
        nin = 1
        for (i, k) in enumerate(kernels)
            nout = filters[i]
            s    = i <= length(strides) ? strides[i] : 1
            push!(layers, Conv((k,), nin => nout, activation; pad=SamePad(), stride=s))
            if use_bn && i < length(kernels)
                push!(layers, BatchNorm(nout))
            end
            nin = nout
        end
        trunk      = Chain(layers...)
        conv_outsize = Flux.outputsize(trunk, (nt, 1); padbatch=true)
        last_layer, output_length = if p === nothing
            identity, prod(conv_outsize)
        else
            Dense(prod(conv_outsize) => p, activation), p
        end
        return Conv1DChain(Chain(trunk..., Flux.flatten, last_layer)), output_length
    end
end

# ╔═╡ 64430447-c267-4eec-8d38-63ccf91d82c4
function get_conv_decoder(nt, pq; kernels=[4, 8, 16], filters=[64, 48, 16, 1], upstrides=[2, 2, 1], use_bn=false, cond_dim::Int=0)
    @assert length(kernels) == length(upstrides) "kernels and upstrides must align"
    @assert length(filters) == length(kernels) + 1 "filters length should be stages+1 (bottleneck..final)"

    total_stride = prod(upstrides)
    @assert nt % total_stride == 0 "nt must be divisible by product(upstrides)"
    bottleneck_len = div(nt, total_stride)

    if cond_dim > 0
        proj = Dense(pq + cond_dim, bottleneck_len * filters[1], activation)

        stages = Any[]
        film_layers = Any[]
        stage_activation = Bool[]
        nin = filters[1]
        for i in 1:length(kernels)
            nout = filters[i + 1]
            k = kernels[i]
            s = upstrides[i]
            stage_layers = if i == length(kernels)
                Any[ConvTranspose((k,), nin => nout; stride=s, pad=SamePad())]
            else
                local tmp = Any[ConvTranspose((k,), nin => nout; stride=s, pad=SamePad())]
                if use_bn
                    push!(tmp, BatchNorm(nout))
                end
                tmp
            end
            push!(stages, Chain(stage_layers...))
            push!(film_layers, FiLMWithMLP(cond_dim, nout))
            push!(stage_activation, i != length(kernels))
            nin = nout
        end

        return FiLMConvDecoder1DChain(proj, Tuple(stages), Tuple(film_layers), Tuple(stage_activation), bottleneck_len, filters[1])
    else
        layers = Any[]
        # project latent PQ -> (bottleneck_len × filters[1]) feature map
        push!(layers, Dense(pq, bottleneck_len * filters[1], activation))

        btl = bottleneck_len
        f1 = filters[1]
        push!(layers, x -> reshape(x, btl, f1, :))

        nin = filters[1]
        for i in 1:length(kernels)
            nout = filters[i+1]
            k = kernels[i]
            s = upstrides[i]
            # ConvTranspose upsamples by stride s
            if i == length(kernels)
                # final convtranspose -> single-channel linear output (no activation)
                push!(layers, ConvTranspose((k,), nin => nout; stride=s, pad=SamePad()))
            else
                push!(layers, ConvTranspose((k,), nin => nout, activation; stride=s, pad=SamePad()))
                if (use_bn)
                    push!(layers, BatchNorm(nout))
                end
            end
            nin = nout
        end

        return Conv1DChain(Chain(layers...))
    end
end

# ╔═╡ 190c8221-c5c7-48f9-b016-36c27fd4528c
function get_symae(t::ConvAE, nt, p, q, k, condition_embed_dim=0)
    P = p * k
    dec_logvar = xpu(cat(1.0f0, dims=3))
    if (condition_embed_dim > 0)
        # ── FiLM conditioning for all encoder/decoder conv blocks and decoder FC ──
        senc, Pout = xpu(get_conv_encoder(nt, nothing; kernels=t.enc_kernels, filters=t.enc_filters, strides=t.enc_strides, use_bn=t.use_bn, cond_dim=condition_embed_dim))
        nenc, qout = xpu(get_conv_encoder(nt, nothing; kernels=t.enc_kernels, filters=t.enc_filters, strides=t.enc_strides, use_bn=t.use_bn, cond_dim=condition_embed_dim))
        senc_λlogits = xpu(Chain(Dense(Pout + condition_embed_dim, div(Pout, 2), activation), Dense(div(Pout, 2), k)))
        senc_μ        = xpu(Dense(Pout, P))
        senc_loginvvar = xpu(Dense(Pout, P))
        nenc_μ   = xpu(Dense(qout, q))
        nenc_logσ = xpu(Dense(qout, q))
        dec = xpu(get_conv_decoder(nt, p + q; kernels=t.dec_kernels, filters=t.dec_filters, upstrides=t.dec_upstrides, use_bn=t.use_bn, cond_dim=condition_embed_dim))
    else
        # ── Unconditioned ConvAE path ──────────────────────────────────────────
        senc, Pout = xpu(get_conv_encoder(nt, nothing; kernels=t.enc_kernels, filters=t.enc_filters, strides=t.enc_strides, use_bn=t.use_bn))
        nenc, qout = xpu(get_conv_encoder(nt, nothing; kernels=t.enc_kernels, filters=t.enc_filters, strides=t.enc_strides, use_bn=t.use_bn))
        senc_λlogits = xpu(Chain(Dense(Pout, div(Pout, 2), activation), Dense(div(Pout, 2), k)))
        senc_μ        = xpu(Dense(Pout, P))
        senc_loginvvar = xpu(Dense(Pout, P))
        nenc_μ   = xpu(Dense(qout, q))
        nenc_logσ = xpu(Dense(qout, q))
        dec = xpu(get_conv_decoder(nt, p + q; kernels=t.dec_kernels, filters=t.dec_filters, upstrides=t.dec_upstrides, use_bn=t.use_bn))
    end
    return (; senc, senc_μ, senc_loginvvar, senc_λlogits, nenc, nenc_μ, nenc_logσ, dec, dec_logvar)
end

# ╔═╡ 747680b0-3469-426c-8b9e-4ab8ca04a6de
md"## Convolutional SymAE"

# ╔═╡ 62681bba-5486-4957-8433-4258657399b8
md"## Dense SymAE"

# ╔═╡ facd01fe-b288-437f-96dd-a8a4d9afd8fe
md"## Removed Architectures"

# ╔═╡ 66bddc43-ca9f-43cd-85a3-d33b11a6c033
md"## Coherent Encoder"

# ╔═╡ 8684c192-d1b9-4821-a02e-2c7300af9b3c
begin
    struct BroadcastCoherentEnc{T1,T2,T3}
        chain::T1
        μ::T2
        loginvvar::T3
    end
    Flux.@layer BroadcastCoherentEnc trainable = (chain, μ, loginvvar)
    function (m::BroadcastCoherentEnc)(x)
        X = m.chain(x)
        μ = m.μ(X)
        loginvvar = m.loginvvar(X)
        return accumulate_Gaussians(μ, loginvvar)
    end
	function (m::BroadcastCoherentEnc)(x, cond)
        X = _maybe_forward(m.chain, x, cond)
        μ = m.μ(X)
        loginvvar = m.loginvvar(X)
        return accumulate_Gaussians(μ, loginvvar)
    end

end

# ╔═╡ e9efedc1-8287-4676-ba64-a4abc77da18d
begin
    struct BroadcastCoherentEncCategorical{T1,T2,T3,T4}
        chain::T1
        μ::T2
        loginvvar::T3
        λlogits::T4
    end
    Flux.@layer BroadcastCoherentEncCategorical trainable = (chain, μ, loginvvar, λlogits)
    function (m::BroadcastCoherentEncCategorical)(x)
        X = m.chain(x)
        μ = m.μ(X)
        loginvvar = m.loginvvar(X)
        λlogits = m.λlogits(X)
        return merge(accumulate_Gaussians(μ, loginvvar), (; λlogits))
    end
	 function (m::BroadcastCoherentEncCategorical)(x, cond)
        X = _maybe_forward(m.chain, x, cond)
        μ = m.μ(X)
        loginvvar = m.loginvvar(X)
        # Concatenate condition embedding to encoder output for λlogits
        # X can be (Pout, ntau, ngroups) [3D] or (Pout, batch) [2D]
        # cond is (cembed, batch_or_ngroups)
        Xc = if cond === nothing
            X
        elseif ndims(X) == 3 && ndims(cond) == 2
            # Reshape cond to (cembed, 1, batch) then expand to (cembed, ntau, batch)
            # matching X = (Pout, ntau, ngroups) where ngroups = size(X,3)
            ngroups = size(X, 3)
            ntau = size(X, 2)
            # cond may already be expanded to ntau*ngroups or just ngroups
            cond3 = if size(cond, 2) == ngroups
                reshape(Flux.stack(fill(cond, ntau), dims=2), size(cond, 1), ntau, ngroups)
            elseif size(cond, 2) == ntau * ngroups
                reshape(cond, size(cond, 1), ntau, ngroups)
            else
                error("cond batch $(size(cond,2)) incompatible with X shape $(size(X))")
            end
            cat(X, cond3, dims=1)
        else
            cat(X, cond, dims=1)
        end
        λlogits = m.λlogits(Xc)
        return merge(accumulate_Gaussians(μ, loginvvar), (; λlogits))
    end

end

# ╔═╡ 06286e75-6fdd-41a3-b80b-0f0ffa4ec603
begin
	"""
	Select waveforms in D that belong to the coherent class kopt with high probability.
	"""
	function filter_mode(D::AbstractVector, model, kopt; p=0.9)
	    X = map(D) do d
	        dnew, indices = filter_mode(d, model, kopt; p=p, condition=nothing)
	        (; dnew, indices)
	    end
	    return first.(X), last.(X)
	end
	
	function filter_mode(D::AbstractVector, model, kopt, condition; p=0.9)
	    cond = Float32.(condition)
	    @assert ndims(cond) == 2 "Vector conditioning must be a matrix with shape (condition_dim, ngroups)."
	    @assert size(cond, 2) == length(D) "Condition group count mismatch. Expected $(length(D)), got $(size(cond, 2))."
	    X = map(enumerate(D)) do (i, d)
	        dnew, indices = filter_mode(d, model, kopt; p=p, condition=reshape(cond[:, i], :, 1))
	        (; dnew, indices)
	    end
	    return first.(X), last.(X)
	end
end

# ╔═╡ 6e3d3148-fc73-4538-b077-2abe1d1721d4
"""
Select waveforms in D that belong to the coherent class kopt with high probability.
"""
function filter_mode(d::AbstractMatrix, model, kopt; p=0.9, condition=nothing)
    coherent_code = model(d, Val(:coherent); condition=condition)
    dnew, indices = if (:λlogits in keys(coherent_code))
        λp = cpu(softmax(coherent_code.λlogits, dims=1))
        Ic = findall(x -> x[kopt] >= p, eachcol(dropdims(λp, dims=3)))
        d[:, Ic], Ic
    else
        d, collect(1:size(d, 2))
    end
    return dnew, indices
end

# ╔═╡ 41e788f4-2e25-4968-8bf4-dcd2f38d24e7
"""
Averaging for samples with probability thresold
"""
function get_cluster_averages_with_high_probability(model, D_to_get_prob, K; D_to_avg=D_to_get_prob, p=0.9, condition=nothing)
    if condition === nothing
    return cat(map(D_to_get_prob, D_to_avg) do d, da
        cat(map(1:K) do k
            _, indices = filter_mode(gpu(collect(d)), model, k; p=p, condition=nothing)
            mean(cpu(da[:, indices]), dims=2)
            end..., dims=2)
        end..., dims=3)
    end

    cond = Float32.(condition)
    @assert ndims(cond) == 2 "Vector conditioning must be a matrix with shape (condition_dim, ngroups)."
    @assert size(cond, 2) == length(D_to_get_prob) "Condition group count mismatch. Expected $(length(D_to_get_prob)), got $(size(cond, 2))."
    return cat(map(enumerate(zip(D_to_get_prob, D_to_avg))) do (i, (d, da))
        cond_i = reshape(cond[:, i], :, 1)
        cat(map(1:K) do k
            _, indices = filter_mode(gpu(collect(d)), model, k; p=p, condition=cond_i)
            mean(cpu(da[:, indices]), dims=2)
        end..., dims=2)
    end..., dims=3)
end

# ╔═╡ f1eddb54-7d90-4a14-9943-053248665e78
"""
Averaging for samples with probability thresold
"""
function get_cluster_percentages_with_high_probability(model, D_to_get_prob, K; p=0.9, condition=nothing)
    if condition === nothing
    return cat(map(D_to_get_prob) do d
        cat(map(1:K) do k
            _, indices = filter_mode(gpu(collect(d)), model, k; p=p, condition=nothing)
            length(indices) / size(d, 2) * 100.0
            end..., dims=1)
        end..., dims=2)
    end

    cond = Float32.(condition)
    @assert ndims(cond) == 2 "Vector conditioning must be a matrix with shape (condition_dim, ngroups)."
    @assert size(cond, 2) == length(D_to_get_prob) "Condition group count mismatch. Expected $(length(D_to_get_prob)), got $(size(cond, 2))."
    return cat(map(enumerate(D_to_get_prob)) do (i, d)
        cond_i = reshape(cond[:, i], :, 1)
        cat(map(1:K) do k
            _, indices = filter_mode(gpu(collect(d)), model, k; p=p, condition=cond_i)
            length(indices) / size(d, 2) * 100.0
        end..., dims=1)
    end..., dims=2)
end

# ╔═╡ 52dc9696-3e0b-42c7-b6cf-7a07ca3cb4dd
md"## Nuisance Encoder"

# ╔═╡ fa646879-158a-4cbb-be0e-d375cf486ba0
begin
    struct BroadcastNuisanceEnc{T1,T2,T3}
        chain::T1
        μ::T2
        logσ::T3
    end
    Flux.@layer BroadcastNuisanceEnc trainable = (chain, μ, logσ)
    (m::BroadcastNuisanceEnc)(::Nothing) = nothing
    function (m::BroadcastNuisanceEnc)(x)
        X = m.chain(x)
        μ = m.μ(X)
        logσ = m.logσ(X)
        return (; μ, logσ)
    end
	 function (m::BroadcastNuisanceEnc)(x, cond)
        X = _maybe_forward(m.chain, x, cond)
        μ = m.μ(X)
        logσ = m.logσ(X)
        return (; μ, logσ)
    end
    (m::BroadcastNuisanceEnc)(::Nothing, cond) = nothing
end

# ╔═╡ eafab001-87a7-423f-917f-1fbd46699186
md"## Decoder"

# ╔═╡ 5ed89ee8-325b-4757-b348-e6c1a3d277ad
md"## SymVAE Model"

# ╔═╡ 371354f6-29e7-4227-b006-f2daacb08ce7
md"""## SymVAETrunk Model
Both coherent and nuisance encoders have a common _trunk_
"""

# ╔═╡ fe87efed-9c40-4869-bdf3-cc60eb5b6436
md"### Sample Posterior and Decode"

# ╔═╡ 78e16b27-85c3-4d4a-9238-3ec2a33c9c88
"""
Coherent code will be divided into K chunks
* λlog are unnormalized logits determining which chunk of the coherent code to be used for the generator (decoder)
* τ is the temperature: non-negative scalar As τ→0, the softmax becomes an argmax and the Gumbel-Softmax distribution becomes the categorical distribution. During training, we let τ>0 to allow gradients past the sample, then gradually anneal the temperature τ (but not completely to 0, as the gradients would blow up).
* if nnoise is false, then the returned λx will be one-hot (used after training)
"""
function sample_q_decode(cμ, clogσ, λlogits, nμ, nlogσ, nnoise, cnoise, decoder, temperature; cond_embedding=nothing)
    if (cnoise)
        cx = cμ + xpu(randn(Float32, size(clogσ))) .* exp.(clogσ)
    else
        cx = cμ
    end
    cx = dropdims(cx, dims=ndims(cμ) - 1)
    cx = Flux.stack(fill(cx, size(nμ, ndims(nμ) - 1)), dims=ndims(nμ) - 1)

    if (nnoise)
        nx = nμ + xpu(randn(Float32, size(nlogσ))) .* exp.(nlogσ)
    else
        nx = nμ
    end

    if (nnoise)
        eps = 1f-6
        gumbel_noise = xpu(-log.(eps .+ -log.(eps .+ rand(Float32, size(λlogits)))))
        λx = softmax((λlogits .+ gumbel_noise) ./ temperature, dims=1)
    else
        max_indices = getindex.(dropdims(CUDA.argmax(λlogits, dims=1), dims=1), 1)
        λx = Flux.onehotbatch(max_indices, 1:size(λlogits, 1))  # OneHotMatrix
    end

    cx_chunks = chunk(cx, size(λlogits, 1), dims=1)
    λ_chunks = chunk(λx, size(λlogits, 1), dims=1)
    cx = sum(map(cx_chunks, λ_chunks) do c, l
        c .* l
    end)

    latent = cat(cx, nx, dims=1)
    xhat, xhat_logvar = decoder(latent, cond_embedding)
    return xhat, xhat_logvar
end

# ╔═╡ efde2571-4e1d-4626-9081-86d513707f5e
md"## Deterministic SymAE with Dropout"

# ╔═╡ 95660df0-088b-49f3-b875-fca19a82d024
md"""## Spatial Transformer Model
Wrap any SymVAE model with a spatial transformer
 - First, predict time shifts with transformerb
 - Then, apply shifts
 - Then, apply SymVAE model
 - Last, apply inv_shifts
"""

# ╔═╡ 5731aea5-af70-4dc3-a505-2f48fce02e8e
md"""
### SpatialTransformer
"""

# ╔═╡ 8abc3a6d-d2f5-4527-a828-793364706fa5

"""
shift traces using Fourier Interpolation
"""
function shift_traces_Fourier(input_traces, shifts, sampling_grid)
    x_fft = fouriertransform1D(input_traces)
    S = reshape(sampling_grid, (length(sampling_grid), ntuple(_ -> 1, ndims(shifts) - 1)...))
    E = exp.(S .* shifts)
    output_traces_fft = x_fft .* E
    output_traces = inversefouriertransform1D(output_traces_fft)
    return reshape(output_traces, size(input_traces))
end


# ╔═╡ c49e3d81-27ba-4a4f-870a-ae218a505dd0
begin
    struct Spatial_Transformer{T1,T2,T3}
        symvae::T1
        transformerb::T2
        sampling_grid::T3
    end
    function (m::Spatial_Transformer)(x, nnoise::Bool, cnoise::Bool, temperature)
        return m(x, nothing, nnoise, cnoise, temperature)
    end
    function (m::Spatial_Transformer)(x, condition, nnoise::Bool, cnoise::Bool, temperature)
        S = m.transformerb(x)
        xt = shift_traces_Fourier(x, S.shifts, m.sampling_grid)
        output = m.symvae(xt, condition, nnoise, cnoise, temperature)
        xhat = shift_traces_Fourier(output.X.xhat, S.inv_shifts, m.sampling_grid)
        return (; X=(; xhat, xshifted=xt, xshifted_reconstructed=output.X.xhat, xhat_logvar=output.X.xhat_logvar), Z=output.Z, shifts=S.shifts)
    end
    function (m::Spatial_Transformer)(xc, xn, nnoise::Bool, cnoise::Bool, temperature)
        return m(xc, xn, nothing, nnoise, cnoise, temperature)
    end
    function (m::Spatial_Transformer)(xc, xn, condition, nnoise::Bool, cnoise::Bool, temperature)
        Sc = m.transformerb(xc)
        Sn = m.transformerb(xn)
        xct = shift_traces_Fourier(xc, Sc.shifts, m.sampling_grid)
        xnt = shift_traces_Fourier(xn, Sn.shifts, m.sampling_grid)
        output = m.symvae(xct, xnt, condition, nnoise, cnoise, temperature)
        xhat = shift_traces_Fourier(output.X.xhat, Sn.inv_shifts, m.sampling_grid)
        return (; X=(; xhat, xshifted=xt, xhat_logvar=output.X.xhat_logvar), Z=output.Z, shifts=S.shifts)
    end
    # estimate either coherent or nuisance code after applying transformer, for each element of D
    function (m::Spatial_Transformer)(D::Vector{T}, code_type::Union{Val{:coherent},Val{:nuisance}}; apply_transformer=true) where {T}
        Dshifted = if (apply_transformer)
            m(D, Val(:transform))
        else
            D
        end
        return m.symvae(Dshifted, code_type)
    end
    # estimate either coherent or nuisance code after applying transformer
    function (m::Spatial_Transformer)(d, code_type::Union{Val{:coherent},Val{:nuisance}}; apply_transformer=true)
        dshifted = if (apply_transformer)
            m(d, Val(:transform))
        else
            d
        end
        return m.symvae(dshifted, code_type)
    end
    function (m::Spatial_Transformer)(D::Vector{T}, ::Val{:transform}) where {T}
        return map(D) do d
            S = m.transformerb(d)
            d = shift_traces_Fourier(d, S.shifts, m.sampling_grid)
        end
    end
    function (m::Spatial_Transformer)(d, ::Val{:transform})
        S = m.transformerb(d)
        return shift_traces_Fourier(d, S.shifts, m.sampling_grid)
    end
    function (m::Spatial_Transformer)(D::Vector{T}, ::Val{:inv_transform}) where {T}
        return map(D) do d
            S = m.transformerb(d)
            d = shift_traces_Fourier(d, S.inv_shifts, m.sampling_grid)
        end
    end
    function (m::Spatial_Transformer)(d, ::Val{:inv_transform})
        S = m.transformerb(d)
        return shift_traces_Fourier(d, S.inv_shifts, m.sampling_grid)
    end
    # sampling grid is not trainable
    Flux.@layer Spatial_Transformer trainable = (symvae, transformerb)
    # if something is not available, search symvae model
    function Base.getproperty(m::Spatial_Transformer, s::Symbol)
        if (s in propertynames(m))
            return getfield(m, s)
        else
            return getfield(m.symvae, s)
        end
    end
end

# ╔═╡ 6720843c-be73-49bc-a9ec-9a7f599a1f98
begin
	"""
	MultiSpatial_Transformer
	"""
    struct MultiSpatial_Transformer{T1,T2,T3}
        symvae::T1
        transformerb::T2
        sampling_grid::T3
    end
    function (m::MultiSpatial_Transformer)(x, nnoise::Bool, cnoise::Bool, temperature)
        return m(x, nothing, nnoise, cnoise, temperature)
    end
    function (m::MultiSpatial_Transformer)(x, condition, nnoise::Bool, cnoise::Bool, temperature)
        S = m.transformerb(x)
		Shifts = chunk(S.shifts, size(S.shifts, 1), dims=1)
		inv_Shifts = chunk(S.inv_shifts, size(S.inv_shifts, 1), dims=1)
		Xt = map(Shifts) do s
			shift_traces_Fourier(x, s, m.sampling_grid)
		end
        xt = cat(Xt..., dims=3)
        output = m.symvae(xt, condition, nnoise, cnoise, temperature)

		Xhat = chunk(output.X.xhat, size(S.shifts, 1), dims=3)
		Xhat_inv_shifted = map(Xhat, inv_Shifts) do x, s
			shift_traces_Fourier(x, s, m.sampling_grid)
		end
        xhat = mean(Xhat_inv_shifted)
        return (; X=(; xhat, xshifted=xt, xdecomposed=Xhat, xhat_logvar=output.X.xhat_logvar), Z=output.Z, shifts=S.shifts)
    end
    # function (m::MultiSpatial_Transformer)(xc, xn, nnoise::Bool, cnoise::Bool, temperature)
    #     Sc = m.transformerb(xc)
    #     Sn = m.transformerb(xn)
    #     xct = shift_traces_Fourier(xc, Sc.shifts, m.sampling_grid)
    #     xnt = shift_traces_Fourier(xn, Sn.shifts, m.sampling_grid)
    #     output = m.symvae(xct, xnt, nnoise, cnoise, temperature)
    #     xhat = shift_traces_Fourier(output.X.xhat, Sn.inv_shifts, m.sampling_grid)
    #     return (; X=(; xhat, xshifted=xt, xhat_logvar=output.X.xhat_logvar), Z=output.Z, shifts=S.shifts)
    # end
    # estimate either coherent or nuisance code after applying transformer, for each element of D
    function (m::MultiSpatial_Transformer)(D::Vector{T}, code_type::Union{Val{:coherent},Val{:nuisance}}; apply_transformer=true) where {T}
        Dshifted = if (apply_transformer)
            m(D, Val(:transform))
        else
            D
        end
        return m.symvae(Dshifted, code_type)
    end
    # estimate either coherent or nuisance code after applying transformer
    function (m::MultiSpatial_Transformer)(d, code_type::Union{Val{:coherent},Val{:nuisance}}; apply_transformer=true)
        dshifted = if (apply_transformer)
            m(d, Val(:transform))
        else
            d
        end
        return m.symvae(dshifted, code_type)
    end
    function (m::MultiSpatial_Transformer)(D::Vector{T}, ::Val{:transform}) where {T}
        return map(D) do d
			S = m.transformerb(d)
			Shifts = chunk(S.shifts, size(S.shifts, 1), dims=1)
			Xt = map(Shifts) do s
			shift_traces_Fourier(d, s, m.sampling_grid)
			end
        	dt = cat(Xt..., dims=3)
        end
    end
    function (m::MultiSpatial_Transformer)(d, ::Val{:transform})
        S = m.transformerb(d)
			Shifts = chunk(S.shifts, size(S.shifts, 1), dims=1)
			Xt = map(Shifts) do s
			shift_traces_Fourier(d, s, m.sampling_grid)
			end
        dt = cat(Xt..., dims=3)
        return dt
    end
    function (m::MultiSpatial_Transformer)(D::Vector{T}, ::Val{:inv_transform}) where {T}
       return map(D) do d
			S = m.transformerb(d)
			Shifts = chunk(S.inv_shifts, size(S.shifts, 1), dims=1)
			Xt = map(Shifts) do s
			shift_traces_Fourier(d, s, m.sampling_grid)
			end
        	dt = cat(Xt..., dims=3)
        end
    end
    function (m::MultiSpatial_Transformer)(d, ::Val{:inv_transform})
       S = m.transformerb(d)
			Shifts = chunk(S.inv_shifts, size(S.shifts, 1), dims=1)
			Xt = map(Shifts) do s
			shift_traces_Fourier(d, s, m.sampling_grid)
			end
        dt = cat(Xt..., dims=3)
        return dt
    end
    # sampling grid is not trainable
    Flux.@layer MultiSpatial_Transformer trainable = (symvae, transformerb)
    # if something is not available, search symvae model
    function Base.getproperty(m::MultiSpatial_Transformer, s::Symbol)
        if (s in propertynames(m))
            return getfield(m, s)
        else
            return getfield(m.symvae, s)
        end
    end
end

# ╔═╡ f4238f8e-3596-4c98-a64e-477c3aa2b054
md"## Nuisance Optimization Model"

# ╔═╡ ab3f0de4-3e7e-4d5f-aa1a-25fe24a52b38
md"## Losses"

# ╔═╡ a636c206-8ded-4f13-b010-9a33dd99f80e
function _split_batch_condition(batch)
    if !(batch isa Tuple)
        return batch, nothing
    end
    x = batch[1]
    if length(batch) == 2
        second = batch[2]
        # Distinguish (x, condition) from (x, class_labels) safely.
        if second === nothing
            return x, nothing
        elseif (second isa AbstractArray) && (ndims(second) <= 2)
            return x, second
        else
            return x, nothing
        end
    elseif length(batch) >= 3
        return x, batch[3]
    end
    return x, nothing
end

# ╔═╡ 6d34151a-b539-4eda-995a-b7f873719a4c
"""
Reconstruction Loss (like deterministic AE)
"""
function loss_mse(model, x)
    xb, condition = _split_batch_condition(x)
    result = model(xb, condition, false, false, 0f0)
    return Flux.mse(result.X.xhat, xb)
end

# ╔═╡ 26aafe3d-e783-4591-959d-910b9c050301
begin
    """
        average_column_correlation(X::AbstractMatrix, ref::AbstractVector)

    Compute the average Pearson correlation coefficient between a reference vector `ref` 
    and each column of the matrix `X`.

    # Arguments
    - `X`: A matrix of size `(m, n)`, where each column is compared to `ref`.
    - `ref`: A vector of length `m`, the reference vector.

    # Returns
    - A scalar: the average Pearson correlation coefficient across all columns.
    """
    function average_column_correlation(X, ref::AbstractVector)
        ref_mean = mean(ref)
        ref_var = sum((ref .- ref_mean) .^ 2)

        total_correlation = 0.0
        n_cols = size(X, 2)

        for col in eachcol(X)
            col_mean = mean(col)
            col_var = sum((col .- col_mean) .^ 2)
            covariance = sum((ref .- ref_mean) .* (col .- col_mean))
            correlation = covariance / sqrt(ref_var * col_var)
            total_correlation += correlation
        end

        return total_correlation / n_cols
    end
    function average_column_correlation(Xv, ref::AbstractMatrix)
        @assert size(ref, 2) == length(Xv)
        return map(Xv, eachcol(ref)) do X, r
            average_column_correlation(X, r)
        end
    end
    """
   for ref: first dim is time, second dim is trial waveforms, and last dims should be the length of Xv
   """
    function average_column_correlation(Xv, ref::AbstractArray)
        @assert ndims(ref) == 3 "ref must be a 3D array"
        Ref = unstack(ref, dims=2)
        corr_vals = map(Ref) do r
            average_column_correlation(Xv, r)
        end
        return hcat(corr_vals...)
    end

end

# ╔═╡ c4de28ea-5af6-4c66-9955-a5d168833036
"""
Train to update nuisance codes
"""
function update_nuisance_codes(
    optimal_nuisance_model,
    D::Vector{T}, # make sure D is already transformed
    alpha;
    nepochs=100,
    learning_rate=0.01,
    p=1, show_corr=false
) where {T}
    # using Adam here is leading to a lot of stochastic behaviour, don't know why
    opt_state = Optimisers.setup(Optimisers.AdamW(learning_rate), optimal_nuisance_model)
    loss_history = []
    corr_loss = []
    Array{Float32,3}(undef, nepochs, size(optimal_nuisance_model.nuisance_codes, 2), length(D))
    coherent_codes = optimal_nuisance_model.model(D, Val(:coherent), apply_transformer=false)
    xsave = optimal_nuisance_model(coherent_codes)
    lkl = loss_kl_Qc_accumulated(optimal_nuisance_model, coherent_codes, alpha; p=p)
    Brefall = lkl.kl
    Cref = lkl.L

    loss(optimal_nuisance_model, coherent_codes) = loss_kl_Qc_accumulated(optimal_nuisance_model, coherent_codes, alpha,
        p=p, Bref=sum(Brefall), Cref=Cref).loss


    @progress for epoch = 1:nepochs
        g = Flux.gradient(loss, optimal_nuisance_model, coherent_codes)[1]
        Optimisers.update!(opt_state, optimal_nuisance_model, g)
        if (mod(epoch, 5) == 0)
            lkl = loss_kl_Qc_accumulated(optimal_nuisance_model, coherent_codes, alpha,
                p=p, Bref=sum(Brefall), Cref=Cref)
            corrl = average_column_correlation(D, lkl.virtualdata)
            @info (; epoch=epoch, correlation_virtualdata=dropdims(maximum(corrl, dims=2), dims=2), loss=lkl.loss, norm_virtualdata=lkl.L)
            push!(loss_history, epoch => lkl.loss)
            push!(corr_loss, epoch => corrl)
        end
    end
    xnew = optimal_nuisance_model(coherent_codes)
    return (; xsave, xnew, loss_history)
end

# ╔═╡ 0f292b15-bc79-4424-b5ae-fded09eb16f0
#=╠═╡
begin
    @time average_column_correlation(Xv[1], ref)
    @time average_column_correlation(Xv, ref2)
    @time average_column_correlation(xpu(Xv), xpu(ref3))
end
  ╠═╡ =#

# ╔═╡ 7931ce6a-a062-4c7f-bb04-82b822e04eab
md"""
## Training
- ntau: number of waveforms used per step (as large as your GPU memory supports)
- beta: custom weight of the KL terms in this order [[nuisance], [coherent, lambda]]
- batchsize: because of stochastic gradient descent
- nsteps: how long you want a single epoch to be?
- gamma: 
- tau:
- nepoch:
- nprint:
- initial_learning_rate:
"""

# ╔═╡ 61fbbf84-1b2b-4de9-8763-800f76c851e0
md"## Redatuming"

# ╔═╡ 5d25db74-dcc2-49d6-b62a-39f612089e9f
md"## Plots"

# ╔═╡ 0a2fb24a-3c7a-46ed-ad54-d35fb72f8031
md"## Misc"

# ╔═╡ 04f9b328-edc8-4b1e-9a7c-79a215b1cf5f
begin
    struct SymVAE{T1,T2,T3,T4}
        sencb::T1
        nencb::T2
        decb::T3
        cond_proj::T4
        condition_dim::Int
    end
    function SymVAE(NN, k=1; condition_dim::Int=0, condition_embed_dim::Int=0)
        senc_core = if (k > 1)
            BroadcastCoherentEncCategorical(NN.senc, NN.senc_μ, NN.senc_loginvvar, NN.senc_λlogits)
        else
            BroadcastCoherentEnc(NN.senc, NN.senc_μ, NN.senc_loginvvar)
        end
        nenc_core = BroadcastNuisanceEnc(NN.nenc, NN.nenc_μ, NN.nenc_logσ)
        # FiLM path: encoder chain is FiLMConv1DChain → pass cond directly via 2-arg callable
        # Concat path: wrap with ConditionConcat so cond is prepended to input
        if (condition_dim <= 0)
            sencb = senc_core
            nencb = nenc_core
        else
            sencb = (senc_core.chain isa FiLMConv1DChain) ? senc_core : ConditionalChain(ConditionConcat(), senc_core)
            nencb = (nenc_core.chain isa FiLMConv1DChain) ? nenc_core : ConditionalChain(ConditionConcat(), nenc_core)
        end
        decb = BroadcastDec(NN.dec, NN.dec_logvar)
        cond_proj = if (condition_dim > 0)
            cembed = condition_embed_dim > 0 ? condition_embed_dim : condition_dim
            xpu(Chain(Dense(condition_dim, cembed, activation), Flux.LayerNorm(cembed)))
        else
            xpu(Chain())
        end

        model = SymVAE(sencb, nencb, decb, cond_proj, condition_dim)
    end
    function (m::SymVAE)(x, nnoise, cnoise, temperature)
        return m(x, nothing, nnoise, cnoise, temperature)
    end
    function (m::SymVAE)(x, condition, nnoise, cnoise, temperature)
        cond_embedding = _get_condition_embedding(m.cond_proj, m.condition_dim, x, condition)
        C = m.sencb(x, cond_embedding)
        N = m.nencb(x, cond_embedding)
        xhat, xhat_logvar = sample_q_decode(C..., N..., nnoise, cnoise, m.decb, temperature; cond_embedding)
        return (;
            Z=(; N, C),
            X=(; xhat, xhat_logvar),
            shifts=[0f0]
        )
    end
    function (m::SymVAE)(xc, xn, condition, nnoise, cnoise, temperature)
        xn_eff = xn === nothing ? xc : xn
        cond_embedding = _get_condition_embedding(m.cond_proj, m.condition_dim, xn_eff, condition)
        C = m.sencb(xc, cond_embedding)
        N = m.nencb(xn_eff, cond_embedding)
        xhat, xhat_logvar = sample_q_decode(C..., N..., nnoise, cnoise, m.decb, temperature; cond_embedding)
        return (;
            Z=(; N, C),
            X=(; xhat, xhat_logvar),
            shifts=[0f0]
        )
    end
    # get coherent code for each element of D
    function (m::SymVAE)(D::Vector{T}, ::Val{:coherent}; condition=nothing, args...) where {T}
        if (condition === nothing)
            return map(D) do d
                m(d, Val(:coherent); condition=nothing, args...)
            end
        end

        cond = Float32.(condition)
        @assert ndims(cond) == 2 "Vector conditioning must be a matrix with shape (condition_dim, ngroups)."
        @assert size(cond, 2) == length(D) "Condition group count mismatch. Expected $(length(D)), got $(size(cond, 2))."
        return map(enumerate(D)) do (i, d)
            m(d, Val(:coherent); condition=reshape(cond[:, i], :, 1), args...)
        end
    end
    # get nuisance code for each element of D
    function (m::SymVAE)(D::Vector{T}, ::Val{:nuisance}; condition=nothing, args...) where {T}
        if (condition === nothing)
            return map(D) do d
                m(d, Val(:nuisance); condition=nothing, args...)
            end
        end

        cond = Float32.(condition)
        @assert ndims(cond) == 2 "Vector conditioning must be a matrix with shape (condition_dim, ngroups)."
        @assert size(cond, 2) == length(D) "Condition group count mismatch. Expected $(length(D)), got $(size(cond, 2))."
        return map(enumerate(D)) do (i, d)
            m(d, Val(:nuisance); condition=reshape(cond[:, i], :, 1), args...)
        end
    end
    # get coherent code for d
    function (m::SymVAE)(d, ::Val{:coherent}; condition=nothing, args...)
        cond_embedding = _get_condition_embedding(m.cond_proj, m.condition_dim, d, condition)
        return m.sencb(d, cond_embedding)
    end
    # get nuisance code for d
    function (m::SymVAE)(d, ::Val{:nuisance}; condition=nothing, args...)
        cond_embedding = _get_condition_embedding(m.cond_proj, m.condition_dim, d, condition)
        return m.nencb(d, cond_embedding)
    end
    function (m::SymVAE)(d, ::Val{:transform})
        return d
    end
    Flux.@layer SymVAE trainable = (sencb, nencb, decb, cond_proj)
end

# ╔═╡ 186713a3-c357-427c-9af4-6739de2d33c5
begin
    struct SymVAETrunk{T1,T2,T3,T4,T5}
        tencb::T1
        sencb::T2
        nencb::T3
        decb::T4
        cond_proj::T5
        condition_dim::Int
    end
    function SymVAETrunk(NN, k=1; condition_dim::Int=0, condition_embed_dim::Int=0)
        tencb = NN.tencb
        senc_core = if (k > 1)
            BroadcastCoherentEncCategorical(NN.senc, NN.senc_μ, NN.senc_loginvvar, NN.senc_λlogits)
        else
            BroadcastCoherentEnc(NN.senc, NN.senc_μ, NN.senc_loginvvar)
        end
        nenc_core = BroadcastNuisanceEnc(NN.nenc, NN.nenc_μ, NN.nenc_logσ)
        if (condition_dim <= 0)
            sencb = senc_core
            nencb = nenc_core
        else
            sencb = (senc_core.chain isa FiLMConv1DChain) ? senc_core : ConditionalChain(ConditionConcat(), senc_core)
            nencb = (nenc_core.chain isa FiLMConv1DChain) ? nenc_core : ConditionalChain(ConditionConcat(), nenc_core)
        end
        decb = BroadcastDec(NN.dec, NN.dec_logvar)
        cond_proj = if (condition_dim > 0)
            cembed = condition_embed_dim > 0 ? condition_embed_dim : condition_dim
            xpu(Chain(Dense(condition_dim, cembed, activation), Flux.LayerNorm(cembed)))
        else
            xpu(Chain())
        end

        model = SymVAETrunk(tencb, sencb, nencb, decb, cond_proj, condition_dim)
    end
    function (m::SymVAETrunk)(x, nnoise, cnoise, temperature)
        return m(x, nothing, nnoise, cnoise, temperature)
    end
    function (m::SymVAETrunk)(x, condition, nnoise, cnoise, temperature)
        cond_embedding = _get_condition_embedding(m.cond_proj, m.condition_dim, x, condition)
        xt = m.tencb(x)
        C = m.sencb(xt, cond_embedding)
        N = m.nencb(xt, cond_embedding)
        xhat, xhat_logvar = sample_q_decode(C..., N..., nnoise, cnoise, m.decb, temperature; cond_embedding)
        return (;
            Z=(; N, C),
            X=(; xhat, xhat_logvar),
            shifts=[0f0]
        )
    end
    function (m::SymVAETrunk)(xc, xn, condition, nnoise, cnoise, temperature)
        xn_eff = xn === nothing ? xc : xn
        cond_embedding = _get_condition_embedding(m.cond_proj, m.condition_dim, xn_eff, condition)
        xtc = m.tencb(xc)
        xtn = m.tencb(xn_eff)
        C = m.sencb(xtc, cond_embedding)
        N = m.nencb(xtn, cond_embedding)
        xhat, xhat_logvar = sample_q_decode(C..., N..., nnoise, cnoise, m.decb, temperature; cond_embedding)
        return (;
            Z=(; N, C),
            X=(; xhat, xhat_logvar),
            shifts=[0f0]
        )
    end
    # get coherent code for each element of D
    function (m::SymVAETrunk)(D::Vector{T}, ::Val{:coherent}; condition=nothing, args...) where {T}
        if (condition === nothing)
            return map(D) do d
                m(d, Val(:coherent); condition=nothing, args...)
            end
        end

        cond = Float32.(condition)
        @assert ndims(cond) == 2 "Vector conditioning must be a matrix with shape (condition_dim, ngroups)."
        @assert size(cond, 2) == length(D) "Condition group count mismatch. Expected $(length(D)), got $(size(cond, 2))."
        return map(enumerate(D)) do (i, d)
            m(d, Val(:coherent); condition=reshape(cond[:, i], :, 1), args...)
        end
    end
    # get nuisance code for each element of D
    function (m::SymVAETrunk)(D::Vector{T}, ::Val{:nuisance}; condition=nothing, args...) where {T}
        if (condition === nothing)
            return map(D) do d
                m(d, Val(:nuisance); condition=nothing, args...)
            end
        end

        cond = Float32.(condition)
        @assert ndims(cond) == 2 "Vector conditioning must be a matrix with shape (condition_dim, ngroups)."
        @assert size(cond, 2) == length(D) "Condition group count mismatch. Expected $(length(D)), got $(size(cond, 2))."
        return map(enumerate(D)) do (i, d)
            m(d, Val(:nuisance); condition=reshape(cond[:, i], :, 1), args...)
        end
    end
    # get coherent code for d
    function (m::SymVAETrunk)(d, ::Val{:coherent}; condition=nothing, args...)
        xt = m.tencb(d)
        cond_embedding = _get_condition_embedding(m.cond_proj, m.condition_dim, d, condition)
        return m.sencb(xt, cond_embedding)
    end
    # get nuisance code for d
    function (m::SymVAETrunk)(d, ::Val{:nuisance}; condition=nothing, args...)
        xt = m.tencb(d)
        cond_embedding = _get_condition_embedding(m.cond_proj, m.condition_dim, d, condition)
        return m.nencb(xt, cond_embedding)
    end
    function (m::SymVAETrunk)(::Any, ::Val{:transform})
        return nothing
    end
    Flux.@layer SymVAETrunk trainable = (tencb, sencb, nencb, decb, cond_proj)
end

# ╔═╡ 56055fa2-8f96-4b43-b1eb-2cd5a9cbadeb
begin
    """
    Spatial State Transformer 
    """
    struct BroadcastSpatialStateTransformer{T1}
        chain::T1
    end
    Flux.@layer BroadcastSpatialStateTransformer trainable = (chain)
    function (m::BroadcastSpatialStateTransformer)(x)
        X = add_dim3_reshape(x)
        s1 = m.chain(X)
        s = mean(s1, dims=ndims(s1) - 1)
        s = dropdims(s, dims=ndims(s) - 1)
        s = Flux.stack(fill(s, size(s1, ndims(s1) - 1)), dims=ndims(s1) - 1)
        return (; shifts=s, inv_shifts=-1.0f0 .* s)
    end
end

# ╔═╡ 90e69a62-b340-45d6-944e-cb56f6f46ca6
begin
    """
    Spatial Transformer 
    """
    struct BroadcastSpatialTransformer{T1}
        chain::T1
    end
    Flux.@layer BroadcastSpatialTransformer trainable = (chain)
    function (m::BroadcastSpatialTransformer)(x)
        X = add_dim3_reshape(x)
        s = m.chain(X)
        return (; shifts=s, inv_shifts=-1.0f0 .* s)
    end
end

# ╔═╡ 1121e34c-ca35-4f68-8283-eca514928654
"""
get symae
"""
function get_symae(para)
    Random.seed!(para.seed)

    condition_embed_dim = if (para.condition_dim > 0)
        if (para.condition_embed_dim > 0)
            para.condition_embed_dim
        else
            para.condition_dim
        end
    else
        0
    end

    model =
        if (typeof(para.network_type) == ConvAE)
            SymVAE(
                get_symae(para.network_type, para.nt, para.p, para.q, para.k, condition_embed_dim),
                para.k;
                condition_dim=para.condition_dim,
                condition_embed_dim=condition_embed_dim,
            )
        else
            SymVAE(
                get_symae(para.network_type, para.nt, para.p, para.q, para.k, condition_embed_dim),
                para.k;
                condition_dim=para.condition_dim,
                condition_embed_dim=condition_embed_dim,
            )
        end
    # commented if we want to use full transformer, not just time shift transformer (later)
    # 		sampling_grid =
    #            cat(
    #                reshape(collect(LinRange(-1.0, 1.0, para.nt)), 1, :, 1),
    #                ones(1, para.nt, 1),
    #                dims=1,
    #            ) |> xpu

    # Wrap with spatial transformer if necessary
    # sampling_grid =
    #     cat(
    #         reshape(collect(LinRange(-1.0, 1.0, para.nt)), 1, :, 1),
    #         ones(1, para.nt, 1),
    #         dims = 1,
    #     ) |> xpu
    sampling_grid = xpu(
        im .*
        Float32.(
            fftfreq(para.nt) * 2.0f0 * pi * para.nt * 0.5f0
        ),
    )
    model =
        if (para.transformer == :spatial)
			transformer = get_dense_transformer(para.nt, nt_out=para.transformer_k, nt_hidden=para.nt)
			transformerb = BroadcastSpatialTransformer(transformer)
			if(para.transformer_k == 1)
            	Spatial_Transformer(model, transformerb, sampling_grid)
			elseif(para.transformer_k > 1)
				MultiSpatial_Transformer(model, transformerb, sampling_grid)
			else
				error("invalid transformer_k")
			end
        elseif (para.transformer == :spatial_state)
            transformer = get_dense_transformer(para.nt, nt_hidden=para.nt)
            transformerb = BroadcastSpatialStateTransformer(transformer)
            Spatial_Transformer(model, transformerb, sampling_grid)
        else
            model
        end

    loss_history = (
        train_mse=Vector{Float32}(undef, 0),
        test_mse=Vector{Float32}(undef, 0),
        train_neg_llh=Vector{Float32}(undef, 0),
        test_neg_llh=Vector{Float32}(undef, 0),
        train_kl=Vector{Float32}(undef, 0),
        test_kl=Vector{Float32}(undef, 0),
        train_neg_elbo=Vector{Float32}(undef, 0),
        test_neg_elbo=Vector{Float32}(undef, 0),
    )
    return model, loss_history
end

# ╔═╡ 84aa58d3-115b-4d2d-a798-fc936f5bb2ca
begin
    struct Model_Optimal_Nuisance{T1,T2,T3,T4}
        nuisance_codes::T1
        model::T2
        shifted_init::T3
        λlogits_cond::T4
    end
    function (m::Model_Optimal_Nuisance)(coherent_codes)
        coherent_codes_λ = apply_λlogits_cond(coherent_codes, m.λlogits_cond)
        nx = add_dim3_reshape(m.nuisance_codes)
        xhat, _ = sample_q_decode(coherent_codes_λ..., nx, nothing, false, false, m.model.decb, 0f0)
        xhat = xhat .- mean(xhat, dims=1) # NECESSARY?
        C_all = m.model(reshape(xhat, size(xhat, 1), 1, size(xhat, 2) * size(xhat, 3)), Val(:coherent), apply_transformer=false)
        C_all_batched = C_all
        # C_all_batched = batch_coherent_codes(C_all, m.λlogits_cond)
        cμ_all = C_all_batched.μ
        clogσ_all = C_all_batched.logσ
        cμ_all = reshape(cμ_all, size(cμ_all, 1), size(xhat, 2), size(xhat, 3))
        clogσ_all = reshape(clogσ_all, size(clogσ_all, 1), size(xhat, 2), size(xhat, 3))
        return xhat, cμ_all, clogσ_all
    end
    function (m::Model_Optimal_Nuisance)(nuisance_codes, C)
        copyto!(m.nuisance_codes, nuisance_codes)
        return m(C)
    end
    Flux.@layer Model_Optimal_Nuisance trainable = (nuisance_codes)
end

# ╔═╡ a46ce69a-3081-4e86-b9eb-c773c772f459
"""
Create an instance of Model_Optimal_Nuisance
"""
function get_Model_Optimal_Nuisance(para, model; kopt=1, init=nothing)
    λlogits_cond = add_dim3_reshape(xpu([i == kopt ? 1f0 : 0f0 for i in 1:para.k]))
    initial_nuisance_codes, shifted_init = if (init === nothing)
        xpu(randn(para.q)), nothing
    else
        output = model(init, Val(:nuisance))
        output.μ, model(init, Val(:transform))
    end
    return Model_Optimal_Nuisance(xpu(initial_nuisance_codes), model, shifted_init, λlogits_cond)
end

# ╔═╡ 8967b2d2-6688-4ac6-a3ec-357d4ce3e6f0
"""
Compute KL divergence between `Q(c|x_i)` and `Q(c|x)` for each i.
Here `Q(c|x)` is the accumulated coherent information across all the instances, and 
`Q(c|x_i)` is the posterior for the ith instance.
- make sure `doptim` is already transformed
"""
function kl_Qc_accumulated(symae_para, model, D, n, doptim, kopt, alpha=0.0)
    coherent_code_all = model(doptim, Val(:coherent), apply_transformer=false)
    # coherent_code_all_batched = batch_coherent_codes(coherent_code_all, kopt)
    coherent_code_all_batched = batch_coherent_codes(coherent_code_all)
    kls = map(D) do d
        optimal_nuisance_model = get_Model_Optimal_Nuisance(symae_para, model, init=d, kopt=kopt)

        lkl = loss_kl_Qc_accumulated(
            optimal_nuisance_model,
            [coherent_code_all_batched],
            alpha,
        )
        return cpu(vec(lkl.kl))
    end
    I = vcat(map(enumerate(D)) do (i, d)
        fill(i, size(d, 2))
    end...)
    J = vcat(map(D) do d
        collect(1:size(d, 2))
    end...)
    perm = sortperm(vcat(kls...))
    return kls, I[perm[1:n]], J[perm[1:n]]
end

# ╔═╡ b3428bd8-c1ec-4e48-97b0-3dbde763292b
"""
**Primary Coherent Information Extraction Function**

This is the main function that implements **Objective 1** of the SymVAE framework: extracting 
coherent information from groups of waveforms. After training, this function performs a 
sophisticated optimization process to recover the pure coherent signal from each group by 
finding optimal nuisance codes that minimize reconstruction error.

## Core Algorithm Steps:

### 1. **Subgroup Filtering (Objective 2 Implementation)**
```julia
_, mode_indices = filter_mode(Doptim, model, kopt, p=filter_probability)
```
- Identifies waveforms belonging to coherent category `kopt` with probability ≥ `filter_probability`  
- Implements automatic subgroup identification within each provided group
- Filters out waveforms that don't strongly belong to the target coherent category

### 2. **Spatial Transformation and Mode Selection**
```julia
Dtransformed = model(Doptim, Val(:transform))
Dcc_mode = map(Dtransformed, mode_indices) do d, I
    d[:, I]  # Select only high-probability waveforms
end
```
- Applies learned spatial transformations (time shifts) to align waveforms
- Extracts subgroups of waveforms with strong coherent category membership
- Results in cleaned datasets for coherent information extraction

### 3. **Initial Nuisance Code Selection**
```julia
kl_Qcs, ideal_pixel_ids, ideal_seismogram_ids = kl_Qc_accumulated(para, model, DN, N, dcc_mode, kopt, 0.0)
```
- Identifies `N` "ideal" waveforms that best represent the coherent information
- Uses KL divergence analysis to find instances closest to the coherent distribution
- These serve as initialization points for nuisance optimization

### 4. **Nuisance Code Optimization**
```julia
optimal_nuisance_result = update_nuisance_codes(optimal_nuisance_model, Dcc_mode, alpha, ...)
```
- **Core Innovation**: Optimizes nuisance codes to minimize KL divergence between:
  - Accumulated coherent distribution q(c|X) across all group instances  
  - Instance-specific coherent distributions q(c|x_i) from reconstructed data
- Iterative optimization finds nuisance codes that best "explain away" non-coherent variations

### 5. **Coherent Signal Recovery**
```julia
coherent_codes_all = model(Dcc_mode, Val(:coherent), apply_transformer=false)
xhat, _, _ = optimal_nuisance_model(coherent_codes_all)
```
- Reconstructs waveforms using learned coherent codes + optimized nuisance codes
- The reconstruction `xhat` represents the **pure coherent information** with instance-specific nuisance effects removed

## Function Arguments:

### Input Data Arguments:
- **`Doptim::Vector`**: Groups of waveforms for coherent information extraction
  - Each element `Doptim[i]` contains all waveforms for group i
  - These groups will have their coherent information extracted simultaneously

- **`Dnuisance`**: Optional groups for nuisance initialization 
  - If `nothing`, uses `Doptim` for both coherent and nuisance learning
  - Allows separation of coherent extraction groups from nuisance reference groups

### Model and Optimization Arguments:
- **`para`**: SymAE parameter configuration (SymAE_Para struct)
- **`model`**: Trained SymVAE model
- **`N=10`**: Number of optimized nuisance codes to initialize
- **`nepochs=100`**: Optimization iterations for nuisance code learning
- **`alpha=0.1`**: Regularization strength (higher = more regularization)
- **`learning_rate=0.001`**: Optimization step size
- **`p=1`**: Norm type for coherent signal regularization (L1 or L2)

### Subgroup Selection Arguments:
- **`kopt=1`**: Target coherent category index (for multi-category models)
- **`filter_probability=0.9`**: Minimum probability threshold for subgroup membership
- **`show_corr=false`**: Display correlation metrics during optimization

## Return Values:

1. **`output`**: Extracted coherent signals for each group
   - Vector of matrices, where `output[i]` contains coherent waveforms for group i
   - These represent the "source signature" with propagation effects removed

2. **`Ideal_seismograms`**: Initial reference waveforms used for optimization
   - Best representative instances identified by KL analysis

3. **`optim_nuisance_loss_history`**: Optimization convergence history
   - Tracks loss reduction during nuisance code optimization

4. **`shifted_init`**: Spatially transformed initial waveforms
   - Shows effect of learned time shift corrections

## Practical Usage Example:

```julia
# After training SymVAE model
para = SymAE_Para(nt=512, p=32, k=2, q=16)
model, _ = get_symae(para)
# ... training completed ...

# Extract coherent information from multiple recording sites
recording_sites = [site1_data, site2_data, site3_data]

coherent_signals, references, history, _ = get_coherent_information(
    para, model, recording_sites,
    N=15,                    # Use 15 optimized nuisance codes
    nepochs=200,            # 200 optimization iterations  
    alpha=0.15,             # Moderate regularization
    kopt=1,                 # Extract category 1 coherent info
    filter_probability=0.85  # 85% probability threshold
)

# coherent_signals[1] contains extracted source signature for site1
# coherent_signals[2] contains extracted source signature for site2  
# coherent_signals[3] contains extracted source signature for site3
```

## Key Insights:

### **Group-Based Processing**: 
Each group in `Doptim` is processed independently, allowing extraction of different coherent signatures from different geological/acquisition contexts.

### **Automatic Subgroup Discovery**:
The `filter_mode` step automatically identifies and processes only the most coherent instances within each group, implementing the subgroup identification objective.

### **Coherent-Nuisance Separation**:
The optimization process finds nuisance codes that, when combined with the learned coherent representation, minimize the difference between the accumulated coherent distribution and instance-specific distributions.

### **Quality Metrics**:
The function monitors "stacking correlation" and optimization loss to ensure successful coherent information extraction.

This function represents the culmination of the SymVAE training process, transforming learned representations into practical coherent signals for seismic analysis and interpretation.
"""
function get_coherent_information(para, model, Doptim::Vector, Dnuisance=nothing; N=10, nepochs=100, alpha=0.1, learning_rate=0.001, initialize_within_state=true, p=1, kopt=1, filter_probability=0.9, show_corr=false)

    _, mode_indices = filter_mode(Doptim, model, kopt, p=filter_probability)
    Dtransformed = model(Doptim, Val(:transform))
    Dcc_mode = map(Dtransformed, mode_indices) do d, I
        d[:, I]
    end
    @info "Stacking correlation" average_leave_one_out_correlation(Dcc_mode)
    Ideal_seismograms = cat(map(Doptim, Dcc_mode) do doptim, dcc_mode
            if (Dnuisance === nothing)
                DN = [doptim]
            else
                DN = Dnuisance
            end
            kl_Qcs, ideal_pixel_ids, ideal_seismogram_ids = kl_Qc_accumulated(para, model, DN, N, dcc_mode, kopt, 0.0)

            ideal_seismograms = mapreduce(hcat, ideal_pixel_ids, ideal_seismogram_ids) do ideal_pixel_id, ideal_seismogram_id
                DN[ideal_pixel_id][:, ideal_seismogram_id]
            end
            return ideal_seismograms
        end..., dims=3)

    optimal_nuisance_model = get_Model_Optimal_Nuisance(
        para,
        model,
        init=Ideal_seismograms,
        kopt=kopt
    )


    optimal_nuisance_result = update_nuisance_codes(optimal_nuisance_model, Dcc_mode, alpha, nepochs=nepochs, learning_rate=learning_rate, p=p, show_corr=show_corr)


    optim_nuisance_loss_history = optimal_nuisance_result.loss_history

    # apply transformer is false because we Dcc_mode is already transformed
    coherent_codes_all = model(Dcc_mode, Val(:coherent), apply_transformer=false)

    xhat, _, _ = optimal_nuisance_model(coherent_codes_all)
    output = cpu(unstack(xhat, dims=3))
    # @show norm(output), norm(optimal_nuisance_model.shifted_init)
    return output, Ideal_seismograms, optim_nuisance_loss_history, optimal_nuisance_model.shifted_init
end

# ╔═╡ d62474f6-48fb-49fe-919b-c4373135067c
"""
First, get optimal nuisance codes and output coherent information in the data space.
# Arguments
- `doptim`: the KL for this state will be minimized
- `Dnuisance`: initial nuisances will be selected from these states
- `Doutput`: after optimization, the coherent information from these states will be output
- `N`: number of initial nuisance codes
"""
function get_coherent_information(para, model, doptim::T, Dnuisance::Vector{T}, Doutput::Vector{T}; N=10, nepochs=100, alpha=0.1, learning_rate=0.001, p=1, kopt=1, filter_probability=0.9, show_corr=false) where {T}

    doptim_transformed = model(doptim, Val(:transform))
    _, mode_indices = filter_mode(doptim, model, kopt, p=filter_probability)
    doptim_transformed_mode = doptim_transformed[:, mode_indices]

    Doutput_transformed = model(Doutput, Val(:transform))
    _, mode_indices_output = filter_mode(Doutput, model, kopt, p=filter_probability)
    Doutput_transformed_mode = map(Doutput_transformed, mode_indices_output) do d, I
        d[:, I]
    end

    @info "Stacking correlation" average_leave_one_out_correlation(Doutput)

    kl_Qcs, ideal_pixel_ids, ideal_seismogram_ids = kl_Qc_accumulated(para, model, Dnuisance, N, doptim_transformed_mode, kopt, 0.0)

    ideal_seismograms = mapreduce(hcat, ideal_pixel_ids, ideal_seismogram_ids) do ideal_pixel_id, ideal_seismogram_id
        Dnuisance[ideal_pixel_id][:, ideal_seismogram_id]
    end


    # optimal_nuisance_results = map(1:size(ideal_seismograms,2)) do I
    optimal_nuisance_model = get_Model_Optimal_Nuisance(
        para,
        model,
        init=ideal_seismograms,
        kopt=kopt,
    )
    optimal_nuisance_result = update_nuisance_codes(optimal_nuisance_model, [doptim_transformed_mode], alpha; nepochs=nepochs, learning_rate=learning_rate, p=p, show_corr=show_corr)
    # return  optimal_nuisance_model, optimal_nuisance_result		
    # end

    optim_nuisance_loss_history = optimal_nuisance_result.loss_history
    # hcat([o[2].loss_history for o in optimal_nuisance_results]...)

    coherent_codes_all = model(Doutput_transformed_mode, Val(:coherent), apply_transformer=false)

    output = map(coherent_codes_all) do C
        xhat, _, _ = optimal_nuisance_model(C)
        # 	Xhat = []
        # 	for I in 1:N
        # 		xhat, _, _ = optimal_nuisance_results[I][1](C.cμ, C.clogσ)
        # 		push!(Xhat, cpu(vec(xhat)))
        # 	end
        # 	return hcat(Xhat...)
        return cpu(dropdims(xhat, dims=3))
    end

    return output, ideal_seismograms, optim_nuisance_loss_history, optimal_nuisance_model.shifted_init
end

# ╔═╡ 9b95b3b6-6c98-4c8b-abdc-bbc1410a0df1
function coherent_prior(K, p, λ=1f0)
    # dense K×K identity matrix (no Diagonal, Zygote-safe)
    I_dense = Float32.(Matrix(1.0I, K, K))   # or Matrix{Float32}(I, K, K)

    # upper block: λ * identity
    upper = λ .* I_dense                     # size K×K

    # lower block: zeros
    lower = zeros(Float32, p - K, K)         # size (p-K)×K

    # stack into a p×K matrix
    M = vcat(upper, lower)                   # size p×K

    # flatten and reshape to (p, K, 1) or whatever add_dim3_reshape expects
    return xpu(add_dim3_reshape(vec(M)))
end

# ╔═╡ 0db6b854-ef75-46c0-8e19-efd8faa037bb
"""
create a probability vector that puts almost all the mass on the first class
"""
function focused_categorical_prior(K::Int, sharpness=2.0)
    return softmax(xpu(add_dim3_reshape([i == 1 ? sharpness : -sharpness for i in 1:K])))
end


# ╔═╡ e29c0103-ab96-413f-a739-cd0f97ff3288
function get_kl(μ, logσ, λlogits, betac, betaλ)
    prior = focused_categorical_prior(size(λlogits, 1), 0.0)
	# @show div(size(μ, 1), size(λlogits, 1)), size(λlogits, 1)
	cμ_prior = coherent_prior(size(λlogits, 1), div(size(μ, 1), size(λlogits, 1)))
    return betac * kl_divergence_multivariate_gaussians(μ, logσ, cμ_prior) +
           betaλ * kl_divergence_categorial_distributions(λlogits, prior)
end

# ╔═╡ f5e6cf15-1072-4037-9a96-91db93e730f0
"""
Loss for Variational SymAE
"""
function loss_sym_vae(model, x, beta, gamma, temperature)
    xb, condition = _split_batch_condition(x)
    result = model(xb, condition, true, true, temperature)
	neg_llh = 0.5f0 * sum(@. abs2(result.X.xhat - xb) * exp(-result.X.xhat_logvar) + result.X.xhat_logvar)
    kl = sum(map(result.Z, beta) do z, b
        get_kl(z..., b...)
    end)
    spatial_transformer_norm = norm(result.shifts, 2)
    neg_elbo = neg_llh + kl + gamma * spatial_transformer_norm
    return (; neg_elbo=neg_elbo, neg_llh, kl, spatial_transformer_norm, gamma)
end

# ╔═╡ 0541d3c6-24cf-46b1-95a1-39f3341aec4f
"""
alternate training between (encoder, decoder) and (transformer) of SymAE
"""
function update_alternating_transformer_batchviews(model, loss_history, data_train, data_test, training_para=Training_Para())
    lr_s = Exp(start=training_para.initial_learning_rate, decay=0.99)
    opt_state = Optimisers.setup(Optimisers.AdamW(eta=training_para.initial_learning_rate), model)
    loss(model, data) = loss_sym_vae(
        model,
        data,
        training_para.beta,
        training_para.gamma,
        training_para.temperature,
    ).neg_elbo
    ntau = min(training_para.ntau, minimum(getindex.(size.(data_train), 2)))
    ntau_test = min(training_para.ntau, minimum(getindex.(size.(data_test), 2)))
    # dup_model = Enzyme.Duplicated(model)
    @progress name = "training" for epoch = 1:training_para.nepoch
        Xtrain = get_batchviews(data_train, ntau)
        Xtest = get_batchviews(data_test, ntau_test)

        # compute losses per epoch for a sample
        xtrain = get_sample(Xtrain, training_para.batchsize)
        xtest = get_sample(Xtest, training_para.batchsize)
        push!(loss_history.train_mse, loss_mse(model, xtrain))
        push!(loss_history.test_mse, loss_mse(model, xtest))
        train_loss = loss_sym_vae(model, xtrain, training_para.beta, training_para.gamma, training_para.temperature)
        test_loss = loss_sym_vae(model, xtest, training_para.beta, training_para.gamma, training_para.temperature)
        push!(loss_history.train_neg_llh, train_loss.neg_llh)
        push!(loss_history.test_neg_llh, test_loss.neg_llh)
        push!(loss_history.train_kl, train_loss.kl)
        push!(loss_history.test_kl, test_loss.kl)
        push!(loss_history.train_neg_elbo, train_loss.neg_elbo)
        push!(loss_history.test_neg_elbo, test_loss.neg_elbo)


        # learning rate depending on how many epochs (from loss_history) we have already run
        Optimisers.adjust!(opt_state, lr_s(epoch))


        N = 2 # N epochs before alternating
        # only update encoder and decoders
        Optimisers.freeze!(opt_state.transformerb)
        for i in 1:N
            for i = 1:training_para.nsteps
                x = get_sample(Xtrain, training_para.batchsize)
                # g = Flux.gradient(loss, dup_model, Const(x))[1]
                g = Flux.gradient(loss, model, x)[1]
                Optimisers.update!(opt_state, model, g)
            end
        end
        Optimisers.thaw!(opt_state.transformerb)


        # only update transformer parameter
        Optimisers.freeze!(opt_state.symvae.sencb)
        Optimisers.freeze!(opt_state.symvae.nencb)
        Optimisers.freeze!(opt_state.symvae.decb)
        for i in 1:N
            for i = 1:training_para.nsteps
                x = get_sample(Xtrain, training_para.batchsize)
                # g = Flux.gradient(loss, dup_model, Const(x))[1]
                g = Flux.gradient(loss, model, x)[1]
                Optimisers.update!(opt_state, model, g)
            end
        end
        Optimisers.thaw!(opt_state.symvae.sencb)
        Optimisers.thaw!(opt_state.symvae.nencb)
        Optimisers.thaw!(opt_state.symvae.decb)



        # print output every 10 epochs
        if (mod(epoch, training_para.nprint) == 0)
            @info merge((; epoch=epoch), map(last, loss_history))
        end
    end
    return nothing
end

# ╔═╡ 483e596b-afbc-45d5-84d4-847959c5b6ea
"""
train SymAE model using `get_batchviews`
"""
function update_with_batchviews(model, loss_history, data_train, data_test, para, training_para=Training_Para())
    opt_state = Optimisers.setup(Optimisers.AdamW(para.learning_rate), model)
    loss(model, data) = loss_sym_vae(
        model,
        data,
        training_para.beta,
        training_para.gamma,
        training_para.temperature,
    ).neg_elbo
    ntau = min(training_para.ntau, minimum(getindex.(size.(data_train), 2)))
    ntau_test = min(training_para.ntau, minimum(getindex.(size.(data_test), 2)))

    @progress name = "training" for epoch = 1:training_para.nepoch
        Xtrain = get_batchviews(data_train, ntau)
        Xtest = get_batchviews(data_test, ntau_test)

        # compute losses per epoch for a sample
        xtrain = get_sample(Xtrain, para.batchsize)
        xtest = get_sample(Xtest, para.batchsize)
        push!(loss_history.train_mse, loss_mse(model, xtrain))
        push!(loss_history.test_mse, loss_mse(model, xtest))
        train_loss = loss_sym_vae(model, xtrain, training_para.beta, training_para.gamma, training_para.temperature)
        test_loss = loss_sym_vae(model, xtest, training_para.beta, training_para.gamma, training_para.temperature)
        push!(loss_history.train_neg_llh, train_loss.neg_llh)
        push!(loss_history.test_neg_llh, test_loss.neg_llh)
        push!(loss_history.train_kl, train_loss.kl)
        push!(loss_history.test_kl, test_loss.kl)
        push!(loss_history.train_neg_elbo, train_loss.neg_elbo)
        push!(loss_history.test_neg_elbo, test_loss.neg_elbo)

        for i = 1:para.nsteps # number of steps per epoch
            x = get_sample(Xtrain, para.batchsize)
            g = Flux.gradient(loss, model, x)[1]
            Optimisers.update!(opt_state, model, g)
        end

        # print output every 10 epochs
        if (mod(epoch, training_para.nprint) == 0)
            @info merge((; epoch=epoch), map(last, loss_history))
        end
    end
    return nothing
end

# ╔═╡ 6f6de313-cd2c-4ed4-beca-7302c4bd7a91
function conditioning_to_matrix_for_lazy(conditioning::Union{Nothing, GroupConditioning}, ngroups::Int)
    if !has_conditioning(conditioning)
        return nothing
    end

    cond_cols = Vector{Vector{Float32}}(undef, ngroups)
    for i in 1:ngroups
        cond_cols[i] = get_group_condition_vector(conditioning, i)
    end

    cond_dim = length(cond_cols[1])
    cond_mat = Matrix{Float32}(undef, cond_dim, ngroups)
    for i in 1:ngroups
        @assert length(cond_cols[i]) == cond_dim "conditioning vectors must have equal length"
        @views cond_mat[:, i] .= cond_cols[i]
    end
    return cond_mat
end

# ╔═╡ 139490d1-5931-4a18-9f74-c391064a3383
"""
**Primary Data Iterator for Group-Based SymVAE Training**

Creates a data iterator where each datapoint consists of `ntau` waveforms randomly sampled 
from a single group. This is the core function that enables the group-based training paradigm
essential for coherent information extraction.

## Critical Training Principle:
During training, it is **essential** to mix instances from different groups in each batch. 
Small batch sizes (e.g., batchsize=1) prevent proper disentanglement of coherent from nuisance 
information. **Always use the largest possible batch size** within memory constraints.

## Group-Based Training Logic:
1. **Within-Group Coherence**: Each datapoint contains `ntau` waveforms from the same group
2. **Cross-Group Learning**: Multiple datapoints (from different groups) in each batch
3. **Coherent Extraction**: Model learns shared features across the `ntau` waveforms
4. **Nuisance Learning**: Model learns instance-specific variations within each group

## Arguments:
* `dvec`: Vector of waveform groups, where `dvec[i]` contains all instances for group i
* `nsteps`: Number of training steps per epoch  
* `ntau`: Number of waveform instances per datapoint (typically 20)
  - Higher values improve coherent learning but require more GPU memory
  - Must be ≤ minimum group size across all groups in `dvec`
* `batchsize`: Number of datapoints per training batch
  - **Critical**: Must be > 1 for proper disentanglement
  - Recommended: 32-256 depending on memory constraints

## Returns:
DataLoader object that yields batches of shape (nt, ntau, batchsize) where:
- nt: time samples per waveform
- ntau: waveforms per group sample  
- batchsize: group samples per batch

## Training Data Flow:
```
Group 1: [w₁₁, w₁₂, ..., w₁ₘ] → Sample ntau → Datapoint 1
Group 2: [w₂₁, w₂₂, ..., w₂ₙ] → Sample ntau → Datapoint 2
   ⋮           ⋮                     ⋮            ⋮
Group G: [wG₁, wG₂, ..., wGₖ] → Sample ntau → Datapoint G

Batch = [Datapoint_i, Datapoint_j, ..., Datapoint_batchsize]
```

## Example Usage:
```julia
# Multiple recording sites (groups)
site_recordings = [site1_waveforms, site2_waveforms, site3_waveforms]

# Create training iterator
train_data = get_data_iterator(
    site_recordings,
    nsteps=1000,     # 1000 steps per epoch
    batchsize=64,    # 64 group samples per batch
    ntau=20          # 20 waveforms per group sample
)

# Each batch contains waveforms from 64 different group samples
for batch in train_data
    # batch shape: (time_samples, 20, 64)
    # 20 waveforms × 64 group samples = 1280 waveforms total
    loss = compute_symvae_loss(model, batch)
end
```
"""
function get_data_iterator(dvec; nsteps=1000, batchsize=256, ntau=20, conditioning=nothing)
    nd = length(dvec)
    conditioning_mat = conditioning_to_matrix_for_lazy(conditioning, nd)
    return DG.get_data_iterator_lazy(
        dvec;
        nsteps=nsteps,
        batchsize=batchsize,
        ntau=ntau,
        conditioning=conditioning_mat,
    )
end


# ╔═╡ 7f8094da-d6a6-4b3d-b16c-cb0d7e928b9d
"""
train SymAE model using `get_data_iterator`
"""
function update(model, loss_history, data_train, data_test, training_para=Training_Para(); conditioning_train=nothing, conditioning_test=conditioning_train)
    lr_s = Exp(start=training_para.initial_learning_rate, decay=0.99)
    opt_state = Optimisers.setup(Optimisers.AdamW(eta=training_para.initial_learning_rate), model)
    ntau = min(training_para.ntau, minimum(getindex.(size.(data_train), 2)))
    ntau_test = min(training_para.ntau, minimum(getindex.(size.(data_test), 2)))
    dup_model = Enzyme.Duplicated(model)
    @progress name = "training" for epoch = 1:training_para.nepoch

        Xtrain = get_data_iterator(
            data_train;
            ntau=ntau,
            batchsize=training_para.batchsize,
            nsteps=training_para.nsteps,
            conditioning=conditioning_train,
        )
        Xtest = get_data_iterator(
            data_test,
            ntau=ntau_test,
            batchsize=training_para.batchsize,
            nsteps=training_para.nsteps,
            conditioning=conditioning_test,
        )

        # compute losses per epoch for a sample
        xtrain = first(Xtrain)
        xtest = first(Xtest)
        push!(loss_history.train_mse, loss_mse(model, xtrain))
        push!(loss_history.test_mse, loss_mse(model, xtest))
        train_loss = loss_sym_vae(model, xtrain, training_para.beta, training_para.gamma, training_para.temperature)
        test_loss = loss_sym_vae(model, xtest, training_para.beta, training_para.gamma, training_para.temperature)
        push!(loss_history.train_neg_llh, train_loss.neg_llh)
        push!(loss_history.test_neg_llh, test_loss.neg_llh)
        push!(loss_history.train_kl, train_loss.kl)
        push!(loss_history.test_kl, test_loss.kl)
        push!(loss_history.train_neg_elbo, train_loss.neg_elbo)
        push!(loss_history.test_neg_elbo, test_loss.neg_elbo)

        # learning rate depending on how many epochs we have already run
        Optimisers.adjust!(opt_state, lr_s(epoch))

        loss(model, data) = loss_sym_vae(
            model,
            data,
            training_para.beta,
            training_para.gamma,
            training_para.temperature,
        ).neg_elbo

        for x in Xtrain
            # g = Flux.gradient(loss, dup_model, Const(x))[1]
            g = Flux.gradient(loss, model, x)[1]
            Optimisers.update!(opt_state, model, g)
        end


        # print output every 10 epochs
        if (mod(epoch, training_para.nprint) == 0)
            @info merge((; epoch=epoch), map(last, loss_history))
        end
    end
    return nothing
end

# ╔═╡ 7e3a8840-ed33-4d7c-a3b0-f6f458e2a72d
"""
alternate training between (encoder, decoder) and (transformer) of SymAE
"""
function update_alternating_transformer(model, loss_history, data_train, data_test, training_para=Training_Para(); conditioning_train=nothing, conditioning_test=conditioning_train)
    lr_s = Exp(start=training_para.initial_learning_rate, decay=0.99)
    opt_state = Optimisers.setup(Optimisers.AdamW(eta=training_para.initial_learning_rate), model)
    loss(model, data) = loss_sym_vae(
        model,
        data,
        training_para.beta,
        training_para.gamma,
        training_para.temperature,
    ).neg_elbo
    ntau = min(training_para.ntau, minimum(getindex.(size.(data_train), 2)))
    ntau_test = min(training_para.ntau, minimum(getindex.(size.(data_test), 2)))
    # dup_model = Enzyme.Duplicated(model)
    @progress name = "training" for epoch = 1:training_para.nepoch
        Xtrain = get_data_iterator(
            data_train;
            ntau=ntau,
            batchsize=training_para.batchsize,
            nsteps=training_para.nsteps,
            conditioning=conditioning_train,
        )
        Xtest = get_data_iterator(
            data_test,
            ntau=ntau_test,
            batchsize=training_para.batchsize,
            nsteps=training_para.nsteps,
            conditioning=conditioning_test,
        )

        # compute losses per epoch for a sample
        xtrain = first(Xtrain)
        xtest = first(Xtest)
        push!(loss_history.train_mse, loss_mse(model, xtrain))
        push!(loss_history.test_mse, loss_mse(model, xtest))
        train_loss = loss_sym_vae(model, xtrain, training_para.beta, training_para.gamma, training_para.temperature)
        test_loss = loss_sym_vae(model, xtest, training_para.beta, training_para.gamma, training_para.temperature)
        push!(loss_history.train_neg_llh, train_loss.neg_llh)
        push!(loss_history.test_neg_llh, test_loss.neg_llh)
        push!(loss_history.train_kl, train_loss.kl)
        push!(loss_history.test_kl, test_loss.kl)
        push!(loss_history.train_neg_elbo, train_loss.neg_elbo)
        push!(loss_history.test_neg_elbo, test_loss.neg_elbo)


        # learning rate depending on how many epochs (from loss_history) we have already run
        Optimisers.adjust!(opt_state, lr_s(epoch))


        N = 2 # N epochs before alternating
        # only update encoder and decoders
        Optimisers.freeze!(opt_state.transformerb)
        for i in 1:N
            for x in Xtrain
                # g = Flux.gradient(loss, dup_model, Const(x))[1]
                g = Flux.gradient(loss, model, x)[1]
                Optimisers.update!(opt_state, model, g)
            end
        end
        Optimisers.thaw!(opt_state.transformerb)


        # only update transformer parameter
        Optimisers.freeze!(opt_state.sencb)
        Optimisers.freeze!(opt_state.nencb)
        Optimisers.freeze!(opt_state.decb)
        for i in 1:N
            for x in Xtrain
                # g = Flux.gradient(loss, dup_model, Const(x))[1]
                g = Flux.gradient(loss, model, x)[1]
                Optimisers.update!(opt_state, model, g)
            end
        end
        Optimisers.thaw!(opt_state.sencb)
        Optimisers.thaw!(opt_state.nencb)
        Optimisers.thaw!(opt_state.decb)



        # print output every 10 epochs
        if (mod(epoch, training_para.nprint) == 0)
            @info merge((; epoch=epoch), map(last, loss_history))
        end
    end
    return nothing
end

# ╔═╡ 63f7880d-9353-4857-b751-36b6f5f45d68
md"""
# JUNK
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
BlackBoxOptim = "a134a8b2-14d6-55f6-9291-3336d3ab0209"
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
DSP = "717857b8-e6f2-59f4-9121-6e50c889abd2"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
Functors = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
Metaheuristics = "bcdb8e00-2c21-11e9-3065-2b553b22f898"
Metalhead = "dbeba491-748d-5e0e-a39e-b530a07fa0cc"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
ParameterSchedulers = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
PlutoHooks = "0ff47ea0-7a50-410d-8455-4348d5de0774"
PlutoLinks = "0ff47ea0-7a50-410d-8455-4348d5de0420"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.6.3"
BlackBoxOptim = "~0.6.3"
CUDA = "~5.9.5"
DSP = "~0.8.4"
Enzyme = "~0.13.109"
FFTW = "~1.10.0"
Flux = "~0.16.7"
Functors = "~0.5.2"
MLUtils = "~0.4.8"
Metaheuristics = "~3.4.1"
Metalhead = "~0.9.5"
Optimisers = "~0.4.6"
ParameterSchedulers = "~0.4.3"
PlutoHooks = "~0.1.0"
PlutoLinks = "~0.1.8"
PlutoPlotly = "~0.6.5"
PlutoUI = "~0.7.75"
ProgressLogging = "~0.1.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.4"
manifest_format = "2.0"
project_hash = "6a1708cc8d296d8db13d1d0a1fa8a4eb5948278e"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "856ecd7cebb68e5fc87abecd2326ad59f0f911f3"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.43"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "e0b47732a192dd59b9d079a06d04235e2f833963"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "29bb0eb6f578a587a49da16564705968667f5fa8"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.2"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "0a6d6d072cb5f2baeba7667023075801f6ea4a7d"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.6.0"

[[deps.BSON]]
git-tree-sha1 = "4c3e506685c527ac6a54ccc0c8c76fd6f91b42fb"
uuid = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
version = "0.3.9"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "a49f9342fc60c2a2aaa4e0934f06755464fcf438"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.6"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BlackBoxOptim]]
deps = ["CPUTime", "Compat", "Distributed", "Distributions", "JSON", "LinearAlgebra", "Printf", "Random", "Requires", "SpatialIndexing", "StatsBase"]
git-tree-sha1 = "9c203a2515b5eeab8f2987614d2b1db83ef03542"
uuid = "a134a8b2-14d6-55f6-9291-3336d3ab0209"
version = "0.6.3"

    [deps.BlackBoxOptim.extensions]
    BlackBoxOptimRealtimePlotServerExt = ["HTTP", "Sockets"]

    [deps.BlackBoxOptim.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
    Sockets = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUTime]]
git-tree-sha1 = "2dcc50ea6a0a1ef6440d6eecd0fe3813e5671f45"
uuid = "a9c8d775-2e2e-55fc-8582-045d282d599e"
version = "1.0.0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Compiler_jll", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "27d1cd229e3e1d5542352a63ad29268439f79fe9"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.9.5"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SparseMatricesCSRExt = "SparseMatricesCSR"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SparseMatricesCSR = "a0a7dd2c-ebf4-11e9-1f05-cf50bc540ca1"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Compiler_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "b63428872a0f60d87832f5899369837cd930b76d"
uuid = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
version = "0.3.0+0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "2023be0b10c56d259ea84a94dbfc021aa452f2c6"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "13.0.2+0"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "f9a521f52d236fe49f1028d69e549e7f2644bb72"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "1.0.0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "92cd84e2b760e471d647153ea5efc5789fc5e8b2"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.19.2+0"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3b704353e517a957323bd3ac70fa7b669b5f48d4"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.72.6"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "b7231a755812695b8046e8471ddc34c8268cbad5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.0"

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

[[deps.Combinatorics]]
git-tree-sha1 = "8010b6bb3388abe68d95743dcbea77650bb2eddf"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.3"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

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

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

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

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.Enzyme]]
deps = ["CEnum", "EnzymeCore", "Enzyme_jll", "GPUCompiler", "InteractiveUtils", "LLVM", "Libdl", "LinearAlgebra", "ObjectFile", "PrecompileTools", "Preferences", "Printf", "Random", "SparseArrays"]
git-tree-sha1 = "73e9cb6bb34e537b0ef3bb5e51b1174160dcc5ec"
uuid = "7da242da-08ed-463a-9acd-ee780be4f1d9"
version = "0.13.109"

    [deps.Enzyme.extensions]
    EnzymeBFloat16sExt = "BFloat16s"
    EnzymeChainRulesCoreExt = "ChainRulesCore"
    EnzymeGPUArraysCoreExt = "GPUArraysCore"
    EnzymeLogExpFunctionsExt = "LogExpFunctions"
    EnzymeSpecialFunctionsExt = "SpecialFunctions"
    EnzymeStaticArraysExt = "StaticArrays"

    [deps.Enzyme.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BFloat16s = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.EnzymeCore]]
git-tree-sha1 = "820f06722a87d9544f42679182eb0850690f9b45"
uuid = "f151be2c-9106-41f4-ab19-57ee4f262869"
version = "0.8.17"
weakdeps = ["Adapt"]

    [deps.EnzymeCore.extensions]
    AdaptExt = "Adapt"

[[deps.Enzyme_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "b7a8c737c8ca2f5ca313e012f212effa1adcbf3a"
uuid = "7cc45869-7501-5eee-bdea-0790c847d4ef"
version = "0.0.229+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

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

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bfcd42851cf2f1b303f51525a54dc5e98d408a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.15.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Flux]]
deps = ["Adapt", "ChainRulesCore", "Compat", "EnzymeCore", "Functors", "LinearAlgebra", "MLCore", "MLDataDevices", "MLUtils", "MacroTools", "NNlib", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "Setfield", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote"]
git-tree-sha1 = "efa66783e2ad06bfd4c148cb34648e24c99f7626"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.16.7"

    [deps.Flux.extensions]
    FluxAMDGPUExt = "AMDGPU"
    FluxCUDAExt = "CUDA"
    FluxCUDAcuDNNExt = ["CUDA", "cuDNN"]
    FluxEnzymeExt = "Enzyme"
    FluxMPIExt = "MPI"
    FluxMPINCCLExt = ["CUDA", "MPI", "NCCL"]

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    MPI = "da04e1cc-30fd-572f-bb4f-1f8673147195"
    NCCL = "3fe64909-d7a1-4096-9b7d-7a0f12cf0f6b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cd33c7538e68650bd0ddbb3f5bd50a4a0fa95b50"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.0"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.Functors]]
deps = ["Compat", "ConstructionBase", "LinearAlgebra", "Random"]
git-tree-sha1 = "60a0339f28a233601cb74468032b5c302d5067de"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.5.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "SparseArrays", "Statistics"]
git-tree-sha1 = "18da8dd0b6aded0c47184e9d2a17573ae8257f36"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.3.1"
weakdeps = ["JLD2"]

    [deps.GPUArrays.extensions]
    JLD2Ext = "JLD2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "Tracy", "UUIDs"]
git-tree-sha1 = "6e5a25bc455da8e8d88b6b7377e341e9af1929f0"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.7.5"

[[deps.GPUToolbox]]
deps = ["LLVM"]
git-tree-sha1 = "9e9186b09a13b7f094f87d1a9bb266d8780e1b1c"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "1.0.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

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

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "57e9ce6cf68d0abf5cb6b3b4abf9bedf05c939c0"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.15"

[[deps.InfiniteArrays]]
deps = ["ArrayLayouts", "FillArrays", "Infinities", "LazyArrays", "LinearAlgebra"]
git-tree-sha1 = "e61675cbcf3ce57ea3566b0abcfe46e4a521bf6f"
uuid = "4858937d-0d70-526a-a4dd-2d5cb5dd786c"
version = "0.12.15"
weakdeps = ["Statistics"]

    [deps.InfiniteArrays.extensions]
    InfiniteArraysStatisticsExt = "Statistics"

[[deps.Infinities]]
git-tree-sha1 = "4495006c20b2fd27b8c453a1dd31d423654f3772"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.12"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues", "TranscodingStreams"]
git-tree-sha1 = "d97791feefda45729613fafeccc4fbef3f539151"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.5.15"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JMcDM]]
deps = ["Requires"]
git-tree-sha1 = "e26d5db41aa1b96d4ed23b46eeeca34116214661"
uuid = "358108f5-d052-4d0a-8344-d5384e00c0e5"
version = "0.7.24"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "3d3b79166e2a0afcf875df20db110af91ad3ab61"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.11"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b5a371fcd1d989d844a4354127365611ae1e305f"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.39"
weakdeps = ["EnzymeCore", "LinearAlgebra", "SparseArrays"]

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "ce8614210409eaa54ed5968f4b50aa96da7ae543"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.4.4"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "8e76807afb59ebb833e9b131ebf1a8c006510f33"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.38+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "MatrixFactorizations", "SparseArrays"]
git-tree-sha1 = "35079a6a869eecace778bcda8641f9a54ca3a828"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "1.10.0"
weakdeps = ["StaticArrays"]

    [deps.LazyArrays.extensions]
    LazyArraysStaticArraysExt = "StaticArrays"

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

[[deps.LibTracyClient_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d2bc4e1034b2d43076b50f0e34ea094c2cb0a717"
uuid = "ad6e5548-8b26-5c9f-8ef3-ef0ad883f3a5"
version = "0.9.1+6"

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

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "5d4278f755440f70648d80cc6225f51e78e94094"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.1"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MLCore]]
deps = ["DataAPI", "SimpleTraits", "Tables"]
git-tree-sha1 = "73907695f35bc7ffd9f11f6c4f2ee8c1302084be"
uuid = "c2834f40-e789-41da-a90e-33b280584a8c"
version = "1.0.0"

[[deps.MLDataDevices]]
deps = ["Adapt", "Functors", "Preferences", "Random", "SciMLPublic"]
git-tree-sha1 = "7268ed353d29d817427e57298a27972dbdd19045"
uuid = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
version = "1.15.3"

    [deps.MLDataDevices.extensions]
    MLDataDevicesAMDGPUExt = "AMDGPU"
    MLDataDevicesCUDAExt = "CUDA"
    MLDataDevicesChainRulesCoreExt = "ChainRulesCore"
    MLDataDevicesChainRulesExt = "ChainRules"
    MLDataDevicesComponentArraysExt = "ComponentArrays"
    MLDataDevicesFillArraysExt = "FillArrays"
    MLDataDevicesGPUArraysExt = "GPUArrays"
    MLDataDevicesMLUtilsExt = "MLUtils"
    MLDataDevicesMetalExt = ["GPUArrays", "Metal"]
    MLDataDevicesOneHotArraysExt = "OneHotArrays"
    MLDataDevicesReactantExt = "Reactant"
    MLDataDevicesRecursiveArrayToolsExt = "RecursiveArrayTools"
    MLDataDevicesReverseDiffExt = "ReverseDiff"
    MLDataDevicesSparseArraysExt = "SparseArrays"
    MLDataDevicesTrackerExt = "Tracker"
    MLDataDevicesZygoteExt = "Zygote"
    MLDataDevicescuDNNExt = ["CUDA", "cuDNN"]
    MLDataDevicesoneAPIExt = ["GPUArrays", "oneAPI"]

    [deps.MLDataDevices.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ComponentArrays = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
    FillArrays = "1a297f60-69ca-5386-bcde-b61e274b549b"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OneHotArrays = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "MLCore", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "a772d8d1987433538a5c226f79393324b55f7846"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.8"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "6731e0574fa5ee21c02733e397beb133df90de35"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "2.2.0"

[[deps.Metaheuristics]]
deps = ["Distances", "JMcDM", "LinearAlgebra", "Pkg", "Printf", "Random", "Reexport", "Requires", "SearchSpaces", "SnoopPrecompile", "Statistics"]
git-tree-sha1 = "4362c6421e39755991b174d185e89e841c7c37bf"
uuid = "bcdb8e00-2c21-11e9-3065-2b553b22f898"
version = "3.4.1"

[[deps.Metalhead]]
deps = ["Artifacts", "BSON", "ChainRulesCore", "Flux", "Functors", "JLD2", "LazyArtifacts", "MLUtils", "NNlib", "PartialFunctions", "Random", "Statistics"]
git-tree-sha1 = "7d3cdd8acb8ccdf82bb80d07f33f020b0976ddc5"
uuid = "dbeba491-748d-5e0e-a39e-b530a07fa0cc"
version = "0.9.5"
weakdeps = ["CUDA"]

    [deps.Metalhead.extensions]
    MetalheadCUDAExt = "CUDA"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

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

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "ScopedValues", "Statistics"]
git-tree-sha1 = "09701dc1df4281fa9212b269a69210dfa81ee52a"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.32"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"
    NNlibMetalExt = "Metal"
    NNlibSpecialFunctionsExt = "SpecialFunctions"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "6b573a3e66decc7fc747afd1edbf083ff78c813a"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.1"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "af2232f69447494514c25742ba1503ec7e9877fe"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.2.2+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.ObjectFile]]
deps = ["Reexport", "StructIO"]
git-tree-sha1 = "22faba70c22d2f03e60fbc61da99c4ebfc3eb9ba"
uuid = "d8793406-e978-5875-9003-1fc021f44a92"
version = "0.5.0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "bfe8e84c71972f77e775f75e6d8048ad3fdbe8bc"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.10"

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

[[deps.Optimisers]]
deps = ["ChainRulesCore", "ConstructionBase", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "131dc319e7c58317e8c6d5170440f6bdaee0a959"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.4.6"

    [deps.Optimisers.extensions]
    OptimisersAdaptExt = ["Adapt"]
    OptimisersEnzymeCoreExt = "EnzymeCore"
    OptimisersReactantExt = "Reactant"

    [deps.Optimisers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d922b4d80d1e12c658da7785e754f4796cc1d60d"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.36"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.ParameterSchedulers]]
deps = ["InfiniteArrays", "Optimisers"]
git-tree-sha1 = "c62f0da0663704d0472ae578c9bb802c44e70a4c"
uuid = "d7d3b36b-41b8-4d0d-a2bf-768c6151755e"
version = "0.4.3"

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

[[deps.PartialFunctions]]
deps = ["MacroTools"]
git-tree-sha1 = "ba0ea009d9f1e38162d016ca54627314b6d8aac8"
uuid = "570af359-4316-4cb7-8c74-252c00c2016b"
version = "1.2.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "49c457ee4c9c6f5bdf2f6f1a69e66976aaecfcdb"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.22"

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

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "844a829c8dc9fd0fe62eced22bc2d0dfd66a3f51"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.1.0"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "aea4eede5ab3ee188906d0cf3bbfa36eb543dccc"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.8"

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
git-tree-sha1 = "db8a06ef983af758d285665a0398703eb5bc1d66"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.75"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "c5a07210bd060d6a8491b0ccdee2fa0235fc00bf"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.1.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "f0803bc1171e455a04124affa9c21bba5ac4db32"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.6"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"
weakdeps = ["Enzyme"]

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "dbe5fd0b334694e905cb9fda73cd8554333c46e2"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.1"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "d97d78d4fc5f858d8ce44f6b88bc972f2023f51d"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.0"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SciMLPublic]]
git-tree-sha1 = "ed647f161e8b3f2973f24979ec074e8d084f1bee"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.0"

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

[[deps.SearchSpaces]]
deps = ["Combinatorics", "Random"]
git-tree-sha1 = "2662fd537048fb12ff34fabb5249bf50e06f445b"
uuid = "eb7571c6-2196-4f03-99b8-52a5a35b3163"
version = "0.2.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

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

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpatialIndexing]]
git-tree-sha1 = "84efe17c77e1f2156a7a0d8a7c163c1e1c7bdaed"
uuid = "d4ead438-fe20-5cc5-a293-4fd39a41b74c"
version = "0.1.6"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

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
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "be5733d4a2b03341bdcab91cea6caa7e31ced14b"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.9"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a3c1536470bf8c5e02096ad4853606d7c8f62721"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.2"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"
weakdeps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "SparseArrays", "StaticArrays"]

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

[[deps.StructIO]]
git-tree-sha1 = "c581be48ae1cbf83e899b14c07a807e1787512cc"
uuid = "53d494c1-5632-5724-8f4c-31dff12d585f"
version = "0.3.1"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tracy]]
deps = ["ExprTools", "LibTracyClient_jll", "Libdl"]
git-tree-sha1 = "73e3ff50fd3990874c59fef0f35d10644a1487bc"
uuid = "e689c965-62c8-4b79-b2c5-8359227902fd"
version = "0.1.6"

    [deps.Tracy.extensions]
    TracyProfilerExt = "TracyProfiler_jll"

    [deps.Tracy.weakdeps]
    TracyProfiler_jll = "0c351ed6-8a68-550e-8b79-de6f926da83c"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "SplittablesBase", "Tables"]
git-tree-sha1 = "4aa1fdf6c1da74661f6f5d3edfd96648321dade9"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.85"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

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

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "PrecompileTools", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a29cbf3968d36022198bcc6f23fdfd70f7caf737"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.7.10"

    [deps.Zygote.extensions]
    ZygoteAtomExt = "Atom"
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Atom = "c52e3926-4ff0-5f6e-af25-54175e0327b1"
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "434b3de333c75fc446aa0d19fc394edafd07ab08"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.7"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

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
# ╟─461f0505-2230-4b84-b6c6-1a9730808437
# ╠═97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
# ╠═cc11647d-1c56-4ceb-9677-703aca03c9f4
# ╠═d73472ff-9e09-45b0-8811-b7dd8d820358
# ╠═76dbf599-a9b3-459f-992b-16ab2f7b74f1
# ╠═4a95997e-5c12-4658-9b8e-a5065328e1c1
# ╠═26fb86d5-c844-469a-aef5-ed3c2a9ba949
# ╟─3983e7d0-9ad0-11f0-0a96-7d2d98772fd2
# ╟─dc2cd512-9ad0-11f0-1dca-71ffa60b282a
# ╟─a91e28fb-e769-418d-953f-0e0bb366d853
# ╠═91a25156-e121-4d53-a5a1-422f1230d235
# ╠═631e5584-c4c3-4320-9a2d-477b7945c684
# ╠═96df0a0a-fd55-46b1-95af-067d02558380
# ╠═8fefefd8-b63a-4be7-92e3-8ed7c3f88865
# ╠═f49d0e70-57e6-44a0-8e46-4d801c4b3f85
# ╠═065f859c-e7cf-4b42-bcc8-dfbf63104d26
# ╠═17f60265-fac3-411c-b1dd-49b284d23597
# ╠═10429267-5808-4840-8678-f9dbf5b453c5
# ╟─29d41554-3a0a-4972-a9e5-54c998429acd
# ╠═dc7e0a28-2739-44c2-9a44-c66079aaae17
# ╠═139490d1-5931-4a18-9f74-c391064a3383
# ╟─ce690827-fa3f-48bc-bc09-1df5ee15f683
# ╠═6affb3b3-9dc4-4bbc-a582-495fc1783a7a
# ╠═a5302fa2-4f67-4ed6-96ce-dda78a160ffe
# ╠═2df712f5-6969-4ceb-98c5-82415718b740
# ╠═d86a890f-942f-4e3a-8405-709507450903
# ╠═5a047afc-ec40-4541-ad50-0518beba3f2e
# ╠═1121e34c-ca35-4f68-8283-eca514928654
# ╟─ae96f920-5828-4c5f-b69f-48d8c4fee378
# ╠═96aebf7d-2112-4a4d-9993-6f53f40ffca5
# ╟─ea372a8f-212f-425d-947c-b57bba6b5574
# ╠═32baae21-b5fa-4391-9066-23436a5a2b1d
# ╠═89599b3f-8c20-46c5-8f5c-ccbb71b26b36
# ╠═64430447-c267-4eec-8d38-63ccf91d82c4
# ╠═5847ea08-43ca-4c6d-a694-0017d7396f60
# ╠═4fb77fae-61bb-4484-b733-82e1d1002371
# ╟─747680b0-3469-426c-8b9e-4ab8ca04a6de
# ╠═bf31f347-bc9a-4bf8-a086-99dba2f6fea0
# ╠═190c8221-c5c7-48f9-b016-36c27fd4528c
# ╟─62681bba-5486-4957-8433-4258657399b8
# ╠═0fbf59a9-74bc-479f-879c-3f72f7c76489
# ╠═0dd8d418-a664-4ecd-ac90-bab97861f9f0
# ╟─facd01fe-b288-437f-96dd-a8a4d9afd8fe
# ╠═06a8d9e2-495c-4a25-8c23-527ba1b8e089
# ╟─66bddc43-ca9f-43cd-85a3-d33b11a6c033
# ╠═8684c192-d1b9-4821-a02e-2c7300af9b3c
# ╠═e9efedc1-8287-4676-ba64-a4abc77da18d
# ╠═b65ae9dc-dd50-4007-9894-cadef28e0552
# ╠═06286e75-6fdd-41a3-b80b-0f0ffa4ec603
# ╠═6e3d3148-fc73-4538-b077-2abe1d1721d4
# ╠═8c54f11c-51b2-4500-9923-d3d38aa91e9b
# ╠═41e788f4-2e25-4968-8bf4-dcd2f38d24e7
# ╠═f1eddb54-7d90-4a14-9943-053248665e78
# ╟─52dc9696-3e0b-42c7-b6cf-7a07ca3cb4dd
# ╠═fa646879-158a-4cbb-be0e-d375cf486ba0
# ╟─eafab001-87a7-423f-917f-1fbd46699186
# ╠═237ad98e-75db-41fc-b378-b895facdd8d9
# ╟─5ed89ee8-325b-4757-b348-e6c1a3d277ad
# ╠═04f9b328-edc8-4b1e-9a7c-79a215b1cf5f
# ╟─371354f6-29e7-4227-b006-f2daacb08ce7
# ╠═186713a3-c357-427c-9af4-6739de2d33c5
# ╟─fe87efed-9c40-4869-bdf3-cc60eb5b6436
# ╠═76c4f167-11b5-48a3-b3b3-4fe8fd9646c1
# ╠═78e16b27-85c3-4d4a-9238-3ec2a33c9c88
# ╟─efde2571-4e1d-4626-9081-86d513707f5e
# ╠═d74b7838-98c4-4356-8a0d-1a2388369788
# ╟─95660df0-088b-49f3-b875-fca19a82d024
# ╠═c49e3d81-27ba-4a4f-870a-ae218a505dd0
# ╠═6720843c-be73-49bc-a9ec-9a7f599a1f98
# ╟─5731aea5-af70-4dc3-a505-2f48fce02e8e
# ╠═56055fa2-8f96-4b43-b1eb-2cd5a9cbadeb
# ╠═90e69a62-b340-45d6-944e-cb56f6f46ca6
# ╠═e80dd767-72ef-410b-b7a2-38e0545a5df3
# ╠═1f139691-e19f-41e5-8113-3cc00e8fe2b8
# ╠═825dda0d-6472-405c-b149-5c4d2202963f
# ╠═8abc3a6d-d2f5-4527-a828-793364706fa5
# ╠═84d56fa3-50de-48ad-8e07-dfaecc1cfdf3
# ╠═86a120ae-8865-4e99-a028-f567f3c1bbad
# ╟─f4238f8e-3596-4c98-a64e-477c3aa2b054
# ╠═a48d23b6-86d6-4232-a1b9-300e65b264ff
# ╠═84aa58d3-115b-4d2d-a798-fc936f5bb2ca
# ╠═a46ce69a-3081-4e86-b9eb-c773c772f459
# ╠═afb6c675-0ef8-445c-a204-795e300c8589
# ╠═8967b2d2-6688-4ac6-a3ec-357d4ce3e6f0
# ╠═a85b6958-d894-45cc-86c6-933fba757e1c
# ╠═74c8e79f-b1c8-484b-86bf-65c2a2831b53
# ╠═073c77d1-7598-45d3-858d-9222f2e4c590
# ╠═c4de28ea-5af6-4c66-9955-a5d168833036
# ╠═b3428bd8-c1ec-4e48-97b0-3dbde763292b
# ╠═d62474f6-48fb-49fe-919b-c4373135067c
# ╟─ab3f0de4-3e7e-4d5f-aa1a-25fe24a52b38
# ╠═f5e6cf15-1072-4037-9a96-91db93e730f0
# ╠═d966211f-012e-45f2-b9ae-abf599666edf
# ╠═e29c0103-ab96-413f-a739-cd0f97ff3288
# ╠═a636c206-8ded-4f13-b010-9a33dd99f80e
# ╠═6d34151a-b539-4eda-995a-b7f873719a4c
# ╠═4ec8578c-a202-4a3d-9425-60bf623a3a02
# ╠═9b95b3b6-6c98-4c8b-abdc-bbc1410a0df1
# ╠═0db6b854-ef75-46c0-8e19-efd8faa037bb
# ╠═41b5d143-6a0f-4f4c-8e62-8483d9e0d5f6
# ╠═e57556e8-b287-4bc5-9ea1-4f68948eccf2
# ╠═6e7c70c8-cffe-4b30-85b4-5942fbb60da8
# ╠═e46630eb-5e29-4c5e-b792-0f7ca0af0ff0
# ╠═361ead1b-3fe3-4ae2-b018-2c6904d0f889
# ╠═6556c7c7-9934-49be-8f8a-423a0d16e57e
# ╠═4e527961-5734-4294-9f3c-e2aa8314eef6
# ╠═26aafe3d-e783-4591-959d-910b9c050301
# ╠═c5d5be4d-882e-4c34-ae8f-1a40aa4cf215
# ╠═0f292b15-bc79-4424-b5ae-fded09eb16f0
# ╠═b9b5f42d-4ec1-43b5-9484-d3ec47dea61a
# ╠═ae75eee7-ed34-4a2a-8aa3-08a06f504d36
# ╠═7931ce6a-a062-4c7f-bb04-82b822e04eab
# ╠═eafe181e-19e9-409e-ad1d-ce859cf0e672
# ╠═7f8094da-d6a6-4b3d-b16c-cb0d7e928b9d
# ╠═7e3a8840-ed33-4d7c-a3b0-f6f458e2a72d
# ╠═0541d3c6-24cf-46b1-95a1-39f3341aec4f
# ╠═483e596b-afbc-45d5-84d4-847959c5b6ea
# ╟─61fbbf84-1b2b-4de9-8763-800f76c851e0
# ╠═0f0fe0e8-a239-474b-b9af-54507cb968aa
# ╠═7531c2eb-5505-4ad2-9f79-111bed3bef74
# ╟─5d25db74-dcc2-49d6-b62a-39f612089e9f
# ╠═c9f4b3e6-6691-4e7f-86c7-dcacdde924a5
# ╟─0a2fb24a-3c7a-46ed-ad54-d35fb72f8031
# ╠═1a8619c6-4c19-4f55-a968-d18a4f1016e7
# ╠═80f77b52-84e0-4664-8aa0-3d79fded40de
# ╠═6ba143e2-50df-441a-8f38-3ea8d9edd4d8
# ╠═6f6de313-cd2c-4ed4-beca-7302c4bd7a91
# ╟─63f7880d-9353-4857-b751-36b6f5f45d68
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
