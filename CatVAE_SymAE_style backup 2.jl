### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ d73472ff-9e09-45b0-8811-b7dd8d820358
using CUDA,
cuDNN,
    Enzyme,
    Flux,
    MLUtils,
    Statistics,
    PlutoUI,
    LinearAlgebra,
    ProgressLogging,
    Optimisers,
    Random,
    ParameterSchedulers

# ╔═╡ 461f0505-2230-4b84-b6c6-1a9730808437
md"""# Categorical Variational Autoencoders (CatVAE)"""

# ╔═╡ 97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
TableOfContents(include_definitions=true)

# ╔═╡ 26fb86d5-c844-469a-aef5-ed3c2a9ba949
xpu = gpu

# ╔═╡ 3983e7d0-9ad0-11f0-0a96-7d2d98772fd2
md"""
## CatVAE Overview: Categorical Variational Autoencoder

### Core Architecture Philosophy
CatVAE employs a categorical latent variable structure designed to learn discrete-continuous representations from data.

#### **Objective: Learning Categorical Structure in Latent Space**
The primary goal is to learn a latent space with the following structure:
- **Discrete Categories (c)**: A categorical variable determining the active mode
- **Continuous Code (z)**: A continuous latent vector conditioned on the category
- **Multimodal Prior**: Fixed prior means μ[c] for each category to encourage separation

### Model Architecture

#### **Encoding Structure**
```
Input x → Encoder Networks → q(c|x), q(z|x,c)
```

#### **Decoding Structure**
```
Latent z → Decoder Network → Reconstructed x̂
```

### Mathematical Foundation

The CatVAE loss function (ELBO):

```
L = E_{q(c|x)q(z|x,c)}[-log p(x|z)] + KL(q(c|x)||p(c)) + E_{q(c|x)}[KL(q(z|x,c)||p(z|c))]
```

Where:
- **Reconstruction term**: Ensures faithful data reproduction
- **Discrete KL term**: Regularizes category assignment (uniform prior)
- **Continuous KL term**: Regularizes continuous code per category (Gaussian prior with category-specific means)

### Prior Structure

For K categories with latent dimension d = K × δ:
- **p(c)**: Uniform categorical prior over K categories
- **p(z|c)**: Gaussian N(μ[c], I) where μ[c] has non-zero values only in the c-th block

This prior structure encourages:
- Clear separation between categories in latent space
- Specialization of latent dimensions to specific categories
"""

# ╔═╡ a91e28fb-e769-418d-953f-0e0bb366d853
md"""
## Parameters
- `input_dim`: dimensionality of input data (e.g., 784 for MNIST)
- `K`: number of discrete categories
- `δ`: latent block size per category
- `d`: total latent dimension (d = K × δ)
- `λ`: prior mean magnitude for each category
- `hidden_dim`: hidden layer size for encoders/decoders
- `network_type`: 
  - `DenseCatVAE()` fully connected architecture
  - `ConvCatVAE()` convolutional architecture (for image data)
"""

# ╔═╡ 6db97fc1-8f11-42df-bffe-f86b8619a399
Base.@kwdef struct CatVAE_Para
    input_dim::Int
    K::Int
    δ::Int
    λ::Float32 = 4f0
    hidden_dim::Int = 400
    network_type = DenseCatVAE()
    seed = nothing
end

# ╔═╡ fc228dea-21fc-4fcd-82a9-7ac3bc7ee722
"""
Compute latent dimension from parameters
"""
get_latent_dim(para::CatVAE_Para) = para.K * para.δ

# ╔═╡ 7c39a024-bf46-4024-b0da-a4d6092e864d
"""
Create multimodal prior means for CatVAE
Each category c has non-zero prior mean only in its corresponding block
"""
function get_prior_means(para::CatVAE_Para)
    d = get_latent_dim(para)
    μ_prior = zeros(Float32, para.K, d)
    for c in 1:para.K
        start_idx = (c-1) * para.δ + 1
        end_idx = c * para.δ
        μ_prior[c, start_idx:end_idx] .= para.λ
    end
    return xpu(μ_prior)
end

# ╔═╡ ce690827-fa3f-48bc-bc09-1df5ee15f683
md"## Network Architecture Types"

# ╔═╡ a5302fa2-4f67-4ed6-96ce-dda78a160ffe
activation = x -> leakyrelu(x, 0.1f0)

# ╔═╡ bf31f347-bc9a-4bf8-a086-99dba2f6fea0
begin
    Base.@kwdef struct DenseCatVAE
        hidden_dim::Int = 400
    end
    
    Base.@kwdef struct ConvCatVAE
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

# ╔═╡ ae96f920-5828-4c5f-b69f-48d8c4fee378
md"## Dense Networks"

# ╔═╡ 96aebf7d-2112-4a4d-9993-6f53f40ffca5
function generate_dense_network(input_dim, output_dim, hidden_dim, output_activation=identity)
    return Chain(
        Dense(input_dim, hidden_dim, activation),
        Dense(hidden_dim, output_dim, output_activation)
    )
end

# ╔═╡ 190c8221-c5c7-48f9-b016-36c27fd4528c
"""
Build CatVAE networks using Dense architecture
"""
function get_catvae_networks(t::DenseCatVAE, para::CatVAE_Para)
    d = get_latent_dim(para)
    
    # q(c|x): categorical encoder
    enc_c = xpu(Chain(
        Dense(para.input_dim, para.hidden_dim, activation),
        Dense(para.hidden_dim, para.K),
        softmax
    ))
    
    # q(z|x,c): continuous encoder - mean
    enc_z_mean = xpu(Chain(
        Dense(para.input_dim + para.K, para.hidden_dim, activation),
        Dense(para.hidden_dim, d)
    ))
    
    # q(z|x,c): continuous encoder - logvar
    enc_z_logvar = xpu(Chain(
        Dense(para.input_dim + para.K, para.hidden_dim, activation),
        Dense(para.hidden_dim, d)
    ))
    
    # p(x|z): decoder
    dec = xpu(Chain(
        Dense(d, para.hidden_dim, activation),
        Dense(para.hidden_dim, para.input_dim),
        σ  # sigmoid for pixel probabilities
    ))
    
    return (; enc_c, enc_z_mean, enc_z_logvar, dec)
end

# ╔═╡ 66bddc43-ca9f-43cd-85a3-d33b11a6c033
md"## KL Divergence Functions"

# ╔═╡ 8684c192-d1b9-4821-a02e-2c7300af9b3c
"""
KL(q(z|x,c) || p(z|c))
Gaussian KL divergence with category-specific prior mean
"""
function kl_gaussian(μq, logvarq, μp)
    σq2 = exp.(logvarq)
    return 0.5f0 * sum(σq2 .+ (μq .- μp).^2 .- 1f0 .- logvarq)
end

# ╔═╡ e9efedc1-8287-4676-ba64-a4abc77da18d
"""
KL(q(c|x) || p(c))
Discrete KL divergence with uniform prior
"""
function kl_discrete(qcx, K)
    # Clamp to avoid log(0)
    qcx_safe = clamp.(qcx, 1f-8, 1f0)
    return sum(qcx_safe .* log.(qcx_safe .* K))
end

# ╔═╡ 5ed89ee8-325b-4757-b348-e6c1a3d277ad
md"## CatVAE Model Structure"

# ╔═╡ 04f9b328-edc8-4b1e-9a7c-79a215b1cf5f
begin
    struct CatVAE{T1,T2,T3,T4,T5}
        enc_c::T1           # q(c|x)
        enc_z_mean::T2      # q(z|x,c) mean
        enc_z_logvar::T3    # q(z|x,c) logvar
        dec::T4             # p(x|z)
        μ_prior::T5         # prior means per category
    end
    
    Flux.@layer CatVAE trainable = (enc_c, enc_z_mean, enc_z_logvar, dec)
    
    """
    Forward pass through CatVAE
    Returns reconstructed x and intermediate variables for loss computation
    """
    function (m::CatVAE)(x, sample::Bool=true)
        batch_size = size(x, 2)
        K = size(m.μ_prior, 1)
        d = size(m.μ_prior, 2)
        
        # Encode: q(c|x)
        qcx = m.enc_c(x)
        
        # For each category, compute q(z|x,c)
        μqs = Vector{Any}(undef, K)
        logvarqs = Vector{Any}(undef, K)
        
        for c in 1:K
            # One-hot encode category
            ohc = Float32.(Flux.onehotbatch(fill(c, batch_size), 1:K))
            xc = vcat(x, ohc)
            
            # Encode z given category c
            μqs[c] = m.enc_z_mean(xc)
            logvarqs[c] = m.enc_z_logvar(xc)
        end
        
        # Compute expected latent code: E_q(c|x)[μ_z]
        z_expected = sum(qcx[c, :, :]' .* μqs[c] for c in 1:K)
        
        # Optional: add reparameterization noise during training
        z = if sample
            # Sample from mixture
            z_sample = zeros(Float32, size(z_expected)...) |> xpu
            for c in 1:K
                ε = randn(Float32, size(μqs[c])...) |> xpu
                z_c = μqs[c] .+ exp.(0.5f0 .* logvarqs[c]) .* ε
                z_sample .+= qcx[c, :, :]' .* z_c
            end
            z_sample
        else
            z_expected
        end
        
        # Decode
        x_recon = m.dec(z)
        
        return (; x_recon, qcx, μqs, logvarqs, z)
    end
    
    """
    Encode input to get categorical distribution q(c|x)
    """
    function (m::CatVAE)(x, ::Val{:encode_c})
        return m.enc_c(x)
    end
    
    """
    Decode latent code z to reconstruction
    """
    function (m::CatVAE)(z, ::Val{:decode})
        return m.dec(z)
    end
end

# ╔═╡ 1121e34c-ca35-4f68-8283-eca514928654
"""
Build complete CatVAE model
"""
function get_catvae(para::CatVAE_Para)
    Random.seed!(para.seed)
    
    # Get network architectures
    networks = get_catvae_networks(para.network_type, para)
    
    # Get prior means
    μ_prior = get_prior_means(para)
    
    # Construct model
    model = CatVAE(
        networks.enc_c,
        networks.enc_z_mean,
        networks.enc_z_logvar,
        networks.dec,
        μ_prior
    ) |> xpu
    
    # Initialize loss history
    loss_history = (
        train_recon=Vector{Float32}(undef, 0),
        test_recon=Vector{Float32}(undef, 0),
        train_kl_cont=Vector{Float32}(undef, 0),
        test_kl_cont=Vector{Float32}(undef, 0),
        train_kl_disc=Vector{Float32}(undef, 0),
        test_kl_disc=Vector{Float32}(undef, 0),
        train_elbo=Vector{Float32}(undef, 0),
        test_elbo=Vector{Float32}(undef, 0),
    )
    
    return model, loss_history
end

# ╔═╡ f5e6cf15-1072-4037-9a96-91db93e730f0
"""
CatVAE loss function (negative ELBO)
Implements Equation 5 from the CatVAE paper
"""
function loss_catvae(model::CatVAE, x, beta_cont=1f0, beta_disc=1f0)
    # Forward pass
    result = model(x, true)
    
    K = size(model.μ_prior, 1)
    batch_size = size(x, 2)
    
    # Reconstruction loss (negative log-likelihood)
    # Using binary cross-entropy for pixel probabilities
    recon_loss = Flux.Losses.binarycrossentropy(result.x_recon, x, agg=sum)
    
    # Discrete KL: KL(q(c|x) || Uniform(K))
    disc_kl = kl_discrete(result.qcx, K)
    
    # Continuous KL: E_q(c|x)[KL(q(z|x,c) || p(z|c))]
    cont_kl = 0f0
    for c in 1:K
        # Average q(c|x) over batch
        qcx_c = mean(result.qcx[c, :, :])
        # Weighted KL for this category
        cont_kl += qcx_c * kl_gaussian(result.μqs[c], result.logvarqs[c], model.μ_prior[c, :])
    end
    
    # Total loss (negative ELBO)
    elbo = recon_loss + beta_disc * disc_kl + beta_cont * cont_kl
    
    return (; elbo, recon_loss, cont_kl, disc_kl)
end

# ╔═╡ d966211f-012e-45f2-b9ae-abf599666edf
"""
Reconstruction loss only (for evaluation)
"""
function loss_reconstruction(model::CatVAE, x)
    result = model(x, false)
    return Flux.Losses.binarycrossentropy(result.x_recon, x, agg=mean)
end

# ╔═╡ 7931ce6a-a062-4c7f-bb04-82b822e04eab
md"""
## Training Parameters
- `nepoch`: number of training epochs
- `batchsize`: batch size for SGD
- `beta_cont`: weight for continuous KL term
- `beta_disc`: weight for discrete KL term
- `initial_learning_rate`: starting learning rate
- `nprint`: print frequency (epochs)
"""

# ╔═╡ eafe181e-19e9-409e-ad1d-ce859cf0e672
Base.@kwdef struct CatVAE_Training_Para
    nepoch::Int = 10
    batchsize::Int = 128
    beta_cont::Float32 = 1f0
    beta_disc::Float32 = 1f0
    initial_learning_rate::Float64 = 1e-3
    nprint::Int = 1
end

# ╔═╡ 7f8094da-d6a6-4b3d-b16c-cb0d7e928b9d
"""
Train CatVAE model
"""
function update(model::CatVAE, loss_history, data_train, data_test, training_para::CatVAE_Training_Para)
    # Setup optimizer with learning rate schedule
    lr_s = Exp(start=training_para.initial_learning_rate, decay=0.99)
    opt_state = Optimisers.setup(Optimisers.Adam(training_para.initial_learning_rate), model)
    
    @progress name = "training CatVAE" for epoch = 1:training_para.nepoch
        
        # Create data loaders
        train_loader = DataLoader(data_train, batchsize=training_para.batchsize, shuffle=true)
        test_loader = DataLoader(data_test, batchsize=training_para.batchsize, shuffle=false)
        
        # Evaluate on first batch
        xtrain = first(train_loader)
        xtest = first(test_loader)
        
        # Compute and record losses
        train_losses = loss_catvae(model, xtrain, training_para.beta_cont, training_para.beta_disc)
        test_losses = loss_catvae(model, xtest, training_para.beta_cont, training_para.beta_disc)
        
        push!(loss_history.train_recon, train_losses.recon_loss)
        push!(loss_history.test_recon, test_losses.recon_loss)
        push!(loss_history.train_kl_cont, train_losses.cont_kl)
        push!(loss_history.test_kl_cont, test_losses.cont_kl)
        push!(loss_history.train_kl_disc, train_losses.disc_kl)
        push!(loss_history.test_kl_disc, test_losses.disc_kl)
        push!(loss_history.train_elbo, train_losses.elbo)
        push!(loss_history.test_elbo, test_losses.elbo)
        
        # Adjust learning rate
        Optimisers.adjust!(opt_state, lr_s(epoch))
        
        # Define loss function for gradient computation
        loss_fn(m, x) = loss_catvae(m, x, training_para.beta_cont, training_para.beta_disc).elbo
        
        # Training loop
        for x in train_loader
            g = Flux.gradient(loss_fn, model, x)[1]
            Optimisers.update!(opt_state, model, g)
        end
        
        # Print progress
        if mod(epoch, training_para.nprint) == 0
            @info "Epoch $epoch" train_elbo=train_losses.elbo test_elbo=test_losses.elbo train_recon=train_losses.recon_loss test_recon=test_losses.recon_loss
        end
    end
    
    return nothing
end

# ╔═╡ 234d7462-e4bd-4a7c-9ed4-7160a6a9ffb9
md"## Utility Functions"

# ╔═╡ 26aafe3d-e783-4591-959d-910b9c050301
"""
Sample from the prior p(z|c) for a given category
"""
function sample_prior(model::CatVAE, category::Int, n_samples::Int=1)
    d = size(model.μ_prior, 2)
    μ_c = model.μ_prior[category, :]
    ε = randn(Float32, d, n_samples) |> xpu
    z = μ_c .+ ε
    return model(z, Val(:decode))
end

# ╔═╡ c5d5be4d-882e-4c34-ae8f-1a40aa4cf215
"""
Classify input data into categories
Returns category assignments (1:K) for each input
"""
function classify(model::CatVAE, x)
    qcx = model(x, Val(:encode_c))
    # Get most likely category for each sample
    return vec(getindex.(argmax(qcx, dims=1), 1))
end

# ╔═╡ 0f292b15-bc79-4424-b5ae-fded09eb16f0
"""
Compute category probabilities for input data
"""
function get_category_probs(model::CatVAE, x)
    return cpu(model(x, Val(:encode_c)))
end

# ╔═╡ b9b5f42d-4ec1-43b5-9484-d3ec47dea61a
"""
Reconstruct input through the full model
"""
function reconstruct(model::CatVAE, x, sample::Bool=false)
    result = model(x, sample)
    return cpu(result.x_recon)
end

# ╔═╡ ae75eee7-ed34-4a2a-8aa3-08a06f504d36
"""
Generate samples by sampling category then latent code
"""
function generate_samples(model::CatVAE, n_samples::Int=1; category::Union{Nothing,Int}=nothing)
    K = size(model.μ_prior, 1)
    
    if category === nothing
        # Sample categories uniformly
        categories = rand(1:K, n_samples)
    else
        categories = fill(category, n_samples)
    end
    
    samples = map(categories) do c
        sample_prior(model, c, 1)
    end
    
    return cat(samples..., dims=2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MLUtils = "f1d291b0-491e-4a28-83b9-f70985020b54"
Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
ParameterSchedulers = "2abc6bf7-7a09-4ab3-b485-b963c0c8e2a4"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[compat]
CUDA = "~5.8.5"
Enzyme = "~0.13.93"
Flux = "~0.16.5"
MLUtils = "~0.4.4"
Optimisers = "~0.4.0"
ParameterSchedulers = "~0.3.7"
PlutoUI = "~0.7.59"
ProgressLogging = "~0.1.4"
cuDNN = "~1.4.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
"""

# ╔═╡ Cell order:
# ╟─461f0505-2230-4b84-b6c6-1a9730808437
# ╠═d73472ff-9e09-45b0-8811-b7dd8d820358
# ╠═97ae4222-5a3e-4cbd-b4d1-aa028d3e4ca8
# ╠═26fb86d5-c844-469a-aef5-ed3c2a9ba949
# ╟─3983e7d0-9ad0-11f0-0a96-7d2d98772fd2
# ╟─a91e28fb-e769-418d-953f-0e0bb366d853
# ╠═6db97fc1-8f11-42df-bffe-f86b8619a399
# ╠═fc228dea-21fc-4fcd-82a9-7ac3bc7ee722
# ╠═7c39a024-bf46-4024-b0da-a4d6092e864d
# ╟─ce690827-fa3f-48bc-bc09-1df5ee15f683
# ╠═a5302fa2-4f67-4ed6-96ce-dda78a160ffe
# ╠═bf31f347-bc9a-4bf8-a086-99dba2f6fea0
# ╟─ae96f920-5828-4c5f-b69f-48d8c4fee378
# ╠═96aebf7d-2112-4a4d-9993-6f53f40ffca5
# ╠═190c8221-c5c7-48f9-b016-36c27fd4528c
# ╟─66bddc43-ca9f-43cd-85a3-d33b11a6c033
# ╠═8684c192-d1b9-4821-a02e-2c7300af9b3c
# ╠═e9efedc1-8287-4676-ba64-a4abc77da18d
# ╟─5ed89ee8-325b-4757-b348-e6c1a3d277ad
# ╠═04f9b328-edc8-4b1e-9a7c-79a215b1cf5f
# ╠═1121e34c-ca35-4f68-8283-eca514928654
# ╠═f5e6cf15-1072-4037-9a96-91db93e730f0
# ╠═d966211f-012e-45f2-b9ae-abf599666edf
# ╟─7931ce6a-a062-4c7f-bb04-82b822e04eab
# ╠═eafe181e-19e9-409e-ad1d-ce859cf0e672
# ╠═7f8094da-d6a6-4b3d-b16c-cb0d7e928b9d
# ╟─234d7462-e4bd-4a7c-9ed4-7160a6a9ffb9
# ╠═26aafe3d-e783-4591-959d-910b9c050301
# ╠═c5d5be4d-882e-4c34-ae8f-1a40aa4cf215
# ╠═0f292b15-bc79-4424-b5ae-fded09eb16f0
# ╠═b9b5f42d-4ec1-43b5-9484-d3ec47dea61a
# ╠═ae75eee7-ed34-4a2a-8aa3-08a06f504d36
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
