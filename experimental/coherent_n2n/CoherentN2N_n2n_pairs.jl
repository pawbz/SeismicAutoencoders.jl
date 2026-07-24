# Noise2Noise pair sampling for CoherentN2N.
#
# Confirmed data convention (single-station, multi-earthquake): "two
# independent looks of the same clean signal" means randomly selecting two
# *different* earthquakes recorded at the same station, aligned into the
# same frame. Each is an independent noisy realization (different noise,
# different radiation-pattern/path effects) of arrivals that should agree
# once aligned — so one earthquake's aligned trace can serve as the N2N
# input and a different, randomly-paired earthquake's aligned trace as the
# target.

using Random

"""
    sample_n2n_pairs(X̂_aligned::AbstractMatrix{ComplexF32}, n_samples::Int; rng=Random.default_rng()) -> (input, target)

Randomly draw `n_samples` (input, target) column pairs from the `R` aligned
earthquake spectra in `X̂_aligned` (`(nt, R)`), where input and target are
two *distinct* earthquake columns each time. Returns two `(nt, n_samples)`
ComplexF32 matrices. Re-call each epoch to redraw the pairing (and swap
input/target roles) so the denoiser never fits a fixed pairing.
"""
function sample_n2n_pairs(X̂_aligned::AbstractMatrix{ComplexF32}, n_samples::Int;
                           rng::AbstractRNG=Random.default_rng())
    nt, R = size(X̂_aligned)
    @assert R >= 2 "Need at least 2 earthquakes to form an N2N pair"

    # Draw all (a,b) index pairs on the CPU first, then gather columns in
    # one vectorized indexing op — column-by-column scalar indexing (the
    # previous approach) is fine for a CPU Matrix but is either extremely
    # slow or unsupported for a GPU CuArray (each column access would be a
    # separate kernel launch / scalar D2H transfer).
    a_idx, b_idx = draw_distinct_pair_indices(R, n_samples, rng)
    input = X̂_aligned[:, a_idx]
    target = X̂_aligned[:, b_idx]
    return input, target
end

"""
    draw_distinct_pair_indices(R::Int, n::Int, rng) -> (a_idx, b_idx)

Draw `n` `(a, b)` column-index pairs from `1:R` with `b != a` each time
(the N2N "two distinct looks" constraint). Returns two `Vector{Int}`. CPU-only
integer work, shared by the single-matrix and grouped `sample_n2n_pairs`
methods so the pairing rule lives in exactly one place.
"""
function draw_distinct_pair_indices(R::Int, n::Int, rng::AbstractRNG)
    a_idx = Vector{Int}(undef, n)
    b_idx = Vector{Int}(undef, n)
    for i in 1:n
        a = rand(rng, 1:R)
        b = rand(rng, 1:R)
        while b == a
            b = rand(rng, 1:R)
        end
        a_idx[i] = a
        b_idx[i] = b
    end
    return a_idx, b_idx
end

"""
    sample_n2n_pairs(groups::Vector{<:AbstractMatrix{ComplexF32}}, n_samples_per_group::Int; rng)
        -> (input, target)

Grouped N2N pair sampling for the per-receiver (multi-station) mode. Each
element of `groups` is one receiver's `(nt, R_g)` aligned spectrum matrix; a
"group" is a receiver and its columns are that receiver's events. Draws
`n_samples_per_group` *within-group* distinct `(a, b)` pairs from every group
with `R_g >= 2`, then horizontally concatenates all groups' inputs (and all
groups' targets) into two `(nt, Σ)` ComplexF32 matrices, `Σ = (# groups with
R_g>=2) * n_samples_per_group`.

Every pair is within a single receiver by construction — receivers are mixed
only at the final `hcat`, so a downstream minibatch may contain columns from
several receivers but no pair ever straddles two receivers. Groups with
`R_g < 2` (cannot form a pair) are silently skipped here; the caller
(`run_coherent_n2n_grouped`) is responsible for warning about / excluding them.
Equal count per group (not proportional to `R_g`) keeps any one large receiver
from dominating the gradient step.
"""
function sample_n2n_pairs(groups::Vector{<:AbstractMatrix{ComplexF32}}, n_samples_per_group::Int;
                           rng::AbstractRNG=Random.default_rng())
    inputs = Vector{Matrix{ComplexF32}}()
    targets = Vector{Matrix{ComplexF32}}()
    for X̂_g in groups
        R_g = size(X̂_g, 2)
        R_g >= 2 || continue
        a_idx, b_idx = draw_distinct_pair_indices(R_g, n_samples_per_group, rng)
        push!(inputs, X̂_g[:, a_idx])
        push!(targets, X̂_g[:, b_idx])
    end
    @assert !isempty(inputs) "No group has >= 2 columns; cannot form any N2N pair"
    return reduce(hcat, inputs), reduce(hcat, targets)
end
