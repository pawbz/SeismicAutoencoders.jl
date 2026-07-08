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
    a_idx = Vector{Int}(undef, n_samples)
    b_idx = Vector{Int}(undef, n_samples)
    for i in 1:n_samples
        a = rand(rng, 1:R)
        b = rand(rng, 1:R)
        while b == a
            b = rand(rng, 1:R)
        end
        a_idx[i] = a
        b_idx[i] = b
    end
    input = X̂_aligned[:, a_idx]
    target = X̂_aligned[:, b_idx]
    return input, target
end
