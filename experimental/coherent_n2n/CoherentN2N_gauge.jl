# Zero-sum gauge fixing for per-earthquake shift vectors.
#
# The alternating loop's shift updates (Block B) have an inherent gauge
# freedom: adding a constant to every τ_r for a given station's earthquake
# set doesn't change any pairwise alignment. We fix this gauge by requiring
# the mean shift to be zero, and separately store what was subtracted so
# absolute timing can be restored at the end (see run_coherent_n2n).

using Statistics: mean

"""
    enforce_zero_sum_gauge!(τ::AbstractVector{Float32}) -> Float32

Subtract the mean of `τ` in place (zero-sum gauge). Returns the subtracted
mean — the "anchor" — which the caller should accumulate and reapply at the
end of the outer loop to restore an absolute time reference (anchored to
whatever the initial coarse cross-correlation lags implied).
"""
function enforce_zero_sum_gauge!(τ::AbstractVector{Float32})
    m = Float32(mean(τ))
    τ .-= m
    return m
end
