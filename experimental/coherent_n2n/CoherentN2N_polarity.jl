# Per-earthquake complex gain (polarity + amplitude) estimation.
#
# Radiation-pattern effects can flip an earthquake's arrival polarity and/or
# scale its amplitude relative to other earthquakes at the same station.
# Left uncorrected, a polarity flip would partially cancel in the coherent
# stack. This closed-form one-tap complex least-squares fit (a matched
# filter / Wiener coefficient) recovers both sign and gain per earthquake.

"""
    estimate_earthquake_gain(d::AbstractVector{<:Complex}, ŝ_shifted::AbstractVector{<:Complex}) -> ComplexF32

Closed-form complex least-squares projection of the observed (shifted)
spectrum `d` onto the current source-estimate spectrum `ŝ_shifted` (already
shifted into `d`'s frame): `g = argmin_g ||d - g*ŝ_shifted||^2`, solved by
`g = (ŝ_shifted' * d) / (ŝ_shifted' * ŝ_shifted)`.
"""
function estimate_earthquake_gain(d::AbstractVector{<:Complex}, ŝ_shifted::AbstractVector{<:Complex})
    num = sum(conj.(ŝ_shifted) .* d)
    den = sum(abs2, ŝ_shifted)
    return ComplexF32(num / den)
end
