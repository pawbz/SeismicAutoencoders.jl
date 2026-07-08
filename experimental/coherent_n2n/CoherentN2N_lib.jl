# Single entry point for @ingredients (PlutoLinks) to pull in the whole
# CoherentN2N implementation as one namespace, with Revise tracking each
# included file for hot-reload. Include order matches dependency order:
# shift/gauge/polarity have no cross-file deps; denoiser depends on Flux
# only; n2n_pairs is standalone; train_denoiser calls sample_n2n_pairs and
# complex_mse_loss; outer_loop calls into all of the above.

include(joinpath(@__DIR__, "CoherentN2N_shift.jl"))
include(joinpath(@__DIR__, "CoherentN2N_gauge.jl"))
include(joinpath(@__DIR__, "CoherentN2N_polarity.jl"))
include(joinpath(@__DIR__, "CoherentN2N_denoiser.jl"))
include(joinpath(@__DIR__, "CoherentN2N_n2n_pairs.jl"))
include(joinpath(@__DIR__, "CoherentN2N_train_denoiser.jl"))
include(joinpath(@__DIR__, "CoherentN2N_outer_loop.jl"))
