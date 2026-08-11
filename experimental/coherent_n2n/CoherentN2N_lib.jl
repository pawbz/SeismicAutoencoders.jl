# Single entry point for @ingredients (PlutoLinks) to pull in the whole
# CoherentN2N implementation as one namespace, with Revise tracking each
# included file for hot-reload. Include order matches dependency order:
# shift/gauge/polarity/binning have no cross-file deps; denoiser and unet depend
# on Flux only (unet also uses activation_cn2n from denoiser); n2n_pairs is
# standalone; train_denoiser calls sample_n2n_pairs and complex_mse_loss;
# outer_loop calls into all of the above; time_loop comes LAST because it reuses
# outer_loop's CoherentN2N_Para / CoherentN2N_Outer_Para / _noalign_warn_ignored
# and n2n_pairs' draw_distinct_pair_indices.

include(joinpath(@__DIR__, "CoherentN2N_shift.jl"))
include(joinpath(@__DIR__, "CoherentN2N_binning.jl"))
include(joinpath(@__DIR__, "CoherentN2N_gauge.jl"))
include(joinpath(@__DIR__, "CoherentN2N_polarity.jl"))
include(joinpath(@__DIR__, "CoherentN2N_denoiser.jl"))
include(joinpath(@__DIR__, "CoherentN2N_unet.jl"))
include(joinpath(@__DIR__, "CoherentN2N_n2n_pairs.jl"))
include(joinpath(@__DIR__, "CoherentN2N_train_denoiser.jl"))
include(joinpath(@__DIR__, "CoherentN2N_outer_loop.jl"))
include(joinpath(@__DIR__, "CoherentN2N_time_loop.jl"))
