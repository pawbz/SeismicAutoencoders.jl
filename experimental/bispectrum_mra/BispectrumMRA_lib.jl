# Single entry point for @ingredients (PlutoLinks) to pull in the whole
# BispectrumMRA implementation as one namespace, with Revise tracking the
# included file for hot-reload. Mirrors CoherentN2N_lib.jl's role for the
# experimental/coherent_n2n module.
#
# There is only one file: BispectrumMRA.jl is self-contained (FFTW + Statistics
# only) and deliberately shares no code with CoherentN2N — the two are
# alternative solvers for the same problem, compared against each other in
# BispectrumMRA_main.jl rather than layered.

include(joinpath(@__DIR__, "BispectrumMRA.jl"))
