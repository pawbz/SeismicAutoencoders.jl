# Time-domain Wave-U-Net denoiser for CoherentN2N - real waveforms in, real
# waveforms out, no FFT anywhere.
#
# This file intentionally ports the architecture from
# waveunet_noise2self_noisy_input_denoising.ipynb while keeping CoherentN2N's
# existing data formulation. The attached notebook uses receiver-held-out
# masked-neighbor inputs; the binned CoherentN2N path still trains with grouped
# Noise2Noise pairs over columns inside each station/bin. Only the network shape
# is borrowed here.

using Flux

const waveunet_activation_cn2n = x -> leakyrelu(x, 0.2f0)

"""
    TimeUNet

Wave-U-Net-style real-valued denoiser. Public CoherentN2N calls use
`model(x::Matrix{Float32})` with shape `(nt, B)` and receive the same shape back.
For architecture checks, a 3-D `(nt, channels, B)` input is also accepted.

The architecture mirrors the attached PyTorch notebook:
- same-padding Conv1D encoder blocks with linearly growing widths;
- downsampling by `x[..., ::2]` / `h[1:2:end, :, :]` after each encoder block;
- bottleneck Conv1D at `num_initial_filters * (num_layers + 1)`;
- decoder interpolation to each skip length, skip concatenation, merge Conv1D;
- final `1x1` output head over `[original_padded_input; decoder_state]`;
- symmetric pad to a multiple of `2^num_layers`, center-trim to `nt`;
- zero temporal mean projection on each returned waveform.
"""
struct TimeUNet{E,D,B,O}
    encoders::E
    decoders::D
    bottleneck::B
    output_head::O
    nt::Int
    pad_to::Int
    input_channels::Int
    output_channels::Int
    num_layers::Int
end
Flux.@layer TimeUNet trainable = (encoders, decoders, bottleneck, output_head)

"""
    pad_crop_length(nt, depth) -> pad_to

Smallest length at least `nt` and at least `2^depth`, divisible by `2^depth`.
This matches the attached notebook's `padded_length`, e.g. `4000 -> 4096` for
nine layers.
"""
pad_crop_length(nt::Int, depth::Int) = max(2^depth, cld(nt, 2^depth) * 2^depth)

center_trim_time(y, nt::Int, pl::Int) = y[(pl + 1):(pl + nt), :, :]
zero_mean_time(y) = y .- sum(y; dims=1) ./ size(y, 1)

function (m::TimeUNet)(x::AbstractMatrix{Float32})
    @assert m.input_channels == 1 "2-D TimeUNet input requires input_channels=1"
    y = m(reshape(x, size(x, 1), 1, size(x, 2)))
    @assert m.output_channels == 1 "2-D TimeUNet output requires output_channels=1"
    return dropdims(y; dims=2)
end

function (m::TimeUNet)(x::AbstractArray{Float32,3})
    nt, C, B = size(x)
    @assert nt == m.nt "TimeUNet built for nt=$(m.nt) but got $(nt)"
    @assert C == m.input_channels "TimeUNet built for $(m.input_channels) input channel(s) but got $C"

    npad = m.pad_to - nt
    pl = npad ÷ 2
    pr = npad - pl
    h = npad > 0 ? Flux.pad_zeros(x, (pl, pr); dims=1) : x
    # Keep the top-level input skip as its own AD value. Reusing the same `h`
    # binding that is later decimated through the encoder makes Zygote try to
    # accumulate gradients for different time lengths.
    original_input = copy(h)

    skips = ()
    for enc in m.encoders
        h = enc(h)
        skips = (skips..., h)
        h = h[1:2:end, :, :]
    end

    h = m.bottleneck(h)

    for (i, dec) in enumerate(m.decoders)
        skip = skips[end - i + 1]
        h = Upsample(:bilinear; size=size(skip)[1:1])(h)
        h = cat(skip, h; dims=2)
        h = dec(h)
    end

    y = m.output_head(cat(original_input, h; dims=2))
    y = npad > 0 ? center_trim_time(y, nt, pl) : y
    return zero_mean_time(y)
end

"""
    build_time_unet(nt; kwargs...) -> TimeUNet

Build the Wave-U-Net architecture used by the binned time-domain CoherentN2N
path. Preferred keywords mirror the attached notebook:

- `input_channels` / `output_channels`
- `num_layers`
- `num_initial_filters`
- `filter_size`
- `merge_filter_size`

Backward-compatible aliases are kept: `depth`, `width`, and `kernel_size` map to
`num_layers`, `num_initial_filters`, and `filter_size`. `growth` and
`taper_fraction` are accepted but ignored so older notebook calls do not break.
"""
function build_time_unet(nt::Int;
                          input_channels::Int=1,
                          output_channels::Int=1,
                          num_layers::Union{Nothing,Int}=nothing,
                          num_initial_filters::Union{Nothing,Int}=nothing,
                          filter_size::Union{Nothing,Int}=nothing,
                          merge_filter_size::Int=5,
                          depth::Int=9,
                          width::Int=24,
                          kernel_size::Int=15,
                          growth::Int=0,
                          taper_fraction::Real=0)
    nl = num_layers === nothing ? depth : num_layers
    nf = num_initial_filters === nothing ? width : num_initial_filters
    fs = filter_size === nothing ? kernel_size : filter_size

    @assert input_channels >= 1 "input_channels must be >= 1"
    @assert output_channels >= 1 "output_channels must be >= 1"
    @assert nl >= 1 "num_layers/depth must be >= 1"
    @assert nf >= 1 "num_initial_filters/width must be >= 1"
    @assert isodd(fs) "filter_size/kernel_size must be odd for same padding"
    @assert isodd(merge_filter_size) "merge_filter_size must be odd for same padding"

    pad_to = pad_crop_length(nt, nl)
    widths = [nf * level for level in 1:nl]

    encoders = Any[]
    encoder_inputs = [input_channels; widths[1:end-1]]
    for (cin, cout) in zip(encoder_inputs, widths)
        push!(encoders, Conv((fs,), cin => cout, waveunet_activation_cn2n;
                             pad=SamePad()))
    end

    bottleneck_width = nf * (nl + 1)
    bottleneck = Conv((fs,), widths[end] => bottleneck_width,
                      waveunet_activation_cn2n; pad=SamePad())

    decoders = Any[]
    current_width = bottleneck_width
    for skip_width in reverse(widths)
        push!(decoders, Conv((merge_filter_size,),
                             (current_width + skip_width) => skip_width,
                             waveunet_activation_cn2n; pad=SamePad()))
        current_width = skip_width
    end

    output_head = Conv((1,), (widths[1] + input_channels) => output_channels;
                       pad=SamePad())

    return TimeUNet(encoders, decoders, bottleneck, output_head, nt, pad_to,
                    input_channels, output_channels, nl)
end
