# Version utilities for SeismicAutoencoders

const VERSION = "2026.06"

function get_version()::String
    VERSION
end

function version_string()::String
    "Symmetric VQ-VAE v$(VERSION)"
end
