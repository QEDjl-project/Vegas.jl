module MetalExt

using Vegas
using Vegas.TestUtils
using Metal

@inline function Vegas.TestUtils.get_test_setup(backend::MetalBackend)
    return TestSetup(backend, (MtlVector,), (Float16, Float32))
end

function binning_vegas!(
        backend::MetalBackend,
        bins_buffer::AbstractMatrix{T},
        ndi_buffer::AbstractMatrix{I},
        buffer::VegasBatchBuffer{T, N, D, V, W, J},
        grid::VegasGrid{T, Ng, D, G},
        func::Function,
    ) where {T <: AbstractFloat, I <: Integer, N, D, V, W, J, Ng, G}
    # bins_buffer = Ng-1 x D
    # buffer.values = N x D
    # grid.jacobians = N
    # grid.nodes = Ng x D

    nbins = Ng - 1

    # NOTE: use unbatched implementation because Metal does not support atomix
    @debug "Calling unbatched binning kernel with $(nbins) bins * $(D) dims = $(nbins * D) threads"
    vegas_binning_kernel!(backend)(
        bins_buffer,
        ndi_buffer,
        buffer.values,
        grid.nodes,
        func,
        Ng,
        Val(D),
        ndrange = Int32(nbins) * Int32(D)
    )

    synchronize(backend)

    return nothing
end

end
