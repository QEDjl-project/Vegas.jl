# TBW

function stencil_vegas!(backend, bins_buffer::AbstractVecOrMat, alpha::Real)
    # write the stencil/compression code here
    # you will need a kernel implementation and pass it the raw vectors/matrices
    # the bins_buffer is both input and output and should be overwritten with the updated values
    # example kernels like this are in src/grid.jl: @kernel function _fill_uniformly_kernel

    # it's a good idea to add some sanity checks too, like asserting that the buffers exist on the same backend, have matching dimensionalities where applicable, etc.

    return nothing
end

function scan_vegas!(backend, bins_buffer::AbstractVecOrMat)
    # write the scanning code here
    # bins_buffer is both input and output, override it with the result

    # caluclate this value too, take care it has the right element type
    T = eltype(bins_buffer)
    avg_d = T(0.0)
    return avg_d
end

function refine_vegas!(backend, grid::VegasGrid, bins_buffer::AbstractVecOrMat, avg_d::Real)
    # write the refining code here
    # grid is both input and output, override it with the result

    return nothing
end
