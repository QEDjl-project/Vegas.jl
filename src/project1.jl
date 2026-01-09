# TBW

function sample_vegas!(backend, buffer::VegasBatchBuffer, grid::VegasGrid)
    # write the sampling code here, filling the buffer
    # you will need a kernel implementation and pass it the raw vectors/matrices
    # expect the buffer to be allocated, but otherwise uninitialized. the output should be written into it
    # example kernels like this are in src/grid.jl: @kernel function _fill_uniformly_kernel

    # it's a good idea to add some sanity checks too, like asserting that the buffers exist on the same backend, have matching dimensionalities where applicable, etc.

    return nothing
end

function binning_vegas!(backend, bins_buffer::AbstractVecOrMat, buffer::VegasBatchBuffer, grid::VegasGrid)
    # write the binning code here, the bins_buffer is the output

    return nothing
end
