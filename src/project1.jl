# TBW

@kernel function vegas_sampling_kernel!(values, jacobians, weights, yi_sammples, yd_sammples, grid_nodes, @Const(Ng), @Const(D))
    
    i = @index(Global)

    Jac = one(eltype(jacobians))

    for d in 1:D
        yi = yi_sammples[i, d]
        yd = yd_sammples[i, d]

        x_start = grid_nodes[yi+1, d]
        x_end = grid_nodes[yi+2, d]
        width = x_end-x_start

        values[i, d] = x_start + width * yd

        Jac *= Ng*width
    end

    jacobians[i] = Jac
end

function sample_vegas!(backend, buffer::VegasBatchBuffer{T,N,D}, grid::VegasGrid{T,Ng,D}) where {T,N,D,Ng}
    # write the sampling code here, filling the buffer
    # you will need a kernel implementation and pass it the raw vectors/matrices
    # expect the buffer to be allocated, but otherwise uninitialized. the output should be written into it
    # example kernels like this are in src/grid.jl: @kernel function _fill_uniformly_kernel

    # it's a good idea to add some sanity checks too, like asserting that the buffers exist on the same backend, have matching dimensionalities where applicable, etc.
    # @assert get_backend(buffer)==backend
    # @assert ndims(buffer) == ndims(grid)

    yi_samples = rand(0:Ng-2, N, D)
    yd_samples = rand(T, N, D)

    yi_device = KernelAbstractions.allocate(backend, Int, N, D)
    yd_device = KernelAbstractions.allocate(backend, T, N, D)
    copyto!(yi_device, yi_samples)
    copyto!(yd_device, yd_samples)

    vegas_sampling_kernel!(backend)(buffer.values, buffer.jacobians, buffer.target_weights, yi_device, yd_device, grid.nodes, Ng, D, ndrange=N)
    return nothing
end

function binning_vegas!(backend, bins_buffer::AbstractVecOrMat, buffer::VegasBatchBuffer, grid::VegasGrid)
    # write the binning code here, the bins_buffer is the output

    return nothing
end
