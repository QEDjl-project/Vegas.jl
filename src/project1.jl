using Atomix: @atomic

@kernel function vegas_sampling_kernel!(values, target_weights, jacobians, grid_lines, yi_samples, yd_samples, func, @Const(Ng), @Const(D))
    
    i = @index(Global)

    @private jac = one(eltype(jacobians))

    # TODO: generate random values with rand()
    # TODO: sample just one f32 from 0 to 1, scale it to Ng to calculate both yi and yd

    for d in 1:D
        yi = yi_samples[i, d]
        yd = yd_samples[i, d]

        x_start = grid_lines[yi, d]
        x_end = grid_lines[yi + 1, d]
        width = x_end - x_start
        
        values[i, d] = x_start + width * yd
        jac *= Ng * width
        
    end
    
    jacobians[i] = jac

    # Broadcasting with prod() doesn't work here because it seems to dynamically allocate memory?
    # Reason: unsupported call to an unknown function (call to jl_alloc_genericmemory_unchecked)
    target_weight = one(eltype(values))
    for d in 1:D
        # TODO: multiply jacobian always or once?
        target_weight *= jac * func(values[i,d])
    end

    target_weights[i] = target_weight

end

function sample_vegas!(backend, buffer::VegasBatchBuffer{T, N, D, V, W, J}, grid::VegasGrid{T, Ng, D, G}, func::Function) where {T <: AbstractFloat, N, D, V, W, J, Ng, G}
    # write the sampling code here, filling the buffer
    # you will need a kernel implementation and pass it the raw vectors/matrices
    # expect the buffer to be allocated, but otherwise uninitialized. the output should be written into it
    # example kernels like this are in src/grid.jl: @kernel function _fill_uniformly_kernel

    # it's a good idea to add some sanity checks too, like asserting that the buffers exist on the same backend, have matching dimensionalities where applicable, etc.

    # buffer.values = N x D
    # grid.target_weights = N
    # grid.jacobians = N
    # grid.nodes = Ng x D

    @assert get_backend(buffer) == get_backend(grid) == backend
    # println("Using backend ", backend)
    
    @assert size(buffer.values, 2) == ndims(grid)
    @assert prod(size(buffer.values)) == N * D
    println("Number of samples / dimensions: ", N, " / ", ndims(grid))

    yi_samples = rand(1:Ng-1, (N, D))
    yd_samples = rand(T, (N, D))
    
    yi_device = KernelAbstractions.allocate(backend, Int, N, D)
    yd_device = KernelAbstractions.allocate(backend, T, N, D)
    copyto!(yi_device, yi_samples)
    copyto!(yd_device, yd_samples)

    println("Calling sampling kernel with ndrange ", N)
    vegas_sampling_kernel!(backend)(buffer.values, buffer.target_weights, buffer.jacobians, grid.nodes, yi_device, yd_device, func::Function, Ng, D, ndrange = N)
    
    synchronize(backend)
    println("Sampling kernel finished")

    return nothing
end

# TODO: completely untested, treat as pseudo-code 
@kernel function vegas_binning_kernel!(bins_buffer, values, target_weights, jacobians, grid_lines, func, @Const(Ng), @Const(D))

    bin = @index(Global, Cartesian)
    
    T = eltype(values)
    nbins = Ng - 1
    Ndi = 0
    # TODO: InexactError: trunc(UInt32, 52879244321342)
    batch_size = (nbins ^ D) / prod(@ndrange())

    for sample in 1:size(values, 1)
        if all(d -> grid_lines[bin[d], d] < values[sample, d] < grid_lines[bin[d] + 1, d], 1:D)
            Ndi += 1
            # bins_buffer[bin[d], d] += func(values[sample, :]) ^ 2
        end
    end


    # for sample in 1:size(values, 1)
        
    #     for d in 1:D
    #         if grid_lines[bin[d], d] < values[sample, d] < grid_lines[bin[d] + 1, d]
    #             Ndi += 1
    #         end
#             bins_buffer[bin[d], d] += func(values[sample, :]) ^ 2
    #     end
    # end

    for d in 1:D
        @atomic bins_buffer[bin[d], d] += Ndi   # TODO: currently just count number of elements in bin, replace with next statement
        # bins_buffer[bin[d], d] *= jacobians[bin[d]] ^ 2 / Ndi
    end


end


function binning_vegas!(backend, bins_buffer::AbstractMatrix{T}, buffer::VegasBatchBuffer{T, N, D, V, W, J}, grid::VegasGrid{T, Ng, D, G}, func::Function) where {T <: AbstractFloat, N, D, V, W, J, Ng, G}

    # Approach fürs Thread Layout beim Binning: pro Bin ein Thread, welcher alle Samples iteriert und schnell prüft ob das sample in den Grenzen zum eigenen Bin liegt
    # Refinement: Block Size für Samples, damit sich mehrere Threads einen Bin teilen und und jeweils nen Subset aller Samples prüfen (nicht sicher ob richtig verstanden, damit wird sync belastender)
    

    # bins_buffer = Ng-1 x D
    # buffer.values = N x D
    # grid.jacobians = N
    # grid.nodes = Ng x D


    bins_per_thread = 1
    nbins = Ng - 1
    ndranges = Tuple(nbins ÷ bins_per_thread for _ in 1:D)
    
    println("Calling binning kernel with ndranges ", ndranges)
    
    vegas_binning_kernel!(backend)(bins_buffer, buffer.values, buffer.target_weights, buffer.jacobians, grid.nodes, func, Ng, D, ndrange = ndranges)
    
    synchronize(backend)
    println("Binning kernel finished")

    return nothing
end
