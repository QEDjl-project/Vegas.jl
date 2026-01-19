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

    f_val = one(eltype(values))
    for d in 1:D
        f_val *= jac * func(values[i,d])
    end

    target_weights[i] = f_val

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
    println("Number of samples, dimensions: ", N, ", ", ndims(grid))

    yi_samples = rand(1:Ng-1, (N, D))
    yd_samples = rand(T, (N, D))
    
    yi_device = KernelAbstractions.allocate(backend, Int, N, D)
    yd_device = KernelAbstractions.allocate(backend, T, N, D)
    copyto!(yi_device, yi_samples)
    copyto!(yd_device, yd_samples)

    println("Calling sampling kernel")
    vegas_sampling_kernel!(backend)(buffer.values, buffer.target_weights, buffer.jacobians, grid.nodes, yi_device, yd_device, func::Function, Ng, D, ndrange = N)
    
    synchronize(backend)
    println("Kernel finished")

    return nothing
end

# TODO: completely untested, treat as pseudo-code 
@kernel function vegas_binning_kernel!(bins_buffer, values, target_weights, jacobians, grid_lines, func, @Const(Ng), @Const(D))

    i = @index(Global)

    # buffer.values = N x D
    # grid.jacobians = N
    # grid.nodes = Ng x D


    eltype = eltype(values[first, first])
    nbins = Ng - 1
    bin = ones(eltype, d) # get this from @index
    Ndi = zeros(D)
    bounds = Array{Tuple(eltype)}(undef, D)

    # lower and upper bound for my bin
    for d in 1:D
    
        bounds[d] = (grid_lines[bin[d], d], grid_lines[bin[d] + 1, d])
        bins_buffer[bin[d], d] = 0

    end
    
    # TODO: need batch size here
    for sample in 1:size(values, 1)
        
        for d in 1:D

            if bound_lower < values[sample, d] < bound_upper

                bins_buffer[bin, d] += func(values[sample, :]) ^ 2
                Ndi += 1
            end
        end
    end

    for d in 1:D
    
        bins_buffer[bin[d], dim] *= jacobians[bin[d]] ^ 2 / Ndi[d]

    end


end



function binning_vegas!(backend, bins_buffer::AbstractVecOrMat{T}, buffer::VegasBatchBuffer{T, N, D, V, W, J}, grid::VegasGrid{T, Ng, D, G}, func::Function) where {T <: AbstractFloat, N, D, V, W, J, Ng, G}

    # Approach fürs Thread Layout beim Binning: pro Bin ein Thread, welcher alle Samples iteriert und schnell prüft ob das sample in den Grenzen zum eigenen Bin liegt
    # Refinement: Block Size für Samples, damit sich mehrere Threads einen Bin teilen und und jeweils nen Subset aller Samples prüfen (nicht sicher ob richtig verstanden, damit wird sync belastender)
    

    # bins_buffer = Ng-1 x D

    batch_size = N
    number_of_bins = (Ng - 1) ^ D
    threads_per_bin = N / batch_size
    
    println("Calling binning kernel")
    
    vegas_binning_kernel!(backend)(bins_buffer, buffer.values, buffer.target_weights, buffer.jacobians, grid.nodes, func, Ng, D, ndrange = (number_of_bins, threads_per_bin))
    
    synchronize(backend)
    println("Kernel finished")

    return nothing
end
