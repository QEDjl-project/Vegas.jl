using Atomix: @atomic

@kernel function vegas_sampling_kernel!(values, target_weights, jacobians, grid_lines, func::Function, @Const(Ng), d::Val{D}) where {D}
    
    i = @index(Global)

    jac = one(eltype(jacobians))

    for d in 1:D
        
        # calling rand() without explicit type seemed to generate values including 1, not sure why
        yn = rand(Float32) * (Ng - 1)   # generate random float in [0:1), scale it to grid
        yi = unsafe_trunc(Int, yn) + 1  # use its integer part as bin index
        yd = yn + 1 - yi                # and its fractional part as shift inside the bin
        
        x_start = grid_lines[yi, d]
        x_end = grid_lines[yi + 1, d]
        width = x_end - x_start
        
        values[i, d] = x_start + width * yd     # transform sample value to grid
        jac *= Ng * width                       # TODO: is this correct?
        
    end
    
    jacobians[i] = jac

    # Workaround by Anton to be able to call function func on sample
    V = ntuple(d -> (@inbounds values[i, d]), Val(D))
    target_weights[i] = jac * func(V)

end

function sample_vegas!(backend, buffer::VegasBatchBuffer{T, N, D, V, W, J}, grid::VegasGrid{T, Ng, D, G}, func::Function) where {T <: AbstractFloat, N, D, V, W, J, Ng, G}
    # buffer.values = N x D
    # grid.target_weights = N
    # grid.jacobians = N
    # grid.nodes = Ng x D

    @assert get_backend(buffer) == get_backend(grid) == backend
    
    @assert size(buffer.values, 2) == ndims(grid)
    @assert prod(size(buffer.values)) == N * D
    

    println("Calling sampling kernel with $(Ng) bins, $(D) dims and $(N) samples = $N threads")
    vegas_sampling_kernel!(backend)(buffer.values, buffer.target_weights, buffer.jacobians, grid.nodes, func, Ng, Val(D), ndrange = N)

    synchronize(backend)
    println("Sampling kernel finished")

    return nothing
end

@kernel function vegas_binning_kernel!(bins_buffer, ndi_buffer, values, target_weights, grid_lines, func::Function, @Const(Ng), ::Val{D}) where {D}

    nbins = Ng - 1

    # this is why we cant have nice things
    i = @index(Global) - 1
    bin = (i % nbins) + 1
    dim = div(i, nbins) + 1
    
    ndi = 0
    bins_buffer[bin, dim] = 0

    lower_bound = grid_lines[bin, dim]
    upper_bound = grid_lines[bin + 1, dim]

    for sample in 1:size(values, 1)
        in_bin = if bin == nbins
            lower_bound <= values[sample, dim] <= upper_bound
        else
            lower_bound <= values[sample, dim] < upper_bound
        end

        if in_bin
            ndi += 1
            V = ntuple(d -> (@inbounds values[sample, d]), Val(D))
            bins_buffer[bin, dim] += func(V) ^ 2
        end
    end

    # jacobian is the same for all samples in this bin/dim, so no need to sum it up
    jac = Ng * (grid_lines[bin + 1, dim] - grid_lines[bin, dim])
    
    bins_buffer[bin, dim] *= (jac ^ 2) / ndi
    
    ndi_buffer[bin, dim] = ndi  # used for sanity check that all samples are binned and none is lost

end


function binning_vegas!(backend, bins_buffer::AbstractMatrix{T}, ndi_buffer::AbstractMatrix{Int}, buffer::VegasBatchBuffer{T, N, D, V, W, J}, grid::VegasGrid{T, Ng, D, G}, func::Function) where {T <: AbstractFloat, N, D, V, W, J, Ng, G}

    # Approach fürs Thread Layout beim Binning: pro Bin ein Thread, welcher alle Samples iteriert und schnell prüft ob das sample in den Grenzen zum eigenen Bin liegt
    # Refinement: Block Size für Samples, damit sich mehrere Threads einen Bin teilen und und jeweils nen Subset aller Samples prüfen (nicht sicher ob richtig verstanden, damit wird sync belastender)

    # bins_buffer = Ng-1 x D
    # buffer.values = N x D
    # grid.jacobians = N
    # grid.nodes = Ng x D

    # TODO: optimize for large batch sizes? (multiple threads per bin)
    # bins_per_thread = 1
    
    nbins = Ng - 1

    println("Calling binning kernel with $(nbins) bins * $(D) dims = $(nbins * D) threads")

    vegas_binning_kernel!(backend)(bins_buffer, ndi_buffer, buffer.values, buffer.target_weights, grid.nodes, func, Ng, Val(D), ndrange = nbins * D)
    synchronize(backend)
    
    println("Binning kernel finished")

    return nothing
end
