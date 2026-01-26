using Atomix: @atomic

@kernel function vegas_sampling_kernel!(values::AbstractMatrix{T}, target_weights::AbstractVector{T}, jacobians::AbstractVector{T}, grid_lines, func::Function, @Const(Ng), d::Val{D}) where {T<:Number, D}
    
    i = @index(Global)

    jac = one(T)

    for d in 1:D
        
        # yn = randoms[i, d] * (Ng - 1)         # would be used instead of next line in case of host generated randoms
        yn = (rand() % 1) * (Ng - 1)            # generate random float in [0:1), scale it to grid
        yi = unsafe_trunc(Int, yn) + 1          # use its integer part as bin index
        yd = yn + 1 - yi                        # and its fractional part as shift inside the bin
        
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

    # CUDABackend() != CUDABackend(prefer_blocks = true)
    @assert typeof(get_backend(buffer)) == typeof(get_backend(grid)) == typeof(backend)
    @assert size(buffer.values, 2) == ndims(grid)
    @assert prod(size(buffer.values)) == N * D

    # TODO: enables oneAPI, breaks AMDGPU (._.)
    # randoms = rand!(allocate(backend, T, (N, D)))
    
    @debug "Calling sampling kernel with $(Ng - 1) bins, $(D) dims and $(N) samples = $N threads"
    vegas_sampling_kernel!(backend)(buffer.values, buffer.target_weights, buffer.jacobians, grid.nodes, func, Ng, Val(D), ndrange = N)

    synchronize(backend)
    
    return nothing
end

@kernel function vegas_binning_kernel!(bins_buffer::AbstractMatrix{T}, ndi_buffer::AbstractMatrix{Int}, values::AbstractMatrix{T}, grid_lines, func::Function, @Const(Ng), ::Val{D}) where {T<:Number, D}

    bin, dim, batch = @index(Global, NTuple)
    batch -= 1

    bin_sum = 0
    ndi = 0

    lower_bound = grid_lines[bin, dim]
    upper_bound = grid_lines[bin + 1, dim]

    threads_per_bin = @ndrange()[3]
    batch_size = div(size(values, 1), threads_per_bin)
    batch_start = batch * batch_size + 1
    batch_end = (batch + 1) * batch_size

    is_last_bin = bin == Ng - 1

    for sample in batch_start:batch_end

        in_bin = if is_last_bin
            lower_bound <= values[sample, dim] <= upper_bound
        else
            lower_bound <= values[sample, dim] < upper_bound
        end
        
        if in_bin
        # if lower_bound <= values[sample, dim] < upper_bound || (is_last_bin && values[sample, dim] == upper_bound)
            V = ntuple(d -> (@inbounds values[sample, d]), Val(D))
            bin_sum += func(V)^2
            ndi += 1
        end
    end

    # jacobian is the same for all samples in this bin/dim, so no need to sum it up
    jac = Ng * (grid_lines[bin + 1, dim] - grid_lines[bin, dim])
    @atomic bins_buffer[bin, dim] += (jac^2) / ndi * bin_sum / threads_per_bin

    # used for sanity check that all samples are binned and none is lost
    @atomic ndi_buffer[bin, dim] += ndi

end


function binning_vegas!(backend, bins_buffer::AbstractMatrix{T}, ndi_buffer::AbstractMatrix{Int}, buffer::VegasBatchBuffer{T, N, D, V, W, J}, grid::VegasGrid{T, Ng, D, G}, func::Function, threads_per_bin::Int = 1024) where {T <: AbstractFloat, N, D, V, W, J, Ng, G}

    # bins_buffer = Ng-1 x D
    # buffer.values = N x D
    # grid.jacobians = N
    # grid.nodes = Ng x D

    nbins = Ng - 1

    # need to be zeroed because every thread just adds its calculation
    fill!(bins_buffer, 0)
    fill!(ndi_buffer, 0)

    @debug "Calling binning kernel with $(nbins) bins * $(D) dims * $(threads_per_bin) threads/bin = $(nbins * D * threads_per_bin) threads"

    vegas_binning_kernel!(backend)(bins_buffer, ndi_buffer, buffer.values, grid.nodes, func, Ng, Val(D), ndrange = (nbins, D, threads_per_bin))
    
    synchronize(backend)

    return nothing
end