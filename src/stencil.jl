@kernel function _vegas_stencil_kernel!(bins_buffer::AbstractMatrix{T}, @Const(sums::AbstractMatrix{T}), @Const(alpha::T)) where {T <: Number}
    (loc_bin, loc_dim) = @index(Local, NTuple)
    (glob_bin, glob_dim) = @index(Global, NTuple)

    # commented out checks cause GPU doesn't like asserts, but keep it here as documentation

    # loc_bin and glob_bin must be identical or the started block wasn't large enough!
    #@assert loc_bin == glob_bin

    # loc_dim should be 1, 1 dimension is handled by one block
    #@assert loc_dim == 1

    (no_bins, _) = @uniform @groupsize()
    local_buffer = @localmem T no_bins

    # load into localmem
    @inbounds local_buffer[loc_bin] = bins_buffer[glob_bin, glob_dim]

    # sync
    @synchronize

    dim_sum = @inbounds sums[begin, glob_dim]

    # load middle value, and conditionally l and r
    m = local_buffer[loc_bin]
    l = loc_bin == 1 ? m : local_buffer[loc_bin - 1]
    r = loc_bin == size(bins_buffer, 1) ? m : local_buffer[loc_bin + 1]

    # calculate new value and write back
    bins_buffer[glob_bin, glob_dim] = _vegas_stencil(l, m, r, dim_sum, alpha)
end

function _vegas_stencil(left::T, middle::T, right::T, sum::T, alpha::T) where {T <: Number}
    smoothed = (left + 6 * middle + right) / 8
    normalized = smoothed / sum
    compressed = ((one(T) - normalized) / (log(inv(normalized))))^alpha
    return compressed
end

function stencil_vegas!(backend, bins_buffer::AbstractMatrix{T}, alpha::Number) where {T <: Number}
    if typeof(get_backend(bins_buffer)) != typeof(backend)
        throw(ArgumentError("buffer does not belong to the passed backend"))
    end

    (no_bins, dims) = size(bins_buffer)
    sums = allocate(backend, eltype(bins_buffer), (1, dims))

    sum!(sums, bins_buffer)                 # uses GPU implementation
    kernel_size = (no_bins, 1)
    problem_size = (no_bins, dims)
    _vegas_stencil_kernel!(backend, kernel_size)(bins_buffer, sums, T(alpha), ndrange = problem_size)
    return nothing
end
