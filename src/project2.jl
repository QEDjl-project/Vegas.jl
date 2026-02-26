import AcceleratedKernels as AK

@kernel function _vegas_stencil_kernel!(bins_buffer::AbstractVecOrMat{T}, @Const(sums::AbstractMatrix), @Const(alpha::Real)) where {T <: Number}
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
    # TODO: could probably be improved by using more threads to append the border values left and right and remove this condition
    m = local_buffer[loc_bin]
    l = loc_bin == 1 ? m : local_buffer[loc_bin - 1]
    r = loc_bin == size(bins_buffer, 1) ? m : local_buffer[loc_bin + 1]

    # calculate new value and write back
    bins_buffer[glob_bin, glob_dim] = _vegas_stencil(l, m, r, dim_sum, alpha)
end
#=
@kernel function _vegas_stencil_kernel!(bins_buffer::AbstractVecOrMat, @Const(sums::AbstractMatrix), @Const(alpha::Real))
    first_bin = firstindex(bins_buffer, 1)
    last_bin = lastindex(bins_buffer, 1)
    local_bin = @index(Local, Linear)
    dim = @index(Group, Linear)
    dim_sum = @inbounds sums[begin, dim]

    # use local memory to reduce loads from global memory
    @uniform batch_size = prod(@groupsize())
    local_buffer = @localmem eltype(bins_buffer) (batch_size + 2)

    # keep last middle value to prevent loading modified data from global memory
    @private middle_value = @inbounds bins_buffer[begin, dim]
    for starting_bin in first_bin:batch_size:last_bin
        # load new batch from global memory into local memory
        @private global_index = starting_bin + local_bin - Int32(1)
        @private local_index = firstindex(local_buffer) + local_bin

        # load left and right value of boundary threads
        if local_bin == batch_size
            @inbounds local_buffer[begin] = middle_value
            @inbounds local_buffer[end] = _vegas_stencil_get_value(bins_buffer, nextind(bins_buffer, global_index), dim)
        end

        # load middle value
        @private middle_value = _vegas_stencil_get_value(bins_buffer, global_index, dim)
        @inbounds local_buffer[local_index] = middle_value

        # wait until data in local buffer is ready
        @synchronize()

        if first_bin <= global_index <= last_bin
            # load left and right values
            left_value = @inbounds local_buffer[prevind(local_buffer, local_index)]
            right_value = @inbounds local_buffer[nextind(local_buffer, local_index)]

            # execute stencil and write result to global memory
            result = _vegas_stencil(left_value, middle_value, right_value, dim_sum, alpha)
            @inbounds bins_buffer[global_index, dim] = result
        end

        # wait until data in local buffer is no longer needed
        @synchronize()
    end
end
=#

function _vegas_stencil_get_value(bins_buffer::AbstractVecOrMat, bin::Integer, dim::Integer)
    first_bin = firstindex(bins_buffer, 1)
    last_bin = lastindex(bins_buffer, 1)
    return @inbounds bins_buffer[clamp(bin, first_bin, last_bin), dim]
end

function _vegas_stencil(left, middle, right, sum, alpha::Real)
    T = promote_type(typeof(middle), typeof(alpha))
    smoothed = (left + T(6) * middle + right) / T(8)
    normalized = smoothed / sum
    compressed = ((one(T) - normalized) / (log(inv(normalized))))^alpha
    return compressed
end

function stencil_vegas!(backend, bins_buffer::AbstractVecOrMat, alpha::Real)
    if typeof(get_backend(bins_buffer)) != typeof(backend)
        throw(ArgumentError("buffer does not belong to the passed backend"))
    end

    (no_bins, dims) = size(bins_buffer)
    sums = allocate(backend, eltype(bins_buffer), (1, dims))

    sum!(sums, bins_buffer)                 # uses GPU implementation
    kernel_size = (no_bins, 1)
    problem_size = (no_bins, dims)
    _vegas_stencil_kernel!(backend, kernel_size)(bins_buffer, sums, alpha, ndrange = problem_size)
    return nothing
end

@kernel function _get_last_col!(lastvals, A, N::Int32, D::Int32)
    d = @index(Global)
    if d <= D
        @inbounds lastvals[d] = A[N, d]
    end
end

@kernel function _scale_by_invN!(out, lastvals, invN, D::Int32)
    d = @index(Global)
    if d <= D
        @inbounds out[d] = lastvals[d] * invN
    end
end

function scan_vegas!(backend, bins_buffer::AbstractVecOrMat)
    T = eltype(bins_buffer)

    # Handle the vector case
    A = bins_buffer isa AbstractVector ? reshape(bins_buffer, :, 1) : bins_buffer

    N = Int32(size(A, 1))
    D = Int32(size(A, 2))
    # 1) Inclusive scan per column (in-place) using AcceleratedKernels
    #    dims=1 means scanning along the first dimension (column-wise)
    AK.accumulate!(+, A, A; dims = 1, init = zero(T))

    # 2) Get the last value of each column
    lastvals = similar(A, T, (Int(D),))
    kernel_last = _get_last_col!(backend)
    kernel_last(lastvals, A, N, D; ndrange = Int(D))
    KernelAbstractions.synchronize(backend)

    # 3) Compute avg_d = lastvals / N
    avg_d = similar(A, T, (Int(D),))
    invN = T(1) / T(N)
    kernel_scale = _scale_by_invN!(backend)
    kernel_scale(avg_d, lastvals, invN, D; ndrange = Int(D))
    KernelAbstractions.synchronize(backend)

    return avg_d
end

@kernel function _copy_bounds_nodes!(new_nodes, old_nodes, Np1::Int32, D::Int32)
    d = @index(Global)

    new_nodes[1, d] = old_nodes[1, d]
    new_nodes[Np1, d] = old_nodes[Np1, d]

end

# refine internal nodes using prefix array (bins_buffer) and avg_d per dimension
@kernel function _refine_nodes_bs!(
        new_nodes, old_nodes, prefix, avg_d, N::Int32, D::Int32
    )
    tid = @index(Global)

    #decode(k,d)
    d = Int32(((tid - 1) % D) + 1)     # Dimension Index; 1..D
    k = Int32(((tid - 1) ÷ D) + 1)     # internal node; 1..N-1
    i = k + Int32(1)                   # New node number; 2..N

    target = eltype(new_nodes)(k) * avg_d[d]

    # binary search in prefix[:, d] for smallest j such that prefix[j,d] >= goal
    lo = Int32(1)
    hi = N
    while lo < hi
        mid = (lo + hi) >>> 1
        if prefix[mid, d] < target
            lo = mid + Int32(1)
        else
            hi = mid
        end
    end
    j = Int32(lo)

    prev = j == 1 ? zero(eltype(prefix)) : prefix[j - 1, d]
    dj = prefix[j, d] - prev
    t = (target - prev) / dj

    x0 = old_nodes[j, d]
    x1 = old_nodes[j + 1, d]
    new_nodes[i, d] = x0 + t * (x1 - x0)

end

function refine_vegas!(backend, grid::VegasGrid, bins_buffer::AbstractVecOrMat, avg_d::AbstractVector)
    # write the refining code here
    # grid is both input and output, override it with the result
    @assert bins_buffer isa AbstractVecOrMat

    old_nodes = grid.nodes               # (N+1, D)
    new_nodes = similar(old_nodes)

    N = Int32(size(bins_buffer, 1))      # nbins
    D = Int32(size(bins_buffer, 2))      # dim
    Np1 = Int32(size(old_nodes, 1))      # N+1

    _copy_bounds_nodes!(backend)(new_nodes, old_nodes, Np1, D; ndrange = Int(D))

    total = Int((N - Int32(1)) * D)
    _refine_nodes_bs!(backend)(new_nodes, old_nodes, bins_buffer, avg_d, N, D; ndrange = total)

    KernelAbstractions.synchronize(backend)
    copyto!(old_nodes, new_nodes)
    return nothing
end
