import AcceleratedKernels as AK

@kernel function _vegas_stencil_kernel!(output_buffer::AbstractVecOrMat, @Const(bins_buffer::AbstractVecOrMat), @Const(sums::AbstractMatrix), @Const(alpha::Real))
    T = promote_type(eltype(bins_buffer), typeof(alpha))
    bin_idx, dim_idx = @index(Global, NTuple)

    # smoothing
    @inbounds smoothed = if bin_idx == firstindex(bins_buffer, 1)
        (T(7) * bins_buffer[bin_idx, dim_idx] + bins_buffer[nextind(bins_buffer, bin_idx), dim_idx]) / T(8)
    elseif bin_idx == lastindex(bins_buffer, 1)
        (T(7) * bins_buffer[bin_idx, dim_idx] + bins_buffer[prevind(bins_buffer, bin_idx), dim_idx]) / T(8)
    else
        (T(6) * bins_buffer[bin_idx, dim_idx] + bins_buffer[prevind(bins_buffer, bin_idx), dim_idx] + bins_buffer[nextind(bins_buffer, bin_idx), dim_idx]) / T(8)
    end

    # normalization
    normalized = smoothed / @inbounds sums[begin, dim_idx]

    # compression
    compressed = ((one(T) - normalized) / (log(inv(normalized))))^alpha

    @inbounds output_buffer[bin_idx, dim_idx] = compressed
end

function stencil_vegas!(backend, bins_buffer::AbstractVecOrMat, alpha::Real)
    if typeof(get_backend(bins_buffer)) != typeof(backend)
        throw(ArgumentError("buffer does not belong to the passed backend"))
    end

    bins = size(bins_buffer, 1)
    dims = size(bins_buffer, 2)
    if bins < 2
        throw(ArgumentError("less than two bins specified"))
    end

    sums = allocate(backend, eltype(bins_buffer), (1, dims))
    output_buffer = allocate(backend, eltype(bins_buffer), size(bins_buffer))

    sum!(sums, bins_buffer)                 # uses GPU implementation
    _vegas_stencil_kernel!(backend)(output_buffer, bins_buffer, sums, alpha, ndrange = (bins, dims))
    copyto!(bins_buffer, output_buffer)     # uses GPU implementation
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
    if d <= D
        new_nodes[1, d]   = old_nodes[1, d]
        new_nodes[Np1, d] = old_nodes[Np1, d]
    end
end

@kernel function _refine_nodes_bs!(
    new_nodes, old_nodes, prefix, avg_d, N::Int32, D::Int32
)
    tid = @index(Global)
    total = (N - Int32(1)) * D

    if tid <= total
        d = Int32(((tid - 1) % D) + 1)     # Dimension Index; 1..D 
        k = Int32(((tid - 1) ÷ D) + 1)     # internal node; 1..N-1
        i = k + Int32(1)                   # New node number; 2..N

        target = float(k) * avg_d

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
        j = Int(lo)

        prev = j == 1 ? zero(eltype(prefix)) : prefix[j-1, d]
        dj   = prefix[j, d] - prev
        t = (target - prev) / dj

        x0 = old_nodes[j,   d]
        x1 = old_nodes[j+1, d]
        new_nodes[i, d] = x0 + t * (x1 - x0)
    end
end

function refine_vegas!(backend, grid::VegasGrid, bins_buffer::AbstractVecOrMat, avg_d::AbstractVector)
    # write the refining code here
    # grid is both input and output, override it with the result
    @assert bins_buffer isa AbstractMatrix

    old_nodes = grid.nodes               # (N+1, D)
    new_nodes = similar(old_nodes)

    N = Int32(size(bins_buffer, 1))      # nbins
    D = Int32(size(bins_buffer, 2))      # dim
    Np1 = Int32(size(old_nodes, 1))      # N+1

    _copy_bounds_nodes!(backend)(new_nodes, old_nodes, Np1, D; ndrange=Int(D))

    total = Int((N - Int32(1)) * D)
    _refine_nodes_bs!(backend)(new_nodes, old_nodes, bins_buffer, avg_d, N, D; ndrange=total)

    KernelAbstractions.synchronize(backend)
    copyto!(old_nodes, new_nodes)
    return nothing
end
