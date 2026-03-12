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
