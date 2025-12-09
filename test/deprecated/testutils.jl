using Distributions

_scalarize(x) = x
_scalarize(x::AbstractVector) = length(x) == 1 ? x[1] : x

"""
Transform scalar data from (0,1) to (l,r).
"""
function _transform_unit_data(data, l, r)
    return (r - l) * data + l
end

"""
Build random grid with boundaries low and up, dimension d and number of nodes n
"""
function _rand_grid(rng, n, d, low, up)

    if d == 1
        nodes = rand(rng, Uniform(low, up), n)
        nodes[1] = low
        nodes[end] = up
        return nodes
    end

    unit_nodes = rand(rng, n, d)
    for i in 1:d
        unit_nodes[:, i] .= _transform_unit_data.(unit_nodes[:, i], low[i], up[i])
        unit_nodes[1, i] = low[i]
        unit_nodes[end, i] = up[i]
    end
    return sort(unit_nodes, dims = 1)
end

function _all_approx(vec, kwargs...)
    el1 = vec[1]
    for i in eachindex(vec)
        isapprox(el1, vec[i], kwargs...) || return false
    end
    return true
end


###
# Old implementation
###
_get_index(y::Real, Ng) = floor(Int64, y * Ng) + 1
_get_index(y::AbstractVector, Ng) = @. _get_index(y, Ng)
function _get_index(y::AbstractMatrix, Ng)
    idx = Matrix{Int64}(undef, size(y)...)
    for dim in 1:size(y, 1)
        idx[dim, :] = _get_index(view(y, dim, :), Ng)
    end
    return idx
end

function _subinterval_avg(jac_x_f, y, Ng, Nev, alpha; kwargs...)
    idx = _get_index(y, Ng)
    return _subinterval_avg_from_idx(jac_x_f, idx, Ng, Nev, alpha; kwargs...)
end

function _subinterval_avg_from_idx(
        jac_x_f,
        idx,
        Ng,
        Nev,
        alpha;
        smooth = false,
        compress = false,
    )
    ni = Nev / Ng
    d = zeros(Ng)
    for i in 1:Nev
        d[idx[i]] += jac_x_f[i]^2
    end
    d ./= ni

    dreg = similar(d)

    if smooth
        sumd = sum(d)
        dreg[1] = (7d[1] + d[2]) / 8
        for j in 2:(Ng - 1)
            dreg[j] = (d[j - 1] + 6d[j] + d[j + 1]) / 8
        end
        dreg[end] = (d[end - 1] + 7d[end]) / 8
        dreg[:] ./= sumd
    else
        dreg = d
    end

    if compress
        dreg[:] = @. ((1 - dreg) / log(1 / dreg))^alpha
    end

    return dreg
end

function _refine_grid(vg::VegasGrid, sub_avg)
    Ng = nbins(vg)
    grid = vg.nodes
    sp = collect(spacing(vg, i) for i in 1:nbins(vg))
    return _refine_grid(grid, sp, sub_avg, Ng)
end

function _refine_grid(grid::AbstractVector, spacing, sub_avg, Ng)

    newx = copy(grid)
    i = 1
    j = 1
    Sd = zero(eltype(sub_avg))
    delta_d = sum(sub_avg) / Ng
    i += 1
    while i < (Ng + 1)
        while Sd < delta_d
            Sd += sub_avg[j]
            j += 1
        end
        Sd -= delta_d
        newx[i] = grid[j] - ((Sd * spacing[j - 1]) / sub_avg[j - 1])
        i += 1
    end
    return newx
end
