# TODO:
# - think of an user interface with options for smoothing and compression


# effective bin average value of J x F ^2

function _eff_bin_avg!(
        vg::VegasGrid{1},
        dest::AbstractVector,
        yvec::AbstractVector,
        fxj2::AbstractVector,
    )

    Ng = nbins(vg)
    Neval = length(yvec)
    ni_avg = Neval / Ng

    for i in eachindex(yvec)
        idx = floor(Int64, yvec[i] * Ng) + 1
        dest[idx] += fxj2[i] / ni_avg
    end

    return dest
end


function _eff_bin_avg(vg::VegasGrid{1}, yvec::AbstractVector, fxj2::AbstractVector)
    return _eff_bin_avg!(vg, zeros(nbins(vg)), yvec, fxj2)
end

function _eff_bin_avg!(
        vg::VegasGrid{N},
        dest::AbstractMatrix,
        ymat::AbstractMatrix,
        fxj2::AbstractVector,
    ) where {N}
    Ng = nbins(vg)
    Neval = size(ymat, 1)
    ni_avg = Neval / Ng

    for i in CartesianIndices(ymat)
        idx = floor(Int64, ymat[i] * Ng) + 1
        dest[idx, i[2]] += fxj2[i[1]] / ni_avg
    end
    return dest
end

function _eff_bin_avg(
        vg::VegasGrid{N},
        ymat::AbstractMatrix,
        fxj2::AbstractVector,
    ) where {N}
    return _eff_bin_avg!(vg, zeros(nbins(vg), ndims(vg)), ymat, fxj2)
end

# smoothing of the bin average

function _smooth_bin_avg(d::AbstractVector)
    s = sum(d)
    Ng = length(d)
    res_d = similar(d)

    res_d[1] = (7 * d[1] + d[2]) / (8 * s)
    res_d[end] = (d[end - 1] + 7 * d[end]) / (8 * s)

    for i in 2:(Ng - 1)
        res_d[i] = (d[i - 1] + 6 * d[i] + d[i + 1]) / (8 * s)
    end

    return res_d
end

function _smooth_bin_avg(d::AbstractMatrix)
    s = sum(d, dims = 1)
    Ng, D = size(d)
    res_d = similar(d)

    for j in 1:D
        res_d[1, j] = (7 * d[1, j] + d[2, j]) / (8 * s[j])

        res_d[end, j] = (d[end - 1, j] + 7 * d[end, j]) / (8 * s[j])

        for i in 2:(Ng - 1)
            res_d[i, j] = (d[i - 1, j] + 6 * d[i, j] + d[i + 1, j]) / (8 * s[j])
        end
    end

    return res_d
end

# compression of the bin average

function _compress_bin_avg!(d::AbstractVector, alpha::Real)
    map!(x -> ((1 - x) / log(1 / x))^alpha, d, d)
    return d
end

function _compress_bin_avg(d::AbstractVecOrMat, alpha::Real)
    return _compress_bin_avg!(copy(d), alpha)
end

function _compress_bin_avg!(d::AbstractMatrix, alpha::Real)
    for i in eachindex(d)
        d[i] = ((1 - d[i]) / log(1 / d[i]))^alpha
    end
    return d
end

# refinement of the grid based on bin average

function _refine_nodes!(vg::VegasGrid{1}, bin_avg::AbstractVector{T}) where {T <: Real}
    Ng = nbins(vg)
    new_grid = copy(vg.nodes)
    j = 1
    s = zero(T)
    delta = sum(bin_avg) / Ng
    for i in 2:Ng
        while s < delta
            s += bin_avg[j]
            j += 1
        end
        s -= delta
        new_grid[i] = nodes(vg, j) - s / bin_avg[j - 1] * spacing(vg, j - 1)
    end
    vg.nodes[:] .= new_grid[:]
    return vg
end

function _refine_nodes!(vg::VegasGrid{N}, bin_avg::AbstractMatrix{T}) where {N, T <: Real}
    Ng = nbins(vg)
    delta = sum(bin_avg, dims = 1) ./ Ng
    new_grid = copy(vg.nodes)

    # iterate over ndimss
    for d in 1:N
        j = 1
        s = zero(T)
        # iterate over bins
        for i in 2:Ng

            while s < delta[d]
                s += bin_avg[j, d]
                j += 1
            end
            s -= delta[d]
            new_grid[i, d] = nodes(vg, j, d) - s / bin_avg[j - 1, d] * spacing(vg, j - 1, d)
        end
    end
    vg.nodes[:] .= new_grid[:]
    return vg
end
