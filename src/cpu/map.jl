# TODO:
# - consider different output: Vector{NTuple{ndims,Float64}} would be faster for
# construction of PSPs and differential cross sections. Maybe the NTuple could be its own
# type, e.g. CoordinatePoint() or something like that.
# - See DEV-old-vegas for details
#
# the vegas map

@inline function _vegas_map(vg::VegasGrid{1}, y::Real)
    Ng = nbins(vg)

    isone(y) ? (return nodes(vg, Ng + 1)) : nothing
    yn = y * Ng
    idx = floor(Int64, yn) + 1
    d = yn + 1 - idx

    return nodes(vg, idx) + spacing(vg, idx) * d
end

@inline function _vegas_map!(vg::VegasGrid{1}, dest::AbstractVector, y::AbstractVector)

    # iterates over points
    for i in eachindex(y)
        dest[i] = _vegas_map(vg, y[i])
    end
    return dest
end

@inline function _vegas_map(vg::VegasGrid{1}, y::AbstractVector)
    return _vegas_map!(vg, similar(y), y)
end

@inline function _vegas_map(vg::VegasGrid{N}, y_vec::AbstractVector) where {N}
    Ng = nbins(vg)

    # iterates over ndims
    res = ntuple(ndims(vg)) do i
        y = @inbounds y_vec[i]
        yn = y * Ng
        idx = floor(Int64, yn) + 1
        d = yn + 1 - idx
        nodes(vg, idx, i) + spacing(vg, idx, i) * d
    end
    return res
end

@inline function _vegas_map(vg::VegasGrid{N}, ymat::AbstractMatrix{T}) where {N, T <: Real}
    return _vegas_map.(Ref(vg), eachrow(ymat))
end

# TODO: public interface
# - consider implementing some input validation and user interface for export
# - consider implementing a sampler based in the VegasGrid (however, maybe its better to
# stay based on the VegasProposal)


# Jacobian of the vegas map

@inline function _jac_vegas_map(vg::VegasGrid{1}, y::Real)
    Ng = nbins(vg)
    idx = floor(Int64, y * Ng) + 1

    return Ng * spacing(vg, idx)
end

function _jac_vegas_map!(vg::VegasGrid{1}, dest::AbstractVector, y::AbstractVector)

    # iterates over points
    for i in eachindex(y)
        dest[i] = _jac_vegas_map(vg, y[i])
    end
    return dest
end
@inline function _jac_vegas_map(vg::VegasGrid{1}, y::AbstractVector{T}) where {T <: Real}
    return _jac_vegas_map!(vg, ones(T, size(y, 1)), y)
end

@inline function _jac_vegas_map(vg::VegasGrid{N}, y::AbstractMatrix{T}) where {N, T <: Real}
    return _jac_vegas_map!(vg, ones(T, size(y, 1)), y)
end

function _jac_vegas_map(vg::VegasGrid{N}, yvec::AbstractVector{T}) where {N, T <: Real}
    Ng = nbins(vg)

    res = one(T)
    # iterates over ndims
    for d in eachindex(yvec)
        idx = floor(Int64, yvec[d] * Ng) + 1

        res *= Ng * spacing(vg, idx, d)

    end

    return res
end

function _jac_vegas_map!(
        vg::VegasGrid{N},
        dest::AbstractVector{T},
        ymat::AbstractMatrix,
    ) where {N, T <: Real}
    Ng = nbins(vg)

    # iterates over (points,ndims)
    for i in CartesianIndices(ymat)
        idx = floor(Int64, ymat[i] * Ng) + 1

        dest[i[1]] *= Ng * spacing(vg, idx, i[2])
    end

    # TODO: Think about returning the prod of jacs along the ndims axis
    return dest
end

# TODO: public interface
# - consider implementing some input validation and user interface for export
