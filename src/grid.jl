abstract type AbstractVegasGrid end

struct VegasGrid{T, N, D, G} <: AbstractVegasGrid
    nodes::G

    function VegasGrid(nodes::G) where {T <: Real, G <: AbstractMatrix{T}}
        N, D = size(nodes)
        return new{T, N, D, G}(nodes)
    end
end

KernelAbstractions.get_backend(grid::VegasGrid) = get_backend(grid.nodes)
Base.eltype(::VegasGrid{T}) where {T} = T
nbins(::VegasGrid{T, N}) where {T, N} = N - one(N)

# TODO: discuss this semantically
Base.ndims(::VegasGrid{T, N, D}) where {T, N, D} = D

function _allocate_vegas_grid(backend, el_type, nbins, dim)
    return VegasGrid(
        allocate(backend, el_type, (nbins + 1, dim))
    )
end

function allocate_vegas_grid(
        backend::KernelAbstractions.Backend,
        el_type::Type{T},
        nbins::Int,
        dim::Int
    ) where {T <: Real}

    nbins > zero(nbins) || throw(
        ArgumentError(
            "nbins must be positive"
        )
    )

    dim > zero(dim) || throw(
        ArgumentError(
            "dimension must be positive"
        )
    )

    return _allocate_vegas_grid(backend, el_type, nbins, dim)
end

# build equidistant grid nodes

@kernel function _fill_uniformly_kernel(grid_nodes, @Const(lower), @Const(upper))

    # TODO: check if this is statically determined
    nbins = size(grid_nodes, 1)

    bin_idx, dim_idx = @index(Global, NTuple)

    # TODO: check for float precision
    grid_nodes[bin_idx, dim_idx] = lower[dim_idx] + (bin_idx - one(bin_idx)) * (upper[dim_idx] - lower[dim_idx]) / (nbins - one(bin_idx))

end

function _fill_uniformly!(grid::VegasGrid, lower::NTuple{N, T}, upper::NTuple{N, T}) where {N, T <: Real}

    # TODO: use different blocksizes for CPU or GPU!
    _fill_uniformly_kernel(get_backend(grid), 32)(grid.nodes, lower, upper, ndrange = size(grid.nodes))

    return nothing
end

function fill_uniformly!(grid::VegasGrid, lower::NTuple{N, T}, upper::NTuple{N, T}) where {N, T <: Real}
    _assert_correct_boundaries(lower, upper)
    _fill_uniformly!(grid, lower, upper)

    return nothing
end

# out-of-place version
function uniform_vegas_grid(
        backend::KernelAbstractions.Backend,
        lower::NTuple{N, T},
        upper::NTuple{N, T},
        nbins::Int,
    ) where {
        N, T <: Real,
    }

    # TODO: check if T and backend are compatible
    grid = allocate_vegas_grid(backend, T, nbins, N)
    fill_uniformly!(grid, lower, upper)
    return grid
end
