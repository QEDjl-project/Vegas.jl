struct VegasBatchBuffer{T, N, D, V, W, J}
    values::V
    target_weights::W
    jacobians::J

    function VegasBatchBuffer(values::V, target_weights::W, jacs::J) where {
            T, V <: AbstractVecOrMat{T}, W <: AbstractVector{T}, J <: AbstractVector{T},
        }
        N = length(target_weights)

        size(values, 1) == N || throw(
            ArgumentError(
                "the first dimension of the values matrix must match the length of the target weight vector"
            )
        )
        length(jacs) == N || throw(
            ArgumentError(
                "target weight vector and jacobians vector must have the same length"
            )
        )

        D = ndims(values) == 1 ? 1 : size(values, 2)
        return new{T, N, D, V, W, J}(values, target_weights, jacs)
    end
end

function _allocate_vegas_batch(backend, el_type, dim, batch_size)
    return VegasBatchBuffer(
        allocate(backend, el_type, (batch_size, dim)), # values
        allocate(backend, el_type, (batch_size,)), # weights
        allocate(backend, el_type, (batch_size,)), # jacobians
    )
end

function allocate_vegas_batch(
        backend::KernelAbstractions.Backend,
        el_type::Type{T},
        dim::Int,
        batch_size::Int
    ) where {T <: Number}

    dim > zero(dim) || throw(
        ArgumentError(
            "dimension must be positive"
        )
    )

    batch_size > zero(batch_size) || throw(
        ArgumentError(
            "batch_size must be positive"
        )
    )

    return _allocate_vegas_batch(backend, el_type, dim, batch_size)
end

Base.eltype(buf::VegasBatchBuffer{T}) where {T} = T
Base.length(buf::VegasBatchBuffer{T, N}) where {T, N} = N
Base.size(buf::VegasBatchBuffer{T, N, D}) where {T, N, D} = (N, D)

struct VegasOutBuffer{T, V}
    weighted_mean::V
    variance::V
    chi_square::V

    function VegasOutBuffer(wmean::V, var::V, chisq::V) where {T, V <: AbstractVector{T}}
        return new{T, V}(wmean, var, chisq)
    end
end

function _allocate_vegas_output(backend, dtype)
    return VegasOutBuffer(
        allocate(backend, dtype, (1,)),  # weighted mean
        allocate(backend, dtype, (1,)),  # variance
        allocate(backend, dtype, (1,)),  # chi square
    )
end
