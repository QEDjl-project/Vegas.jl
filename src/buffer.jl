struct VegasBatchBuffer{T, D, V, W, J}
    values::V
    target_weights::W
    jacobians::J

    function VegasBatchBuffer(values::V, target_weights::W, jacs::J) where {
            T, V <: AbstractVecOrMat{T}, W <: AbstractVector{T}, J <: AbstractVector{T},
        }
        D = ndims(values) == 1 ? 1 : size(values, 2)
        return new{T, D, V, W, J}(values, target_weights, jacs)
    end
end

function _allocate_vegas_batch(backend, el_type, dof, batch_size)
    return VegasBatchBuffer(
        allocate(backend, el_type, (batch_size, dof)), # values
        allocate(backend, el_type, (batch_size,)), # weights
        allocate(backend, el_type, (batch_size,)), # jacobians
    )
end

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
