struct VegasBatchBuffer{Ts, Tw, V, C, J, F}
    vegas_samples::V
    target_samples::C
    jacobians::J
    func_values::F

    function VegasBatchBuffer(vsamples::V, tsamples::C, jacs::J, fvalues::F) where {
            Ts, Tw, V <: AbstractVector{Ts}, C <: AbstractVector{Ts}, J <: AbstractVector{Tw}, F <: AbstractVector{Tw},
        }

        return new{Ts, Tw, V, C, J, F}(vsamples, tsamples, jacs, fvalues)
    end
end

function _allocate_vegas_batch(backend, sample_type, weight_type, batch_size)
    return VegasBatchBuffer(
        allocate(backend, sample_type, (batch_size,)), # vsamples
        allocate(backend, sample_type, (batch_size,)), # tsamples
        allocate(backend, weight_type, (batch_size,)), # jacobians
        allocate(backend, weight_type, (batch_size,))  # fvalues
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
