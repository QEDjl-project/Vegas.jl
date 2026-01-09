"""
    VegasBatchBuffer{T, N, D, V, W, J}

Buffer object for Vegas on the GPU.

Template parameters:
- `T`: Basic data type used throughout, e.g. `Float32`
- `N`: Size of the buffer, "batch_size", for example `1024`
- `D`: Dimensionality of the samples, for example `3`
- `V`: The type of the values (samples), backend specific vector or matrix type, size according to previously defined N and D
- `W`: The type of the weights, backend specific vector type, size according to previously defined N
- `J`: The type of the jacobians, backend specific vector type, size according to previously defined N

Members:
- `values::V`: The sampled values
- `target_weights::W`: The calculated weights of the samples
- `jacobians::J`: The calculated jacobians of the samples

Allocate this through `allocate_vegas_batch`.

`eltype()`, `length()`, `size()`, and `get_backend()` are statically defined on this buffer.
"""
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
KernelAbstractions.get_backend(buf::VegasBatchBuffer) = get_backend(buf.values)

"""
    VegasOutBuffer{T, V}

The GPU buffer for some statistical values used during training.

Templates:
- `T`: The underlying element type, for example `Float32`
- `V`: The actual backend aware vector type

Members:
- `weighted_mean::V`: The weighted mean, only one value so the vector has length 1
- `variance::V`: The variance, only one value so the vector has length 1
- `chi_squared::V`: The chi squared statistic, only one value so the vector has length 1

`eltype()` is statically defined on this type.

Allocate using `_allocate_vegas_output`.
"""
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

Base.eltype(buf::VegasOutBuffer{T}) where {T} = T
