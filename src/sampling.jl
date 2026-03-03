@kernel function vegas_sampling_kernel!(
        values::AbstractMatrix{T},
        target_weights::AbstractVector{T},
        jacobians::AbstractVector{T},
        grid_lines::AbstractMatrix{T},
        func::Function,
        @Const(Ng),
        d::Val{D}
    ) where {T <: Number, D}
    i::Int32 = @index(Global)

    jac = one(T)

    for d in one(Int32):D
        # generate random yi and yd
        yn = (rand(T) % one(Int32)) * (Ng - one(Int32))
        yi = unsafe_trunc(Int, yn) + one(Int32)
        yd = yn + one(Int32) - yi

        x_start = grid_lines[yi, d]
        x_end = grid_lines[yi + one(Int32), d]
        width = x_end - x_start

        # transform sample value to grid
        values[i, d] = x_start + width * yd

        # calculate jacobian
        jac *= Ng * width
    end

    jacobians[i] = jac

    # manually unroll ntuple from Val{D} to make sure everything is type stable
    V = ntuple(d -> (@inbounds values[i, Int32(d)]), Val(D))
    target_weights[i] = jac * func(V)
end

function sample_vegas!(
        backend,
        buffer::VegasBatchBuffer{T, N, D, V, W, J},
        grid::VegasGrid{T, Ng, D, G},
        func::Function
    ) where {T <: AbstractFloat, N, D, V, W, J, Ng, G}
    # buffer.values = N x D
    # grid.target_weights = N
    # grid.jacobians = N
    # grid.nodes = Ng x D

    @assert typeof(get_backend(buffer)) == typeof(get_backend(grid)) == typeof(backend)
    @assert size(buffer.values, 2) == ndims(grid)
    @assert prod(size(buffer.values)) == N * D

    # TODO: currently, oneAPI does not support device-side RNG
    @debug "Calling sampling kernel with $(Ng - 1) bins, $(D) dims and $(N) samples = $N threads"
    vegas_sampling_kernel!(backend)(
        buffer.values,
        buffer.target_weights,
        buffer.jacobians,
        grid.nodes,
        func,
        Ng,
        Val(D),
        ndrange = N
    )

    synchronize(backend)

    return nothing
end
