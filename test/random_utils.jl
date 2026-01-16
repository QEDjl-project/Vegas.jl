# definition of a simple E = 0, σ = 1 normal distribution
normal_distribution(u::T) where {T <: Number} = inv(sqrt(T(2) * π)) * exp(-(u^2 / T(2)))

# can be used as a target function
normal_distribution(u::NTuple{N, T}) where {N, T <: Number} = prod(normal_distribution.(u))
