"""
    AbstractTargetDistribution

Abstract base type for target distributions used in sampling.

Concrete subtypes represent functions defined on an `N`-dimensional real
domain and must implement the following interface:

# Required Methods
- `_compute(dist, x::NTuple{N,<:Real})::Real`
  Evaluate the (possibly unnormalized) target distribution at the point `x`.

- `degrees_of_freedom(dist)::Int`
  Return the number of degrees of freedom, i.e. the dimensionality of the
  domain on which the distribution is defined.

# Notes
- Implementations are expected to be side-effect free and compatible with
  GPU execution.
- Broadcasting over an `AbstractTargetDistribution` treats the distribution
  as a scalar.

# See also
[`_compute`](@ref), [`degrees_of_freedom`](@ref)
"""
abstract type AbstractTargetDistribution end
Base.broadcastable(dist::AbstractTargetDistribution) = Ref(dist)

"""

    _compute(dist::AbstractTargetDistribution,x::NTuple{N,<:Real})::Real

Interface function: Return value of `dist` computed at `x`.
"""
function _compute end

"""

    degrees_of_freedom(dist::AbstractTargetDistribution)::Int

Interface function: return the degrees of freedom of `dist`, e.g. the dimension of the domain.
"""
function degrees_of_freedom end

@kernel inbounds = true function _compute_kernel(dest, @Const(dist), ::Val{dim}, @Const(xmat)) where {dim}
    I = @index(Global)
    x = ntuple(i -> xmat[I, i], Val(dim))
    @inbounds dest[I] = _compute(dist, x)
end

function _compute!(backend::KernelAbstractions.Backend, buf::VegasBatchBuffer, dist::AbstractTargetDistribution)
    dim = degrees_of_freedom(dist)

    return _compute_kernel(backend, 32)(
        buf.target_weights,
        dist,
        Val(dim),
        buf.values,
        ndrange = size(buf.target_weights),
    )
end
