function _assert_correct_boundaries(::Tuple{}, ::Tuple{}) end

function _assert_correct_boundaries(
        low::Tuple{Vararg{T, N}},
        up::Tuple{Vararg{T, N}},
    ) where {T <: Real, N}
    first(low) <= first(up) || throw(
        ArgumentError(
            "lower boundary need to be smaller or equal to the respective upper boundary",
        ),
    )
    return _assert_correct_boundaries(low[2:end], up[2:end])
end

"""
    _gen_grid(bins, dists::Tuple)

Sets up the D-dimensional grid with the distributions, where dists is a tuple of D distributions that implement the `quantile` and `cdf` functions.

!!! warning
    Currently only works on CPU, the finished grid can be moved to the GPU
"""
function _gen_grid(bins::AbstractMatrix, dists::Tuple, lo::NTuple{N, T}, hi::NTuple{N, T}) where {N, T <: Number}
    (nbins, ndims) = size(bins)

    @assert ndims == length(dists) "grid dimensionality doesn't match number of given dists"
    @assert N == ndims "number of hi and lo limits don't match the distributions"
    _assert_correct_boundaries(lo, hi)

    for d in 1:ndims
        lo_q = cdf(dists[d], lo[d])
        hi_q = cdf(dists[d], hi[d])

        for n in 1:nbins
            # need to clamp to 1 (or figure out better precision for this at some point)
            q = n == nbins ? one(T) : lo_q + ((n - 1) / (nbins - 1)) * (hi_q - lo_q)
            bins[n, d] = quantile(dists[d], q)
        end
    end

    return nothing
end
