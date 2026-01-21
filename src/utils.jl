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
    _find_quantile(dist, q)

For a given distribution `d` and quantile `q`, find `v` such that \$quantile(dist, v) = q\$
"""
function _find_quantile(dist, q::T; approx = sqrt(eps(T))) where {T <: Number}
    lo = zero(T)
    hi = one(T)

    lo_q = quantile(dist, lo)
    hi_q = quantile(dist, hi)

    max_runs = Base.significand_bits(T) # number precision bits
    for _ in 1:max_runs
        v = lo + (hi - lo) / 2
        v_q = quantile(dist, v)

        if isapprox(v_q, q; rtol = zero(T), atol = approx)
            break
        elseif v_q > q
            hi = v
            hi_q = v_q
        else # v_q < q
            lo = v
            lo_q = v_q
        end
    end

    return lo + (hi - lo) / 2
end

"""
    _gen_grid(bins, dists::Tuple)

Sets up the D-dimensional grid with the distributions, where dists is a tuple of D distributions that implement the `quantile` function.

!!! warning
    Currently only works on CPU, the finished grid can be moved to the GPU
"""
function _gen_grid(bins::AbstractMatrix, dists::Tuple, lo::NTuple{N, T}, hi::NTuple{N, T}) where {N, T <: Number}
    (nbins, ndims) = size(bins)

    @assert ndims == length(dists) "grid dimensionality doesn't match number of given dists"
    @assert N == ndims "number of hi and lo limits don't match the distributions"
    _assert_correct_boundaries(lo, hi)

    for d in 1:ndims
        lo_q = _find_quantile(dists[d], lo[d])
        hi_q = _find_quantile(dists[d], hi[d])

        for n in 1:nbins
            q = lo_q + ((n - 1) / (nbins - 1)) * (hi_q - lo_q)
            bins[n, d] = quantile(dists[d], q)
        end
    end

    return nothing
end
