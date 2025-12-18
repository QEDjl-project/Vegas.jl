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
