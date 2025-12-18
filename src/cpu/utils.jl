# infer nodes from spacing

function _nodes_from_spacing(stepsizes::VecOrMat, init::VecOrMat)
    return vcat(init', reshape(init, 1, :) .+ accumulate(+, stepsizes, dims = 1))
end

function _nodes_from_spacing(stepsizes::VecOrMat, init::Real)
    return _nodes_from_spacing(stepsizes, fill(init, size(stepsizes, 2)))
end

# tot_cs, cum_std_dev, chi_sq
@inline _vp_init_values(::Type{T}) where {T <: Real} = ntuple(i -> zero(T), 3)
