# TBW

@kernel function _vegas_stencil_kernel!(output_buffer::AbstractVecOrMat, @Const(bins_buffer::AbstractVecOrMat), @Const(sums::AbstractMatrix), @Const(alpha::Real))
    T = promote_type(eltype(bins_buffer), typeof(alpha))
    bin_idx, dim_idx = @index(Global, NTuple)

    # smoothing
    @inbounds smoothed = if bin_idx == firstindex(bins_buffer, 1)
        (T(7) * bins_buffer[bin_idx, dim_idx] + bins_buffer[nextind(bins_buffer, bin_idx), dim_idx]) / T(8)
    elseif bin_idx == lastindex(bins_buffer, 1)
        (T(7) * bins_buffer[bin_idx, dim_idx] + bins_buffer[prevind(bins_buffer, bin_idx), dim_idx]) / T(8)
    else
        (T(6) * bins_buffer[bin_idx, dim_idx] + bins_buffer[prevind(bins_buffer, bin_idx), dim_idx] + bins_buffer[nextind(bins_buffer, bin_idx), dim_idx]) / T(8)
    end

    # normalization
    normalized = smoothed / @inbounds sums[begin, dim_idx]

    # compression
    compressed = ((one(T) - normalized) / (log(inv(normalized))))^alpha

    @inbounds output_buffer[bin_idx, dim_idx] = compressed
end

function stencil_vegas!(backend, bins_buffer::AbstractVecOrMat, alpha::Real)
    if typeof(get_backend(bins_buffer)) != typeof(backend)
        throw(ArgumentError("buffer does not belong to the passed backend"))
    end

    bins = size(bins_buffer, 1)
    dims = size(bins_buffer, 2)
    if bins < 2
        throw(ArgumentError("less than two bins specified"))
    end

    sums = allocate(backend, eltype(bins_buffer), (1, dims))
    output_buffer = allocate(backend, eltype(bins_buffer), size(bins_buffer))

    sum!(sums, bins_buffer)                 # uses GPU implementation
    _vegas_stencil_kernel!(backend)(output_buffer, bins_buffer, sums, alpha, ndrange = (bins, dims))
    copyto!(bins_buffer, output_buffer)     # uses GPU implementation
    return nothing
end

function scan_vegas!(backend, bins_buffer::AbstractVecOrMat)
    # write the scanning code here
    # bins_buffer is both input and output, override it with the result

    # caluclate this value too, take care it has the right element type
    T = eltype(bins_buffer)
    avg_d = T(0.0)
    return avg_d
end

function refine_vegas!(backend, grid::VegasGrid, bins_buffer::AbstractVecOrMat, avg_d::Real)
    # write the refining code here
    # grid is both input and output, override it with the result

    return nothing
end
