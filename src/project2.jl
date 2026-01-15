# TBW

@kernel function _vegas_stencil_kernel!(@Const(bins_buffer::AbstractVecOrMat), @Const(sums::AbstractMatrix), @Const(alpha::Real), output_buffer::AbstractVecOrMat)
    bins, _ = @ndrange()
    bin_idx, dim_idx = @index(Global, NTuple)

    # smoothing
    @inbounds smoothed = if bin_idx == 1
        (7 * bins_buffer[1, dim_idx] + bins_buffer[2, dim_idx]) / 8
    elseif bin_idx == bins
        (bins_buffer[bins - 1, dim_idx] + 7 * bins_buffer[bins, dim_idx]) / 8
    else
        (bins_buffer[bin_idx - 1, dim_idx] + 6 * bins_buffer[bin_idx, dim_idx] + bins_buffer[bin_idx + 1, dim_idx]) / 8
    end

    # normalization
    normalized = smoothed / @inbounds sums[1, dim_idx]

    # compression
    compressed = ((1 - normalized) / (log(1 / normalized)))^alpha

    @inbounds output_buffer[bin_idx, dim_idx] = compressed
end

function stencil_vegas!(backend, bins_buffer::AbstractVecOrMat, alpha::Real)
    if get_backend(bins_buffer) != backend
        throw(ArgumentError("buffer is not associated with passed backend"))
    end

    bins = size(bins_buffer, 1)
    dims = size(bins_buffer, 2)
    if bins < 2
        throw(ArgumentError("less than two bins specified"))
    end

    sums = allocate(backend, eltype(bins_buffer), (1, dims))
    output_buffer = allocate(backend, eltype(bins_buffer), size(bins_buffer))

    sum!(sums, bins_buffer)                 # should be specialized to run on the GPU
    _vegas_stencil_kernel!(backend)(bins_buffer, sums, alpha, output_buffer, ndrange = (bins, dims))
    copyto!(bins_buffer, output_buffer)     # should be specialized to run on the GPU
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
