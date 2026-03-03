import AcceleratedKernels as AK

@kernel function _get_last_col!(lastvals, A, N::Int32, D::Int32)
    d = @index(Global)
    if d <= D
        @inbounds lastvals[d] = A[N, d]
    end
end

@kernel function _scale_by_invN!(out, lastvals, invN, D::Int32)
    d = @index(Global)
    if d <= D
        @inbounds out[d] = lastvals[d] * invN
    end
end

function scan_vegas!(backend, bins_buffer::AbstractVecOrMat)
    T = eltype(bins_buffer)

    # Handle the vector case
    A = bins_buffer isa AbstractVector ? reshape(bins_buffer, :, 1) : bins_buffer

    N = Int32(size(A, 1))
    D = Int32(size(A, 2))
    # 1) Inclusive scan per column (in-place) using AcceleratedKernels
    #    dims=1 means scanning along the first dimension (column-wise)
    AK.accumulate!(+, A, A; dims=1, init=zero(T))

    # 2) Get the last value of each column
    lastvals = similar(A, T, (Int(D),))
    kernel_last = _get_last_col!(backend)
    kernel_last(lastvals, A, N, D; ndrange=Int(D))
    KernelAbstractions.synchronize(backend)

    # 3) Compute avg_d = lastvals / N
    avg_d = similar(A, T, (Int(D),))
    invN = T(1) / T(N)
    kernel_scale = _scale_by_invN!(backend)
    kernel_scale(avg_d, lastvals, invN, D; ndrange=Int(D))
    KernelAbstractions.synchronize(backend)

    return avg_d
end
