using Atomix: @atomic

@kernel function vegas_binning_kernel_batched2!(
        bins_buffer::AbstractMatrix{T}, # out
        ndi_buffer::AbstractMatrix{I},  # out
        values::AbstractMatrix{T},      # in
        grid_lines::AbstractMatrix{T},  # in
        func::Function,
        @Const(batch_size::Int32),
        @Const(Ng),
        ::Val{D}
    ) where {T <: Number, I <: Integer, D}
    # block dim 1 -> bin number including dimension
    # block dim 2 -> batch number

    block_dim_1 = @groupsize()
    (no_bins, _) = size(bins_buffer)
    n_vals = size(values, 1)

    # step 1 get indices ->
    combined_bin_idx = @index(Global, Ntuple)[1]
    # local_D = dimension the bin cares about
    local_D = (combined_bin_idx - one(Int32)) ÷ no_bins + one(Int32)
    # local_S = bin number the bin cares about
    local_S = (combined_bin_idx - one(Int32)) % no_bins + one(Int32)

    # range of samples the thread (and the whole block) cares about -> block idx * batch_size to block idx + 1 * batch_size - 1
    batch_idx_lo = (@index(Global, NTuple)[2] - one(Int32)) * batch_size + one(Int32)
    batch_idx_hi = batch_idx_lo + batch_size - one(Int32)
    batch_idx_hi = min(n_vals, batch_idx_hi)

    # step 2
    # load local values
    # bin_lo, bin_hi -> for this thread local bin/dimension, store the bin limits we care about
    # TODO: limits?
    bin_lo = @inbounds grid_lines[local_S, local_D]
    bin_hi = @inbounds grid_lines[local_S + 1, local_D]

    local_sum = zero(T)
    local_ndi = zero(I)

    # step 3
    # scan the values for samples falling into this bin
    for i in batch_idx_lo:batch_idx_hi
        sample = @inbounds values[i, local_D]
        V = ntuple(d -> (@inbounds values[sample, Int32(d)]), Val(D))
        local_sum += func(V)^2
        local_ndi += one(Int32)
    end

    # step 4
    # write back
    if !iszero(local_ndi)
        # jacobian is the same for all samples in this bin/dim, so no need to sum it up
        jac = Ng * (bin_hi - bin_lo)
        @atomic bins_buffer[bin, dim] += (jac^2) / local_ndi * bin_sum / threads_per_bin

        # used for sanity check that all samples are binned and none is lost
        @atomic ndi_buffer[bin, dim] += local_ndi
    end
end

@kernel function vegas_binning_kernel_batched!(
        bins_buffer::AbstractMatrix{T},
        ndi_buffer::AbstractMatrix{I},
        values::AbstractMatrix{T},
        grid_lines::AbstractMatrix{T},
        func::Function,
        @Const(Ng),
        ::Val{D},
        ::Val{batch_size}
    ) where {T <: Number, I <: Integer, D, batch_size}
    bin::Int32, dim::Int32, batch::Int32 = @index(Global, NTuple)
    batch -= one(Int32)

    bin_sum = zero(T)
    ndi = zero(I)

    lower_bound = grid_lines[bin, dim]
    upper_bound = grid_lines[bin + one(Int32), dim]

    threads_per_bin = @ndrange()[Int32(3)]
    batch_start = batch * batch_size + one(Int32)
    batch_end = (batch + one(Int32)) * batch_size

    is_last_bin = bin == Ng - one(Int32)

    for sample in batch_start:batch_end
        # TODO: with high sample counts some samples fall _on_ the last grid line, despite random sampling excludes 1
        if lower_bound <= values[sample, dim] < upper_bound || (is_last_bin && values[sample, dim] == upper_bound)
            V = ntuple(d -> (@inbounds values[sample, Int32(d)]), Val(D))
            bin_sum += func(V)^2
            ndi += one(Int32)
        end
    end

    # could happen that for our batch no samples fall into the bin
    if !iszero(ndi)
        # jacobian is the same for all samples in this bin/dim, so no need to sum it up
        jac = Ng * (grid_lines[bin + one(Int32), dim] - grid_lines[bin, dim])
        @atomic bins_buffer[bin, dim] += (jac^2) / ndi * bin_sum / threads_per_bin

        # used for sanity check that all samples are binned and none is lost
        @atomic ndi_buffer[bin, dim] += ndi
    end
end

@kernel function vegas_binning_kernel!(
        bins_buffer::AbstractMatrix{T},
        ndi_buffer::AbstractMatrix{I},
        values::AbstractMatrix{T},
        grid_lines::AbstractMatrix{T},
        func::Function,
        @Const(Ng),
        ::Val{D}
    ) where {T <: Number, I <: Integer, D}
    nbins::Int32 = Ng - one(Int32)

    # this is why we cant have nice things
    i::Int32 = @index(Global) - one(Int32)
    bin::Int32 = (i % nbins) + one(Int32)
    dim::Int32 = div(i, nbins) + one(Int32)

    ndi = zero(I)
    bins_buffer[bin, dim] = zero(T)

    lower_bound = grid_lines[bin, dim]
    upper_bound = grid_lines[bin + one(Int32), dim]

    is_last_bin = bin == Ng - one(Int32)

    for sample in one(Int32):size(values, one(Int32))
        # TODO: with high sample counts some samples fall _on_ the last grid line, despite random sampling excludes 1
        if lower_bound <= values[sample, dim] < upper_bound || (is_last_bin && values[sample, dim] == upper_bound)
            ndi += one(Int32)
            V = ntuple(d -> (@inbounds values[sample, Int32(d)]), Val(D))
            bins_buffer[bin, dim] += func(V)^2
        end
    end

    # jacobian is the same for all samples in this bin/dim, so no need to sum it up
    jac = Ng * (grid_lines[bin + one(Int32), dim] - grid_lines[bin, dim])

    bins_buffer[bin, dim] *= (jac^2) / ndi

    # used for sanity check that all samples are binned and none is lost
    ndi_buffer[bin, dim] = ndi
end

function binning_vegas!(
        backend,
        bins_buffer::AbstractMatrix{T},
        ndi_buffer::AbstractMatrix{I},
        buffer::VegasBatchBuffer{T, N, D, V, W, J},
        grid::VegasGrid{T, Ng, D, G},
        func::Function,
        threads_per_bin::Int = 1024
    ) where {T <: AbstractFloat, I <: Integer, N, D, V, W, J, Ng, G}
    # bins_buffer = Ng-1 x D
    # buffer.values = N x D
    # grid.jacobians = N
    # grid.nodes = Ng x D

    nbins = Ng - 1

    # NOTE: use unbatched implementation in case GPU does not support @atomic for the selected float
    if string(typeof(backend)) == "MetalBackend"
        @debug "Calling unbatched binning kernel with $(nbins) bins * $(D) dims = $(nbins * D) threads"
        vegas_binning_kernel!(backend)(
            bins_buffer,
            ndi_buffer,
            buffer.values,
            grid.nodes,
            func,
            Ng,
            Val(Int32(D)),
            ndrange = Int32(nbins) * Int32(D)
        )
    else
        # need to be zeroed because every thread just adds its calculation
        fill!(bins_buffer, zero(T))
        fill!(ndi_buffer, zero(T))

        @debug "Calling batched binning kernel with $(nbins) bins * $(D) dims * $(threads_per_bin) threads/bin = $(nbins * D * threads_per_bin) threads"

        els_per_thread = Int32(256)
        nblocks = ceil(Int32, N / els_per_thread)

        vegas_binning_kernel_batched!(backend, (256, 1, 1))(
            bins_buffer,
            ndi_buffer,
            buffer.values,
            grid.nodes,
            func,
            Ng,
            Val(D),
            Val(els_per_thread),
            ndrange = (Int32(nbins), Int32(D), nblocks)
        )
    end

    synchronize(backend)

    return nothing
end
