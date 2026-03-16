using Atomix: @atomic

"""
Batched binning kernel. The grid layout is three-dimensional, the first dimension is in the number of bins, the second the dimensionality of the samples, the third is the number of batches. The given batch_size times the third block dimension must equal the number of samples.

!!! warn
    The second dimension of the block size for this *must* be D for the shared memory loading to work correctly. The third dimension *should* be 1.
"""
@kernel function vegas_binning_kernel_batched!(
        bins_buffer::AbstractMatrix{T}, # out
        ndi_buffer::AbstractMatrix{I},  # out
        values::AbstractMatrix{T},      # in
        grid_lines::AbstractMatrix{T},  # in
        func::Function,
        @Const(Ng),
        ::Val{D},
        ::Val{batch_size}
    ) where {T <: Number, I <: Integer, D, batch_size}
    # block dim 1 -> bin number
    # block dim 2 -> dim number
    # block dim 3 -> batch number

    bin, dim, batch = @index(Global, NTuple)

    bin_dim = @groupsize()[1]
    t = @index(Local, NTuple)
    local_bin = t[1]

    n_vals = size(values, 1)

    # range of samples the thread (and the whole block) cares about -> block idx * batch_size to block idx + 1 * batch_size - 1
    batch_idx_lo = (batch - one(Int32)) * batch_size + one(Int32)
    batch_idx_hi = batch_idx_lo + batch_size - one(Int32)
    batch_idx_hi = min(n_vals, batch_idx_hi)

    # set up shared memory for the sample batch
    local_sample_batch = @localmem T (batch_size, D)

    # each thread loads in its dimension
    for local_idx in local_bin:bin_dim:batch_size
        global_idx = local_idx + batch_idx_lo - one(Int32)

        # this is not pretty and not performant, but necessary when the n_vals is not a multiple of the batch_size
        if !iszero(n_vals % batch_size) && global_idx > n_vals
            # the iszero *should* be able to happen at compile time so ideally this check doesn't exist when it's not necessary
            break
        end

        local_sample_batch[local_idx, dim] = @inbounds values[global_idx, dim]
    end

    # synchronize because the target function evaluation loops differently and needs these results
    @synchronize

    t = @index(Local, NTuple)
    local_bin = t[1]
    bin_dim = @groupsize()[1]

    # shared memory for the squared target values
    local_targets_sq = @localmem T batch_size

    local_idx_offset = local_bin + (dim - one(Int32)) * bin_dim
    # each thread calculates one squared target function, indices are unfortunately a bit ugly
    for i in local_idx_offset:(bin_dim * D):batch_size
        V = ntuple(d -> (@inbounds local_sample_batch[i, Int32(d)]), Val(D))
        local_targets_sq[i] = func(V)^2
    end

    @synchronize

    # step 2
    # load local values
    # bin_lo, bin_hi -> for this thread local bin/dimension, store the bin limits we care about
    bin_lo = @inbounds grid_lines[bin, dim]
    bin_hi = @inbounds grid_lines[bin + one(Int32), dim]   # TODO: is this correct for last bin?

    local_sum = zero(T)
    local_ndi = zero(I)

    is_last_bin = bin == Ng - one(Int32)

    # step 3
    # scan the values for samples falling into this bin
    for i in 1:batch_size
        sample = @inbounds local_sample_batch[i, dim]

        if bin_lo <= sample && (sample < bin_hi || is_last_bin)
            local_sum += local_targets_sq[i]
            local_ndi += one(Int32)
        end

        if bin_lo <= sample && is_last_bin && sample >= bin_hi
            @show (bin_lo, bin_hi, sample)
        end
    end

    # step 4
    # write back
    if !iszero(local_ndi)
        # jacobian is the same for all samples in this bin/dim, so no need to sum it up
        jac = Ng * (bin_hi - bin_lo)
        @atomic bins_buffer[bin, dim] += (jac^2) / local_ndi * local_sum

        # used for sanity check that all samples are binned and none is lost
        @atomic ndi_buffer[bin, dim] += local_ndi
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
    i::Int32 = @index(Global)
    i -= one(Int32)
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
    ) where {T <: AbstractFloat, I <: Integer, N, D, V, W, J, Ng, G}
    nbins = Ng - 1

    # need to be zeroed because every thread just adds its calculation
    fill!(bins_buffer, zero(T))
    fill!(ndi_buffer, zero(T))

    els_per_thread = 1024
    bins_block_size = min(256, nbins)
    nblocks = ceil(Int32, N / els_per_thread)

    @debug "Calling batched binning kernel with $(els_per_thread) elements per thread"
    vegas_binning_kernel_batched!(backend, (bins_block_size, D, 1))(
        bins_buffer,
        ndi_buffer,
        buffer.values,
        grid.nodes,
        func,
        Ng,
        Val(D),
        Val(els_per_thread),
        ndrange = (Int32(nbins), Int32(D), Int32(nblocks))
    )

    synchronize(backend)

    return nothing
end

function binning_vegas!(
        backend::KernelAbstractions.CPU,
        bins_buffer::AbstractMatrix{T},
        ndi_buffer::AbstractMatrix{I},
        buffer::VegasBatchBuffer{T, N, D, V, W, J},
        grid::VegasGrid{T, Ng, D, G},
        func::Function,
    ) where {T <: AbstractFloat, I <: Integer, N, D, V, W, J, Ng, G}
    nbins = Ng - 1

    # NOTE: use unbatched implementation because the batched version is not good for CPU
    @debug "Calling unbatched binning kernel with $(nbins) bins * $(D) dims = $(nbins * D) threads"
    vegas_binning_kernel!(backend)(
        bins_buffer,
        ndi_buffer,
        buffer.values,
        grid.nodes,
        func,
        Ng,
        Val(D),
        ndrange = Int32(nbins) * Int32(D)
    )

    return synchronize(backend)
end
