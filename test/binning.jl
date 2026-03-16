# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils
using Distributions
using Vegas: sample_vegas!, binning_vegas!

function testsuite_binning(backend, el_type, nbins, dim)
    LOWER = ntuple(_ -> el_type(0.0), dim)
    UPPER = ntuple(_ -> el_type(1.0), dim)

    @testset "batch_size = $batch_size" for batch_size in (2^10, 2^14, 2^18, 1024 * 35)
        buffer = allocate_vegas_batch(backend, el_type, dim, batch_size)
        grid = uniform_vegas_grid(backend, LOWER, UPPER, nbins)

        # fill sampling buffer to have inputs for binning
        sample_vegas!(backend, buffer, grid, normal_distribution)

        # == BINNING ==
        bins_buffer = allocate(backend, el_type, (nbins, dim))
        ndi_buffer = allocate(backend, Int, (nbins, dim))

        @test isnothing(binning_vegas!(backend, bins_buffer, ndi_buffer, buffer, grid, normal_distribution))

        @test if sum(ndi_buffer) == batch_size * dim
            true
        else
            @error "Amount of binned samples wrong, must have lost some in binning kernel: $(sum(ndi_buffer)) out of $(batch_size * dim)"
            false
        end

        # TODO: add statistical tests for these outputs
    end

    return
end
