# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Distributions
using Vegas: sample_vegas!

function testsuite_sampling(backend, el_type, nbins, dim)
    LOWER = ntuple(_ -> el_type(0.0), dim)
    UPPER = ntuple(_ -> el_type(1.0), dim)

    @testset "batch_size = $batch_size" for batch_size in (2^10, 2^14, 2^18, 2^22)
        buffer = allocate_vegas_batch(backend, el_type, dim, batch_size)
        grid = uniform_vegas_grid(backend, LOWER, UPPER, nbins)

        # == SAMPLING ==
        @test isnothing(sample_vegas!(backend, buffer, grid, normal_distribution))

        # TODO: add actual statistical checks for what this is sampling
    end

    return
end
