# test suite for project 1
#
# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Vegas: sample_vegas!, binning_vegas!

# NOTE: The function signature can be changed, but must be adjusted in testuite.jl as well.
function testsuite_project1(backend, el_type, nbins, dim)
    LOWER = ntuple(_ -> el_type(0.0), dim)
    UPPER = ntuple(_ -> el_type(1.0), dim)

    @testset "batch_size = $batch_size" for batch_size in (2^10, 2^14, 2^18, 2^22)
        buffer = allocate_vegas_batch(backend, el_type, dim, batch_size)
        grid = uniform_vegas_grid(backend, LOWER, UPPER, nbins)

        # == SAMPLING ==
        # TODO: implement the `sample_vegas!` call:
        @test isnothing(sample_vegas!(backend, buffer, grid))

        # == BINNING ==
        # TODO: implement the `binning_vegas!` call:
        bins_buffer = allocate(backend, el_type, (nbins, dim))
        @test isnothing(binning_vegas!(backend, bins_buffer, buffer, grid))

        # TODO: add some sanity checks on the results
    end

    return
end
