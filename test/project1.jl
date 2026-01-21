# test suite for project 1
#
# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Vegas: sample_vegas!, binning_vegas!

# NOTE: The function signature can be changed, but must be adjusted in testuite.jl as well.
function testsuite_project1(backend, el_type, nbins, dim)
    grid = allocate_vegas_grid(backend, el_type, nbins, dim)

    # generate some random distributions
    μs = ntuple(_ -> 3 * (rand(el_type) - el_type(0.5)), dim)
    σs = ntuple(_ -> rand(el_type) * 5, dim)
    dists = ntuple(i -> Normal(μs[i], σs[i]), dim)

    # build a grid for these distributions
    # (only possible because we know the distributions are linearly independent and we now their cdfs)
    LOWER = ntuple(_ -> el_type(-5.0), dim)
    UPPER = ntuple(_ -> el_type(5.0), dim)

    bins = fill(zero(el_type), (nbins + 1, dim))
    Vegas._gen_grid(bins, dists, LOWER, UPPER)

    copyto!(grid.nodes, bins)

    @testset "batch_size = $batch_size" for batch_size in (2^10, 2^14, 2^18, 2^22)
        buffer = allocate_vegas_batch(backend, el_type, dim, batch_size)

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
