# test suite for project 1
#
# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Vegas: sample_vegas!, binning_vegas!
using Plots
include("random_utils.jl")


# NOTE: The function signature can be changed, but must be adjusted in testuite.jl as well.
function testsuite_project1(backend, el_type, nbins, dim)
    LOWER = ntuple(_ -> el_type(0.0), dim)
    UPPER = ntuple(_ -> el_type(1.0), dim)
    
    println("Hello from Maria and Artur :>")

    @testset "batch_size = $batch_size" for batch_size in (2^10, 2^14, 2^18, 2^22)
        buffer = allocate_vegas_batch(backend, el_type, dim, batch_size)
        grid = uniform_vegas_grid(backend, LOWER, UPPER, nbins)

        # == SAMPLING ==
        @test isnothing(sample_vegas!(backend, buffer, grid, normal_distribution))

        # samples = Array{el_type, dim}(undef, dim, batch_size)
        samples = zeros(el_type, batch_size, dim)
        D = size(buffer)[2]
        println("Dimensions: ",D, "Samples: " , size(buffer)[1])
        copyto!(samples, buffer.values)
        b_range = range(0, 1, length=100)
        histogram!(samples[:,1], bins=b_range)
        savefig("scatter_$(D)_$(el_type).pdf")
        # plot(1:nbins, [sum([1 for x in samples[ : , 1] if bin < x < bin + 1]) for bin in 1:nbins])
        # plot(1:batch_size, samples[ : , 1])
        # savefig("sampling_$(dim)_$(batch_size).pdf")

        # == BINNING ==
        # bins_buffer = allocate(backend, el_type, (nbins, dim))
        # @test isnothing(binning_vegas!(backend, bins_buffer, buffer, grid, normal_distribution))

        # TODO: add some sanity checks on the results
    end

    return
end
