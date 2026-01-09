# test suite for project 2
#
# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Vegas: stencil_vegas!, scan_vegas!, refine_vegas!

# NOTE: The function signature can be changed, but must be adjusted in testuite.jl as well.
function testsuite_project2(backend, el_type, nbins, dim)
    LOWER = ntuple(_ -> el_type(0.0), dim)
    UPPER = ntuple(_ -> el_type(1.0), dim)

    grid = uniform_vegas_grid(backend, LOWER, UPPER, nbins)

    @testset "ALPHA = $ALPHA" for ALPHA in (
            zero(el_type), # should leave the grid unchanged
            one(el_type),
            el_type(1.5),
        )

        # TODO: this needs to be filled with some sensible data from project 1
        bins_buffer = allocate(backend, el_type, (nbins, dim))

        # == STENCIL + COMPRESSION ==
        # TODO: implement this `stencil_vegas!` call
        @test isnothing(stencil_vegas!(backend, bins_buffer, ALPHA))

        # == SCAN ==
        # TODO: implement this `scan_vegas!` call
        avg_d = scan_vegas!(backend, bins_buffer)
        @test avg_d isa el_type

        # == REFINE ==
        # TODO: implement this `refine_vegas!` call
        @test isnothing(refine_vegas!(backend, grid, bins_buffer, avg_d))

        # TODO: add some sanity checks on the results
    end

    return nothing
end
