# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Vegas: stencil_vegas!, scan_vegas!, refine_vegas!

function testsuite_refine(backend, el_type, nbins, dim)
    LOWER = ntuple(_ -> el_type(-1.0), dim)
    UPPER = ntuple(_ -> el_type(1.0), dim)

    grid = uniform_vegas_grid(backend, LOWER, UPPER, nbins)

    @testset "ALPHA = $ALPHA" for ALPHA in (
            zero(el_type), # should leave the grid unchanged
            one(el_type),
            el_type(1.5),
        )
        bins_buffer = mock_bins_buffer(backend, el_type, nbins, dim)

        stencil_vegas!(backend, bins_buffer, ALPHA)
        avg_d = scan_vegas!(backend, bins_buffer)

        # == REFINE ==
        @test isnothing(refine_vegas!(backend, grid, bins_buffer, avg_d))

        # TODO: add some sanity checks on the results
    end

    return nothing
end
