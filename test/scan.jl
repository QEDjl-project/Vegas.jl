# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Vegas: stencil_vegas!, scan_vegas!

function testsuite_scan(backend, el_type, nbins, dim)
    @testset "ALPHA = $ALPHA" for ALPHA in (
            zero(el_type), # should leave the grid unchanged
            one(el_type),
            el_type(1.5),
        )
        bins_buffer = mock_bins_buffer(backend, el_type, nbins, dim)

        stencil_vegas!(backend, bins_buffer, ALPHA)

        # == SCAN ==
        avg_d = scan_vegas!(backend, bins_buffer)
        @test avg_d isa AbstractVector
        @test length(avg_d) == dim
        @test eltype(avg_d) == el_type

        # TODO: add some sanity checks on the results
    end

    return nothing
end
