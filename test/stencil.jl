# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Vegas: stencil_vegas!

function testsuite_stencil(backend, el_type, nbins, dim)
    @testset "ALPHA = $ALPHA" for ALPHA in (
            zero(el_type),
            one(el_type),
            el_type(1.5),
        )
        bins_buffer = mock_bins_buffer(backend, el_type, nbins, dim)

        # == STENCIL + COMPRESSION ==
        @test isnothing(stencil_vegas!(backend, bins_buffer, ALPHA))

        if iszero(ALPHA)
            # bins will all be set to 1 (because the last compression step does x^alpha)
            if !all(isone.(Matrix(bins_buffer)))
                @show Matrix(bins_buffer)
                @test false
            end
        end
    end

    return nothing
end
