function testsuite_buffer(backend, el_type, dim, batch_size)

    @testset "batch buffer" begin
        buffer = Vegas._allocate_vegas_batch(backend, el_type, dim, batch_size)

        @testset "sizes" begin
            @test size(buffer.values) == (batch_size, dim)
            @test size(buffer.target_weights) == (batch_size,)
            @test size(buffer.jacobians) == (batch_size,)
        end

        @testset "types" begin
            @test eltype(buffer.values) == el_type
            @test eltype(buffer.target_weights) == el_type
            @test eltype(buffer.jacobians) == el_type
        end

    end

    @testset "output buffer" begin
        buffer = Vegas._allocate_vegas_output(backend, el_type)

        @testset "sizes" begin
            @test size(buffer.weighted_mean) == (1,)
            @test size(buffer.variance) == (1,)
            @test size(buffer.chi_square) == (1,)
        end

        @testset "types" begin
            @test eltype(buffer.weighted_mean) == el_type
            @test eltype(buffer.variance) == el_type
            @test eltype(buffer.chi_square) == el_type
        end


    end
    return nothing
end
