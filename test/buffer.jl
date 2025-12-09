function testsuite_buffer(backend, vec_type, sample_type, weight_type, batch_size)

    @testset "batch buffer" begin
        buffer = Vegas._allocate_vegas_batch(backend, sample_type, weight_type, batch_size)

        @testset "sizes" begin
            @test size(buffer.vegas_samples) == (batch_size,)
            @test size(buffer.target_samples) == (batch_size,)
            @test size(buffer.jacobians) == (batch_size,)
            @test size(buffer.func_values) == (batch_size,)
        end

        @testset "types" begin
            @test eltype(buffer.vegas_samples) == sample_type
            @test eltype(buffer.target_samples) == sample_type
            @test eltype(buffer.jacobians) == weight_type
            @test eltype(buffer.func_values) == weight_type
        end

    end

    @testset "output buffer" begin
        buffer = Vegas._allocate_vegas_output(backend, weight_type)

        @testset "sizes" begin
            @test size(buffer.weighted_mean) == (1,)
            @test size(buffer.variance) == (1,)
            @test size(buffer.chi_square) == (1,)
        end

        @testset "types" begin
            @test eltype(buffer.weighted_mean) == weight_type
            @test eltype(buffer.variance) == weight_type
            @test eltype(buffer.chi_square) == weight_type
        end


    end
    return nothing
end
