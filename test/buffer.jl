function testsuite_buffer(backend, el_type, dim, batch_size)

    @testset "batch buffer" begin
        buffer = allocate_vegas_batch(backend, el_type, dim, batch_size)

        @testset "properties" begin
            @test eltype(buffer) == el_type
            @test length(buffer) == batch_size
            @test size(buffer) == (batch_size, dim)
        end

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

        @testset "errors" begin
            # wrong size of values
            @test_throws ArgumentError VegasBatchBuffer(
                allocate(backend, el_type, (batch_size + 1, dim)),
                buffer.target_weights,
                buffer.jacobians
            )

            # wrong length of target weights
            @test_throws ArgumentError VegasBatchBuffer(
                buffer.values,
                allocate(backend, el_type, batch_size + 1),
                buffer.jacobians
            )

            # wrong length of jacobians
            @test_throws ArgumentError VegasBatchBuffer(
                buffer.values,
                buffer.target_weights,
                allocate(backend, el_type, batch_size + 1),
            )

            # wrong element type
            @test_throws MethodError allocate_vegas_batch(
                backend,
                (1,),
                dim,
                batch_size
            )

            # wrong dim (negative)
            @test_throws ArgumentError allocate_vegas_batch(
                backend,
                el_type,
                -1,
                batch_size
            )

            # wrong dim (vanishing)
            @test_throws ArgumentError allocate_vegas_batch(
                backend,
                el_type,
                0,
                batch_size
            )

            # wrong batch size (negative)
            @test_throws ArgumentError allocate_vegas_batch(
                backend,
                el_type,
                dim,
                -1,
            )

            # wrong batch size (vanishing)
            @test_throws ArgumentError allocate_vegas_batch(
                backend,
                el_type,
                dim,
                0,
            )
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
