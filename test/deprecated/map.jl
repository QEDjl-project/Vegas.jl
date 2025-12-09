using Vegas
using Random

include("testutils.jl")

RNG = Xoshiro(137)
DIMS = (2, 5)
NNODES = (2, 8)

@testset "ndims: 1" begin
    LOW = -rand(RNG)
    UP = rand(RNG)

    @testset "n: $N" for N in NNODES
        test_grid = _rand_grid(RNG, N, 1, LOW, UP)
        test_vg = VegasGrid(test_grid)

        @testset "single" begin

            @testset "bounds" begin
                test_u = rand(RNG)
                test_x = Vegas._vegas_map(test_vg, test_u)
                @test LOW <= test_x <= UP
            end

            @testset "on nodes" begin
                for i in 1:nbins(test_vg)
                    u_on_node = (i - 1) / nbins(test_vg)
                    test_x = Vegas._vegas_map(test_vg, u_on_node)

                    @test isapprox(test_x, nodes(test_vg, i))
                end
            end

        end

        @testset "multiple" begin

            @testset "bounds" begin
                test_u_vec = rand(RNG, 2)
                test_x_vec = Vegas._vegas_map(test_vg, test_u_vec)
                for i in eachindex(test_x_vec)
                    @test LOW <= test_x_vec[i] <= UP
                end
            end

            @testset "on nodes" begin
                for i in 1:nbins(test_vg)
                    u_on_node = fill((i - 1) / nbins(test_vg), 2)
                    test_x_vec = Vegas._vegas_map(test_vg, u_on_node)

                    for j in eachindex(test_x_vec)
                        @test isapprox(test_x_vec[j], nodes(test_vg, i))
                    end
                end
            end

        end

    end
end

@testset "ndims: $DIM" for DIM in DIMS
    LOW = -rand(RNG, DIM)
    UP = rand(RNG, DIM)
    @testset "n: $N" for N in NNODES

        test_grid = _rand_grid(RNG, N, DIM, LOW, UP)
        test_vg = VegasGrid(test_grid)

        @testset "single" begin

            @testset "bounds" begin
                test_u = rand(RNG, DIM)
                test_x = Vegas._vegas_map(test_vg, test_u)
                for d in 1:DIM
                    @test LOW[d] <= test_x[d] <= UP[d]
                end
            end

            @testset "on nodes" begin
                for i in 1:nbins(test_vg)
                    u_on_node = fill((i - 1) / nbins(test_vg), DIM)
                    test_x = Vegas._vegas_map(test_vg, u_on_node)
                    @testset "dim: $d" for d in 1:DIM
                        @test isapprox(test_x[d], nodes(test_vg, i, d))
                    end
                end
            end

        end

        @testset "multiple" begin

            @testset "bounds" begin
                test_u = rand(RNG, 2, DIM)
                test_x = Vegas._vegas_map(test_vg, test_u)
                for d in 1:DIM
                    for i in eachindex(test_x)
                        @test LOW[d] <= test_x[i][d] <= UP[d]
                    end
                end
            end

            @testset "on nodes" begin
                for i in 1:nbins(test_vg)
                    u_on_node = fill((i - 1) / nbins(test_vg), 2, DIM)
                    test_x = Vegas._vegas_map(test_vg, u_on_node)
                    @testset "dim: $d" for d in 1:DIM
                        for j in eachindex(test_x)
                            @test isapprox(test_x[j][d], nodes(test_vg, i, d))
                        end
                    end
                end
            end

        end
    end
end
