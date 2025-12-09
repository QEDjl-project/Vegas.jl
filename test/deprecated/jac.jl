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
            for i in 1:nbins(test_vg)
                u_on_node = (i - 1) / nbins(test_vg)
                test_jac = Vegas._jac_vegas_map(test_vg, u_on_node)

                @test isapprox(test_jac, nbins(test_vg) * spacing(test_vg, i))
            end
        end

        @testset "multiple" begin
            for i in 1:nbins(test_vg)
                u_on_node = fill((i - 1) / nbins(test_vg), 2)
                test_jac_vec = Vegas._jac_vegas_map(test_vg, u_on_node)

                for j in eachindex(test_jac_vec)
                    @test isapprox(test_jac_vec[j], nbins(test_vg) * spacing(test_vg, i))
                end
            end
        end

        @testset "sum rule" begin
            test_sum = sum(
                [
                    spacing(test_vg, i) /
                        Vegas._jac_vegas_map(test_vg, (i - 1) / nbins(test_vg)) for
                        i in 1:nbins(test_vg)
                ]
            )
            @test isapprox(test_sum, one(test_sum))
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
            for i in 1:nbins(test_vg)
                u_on_node = fill((i - 1) / nbins(test_vg), DIM)
                test_jac = Vegas._jac_vegas_map(test_vg, u_on_node)
                groundtruth_jac =
                    prod(nbins(test_vg) * spacing(test_vg, i, d) for d in 1:DIM)

                @test isapprox(test_jac, groundtruth_jac)
            end
        end

        @testset "multiple" begin
            for i in 1:nbins(test_vg)
                u_on_node = fill((i - 1) / nbins(test_vg), 2, DIM)
                test_jac_vec = Vegas._jac_vegas_map(test_vg, u_on_node)
                groundtruth_jac =
                    prod(nbins(test_vg) * spacing(test_vg, i, d) for d in 1:DIM)

                for j in eachindex(test_jac_vec)
                    @test isapprox(test_jac_vec[j], groundtruth_jac)
                end
            end
        end

        @testset "sum rule" begin
            test_jac_on_nodes = Vegas._jac_vegas_map(
                test_vg,
                stack(fill((0:(nbins(test_vg) - 1)) / nbins(test_vg), DIM)),
            )

            test_sum = sum(
                [
                    prod(spacing(test_vg, i, d) for d in 1:DIM) / test_jac_on_nodes[i] for
                        i in 1:nbins(test_vg)
                ]
            )
            @test isapprox(test_sum, inv(nbins(test_vg)^(DIM - 1)))
        end

    end
end
