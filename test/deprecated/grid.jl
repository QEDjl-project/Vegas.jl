using Vegas.VegasCPU
using Random

include("testutils.jl")

RNG = Xoshiro(137)
DIMS = (1, 2, 5)
NNODES = (2, 8)


@testset "dim: $DIM" for DIM in DIMS
    @testset "n: $N" for N in NNODES

        LOW = _scalarize(-rand(RNG, DIM))
        UP = _scalarize(rand(RNG, DIM))

        @testset "accessors" begin

            test_grid = _rand_grid(RNG, N, DIM, LOW, UP)
            test_vg = VegasGrid(test_grid)
            @test nbins(test_vg) == N - 1
            @test ndims(test_vg) == DIM

            if DIM == 1
                @test extent(test_vg) == UP - LOW
            else
                @test extent(test_vg) == Tuple(UP .- LOW)
            end

            # TODO: add out of bounds tests
            @testset "indexing: nodes" begin
                idx = rand(RNG, 1:N)
                @test nodes(test_vg, idx) == _scalarize(test_grid[idx, :])

                if DIM != 1
                    for d in 1:DIM
                        @test nodes(test_vg, idx, d) == test_grid[idx, d]
                    end
                end
            end

            @testset "indexing: spacing" begin
                idx = rand(RNG, 1:(N - 1))
                groundtruth_spacing = _scalarize(test_grid[idx + 1, :] .- test_grid[idx, :])
                test_spacing = spacing(test_vg, idx)

                if DIM != 1
                    for d in 1:DIM
                        @test isapprox(test_spacing[d], groundtruth_spacing[d])
                        @test isapprox(spacing(test_vg, idx, d), groundtruth_spacing[d])
                    end
                else
                    @test isapprox(groundtruth_spacing, test_spacing)
                end
            end
        end


        @testset "uniform grid" begin

            test_vg = uniform_vegas_nodes(LOW, UP, N)

            diffs = diff(test_vg.nodes, dims = 1)

            for d in 1:DIM
                @test _all_approx(diffs[:, d])
            end

        end
    end
end
