using Vegas
using Random

include("testutils.jl")

RNG = Xoshiro(137)
#DIMS = (1, 2, 5)
NNODES = (2, 10)
NSAMPLES = (1, 15)
ALPHA_VALUES = (1.0, 0.0, rand(RNG, 2:10))

# TODO: implement multi dim
@testset "ndims: 1" begin
    @testset "n: $N" for N in NNODES
        LOW = -rand(RNG)
        UP = rand(RNG)


        test_grid = _rand_grid(RNG, N, 1, LOW, UP)
        test_vg = VegasGrid(test_grid)

        @testset "nsamples: $ns" for ns in NSAMPLES
            test_u = rand(RNG, ns)
            JxF = rand(RNG, ns)
            JxF2 = JxF .^ 2

            @testset "create" begin
                test_bin_avg = Vegas._eff_bin_avg(test_vg, test_u, JxF2)
                groundtruth_bin_avg = _subinterval_avg(JxF, test_u, nbins(test_vg), ns, 1.0)

                @test isapprox(test_bin_avg, groundtruth_bin_avg)
            end

            @testset "smoothing" begin
                # smoothing is non-sense less that three bins
                if N > 2
                    test_bin_avg = Vegas._eff_bin_avg(test_vg, test_u, JxF2)
                    test_bin_avg = Vegas._smooth_bin_avg(test_bin_avg)
                    groundtruth_bin_avg = _subinterval_avg(
                        JxF,
                        test_u,
                        nbins(test_vg),
                        ns,
                        1.0;
                        smooth = true,
                    )

                    @test isapprox(test_bin_avg, groundtruth_bin_avg)
                end
            end

            @testset "compression" begin
                @testset "alpha: $A" for A in ALPHA_VALUES
                    test_bin_avg = Vegas._eff_bin_avg(test_vg, test_u, JxF2)
                    Vegas._compress_bin_avg!(test_bin_avg, A)
                    groundtruth_bin_avg = _subinterval_avg(
                        JxF,
                        test_u,
                        nbins(test_vg),
                        ns,
                        A;
                        compress = true,
                    )

                    @test isapprox(test_bin_avg, groundtruth_bin_avg)
                end
            end

            @testset "refinement" begin
                @testset "alpha: $A" for A in ALPHA_VALUES
                    # smoothing is non-sense less that three bins
                    if N > 2
                        test_vg2 = deepcopy(test_vg)
                        test_bin_avg = Vegas._eff_bin_avg(test_vg2, test_u, JxF2)
                        test_bin_avg = Vegas._smooth_bin_avg(test_bin_avg)
                        Vegas._compress_bin_avg!(test_bin_avg, A)
                        Vegas._refine_nodes!(test_vg2, test_bin_avg)

                        groundtruth_bin_avg = _subinterval_avg(
                            JxF,
                            test_u,
                            nbins(test_vg),
                            ns,
                            A;
                            smooth = true,
                            compress = true,
                        )
                        groundtruth_refined_grid =
                            _refine_grid(test_vg, groundtruth_bin_avg)
                        @test isapprox(test_vg2.nodes, groundtruth_refined_grid)
                    end
                end
            end
        end

    end
end
