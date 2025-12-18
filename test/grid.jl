RNG = Xoshiro(137137137)

function testsuite_grid(backend, el_type, n_bins, dim)

    @testset "properties" begin

        grid = allocate_vegas_grid(backend, el_type, n_bins, dim)

        # INFO:
        # - backend is not exactly the same
        # - get_backend(grid) returns the backend of the array storing the nodes, which is
        # the default backend for the array type
        @test get_backend(grid) isa typeof(backend)

        @test nbins(grid) == n_bins
        @test eltype(grid) == el_type
        @test ndims(grid) == dim
    end

    @testset "uniform grid" begin
        lower = Tuple(-rand(RNG, el_type, dim))
        upper = Tuple(rand(RNG, el_type, dim))

        d_grid_preinit = allocate_vegas_grid(backend, el_type, n_bins, dim)
        fill_uniformly!(d_grid_preinit, lower, upper)
        h_grid_preinit_nodes = Array(d_grid_preinit.nodes)
        diffs = diff(h_grid_preinit_nodes, dims = 1)
        for d in 1:dim
            _check_all_equal(diffs[:, d])
        end

        d_grid_direct = uniform_vegas_grid(backend, lower, upper, n_bins)
        h_grid_direct_nodes = Array(d_grid_direct.nodes)
        diffs = diff(h_grid_direct_nodes, dims = 1)
        for d in 1:dim
            _check_all_equal(diffs[:, d])
        end
    end

    @testset "error" begin
        @test_throws ArgumentError allocate_vegas_grid(backend, el_type, -1, dim)
        @test_throws ArgumentError allocate_vegas_grid(backend, el_type, n_bins, -1)
        @test_throws ArgumentError allocate_vegas_grid(backend, el_type, 0, dim)
        @test_throws ArgumentError allocate_vegas_grid(backend, el_type, n_bins, 0)

        # one entry with lower >upper
        idx = rand(RNG, 1:dim)
        wrong_low = -rand(RNG, el_type, dim)
        wrong_low[idx] *= -1
        wrong_low = Tuple(wrong_low)
        wrong_up = rand(RNG, el_type, dim)
        wrong_up[idx] *= -1
        wrong_up = Tuple(wrong_up)

        grid = allocate_vegas_grid(backend, el_type, n_bins, dim)
        @test_throws ArgumentError fill_uniformly!(grid, wrong_low, wrong_up)
    end
    return nothing
end
