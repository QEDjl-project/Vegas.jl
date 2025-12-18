include("buffer.jl")
include("target.jl")
include("grid.jl")

function testsuite_run(backend, vec_type, el_type)
    @testset "buffer" begin
        @testset "dim = 1" testsuite_buffer(backend, el_type, 1, 1024)
        @testset "dim = 11" testsuite_buffer(backend, el_type, 11, 1024)
    end

    @testset "target" begin
        @testset "dim = 1" testsuite_target(backend, el_type, 1, 1024)
        @testset "dim = 11" testsuite_target(backend, el_type, 11, 1024)
    end

    @testset "grid" begin
        @testset "dim = 1" testsuite_grid(backend, el_type, 2^5, 1)
        @testset "dim = 11" testsuite_grid(backend, el_type, 2^5, 11)
    end
    return nothing
end
