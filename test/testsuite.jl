include("buffer.jl")
include("target.jl")
include("grid.jl")
include("project1.jl")
include("project2.jl")

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

    @testset "project 1" begin
        @testset "dim = 1" testsuite_project1(backend, el_type, 2^5 - 1, 1) # 2^5 - 1 nbins -> 32 grid lines
        @testset "dim = 11" testsuite_project1(backend, el_type, 2^5 - 1, 11)  # 2^5 - 1 nbins -> 32 grid lines
    end

    @testset "project 2" begin
        @testset "dim = 1" testsuite_project2(backend, el_type, 2^5, 1)
        @testset "dim = 11" testsuite_project2(backend, el_type, 2^5, 11)
    end

    return nothing
end
