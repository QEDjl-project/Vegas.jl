include("buffer.jl")
include("target.jl")

function testsuite_run(backend, vec_type, el_type)
    @testset "buffer" begin
        @testset "dim = 1" testsuite_buffer(backend, el_type, 1, 1024)
        @testset "dim = 4" testsuite_buffer(backend, el_type, 4, 1024)
    end

    @testset "target" begin
        @testset "dim = 1" testsuite_target(backend, el_type, 1, 1024)
        @testset "dim = 4" testsuite_target(backend, el_type, 4, 1024)
    end
    return nothing
end
