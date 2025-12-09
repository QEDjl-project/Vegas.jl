include("buffer.jl")

function testsuite_run(backend, vec_type, el_type)
    @testset "buffer" testsuite_buffer(backend, vec_type, SVector{el_type, 4}, el_type, 1024)

    return nothing
end
