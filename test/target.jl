RNG = Xoshiro(137137)

struct SimpleDist{D} <: AbstractTargetDistribution
    SimpleDist(d::Int) = new{d}()
end

Vegas.degrees_of_freedom(::SimpleDist{D}) where {D} = D
_groundtruth(x) = sum(x)
_groundtruth!(dest::Vector, x::Matrix) = (dest .= sum(x, dims = 2))

Vegas._compute(::SimpleDist, x) = _groundtruth(x)


function testsuite_target(backend, el_type, dim, batch_size)
    test_dist = SimpleDist(dim)
    buffer = Vegas._allocate_vegas_batch(backend, el_type, dim, batch_size)
    rand!(buffer.values) # fill target samples

    @testset "properties" begin
        @test degrees_of_freedom(test_dist) == dim
    end

    return @testset "compute" begin
        Vegas._compute!(backend, buffer, test_dist)
        h_x = Array(buffer.values)
        h_dest = Vector{el_type}(undef, batch_size)
        _groundtruth!(h_dest, h_x)

        @test isapprox(Array(buffer.target_weights), h_dest)
    end


end
