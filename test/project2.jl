# test suite for project 2
#
# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils


# NOTE: The function signature can be changed, but must be adjusted in testuite.jl as well.
function testsuite_project2(backend, el_type, nbins, dim)

    return @test Vegas.dummy_proj2() == "This is the place for project 2"

end
