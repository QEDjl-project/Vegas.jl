module AMDGPUExt

using Vegas
using Vegas.TestUtils
using AMDGPU

@inline function Vegas.TestUtils.get_test_setup(backend::ROCBackend)
    return TestSetup(backend, (ROCVector,), (Float32, Float64))
end

end
