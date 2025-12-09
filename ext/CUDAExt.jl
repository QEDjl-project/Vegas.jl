module CUDAExt

using Vegas
using Vegas.TestUtils
using CUDA

@inline function Vegas.TestUtils.get_test_setup(backend::CUDABackend)
    backends = (
        CUDABackend(),
        CUDABackend(prefer_blocks = true),
        CUDABackend(always_inline = true),
        CUDABackend(prefer_blocks = true, always_inline = true),
    )
    return TestSetup(backends, (CuVector,), (Float16, Float32, Float64))
end

end
