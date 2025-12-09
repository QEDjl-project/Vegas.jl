module MetalExt

using Vegas
using Vegas.TestUtils
using Metal

@inline function Vegas.TestUtils.get_test_setup(backend::MetalBackend)
    return TestSetup(backend, (MtlVector,), (Float16, Float32))
end


end
