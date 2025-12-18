module Vegas

# compute setups
export AbstractComputationSetup, AbstractProcessSetup
export compute, scattering_process, physical_model

# target distributions
export AbstractTargetDistribution
export degrees_of_freedom, compute!

# buffer
export VegasBatchBuffer, VegasOutBuffer
export allocate_vegas_batch

# Vegas Grid
export VegasGrid
export allocate_vegas_grid
export nbins
export fill_uniformly!, uniform_vegas_grid

using QEDbase
using QEDcore
using QEDevents
using Random
using KernelAbstractions
using Adapt


include("utils.jl")

include("buffer.jl")
include("target.jl")
include("grid.jl")
include("testutils/TestUtils.jl")
include("cpu/VegasCPU.jl")

include("project1.jl")
include("project2.jl")

end
