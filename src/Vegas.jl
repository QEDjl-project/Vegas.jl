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

# Vegas Sampler
export VegasGrid, VegasProposal
export nbins, extent, nodes, spacing
export uniform_vegas_nodes
export train!

using QEDbase
using QEDcore
using QEDevents
using Random
using KernelAbstractions

include("setups/interface.jl")
include("setups/generics.jl")


include("utils.jl")
include("types.jl")
include("access.jl")
include("map.jl")
include("refine.jl")
include("training.jl")
include("sampler.jl")

include("buffer.jl")
include("target.jl")
include("testutils/TestUtils.jl")


end
