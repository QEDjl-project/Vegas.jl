module VegasCPU

# compute setups
export AbstractComputationSetup, AbstractProcessSetup
export compute, scattering_process, physical_model

# Vegas Sampler
export VegasGrid, VegasProposal
export nbins, extent, nodes, spacing
export uniform_vegas_nodes
export train!

using QEDbase
using QEDcore
using QEDevents
using Random

include("setups/interface.jl")
include("setups/generics.jl")

include("utils.jl")
include("types.jl")
include("access.jl")
include("map.jl")
include("refine.jl")
include("training.jl")
include("sampler.jl")

end
