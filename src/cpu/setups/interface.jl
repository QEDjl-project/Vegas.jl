abstract type AbstractComputationSetup end
Base.broadcastable(stp::AbstractComputationSetup) = Ref(stp)

"""

    _compute(stp::AbstractComputationSetup,input::Any)

"""
function _compute end


"""

    AbstractProcessSetup{P,M,PSL}

Abstract base type for setups related to scattering processes with

- `P<:AbstractProcessDefinition`,
- `M<:AbstractModelDefinition`,
- `PSL<:AbstractPhaseSpaceLayout`.

Interface function to be implemented:

- `QEDbase.process(stp::AbstractProcessSetup)::P`,
- `QEDbase.model(stp::AbstractProcessSetup)::M`,
- `QEDbase.phase_space_layout(stp::AbstractProcessSetup)::PSL`,
- `_compute(stp::AbstractProcessSetup, psp::AbstractPhaseSpacePoint)::Real`.

Optionally, the following interface functions can be implemented:
- `_build_psp(stp::AbstractProcessSetup,coords::Tuple)::PSP`
- `degree_of_freedom(stp::AbstractProcessSetup)::Int`,
- `coordinate_boundaries(stp::AbstractProcessSetup)::Tuple`,

A leading underscore of an interface function means, there is no input validation
necessary, because there is a generic implementation of the same function without the
underscore, which does basic compiletime checks.

"""
abstract type AbstractProcessSetup{
    P <: AbstractProcessDefinition,
    M <: AbstractModelDefinition,
    PSL <: AbstractPhaseSpaceLayout,
} <: AbstractComputationSetup end

"""

    _build_psp(stp::AbstractProcessSetup, coords::Tuple)

"""
function _build_psp end


"""

    degree_of_freedom(stp::AbstractProcessSetup)

"""
function degree_of_freedom end

"""

    coordinate_boundaries(stp::AbstractProcessSetup)

"""
function coordinate_boundaries end
