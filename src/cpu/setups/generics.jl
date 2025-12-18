# generic fallback for any computing setup
compute(stp::AbstractComputationSetup, input) = _compute(stp, input)

# make setups callable, falls back to `compute`
@inline function (stp::AbstractComputationSetup)(input...)
    return compute(stp, input...)
end

# generics for process setups

# generic implementation which ensures stp and psp have the same process, model and psl
# needs to be specialized, if this is not the case.
function compute(
        stp::STP,
        psp::PSP
    ) where {
        P <: AbstractProcessDefinition,
        M <: AbstractModelDefinition,
        PSL <: AbstractPhaseSpaceLayout,
        STP <: AbstractProcessSetup{P, M, PSL},
        PSP <: AbstractPhaseSpacePoint{P, M, PSL},
    }
    return _compute(stp, psp)
end

# generic implementation of coordinate based compute
# (without input validation)
function _compute(
        stp::AbstractProcessSetup,
        coords::NTuple{N, T}
    ) where {
        N,
        T <: Real,
    }
    psp = _build_psp(stp, coords)
    return _compute(stp, psp)
end

# generic implementation of in and out coords are passed in separately
function _compute(
        stp::AbstractProcessSetup,
        in_coords::NTuple{Nin, T},
        out_coords::NTuple{Nout, T}
    ) where {
        Nin,
        Nout,
        T <: Real,
    }
    return _compute(stp, (in_coords..., out_coords...))
end

# generic implementation of coordinate based compute
# (with input validation)
function compute(
        stp::AbstractProcessSetup,
        coords::NTuple{N, T}
    ) where {
        N,
        T <: Real,
    }
    degree_of_freedom(stp) == N || throw(
        InvalidInputError(
            "the number of coordinates needs to match the degree of freedom for the given setup"
        )
    )
    return _compute(stp, coords)
end

function compute(
        stp::AbstractProcessSetup,
        in_coords::NTuple{Nin, T},
        out_coords::NTuple{Nout, T}
    ) where {
        Nin,
        Nout,
        T <: Real,
    }
    return compute(stp, (in_coords..., out_coords...))
end

# default implementation for process setup, which don't have any cached coordinates, i.e.
# `coords` contain all the coordinate information to build the psp.
# (without input validation)
function _build_psp(stp::AbstractProcessSetup, coords::Tuple)
    return PhaseSpacePoint(
        process(stp),
        model(stp),
        phase_space_layout(stp),
        coords
    )
end

# generic implementation which falls back to `_build_psp`
# (with input validation)
function build_psp(
        stp::AbstractProcessSetup,
        coords::NTuple{N, T}
    ) where {
        N,
        T <: Real,
    }
    degree_of_freedom(stp) == N || throw(
        InvalidInputError(
            "the number of coordinates needs to match the degree of freedom for the given setup"
        )
    )
    return _build_psp(stp, coords)
end

## delegations for process setups

@inline QEDbase.number_incoming_particles(stp::AbstractProcessSetup) =
    number_incoming_particles(process(stp))
@inline QEDbase.number_outgoing_particles(stp::AbstractProcessSetup) =
    number_outgoing_particles(process(stp))
