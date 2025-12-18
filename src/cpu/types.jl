abstract type AbstractVegasGrid end

struct VegasGrid{N, T, G} <: AbstractVegasGrid

    nodes::G

    # 1D case
    VegasGrid(nodes::G) where {T <: Real, G <: AbstractVector{T}} = new{1, T, G}(nodes)

    # nD case: static ndimsension
    function VegasGrid{DIMS}(nodes::G) where {DIMS, T <: Real, G <: AbstractMatrix{T}}
        DIMS == size(nodes, 2) || throw(
            ArgumentError(
                "wrong dimension! Input nodes has dimension $(size(nodes, 2)); $N expected",
            ),
        )

        return new{N, T, G}(nodes)
    end

    function VegasGrid(nodes::G) where {T <: Real, G <: AbstractMatrix{T}}
        N = size(nodes, 2)
        return new{N, T, G}(nodes)
    end
end

# TODO:
# - implement for tuple input, maybe using SVectors ?
# - implement possibility to init with stepsize analog to low:steps:up
# - consider implementing another VegasGrid based on vector of svectors

# equidistant nodes: used for initialization

function _uniform_vegas_nodes(
        lower_bounds::AbstractVector,
        upper_bounds::AbstractVector,
        nbins::Int,
    )
    steps = Matrix(stack(fill((upper_bounds .- lower_bounds) ./ nbins, nbins))')
    g = _nodes_from_spacing(steps, lower_bounds)
    return VegasGrid(g)
end

function _uniform_vegas_nodes(lower_bounds::Tuple, upper_bounds::Tuple, nbins::Int)
    return _uniform_vegas_nodes([lower_bounds...], [upper_bounds...], nbins)
end

function uniform_vegas_nodes(lower::Real, upper::Real, nbins::Int)
    nbins >= 1 || throw(
        InvalidInputError(
            "number of bins `nbins` must be a positiv integer. <$nbins> given.",
        ),
    )

    return VegasGrid(collect((upper - lower) .* range(0, 1, nbins + 1) .+ lower))
end

function uniform_vegas_nodes(lower, upper, nbins)
    length(lower) == length(upper) || throw(
        InvalidInputError(
            """number of lower bounds and upper bounds need to be equal:\n
            \tlength(lower) = (<$(length(lower))>)\n\n
            \tlength(upper) = (<$(length(upper))>)
            """
        ),
    )

    nbins >= 1 || throw(
        InvalidInputError(
            "number of bins `nbins` must be a positiv integer. <$nbins> given.",
        ),
    )
    return _uniform_vegas_nodes(lower, upper, nbins)
end

# Vegas proposal

abstract type AbstractProposalDistribution <: ScatteringProcessDistribution end

struct VegasProposal{STP, T, G} <: AbstractProposalDistribution
    stp::STP
    alpha::T
    weighted_tot_cs::Ref{T}
    cum_variance::Ref{T}
    cum_chi_square::Ref{T}
    vgrid::G

    function VegasProposal(
            stp::STP;
            ret_type::Type{T} = Float64,
            nbins::Int = 1000,
            alpha::T = 1.5,
            #rtol::T = 1e-4,
            #atol::T = 1e-4,
        ) where {T <: Real, STP <: AbstractProcessSetup}
        lower, upper = coordinate_boundaries(stp)
        init_vg = _uniform_vegas_nodes(lower, upper, nbins)
        init_tot_cs, init_variance, init_chi_sq = _vp_init_values(T)

        return new{STP, T, typeof(init_vg)}(
            stp,
            alpha,
            init_tot_cs,
            init_variance,
            init_chi_sq,
            init_vg,
        )
    end
end

setup(d::VegasProposal) = d.stp
QEDbase.process(d::VegasProposal) = process(setup(d))
QEDbase.model(d::VegasProposal) = model(setup(d))
QEDbase.phase_space_layout(d::VegasProposal) = phase_space_layout(setup(d))

function _build_psp(vp::VegasProposal, coords::Tuple)
    return _build_psp(setup(vp), coords)
end

# TODO:
# - implement the ProcessDistribution interface for VegasProposal
#
