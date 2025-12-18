@inline nbins(vg::VegasGrid) = size(vg.nodes, 1) - 1
@inline Base.ndims(vg::VegasGrid{N}) where {N} = N
@inline nodes(vg::VegasGrid, idx, d) = vg.nodes[idx, d]
@inline nodes(vg::VegasGrid, idx) = vg.nodes[idx, :]
@inline nodes(vg::VegasGrid{1}, idx) = vg.nodes[idx]

@inline spacing(vg::VegasGrid, idx, d) = nodes(vg, idx + 1, d) - nodes(vg, idx, d)
@inline spacing(vg::VegasGrid, idx, d::Colon) = spacing(vg, idx)
@inline spacing(vg::VegasGrid, idx) =
    ntuple(d -> nodes(vg, idx + 1, d) - nodes(vg, idx, d), ndims(vg))
@inline spacing(vg::VegasGrid{1}, idx) = nodes(vg, idx + 1) - nodes(vg, idx)

@inline extent(vg::VegasGrid) = ntuple(d -> vg.nodes[end, d] - vg.nodes[1, d], ndims(vg))
@inline extent(vg::VegasGrid{1}) = vg.nodes[end] - vg.nodes[1]


@inline Base.ndims(vp::VegasProposal) = ndims(vp.vgrid)
@inline nbins(vp::VegasProposal) = nbins(vp.vgrid)
@inline QEDbase.total_cross_section(vp::VegasProposal) =
    vp.weighted_tot_cs[] / vp.cum_variance[]
@inline variance(vp::VegasProposal) = inv(sqrt(vp.cum_variance[]))
@inline chi_square(vp::VegasProposal) = vp.cum_chi_square[]
