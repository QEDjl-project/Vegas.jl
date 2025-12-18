function train_iter!(
        rng::AbstractRNG,
        vp::VegasProposal,
        ncalls::Int = 10000,
        # other training parameters
    )
    vg = vp.vgrid

    # generate uniform distributed y
    #
    # TODO: enhancement
    # implement a function `generate_y` which also gets the VegasProposal
    # this allows for generating the y on the correct device
    y = rand(rng, ncalls, ndims(vg))

    # calculate coordinates using vegas map
    coords = _vegas_map(vg, y)
    # TODO: think about constructing psp based on coords here.

    # calculate jacobian
    jac = prod(_jac_vegas_map(vg, y), dims = 2)

    # calculate dcs values
    # TODO: could be done via broadcast to allow work evaluation on GPU (maybe better
    # based on psp, rather than coords, because in_coords do not live on GPU)
    # TODO: basing this line on psp also makes it easier to include the init state of the
    # process
    F = map(Base.Fix1(_compute, setup(vp)), coords)

    # calculate F times jac
    FxJ = F .* jac
    FxJ2 = FxJ .^ 2

    # calculate bin avg
    D = _eff_bin_avg(vg, y, FxJ2)

    D = _smooth_bin_avg(D)
    _compress_bin_avg!(D, vp.alpha)

    # refine grid
    _refine_nodes!(vg, D)

    # calculate iteration estimate and variance
    # TODO:
    # - think about using StatsBase.mean and StatsBase.var here
    I = sum(FxJ) / ncalls
    var = (sum(FxJ2) / ncalls - I^2) / (ncalls - 1)

    # update (totCS, cumj_variance, sum_chi_sq)
    vp.weighted_tot_cs[] += I / var
    vp.cum_variance[] += inv(var)
    vp.cum_chi_square[] += (I - total_cross_section(vp)) / var

    # return the iteration values for analysis of the training process
    return (I, var)
end

# TODO:
# - test this!
# - enhance this with some training analysis (e.g. enable short-cut if training was
# sufficient before reaching niter)
function train!(rng, vp::VegasProposal, niter, ncalls)
    for _ in 1:niter
        train_iter!(rng, vp, ncalls)
    end
    return nothing
end


# TODO: GPU training
# - except _refine_nodes!, everything can be called on GPU
# - for now, we should do the training on CPU, however, one could think of an evaluation
# of the dcs on GPU to increase the performance
# - maybe it is worth to generate CuArrays of PSP directly from the coords returned by the
# vegas map
