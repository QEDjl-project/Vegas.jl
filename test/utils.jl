"""
    _is_test_platform_active(env_vars::AbstractVector{String}, default::Bool)::Bool

# Args
- `env_vars::AbstractVector{String}`: List of the names of environment variables. The value of the
    first defined variable in the list is parsed and returned.
- `default::Bool`: If none of the variables named in `env_vars` are defined, this value is returned.

# Return

Return if platform is active or not.
"""
function _is_test_platform_active(env_vars::AbstractVector{String}, default::Bool)::Bool
    for env_var in env_vars
        if haskey(ENV, env_var)
            return tryparse(Bool, ENV[env_var])
        end
    end
    return default
end

function test_deprecated()
    return @testset "deprecated CPU tests" begin
        @safetestset "Grid" begin
            include("deprecated/grid.jl")
        end

        @safetestset "Vegas Map" begin
            include("deprecated/map.jl")
        end

        @safetestset "Jacobian" begin
            include("deprecated/jac.jl")
        end

        @safetestset "bin average" begin
            include("deprecated/bin_avg.jl")
        end
    end
end


function _check_all_equal(vec, kwargs...)
    el1 = first(vec)
    @test isapprox(fill(el1, length(vec)), vec, kwargs...)
    return nothing
end

# needs a plotting library loaded
function _plot_grid_dimension(bins::AbstractMatrix, d)
    vec = bins[:, d]

    data_x = vec[1:(end - 1)]
    data_y = inv.(diff(vec))
    y_sum = sum(data_y)
    data_y = data_y ./ y_sum

    return lines(data_x, data_y)
end

# needs a plotting library loaded
function _plot_grid(bins::AbstractMatrix)
    f = Figure()

    (nbins, ndims) = size(bins)

    ax = Axis(f[1, 1])

    plots = []
    labels = []

    for d in 1:ndims
        vec = bins[:, d]

        data_x = vec[1:(end - 1)]
        data_y = inv.(diff(vec))
        y_sum = sum(data_y)
        data_y = data_y ./ y_sum

        push!(plots, lines!(f[1, 1], data_x, data_y))
        push!(labels, "Dimension $d")
    end

    Legend(
        f[1, 1],
        plots,
        labels;
        tellheight = false,
        tellwidth = false,
        halign = :center,
        valign = :bottom,
        orientation = :vertical
    )

    return f
end
