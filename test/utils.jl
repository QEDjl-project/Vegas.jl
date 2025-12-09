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
