# test suite for project 1
#
# NOTE: packages/modules already loaded:
# Pkg, Test, SafeTestsets, Random, GPUArrays, KernelAbstractions, StaticArrays, Vegas, Vegas.TestUtils

using Vegas: sample_vegas!, binning_vegas!
using Plots
using StatsPlots
using DataFrames

plotting = true

# definition of a simple E = 0, σ = 1 normal distribution
@inline normal_distribution(u::T) where {T <: Number} = inv(sqrt(T(2) * π)) * exp(-(u^2 / T(2)))


@inline normal_distribution(u::T, args::Vararg{T, N}) where {T <: Number, N} = normal_distribution(u) * normal_distribution(args...)

# can be used as a target function
normal_distribution(u::NTuple{N, T}) where {N, T <: Number} = @inline normal_distribution(u...)


# NOTE: The function signature can be changed, but must be adjusted in testuite.jl as well.
function testsuite_project1(backend, el_type, nbins, dim)
    LOWER = ntuple(_ -> el_type(0.0), dim)
    UPPER = ntuple(_ -> el_type(1.0), dim)
    
    println("Hello from Maria and Artur :>")

    @testset "batch_size = $batch_size" for batch_size in (2^10)#, 2^14, 2^18, 2^22)
        buffer = allocate_vegas_batch(backend, el_type, dim, batch_size)
        grid = uniform_vegas_grid(backend, LOWER, UPPER, nbins)
        
        # == SAMPLING ==
        @test isnothing(sample_vegas!(backend, buffer, grid, normal_distribution))
        
        plot_details = " ($(dim) dims, $(nbins) bins, $(batch_size) samples)"
        if plotting
            println("plotting...")
            samples = zeros(el_type, batch_size, dim)
            copyto!(samples, buffer.values)

            df_samples = DataFrame()
            for d in eachindex(axes(samples, 2))
                append!(df_samples, DataFrame(
                    value = samples[:, d],
                    dimension = "Dim $d"
                ))
            end

            b_range = range(0, 1, length = nbins)

            @df df_samples groupedhist(:value, 
                group = :dimension,
                bar_position = :dodge,
                bins = b_range,
                legend = :topright,
                title = "Sampling Distribution" * plot_details,
                xlabel = "Sample range [0:1)",
                ylabel = "Distribution [#]",
                palette = :seaborn_deep)            
            

            savefig("sampling_$(batch_size)_$(dim)_$(el_type).png")

            weights = zeros(el_type, batch_size)
            copyto!(weights, buffer.target_weights)
            
            scatter(range(0, batch_size), weights, legend = false, title = "Weighted Samples" * plot_details, xlabel = "Sample [i]", ylabel = "Weight [%]")
            savefig("weights_$(batch_size)_$(dim)_$(el_type).png")
        end

        # == BINNING ==
        bins_buffer = allocate(backend, el_type, (nbins, dim))
        @test isnothing(binning_vegas!(backend, bins_buffer, buffer, grid, normal_distribution))

        if plotting
            println("plotting...")
            binned = zeros(el_type, nbins, dim)
            copyto!(binned, bins_buffer)
            
            bar(range(0, nbins), binned[:, 1], legend = false, title = "Binning - Samples per Bin" * plot_details, xlabel = "Bin [$nbins]", ylabel = "Amount of samples per bin [#]")
            
            savefig("binning_$(batch_size)_$(dim)_$(el_type).png")
        end
        
        if sum(bins_buffer) != batch_size * dim
            println("Error: Number of binned samples doesnt match!")
            println(sum(bins_buffer), " / ", batch_size * dim, " = ", sum(bins_buffer) / (batch_size * dim))
            for d in 1:dim
                println(bins_buffer[:, d])
            end
        end
    
        # TODO: add some sanity checks on the results
        @assert sum(bins_buffer) == batch_size * dim
    end

    return
end
