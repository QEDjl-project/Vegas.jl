using Pkg

# targeting the correct source code
# this assumes the make.jl script is located in Vegas.jl/docs
project_path = Base.Filesystem.joinpath(Base.Filesystem.dirname(Base.source_path()), "..")
Pkg.develop(; path = project_path)

using Documenter

using Vegas

# some paths for links
readme_path = joinpath(project_path, "README.md")
index_path = joinpath(project_path, "docs/src/index.md")
license_path = "https://github.com/QEDjl-project/Vegas.jl/blob/main/LICENSE"

# Copy README.md from the project base folder and use it as the start page
open(readme_path, "r") do readme_in
    readme_string = read(readme_in, String)

    # replace relative links in the README.md
    readme_string = replace(readme_string, "[MIT](LICENSE)" => "[MIT]($(license_path))")

    open(index_path, "w") do readme_out
        write(readme_out, readme_string)
    end
end

pages = [
    "Home" => "index.md",
    "refs.md",
]

try
    # generate docs with Documenter.jl
    makedocs(;
        modules = [Vegas],
        checkdocs = :exports,
        authors = "Uwe Hernandez Acosta, Anton Reinhard",
        repo = Documenter.Remotes.GitHub("QEDjl-project", "Vegas.jl"),
        sitename = "Vegas.jl",
        format = Documenter.HTML(;
            prettyurls = get(ENV, "CI", "false") == "true",
            canonical = "https://qedjl-project.gitlab.io/Vegas.jl",
            assets = String[],
            mathengine = Documenter.MathJax2(),
            collapselevel = 1,
        ),
        pages = pages,
    )
finally
    # doing some garbage collection
    @info "GarbageCollection: remove generated landing page"
    rm(index_path)
end

deploydocs(; repo = "github.com/QEDjl-project/Vegas.jl.git", push_preview = false)
