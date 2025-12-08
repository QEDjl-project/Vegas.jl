# Vegas

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://QEDjl-project.github.io/Vegas.jl/stable)
[![Development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://QEDjl-project.github.io/Vegas.jl/dev)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

## Contribution Guide

If this is the first time you work with this repository, follow the instructions below to clone the repository.

1. Fork this repo
2. Clone **your** repo (this will create a `git remote` called `origin`)
3. Add this repo as a remote:

   ```bash
   git remote add upstream https://github.com/QEDjl-project/Vegas.jl
   ```

This will ensure that you have two remotes in your git: `origin` and `upstream`.
You will create branches and push to `origin`, and you will fetch and update your local `main` branch from `upstream`.

## Linting and formatting

Install a plugin on your editor to use [EditorConfig](https://editorconfig.org).
This will ensure that your editor is configured with important formatting settings.

To format the Julia code, we are using [Runic.jl](https://github.com/fredrikekre/Runic.jl), so please install it globally first:

```julia-repl
julia --startup-file=no -e 'using Pkg; Pkg.add("Runic")'
```

We recommend to use [pre-commit](https://pre-commit.com).
To install `pre-commit`, we recommend using [pipx](https://pipx.pypa.io) as follows:

```bash
# Install pipx following the link
pipx install pre-commit
```

With `pre-commit` installed, activate it as a pre-commit hook:

```bash
pre-commit install
```

To run the linting and formatting manually, enter the command below:

```bash
pre-commit run -a
```

## Testing

As with most Julia packages, you can just open Julia in the repository folder, activate
the environment, and run `test`:

```bash
julia --project
```

will open a julia REPL and activate the environment for `Vegas.jl`. Then you can run the
tests locally by using the Pkg mode of the Julia REPL

```julia-repl
julia> # press ]
pkg> test
```

To run the tests for specific backends, you need to set the following environmental
variables:

```bash
TEST_<BACKEND> = 1 julia --project
```

for one of `BACKEND=[CPU, CUDA, AMDGPU, METAL, ONEAPI]`.

## Working on a new issue

We try to keep a linear history in this repo, so it is important to keep your branches up-to-date.

1. Fetch from the remote and fast-forward your local main

   ```bash
   git fetch upstream
   git switch main
   git merge --ff-only upstream/main
   ```

2. Branch from `main` to address the issue (see below for naming)

   ```bash
   git switch -c 42-add-answer-universe
   ```

3. Push the new local branch to your personal remote repository

   ```bash
   git push -u origin 42-add-answer-universe
   ```

4. Create a pull request to merge your remote branch into the org main.

### Before creating a pull request

- Make sure the tests pass locally.
- Make sure the formatter pass.
- Fetch any `main` updates from upstream and rebase your branch, if necessary:

  ```bash
  git fetch upstream
  git rebase upstream/main BRANCH_NAME
  ```

- Then you can open a pull request and work with the reviewer to address any issues.

## Building and viewing the documentation locally

Following the latest suggestions, we recommend using `LiveServer` to build the documentation.
Here is how you do it:

1. Run `julia --project=docs` to open Julia in the environment of the docs.
1. If this is the first time building the docs
   1. Press `]` to enter `pkg` mode
   1. Run `pkg> dev .` to use the development version of your package
   1. Press backspace to leave `pkg` mode
1. Run `julia> using LiveServer`
1. Run `julia> servedocs()`
