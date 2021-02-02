# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Julia 1.5.3
#     language: julia
#     name: julia-1.5
# ---

# # Code snipets
# ## Table of contents
# 1. [Differences from R/Python/MATLAB](#differentfrom)
# 2. [Print contents of a function](#printfunction)

# ## Differences from R/Python/MATLAB <a name="differentfrom"></a>

# Element-wise operations using period:
a = [1 3 4];
b = [1 3 5];
a .== b

# ## Print contents of a function <a name="printfunction"></a>
# ### First time installation

using Pkg
Pkg.add("CodeTracking")

# ### Requirements
# - _Function_ made available by include() / using, or from the base package.
# - _Arguments_ ready for inputing to the function

using CodeTracking
print(@code_string sum(1:3))

include("myFunc.jl")
print(@code_string double_this(3))
