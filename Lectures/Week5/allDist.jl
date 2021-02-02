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

# ### All distances between a one row Array, v, and all rows of an Array, X

using LinearAlgebra
function allDist(X,v) # Calculate all distances between the X-rows and vector v.
    # INputs: X - data matrix containing N p-dimensional observations (rows)
    #         v - p-dimensional row vector
    # Output: d - vector of distances from the various rows of X to v.
    N = size(X,1)
    d = fill(NaN, (N,1))
    for i = 1:N
        d[i] = norm(X[i,:]-v)
    end
    return d
end

# Example:

runExample = false;

if runExample
    X = [1 2 1;
         3 1 4];
    v = X[1,:];
    print(allDist(X,v))
end
