using LinearAlgebra
using Statistics
function my_PCA(X; mc = 2)
# T, σ2, x¯, V, σ, U = my_PCA(X)
# ------------------------------------------------------------------
# Principal Component Analysis of data-matrix X (the input argument)
# ------------------------------------------------------------------
# The function returns
# T  - matrix of PCA-scores
# σ2 - the PCA-variances
# x¯ - row vector of the X-column mean values
# V  - the PCA-loadings
# U  - the normalized prinicpal components
m,n = size(X)
mc  = min(mc, min(n,m)-1)  # Make sure that the maximum number of components is not too large.
x¯  = mean(X, dims=1)
U, σ, V = svd(X.-x¯); U = U[:,1:mc]; σ = σ[1:mc]; V = V[:,1:mc]; # Restriction to the first mc components.
T   = U.*σ'      # the PCA-scores
σ2  = (σ.^2)./m  # the PCA-variances
return T, σ2, x¯, V, σ, U;
end
