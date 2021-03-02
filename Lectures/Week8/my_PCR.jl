using LinearAlgebra
using Statistics
#using LinearAlgebra, Statistics
function my_PCR(X, y; mc = 1)
# b0, B = my_PCR(X,y; mc = 3)
# -------------------------------------------------------------------
# Principal Component Regression with data-matrix X and corresponding
# response vector y for models with up to mc principal components.
# -------------------------------------------------------------------
# This function returns:
# b0 - the constant terms (1×mc vector) of the PCR-models for up to mc components.
# B  - the PC regression coeffs (m×mc matrix) for the X-data for up to mc components.
m,n = size(X)
mc  = min(mc, min(n,m)-1) # Make sure that the maximum number of components is not too large.
x¯  = mean(X, dims=1) # - row vector of the X-column mean values.
y¯  = mean(y, dims=1) # - the mean of the response values y.
y0  = y.- y¯;         # - the centered response vector.
U, σ, V = svd(X.-x¯)  # - SVD of the centered X-data to find the principal components.
U = U[:,1:mc]; σ = σ[1:mc]; V = V[:,1:mc]; # Restriction to the first mc components.
q = (σ.^(-1).*(U'*y0));   # the regression coeffs for the PCA-scores.
B = cumsum(V.*q', dims=2) # the PCR-regression coeffs for the X-data based on up to mc components.
b0 = y¯ .- x¯*B;      # the correspondning constant terms for the PCR-models.
return b0, B;
end
