using LinearAlgebra, Statistics, SparseArrays
function my_RR(X, y; λ = 1)
# β0, β = my_RR(X,y; λ = 3)
# -------------------------------------------------------------------
# Ridge Regression with data-matrix X and corresponding
# response vector y for regularization parameter value λ > 0.
# -------------------------------------------------------------------
# This function returns:
# β0 - the constant term (a number) of the RR-model.
# β  - the vector of RR regression coeffs.
m,n = size(X)
x¯  = mean(X, dims=1) # - row vector of the X-column mean values.
y¯  = mean(y, dims=1) # - the mean of the response values y.
y0  = y.- y¯;         # - the centered response vector.
X0 = (X.- x¯);        # - the centered data matrix.
if m > n # X is tall - solve the stacked system.
    β = [X0; sqrt(λ)*speye(n)]\[y0; zeros(n,1)];
else # X is wide - use the kernel trick:
    β = X0' * ((X0*X0' + λ*speye(m))\y0);
end
β0 = y¯ .- x¯*β;      # the constant term for the RR-model.
return β0[1], β;
end
