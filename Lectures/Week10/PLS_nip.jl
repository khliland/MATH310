using LinearAlgebra, Statistics
function PLS_nip(X, y; mc = 2)
# b0, B, T, W, P, q = PLS_nip(X,y; mc = 3)
"
# -------------------------------------------------------------------
# NIPALS algorithm for Partial Least Squares Regression (PLSR) with
# data-matrix X and corresponding response vector y for models with
# up to mc PLS components.
# Reference: 'Fast and stable partial least squares modelling: A benchmark study with theoretical comments'
#             - https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/cem.2898
# Inputs:
# X  - data-matrix.
# y  - corresponding response vector.
# mc - the maximal number of PLS components to be extracted.
# -------------------------------------------------------------------
# The function returns:
# b0 - the constant terms (1×mc vector) of the PLS-models for up to mc components.
# B  - the PLS regression coeffs (m×mc matrix) for the X-data for up to mc components.
# T  - matrix of PLS-scores.
# W  - matrix of PLS-weights.
# P  - matrix of PLS-loadings.
# q  - vector of regression coeffs for the PLS scores.
"
m,n = size(X)
mc  = min(mc, min(n,m)-1) # Assure that the number of extracted components is consistent with the size of the problem.
T   = zeros(m,mc); W   = zeros(n,mc); P   = zeros(n,mc)
q   = zeros(1,mc);    # - the regression coeffs for the PLS-scores.
x¯  = mean(X, dims=1) # - row vector of the X-column mean values.
y¯  = mean(y, dims=1)[1] # - the mean of the response values y.
y  = y.- y¯;         # - the centered response vector.
X  = X.- x¯;         # - the centered X-data
for a = 1:mc
    w = X'y;  w = w/norm(w);  W[:,a] = w;
    t = X*w;  t = t/norm(t);  T[:,a] = t;
# ------------------- Deflate X and y ----------------------
    P[:,a] = X't;         X = X - t*P[:,a]';
    q[a]   = (y't)[1];    y = y - q[a].*t;
end
# ---------- Calculate regression coefficients -------------
B  = cumsum((W/triu(P'W)).*q, dims = 2); # the PLS-regression coeffs for the X-data based on up to mc components.
#B  = cumsum((W/Bidiagonal(P'W, :U)).*q, dims = 2); # the PLS-regression coeffs for the X-data based on up to mc components.
b0 = y¯ .- x¯*B;      # the correspondning constant terms for the PLS-models.
return b0, B, T, W, P, q;
end
