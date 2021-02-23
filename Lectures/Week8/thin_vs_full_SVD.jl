#
# This is a note regarding the full SVD vs the thin SVD:
#
# ------------------------------------------------------
m = 25; n = 8
A = rand(m, n) # Generate m×n sample matrix A
# ------------------------------------------------------
using LinearAlgebra
# The SVD-function returs the singular values (σ) in an n-vector:
Uf, σ, V = svd(A; full = true)
Σth = diagm(σ); # The square (n×n) diagonal version of Σ
Σf   = [Σth; zeros(m-n,n)]; # Make the full m×n-version of Σ
# Here Uf, Σf and V represent the matrices of the full SVD:
A   ≈ Uf*Σf*V'
# ------------------------------------------------------
Uth = Uf[:,1:n]
# Here Uth, Σth and V represent the matrices of the thin SVD:
A ≈ Uth*Σth*V'
# Note that
Uf*Σf ≈ Uth*Σth
# because of the m×n-matrix (N) of 0's that is
# unnecessarily computed in the full SVD:
N = Uf[:,n+1:m]*Σf[n+1:m,:]
# Algebraically we therefore always have: UΣ = Uth*Σth + N = Uth*Σth
# ------------------------------------------------------
# To obtain the thin SVD directly we just skip the argument "full = true":
U, σ, V = svd(A)
