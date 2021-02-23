using LinearAlgebra
function my_pinv(A)
# pinvA = my_pinv(A)
# ------------------------------------------------------------------
# Pseudo-inverse of matrix A (the input argument)
# ------------------------------------------------------------------
# The function returns
# pinvA  - the pseudo-inverse of the input-matrix A.
U, σ, V = svd(A)
pinvA = (V.*(σ.^(-1))')*U';
return pinvA
end
