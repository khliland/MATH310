using LinearAlgebra
function MGS(B)
# QR-factorizarion of input-matrix B by the modified Gram Schmidt algorithm (MGS).
# The function returns Q and R in the QR-factorizarion.
A = copy(B); # Make local copy (A) of the input-matrix.
m, n = size(A); p = min(m,n);
Q = zeros(m,p); R = zeros(p,n); # Matrices for storing Q and R
for k=1:p
    R[k,k]     = norm(A[:,k]);
    q = A[:,k]./R[k,k];
    R[k,k+1:n] = q'A[:,k+1:n]; # q-coordinates of the columns A[:,k+1:n] as k-th row in R.
    # MGS-step - deflate A[:,k+1:n] by subtracting its projection onto q:
    A[:,k+1:n] = A[:,k+1:n] - q*R[k,k+1:n]';
    Q[:,k] = q;
end
return Q, R;
end
