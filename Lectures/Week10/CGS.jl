using LinearAlgebra
function CGS(A)
# QR-factorizarion of input-matrix A by the classical Gram Schmidt algorithm (CGS).
# The function returns Q and R in the QR-factorizarion.
m, n = size(A);
p = min(m,n)
Q = zeros(m,p);
R = zeros(p,n);
R[1,1] = norm(A[:,1]);
Q[:,1] = A[:,1]./R[1,1];
for i = 2:n
    if i <= p
    R[1:i-1,i] = Q[:,1:i-1]'A[:,i];
    v = A[:,i]-Q[:,1:i-1]*R[1:i-1,i]; # Orthogonalize i-th column wrt Q[:,1:i-1].
    R[i,i] = norm(v);
    Q[:,i] = v./R[i,i];
    else
        R[:,i] = Q'A[:,i];           # Find Q-coordinates of A[:,i]
    end
end
return Q, R;
end
