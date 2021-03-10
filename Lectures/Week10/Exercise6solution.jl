"
Exercise 6 (week 10):
"
using LinearAlgebra
include("CGS.jl")
include("MGS.jl")
ϵ = 1e-7;
A = [1 1 1; ϵ 0 0; 0 ϵ 0; 0 0 ϵ];
x = ones(3,1);
b = A*x;
"
Solve the system Ax = b by the
 - normal equations (A^tA)x=(A^t)b.
 - QR-factorization obtained from the CGS.
 - QR-factorization obtained from the MGS.
 - QR-factorization in Julia.
 - backslash operator.
 - pseudo-inverse of A.
"
x1 = inv(A'A)*(A'b);
Qc,Rc = CGS(A); x2 = Rc\Qc'b; Ic = Qc'Qc;
Qm,Rm = MGS(A); x3 = Rm\Qm'b; Im = Qm'Qm;
Q,R = qr(A); Q = Matrix(Q); x4 = R\Q'b; I = Q'Q;
x5 = A\b;
x6 = pinv(A)*b;

# Compare and comment on the results.
