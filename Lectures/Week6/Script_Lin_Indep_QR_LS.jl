using LinearAlgebra, VMLS

"""
# See Ch 5 - Linear independence, Ch 10.4 - QR-factorization and 12 - Least squares
# in https://web.stanford.edu/~boyd/vmls/vmls-julia-companion.pdf
"""
# -------------------------------
# 5.1 Cash flow replication (VMLS, page 93):
r = 0.05;
e1 = [1,0,0]; l1 = [1,-(1+r),0]; l2 = [0,1,-(1+r)];
"""
# The first vector e1 is a single payment of 1 dollar in period (time) t = 1.
# The second vector l1 is loan of 1 dollar in period t = 1, paid back in
# period t = 2 with interest r. The third vector l2 is a loan of 1 dollar
# in period t = 2, paid back in period t = 3 with interest r.

# We would like to express the cash-flow vector c = (1 2 -3) as a linear
# combination of the basis vectors e1, l1 and l2.
"""
c = [1,2,-3];
# The required coefficients alpha1, alpha2 and alpha3 can be calculated manually
# as:
alpha3 = -c[3]/(1+r);
alpha2 = -c[2]/(1+r) - c[3]/(1+r)^2;
alpha1 = c[1] + c[2]/(1+r) + c[3]/(1+r)^2 # NPV of cash flow
# Verification of c as the the result of
alpha1*e1 + alpha2*l1 + alpha3*l2
# -------------------------------
# The above problem correpsonda to solving the associated linear system in
# matrix-vector representation M*α = c.
# Such systems can be solved by the "backslash"-operator:
M = [e1 l1 l2]
α = M\c

alpha = [alpha1; alpha2; alpha3];
# Check that the two vectors α and alpha are equal by verifying that the distance
# between them (the norm ||α - alpha||) is equal to 0:
checkdist = norm(α - alpha)

# Finally check that c = M*α:
checkdist = norm(c-M*α)
# -------------------------------


# 5.3 Orthonormal vectors
# Define
a1 = [0,0,-1]; a2 = [1,1,0]/sqrt(2); a3 = [1,-1,0]/sqrt(2);
A = [a1 a2 a3] # A has orthogonal columns of unit norm, just check by:
AtA = A'*A
# -------------------------------
# Let
x = 1:3;
# Solve the system Aβ = x:
β = A\x
# Check that the soution is correct:
checkdist = norm(x-A*β)
# -------------------------------

# 10.4 QR factorization
"""
# In Julia, the QR factorization of a matrix A can be found by qr(A), which
# returns a tuple with the Q and R factors. Note that the matrix Q is not returned
# as an array, but in a special compact format. It can be converted to a regular
# matrix variable using the command Matrix(Q). Hence, the QR factorization as
# defined in VMLS is computed by a sequence of two commands:
# julia> Q, R = qr(A);
# julia> Q = Matrix(Q);
"""
# -------------------------------
A = randn(6,4); # A random matrix of numbers for testing the QR-factorization
Q, R = qr(A); Q = Matrix(Q)  # QR
QR = Q*R;
cmp_matrices = norm(A-QR) # Check if the two matrices are equal
# -------------------------------


# 11 Matrix inverse and pseudo-inverse via QR:
# -------------------------------
# Matrix inverse - we generate a random square matrix:
A = randn(5,5); # A is a square Matrix
Q, R = qr(A); Q = Matrix(Q);
Ainv = R\Q'; # The inverse of A can be calculated by the backslash operator (R\Q')
A*Ainv
Ainv*A
# -------------------------------
# Matrix pseudo-inverse:
A = [-3 -4; 4 6; 1 1] # A rectangular matrix
pinvA = pinv(A) # the pseudo-inverse of A

Q, R = qr(A); Q = Matrix(Q)
pinv2A = R\Q';

checkpinv = pinv2A*A
cmp_pinvs = norm(pinvA-pinv2A) # The two pseudo-inverses are identical
# -------------------------------

# 12 Least squares problems
# -------------------------------
# We generate a random matrix of 3 rows and 3 columns:
A = [ 2 0 ; -1 1 ; 0 2 ]
b = [1; 0; -1]

# The system A*b = b is inconsistent but we can find the least squares solution
# in Julia in several ways:

# 1) - backslash
x1 = A\b

# 2) - by solving the norma equations (A'*A)*x = A'*b
x2 = inv(A'*A)*(A'*b) # alternatively by x2 = (A'*A)\A'*b,

# 3) - by the QR-factorization
Q, R = qr(A); Q = Matrix(Q)
x3 = R\(Q'*b)
# -------------------------------

# Example 12.4 (Advertising purchases, see page 234 in VMLS)
R = [ 0.97 1.86 0.41; # Impressions per 1$ spent for 3 advertising channels in 10 demographic groups
1.23 2.18 0.53;
0.80 1.24 0.62;
1.29 0.98 0.51;
1.10 1.23 0.69;
0.67 0.34 0.54;
0.87 0.26 0.62;
1.10 0.16 0.48;
1.92 0.22 0.71;
1.29 0.12 0.62];
# -------------------------------
m, n = size(R);
vdes = 1e3 * ones(m) # The desires number of impressions for each group.
# By solving the system R*s = vds with respect to the budget allocation
# vector s we can find the "optimal" (recommended) spendings:
s = R\vdes
# -------------------------------
