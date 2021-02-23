# Condition numbers and consequences:
# -----------------------------------

# Lets consider the coefficient matrix:
A = [10 7 8 7; 7 5 6 5; 8 6 10 9;  7 5 9 10];
U, σ, V = svd(A) # SVD of A
# opnorm(pinv(A))

# In Julia we can compute the condition number of A by eiter
ct = σ[1]/σ[end]
# or
cond(A)

# Lets solve Ax=b wrt x for
b =[32 23 33 31]';
x = A\b

# Now, consider Ax0 = b0, where b0 is the following perturbation of b:
b0 =[32.1 22.9 33.1 30.9]';
x0 = A\b0

# Note that the two vectors b and b0 are quite similar, so that one would expect
# the corresponding solution vectors x and x0 to be quite similar (?):
norm(b-b0)
norm(x-x0)
# As we can see, the latter is not the case, and we must conclude that
# relatively small errors in b in this particular case implies drastical
# changes in the solution (by comparing x0 to x).

# This phenomenon is a consequence of A beeing poorly conditioned.

# ------------------------------------------------------------------------------

# The following matrix AA is more well-conditioned:
AA = [10 8 5 2; 8 5 6 5; 5 6 10 9;  2 5 9 10];
U1, σ1, V1 = svd(AA) # SVD of A
# The condition number for AA:
ct1 = σ1[1]/σ1[end]
# alternatively
cond(AA)
# is much smaller than for the above A:
cond(A)

# Lets solve AAx=b wrt x for the above
b =[32 23 33 31]';
xx = AA\b
# and for AAx0 = b0, where b0 is is the same perturbation of b
b0 =[32.1 22.9 33.1 30.9]';
xx0 = AA\b0

# The two vectors b and b0 are quite similar,
norm(b-b0)
# and because AA is more well conditioned
# the two solution vectors xx and xx0 are now correspondingly similar:
norm(xx-xx0)
