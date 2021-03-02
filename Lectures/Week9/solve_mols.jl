function solve_mols(As,bs,λs)
# xλ = mols_solve(As,bs,λs)
# ----------------------------------------------------------------------------
# Function for solving a multi-obj. least squares (mols) problem with inputs:
# As - vector array of coeff. matrices.
# bs - corresponding vector array of right hand side vectors.
# λs - weighting paramters for the individual systems of the mols problem.
# ----------------------------------------------------------------------------
# The function returns:
# xλ - the mols solution for the weighting configurations λs.
# ----------------------------------------------------------------------------
k      = length(λs); # The number of linear systems in the mols-problem
Astack = vcat([sqrt(λs[i])*As[i] for i=1:k]...) # Stacking coeff. matrices
bstack = vcat([sqrt(λs[i])*bs[i] for i=1:k]...) # Stacking right hand sides
xλ     = Astack \ bstack                        # The mols solution
return xλ
end
