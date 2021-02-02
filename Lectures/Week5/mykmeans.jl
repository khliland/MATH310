include("allDist.jl")
using Random
function mykmeans(X, k; tol = 1e-5)
#function [Cid, Ccenters, J] = mykmeans(X,k; tol = 1e-5) # (MATLAB-name)
# ------------- Our implementation of the k-means algorithm -------------------
# INPUT:
# X        - data matrix (observed datapoints are rows in X).
# k        - the number of clusters (an integer >= 2, typically 2 to something smaller than the number of rows in X).
# -----------------------------------------------------------------------------
# OUTPUT:
# Cid      - vector of final cluster labels (1,..,k) assigned to each datapoint.
# C        - matrix of final cluster centers (given as rows)
# J        - objective function values describing the clustering process (J small means good clustering)
# cs       - the sizes of each cluster
# -----------------------------------------------------------------------------
J = [];                           # objective function values describing the mean squared distances of the clustering process.
N = size(X,1);                    # number of samples (rows) in tha data matrix
D2 = Array{Float64}(undef, N, k); # Matrix for storing distances between samples and cluster centers. #D2 = zeros(N, k); %D2 = nan(N,k);
Cid = zeros(N, 1);                # Vector of the cluster assignments
cs = zeros(k,1);                  # Vector of the k cluster sizes
msdnew = 0; msdold = -1;          # msdnew and msdold represent objective function values - will be updated during the clustering process.
ids = randperm(N)[1:k];           # Generate k random integers from [1, N] for choosing the initial cluster centers from X.
C   = copy(X[ids,:]);             # The initial cluster centers found by the random rows identified by "ids" above.
# -----------------------------------------------------------------------------

while abs(msdnew-msdold) > tol    # Repeat until convergence of the objective function values
    # Here we calculate all the distances between X-rows and the cluster centers
    for i = 1:k
        D2[:,i] = allDist(X,C[i,:]).^2; # Squared euclidean distances of all samples to i-th cluster center.
    end

    # Identify shortest distance and corresponding cluster number for each observation:
    minD2, Cid = findmin(D2, dims=2); Cid = getindex.(Cid,2) # convert from CartesianIndex to column number

    msdold = msdnew;            # keep the old objective function value in msdold
    msdnew = sum(minD2)/N       # compute the new (updated) objective function value in msdnew (as mean of squared distances)
    J      = vcat(J,msdnew);    #J = [J; msdnew];

    # Update the cluster centers based on the labelling of Cid:
    for i = 1:k
        rows_i = vec(Cid .== i);  # Find row-numbers of the i-th cluster members. Alternative syntax: #rows_i = findall(x->x==i,Cid)
        cs[i]  = sum(rows_i)      # Number of samples in cluster i.
        if cs[i]>0                # Update if i-th cluster is non-empty.
        C[i,:] = sum(X[rows_i,:],dims = 1)./cs[i] # Update cluster centers as the mean of the cluster members.
        end
    end
end

return Cid, C, J, cs;

end
