# include("allDist.jl") # Slow calc. of distances
using Random
function mykmeans(X, k; tol = 5e-3)
# ------------- Our implementation of the k-means algorithm -------------------
# INPUT: ----------------------------------------------------------------------
# X        - data matrix (observed datapoints are rows in X).
# k        - the number of clusters (an integer >= 2, typically 2 to something smaller than the number of rows in X).
# OUTPUT: ---------------------------------------------------------------------
# Cid      - vector of final cluster labels (1,..,k) assigned to each datapoint.
# C        - matrix of final cluster centers (given as rows)
# J        - objective function values describing the clustering process (J small means good clustering)
# cs       - the sizes of each cluster
# -----------------------------------------------------------------------------
J = [];                              # objective function values describing the mean squared distances of the clustering process.
N      = size(X,1);                  # number of samples (rows) in tha data matrix
D2     = Array{Float64}(undef, N, k);# Matrix for storing distances between samples and cluster centers. #D2 = zeros(N, k); %D2 = nan(N,k);
Cid    = zeros(N, 1);                # Vector of the cluster assignments
cs     = zeros(k,1);                 # Vector of the k cluster sizes
Jcurrent = 0; Jprevious = -1;     # Jcurrent and Jprevious represent objective function values - will be updated during the clustering process.
ids = randperm(N)[1:k];           # Generate k random integers from [1, N] for choosing the initial cluster centers from X.
C   = copy(X[ids,:]);             # The initial cluster centers found by the random rows identified by "ids" above.
iter = 1                          # For counting the number of cluster iterations
while abs((Jcurrent-Jprevious)/Jcurrent) > tol    # Repeat until convergence of the objective function values
    # Here we calculate all the distances between X-rows and the cluster centers
    for i = 1:k
        #D2[:,i] = allDist(X,C[i,:]).^2; # Squared euclidean distances of all samples to i-th cluster center.
        D2[:,i] = sum((X.-C[[i],:]).^2, dims =2) # Also the squared euclidean distances of all samples to i-th cluster center.
    end
    # Identify shortest distance and corresponding cluster number for each observation:
    minD2, Cid = findmin(D2, dims=2); Cid = getindex.(Cid,2) # convert from CartesianIndex to column number
    Jprevious = Jcurrent; Jcurrent = sum(minD2)/N # update old and new objective function values
    J      = vcat(J,Jcurrent);    #J = [J; Jcurrent];
    # Update the cluster centers based on the labelling of Cid:
    for i = 1:k
        rows_i = getindex.(findall(x -> x == i, Cid),1) # Find row-numbers of the i-th cluster members.
        cs[i] = length(rows_i)      # Number of samples in cluster i.
        if cs[i]>0                # Update if i-th cluster is non-empty.
        C[i,:] = sum(X[rows_i,:],dims = 1)./cs[i] # Update cluster centers as the mean of the cluster members.
        end
    end, println("Iteration ", iter, ": Jclust = ", Jcurrent, ".");  iter +=1
end, return Cid, C, J, cs;
end
