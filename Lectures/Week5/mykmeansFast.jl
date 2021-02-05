using Random
function mykmeansFast(X, k; tol = 5e-3)
# ---------- A faster implementation of the k-means algorithm -----------------
# INPUT: ----------------------------------------------------------------------
# X        - data matrix (observed datapoints are rows in X).
# k        - the number of clusters (an integer >= 2, typically 2 to something smaller than the number of rows in X).
# OUTPUT: ---------------------------------------------------------------------
# Cid      - vector of final cluster labels (1,..,k) assigned to each datapoint.
# C        - matrix of final cluster centers (given as rows)
# J        - objective function values describing the clustering process (J small means good clustering)
# cs       - the sizes of each cluster
# -----------------------------------------------------------------------------
J = [];                           # objective function values describing the mean squared distances of the clustering process.
N = size(X,1);                    # number of samples (rows) in tha data matrix
Cid = zeros(N, 1);                # Vector of the cluster assignments
cs  = zeros(k,1);                 # Vector of the k cluster sizes
Jcurrent = 0; Jprevious = -1;     # Jcurrent and Jprevious represent objective function values - will be updated during the clustering process.
C  = copy(X[randperm(N)[1:k],:]); # Initial cluster centers selected as k random rows of X.
X2 = sum(X.^2, dims=2);           # Squared norms of the X-rows
iter = 1
while abs((Jcurrent-Jprevious)/Jcurrent) > tol # Repeat until convergence of the objective function values
    # Calculation of all the distances between X-rows and the cluster centers in C:
    D2 = (X2.+sum(C.^2,dims=2)')-2*(X*C') #D2 = avstander(X, C);
    # Identify shortest distance and corresponding cluster number for each sample:
    minD2, Cid = findmin(D2, dims=2); Cid = getindex.(Cid,2) # convert from CartesianIndex to column number
    Jprevious = Jcurrent; Jcurrent = sum(minD2)/N # update old and new objective function values
    J      = vcat(J,Jcurrent);      #J = [J; Jcurrent]; # Collect J-value of the current iteration.
    # Update the cluster centers based on the labelling of Cid:
    for i = 1:k
        rows_i = getindex.(findall(x -> x == i, Cid),1) # Find row-numbers of the samples in the i-th cluster.
        #rows_i = [j for j=1:N if Cid[j] == i]
        cs[i]  = length(rows_i)   # Number of samples in cluster i.
        if cs[i]>0                # Update if i-th cluster is non-empty.
        C[i,:] = sum(X[rows_i,:],dims = 1)./cs[i] # Update cluster centers as the mean of the cluster members.
        end
    end, println("Iteration ", iter, ": Jclust = ", Jcurrent, ".");  iter +=1
end, return Cid, C, J, cs;
end
