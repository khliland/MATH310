using Random
Random.seed!(1234) # This line assures the same random dataset to be generated each time.

nn = 100;     # Generate 3 random "clouds" of datapoints (samples), each of size nn.
X = [randn((nn,2))*0.7 .+ [ 1  1];
    randn((nn,2))*0.7  .+ [-3 -5];
    randn((nn,2))*0.7  .+ [-4 -2]];
# Each row of X corresponds to a datapoint

using Plots #b Precompiles on every startup (~20 secondss)
gr() # Needs modules Plots and GR to be installed, may need a rebuild of GR with ']build GR'
default(size=(600, 400), fmt = :png) # Default plot size, change output format to png

# Define and display a plot of the raw random data
sp = scatter(X[:,1],X[:,2], title = "Random generated pointcloud 2D-data", legend = false)
display(sp)

include("mykmeans.jl")
k = 3;    # The suggested number of clusters (you should also repeatedly try k = 2, 4 and 5)
Cid, CS, J = mykmeans(X, k);

# Define and display a plot of the clustered data and the cluster centers(CS):
p = plot(title = "Clustering example with random generated 2D data",
    label = " ", legend = :bottomright, size = (600, 400))
for i=1:k
    snr = vec(Cid.==i) # the sample numbers of the j-th cluster
    scatter!(p, X[snr,1], X[snr,2], label = string("Cluster ",i))
end
scatter!(p, CS[:,1],CS[:,2], marker = :star, markersize = 8, color = :orange, label = "CS")
display(p)

# Define and display a plot of the objective function values reflecting the clustering process
Jp =plot(J, linestyle = :dashdot, title = "Objective function (J) values - monitoring the clustering process",
    ylabel = "J (the mean squared distace)", xlabel = "Interations", label = "J")
display(Jp)
## Visualization for 3D data
# Here is a corresponding 3-dimensional dataset:
Random.seed!(1234) # This line assures the same random dataset to be generated each time.
nn = 100;
X = [randn(nn,3)*0.7  .+ [1  1  1];
     randn(nn,3)*0.7  .+ [1 -2 -0];
     randn(nn,3)*0.7  .+ [0  0 -3]];

# Define and display a plot of the raw random data
sp = scatter(X[:,1],X[:,2],X[:,3], title = "Random pointcloud 3d-data", legend = false, camera = (75,10))
display(sp)

k = 3;    # The suggested number of clusters (you should also repeatedly try k = 3, 4 and 5)
Cid, CS, J = mykmeans(X, k);

##
# Plotting the clustered data and the cluster-centers:
colors = [:red, :blue, :green, :cyan, :magenta, :black];

# Define and display a plot of the clustered data and the cluster centers(CS):
p = plot(legend = :bottomright, title = "Clustering example with random generated 3D data", size = (600,400))
for j = 1:k
    snr = vec(Cid.==j); # the sample numbers of the j-th cluster
    # Plot only this group:
    scatter!(p, X[snr,1],X[snr,2],X[snr,3],color = colors[j], markersize = 5, label = string("Group ", j))
end
scatter!(p, CS[:,1],CS[:,2], CS[:,3], marker = :star, markersize = 12, color = :orange, label = "CS", camera = (75,10)) # Plotting the cluster centers
display(p)

# Define and display a plot of the objective function values reflecting the clustering process
Jp = plot(J, linestyle = :dashdot, title = "Objective function (J) values - monitoring the clustering process",
    ylabel = "J (the mean squared distace)", xlabel="Interation", label="J")
display(Jp)
