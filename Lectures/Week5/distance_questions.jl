## Calculation of squared distances:
include("allDist.jl")
nn = 100;     # Generate 3 random "clouds" of data points (samples), each of size nn.
X0 = [randn((nn,2))*0.7  .+ [ 1  1];
      randn((nn,2))*0.7  .+ [-3 -5];
      randn((nn,2))*0.7  .+ [-4 -2]];

v = X0[10,:]     # Note: the extracted X-row becomes a column vector
d1 = allDist(X0,v).^2; # squared distances between the X-rows and v

w = X0[[10],:]; # Note: the by using additional brackets the extracted X-row remains a row vector
d2 = sum((X0.-w).^2, dims =2); # this is also the squared distances

## Regarding measuring similarity by angles:
# First note that the squared distance
# ||x-v||^2 = (x-v)'*(x-v)
#           = x'*x - 2*x'*v + v'*v
#           = ||x||^2 - 2*x'*v + ||v||^2,
# this means that the squared distance can be found
# from the squared norms and the inner product between x and v.

# From the cosine-definition of the angle between x and v, the squared cosine becomes
# cos2 = (|x'*v|^2)/((x'*x)*(v'*v))
#      = (|x'*v|^2)/(||x||^2 * ||v||^2),
# this means that the squared cosine also can be found
# from the squared norms and the inner product between x and v.

# ----------------------
## Questions:
# 1) How do we calculate the squared norms (n2X0) of all the X0-rows quickly?
# 2) How do we calculate the inner products between all X0-rows and v quickly?
