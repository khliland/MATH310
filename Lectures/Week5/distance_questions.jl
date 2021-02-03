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
dw = (X0.-w)
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
 # 1)
  n2X0 = sum(X0.^2,dims =2);

 # 2)
  X0v = X0*v;
  vtv = v'*v;
# Now the squared distances become (see line 18):
d3 = (n2X0 - 2*X0v).+vtv;

 ## Some extra comments:
 # The squared cosine values becomes
 cos2vals = (X0v.^2)./(n2X0.*vtv);

 # Vector similarity corresponds to small angles,
 # that can be measured by the squared sine values
 sin2vals = 1 .- (X0v.^2)./(n2X0.*vtv);  # = 1 - cos2vals

 # Note that in the k-means algorithm we calculate squared distances
 # (or the squared cosine of the angles) between the X0-rows and
 # numerous cluster center candidates v.

 # ----------------------------------------------------------------
 # For all these calculations we can consider the squared norms (n2X0)
 # of all the X0-rows as fixed (JUST ONE single calculation of n2X0 is required).
 #
 # For each vector v just one single calculation of the inner products X0v = X0*v is required,
 # together with just one calculation of vtv = v'*v.

 # Try to use these insights to suggest computationally efficient implementations of the k-means algorithm
