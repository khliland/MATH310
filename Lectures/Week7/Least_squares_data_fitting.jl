import Random
using LinearAlgebra, SparseArrays, VMLS, Plots
"""
# Least squares data fitting - Ch. 13 in VMLS
"""

"""
# Straight-line fit:
# ------------------
# Petroleum consumption for the years 1980-2013 in thousand barrels/day by a
# straight-line fit to time series data gives an estimate of a trend line.
# In Figure 13.3 of VMLS we apply this to a time series of petroleum
# consumption. The figure is reproduced by the code below.
"""
consumption = petroleum_consumption_data() # The petroleum consumption data for n = 34 years
n = length(consumption);                   # The number of years with available data
A = [ones(n) 1:n ];                        # Coeff. matrix for fitting the constant and linear terms
x = A \ consumption;                       # Regression coeffs of the linear fit
pconsmp = scatter(1980:2013, consumption, legend=true, title = "Petroleum consump: Linear fit", xlabel="years", ylabel=" In units 1000 barrels/day", label = "Real data")
pconsmp = plot!(1980:2013, A*x, label = "fitted line") # Update pconsmp-plot with fitted line.
display(pconsmp)                           # Display the plot


"""
# Estimation of trend and seasonal component:
# -------------------------------------------
# The next example shows the least squares fit of a linear trend and a periodic
# component to a time series. In VMLS this is illustrated with a time
# series of vehicle-miles traveled in the US, per month, over 15 years
# (2000–2014). The below code generates Figure 13.5 in VMLS.
# Data are imported via the function vehicle_miles_data into a 15×12 matrix
# vmt, with the monthly values for each of the 15 years.
"""
vmt = vehicle_miles_data(); # creates the 15x12 matrix vmt of monthly data for 15 years
n,p = size(vmt)
m = n*p;         # n = 15, p = 12
A = [0:(m-1) vcat([eye(12) for i=1:15]...) ]; # Coeff. matrix for representing the linear and seasonal trends
y = reshape(vmt', m, 1); # reshape vmt-matrix into a time series
θ = A\y;         # Solve the system  Aθ = y to find the time-series model.
tsplt = scatter(1:m, y, markersize = 1.6, legend = true, label = "Real data", title = "Vehicle-miles traveled in the US"); # scatter plot of the real data
tsplt = plot!(1:m, A*θ, xlabel = "Month", ylabel = "Miles (millions)", label = "fitted ts-model") # Update tsplt-plot with fitted model.
display(tsplt)   # Display the plot

"""
# The matrix A in this example has size m × n where m = 15 · 12 = 180 and n = 13.
# The first column has entries 0, 1, 2, . . . , 179 for the linear trend. The remaining
# columns are formed by vertical stacking of 15 identity matrices of size 12 × 12.
# The Julia expression vcat([eye(12) for i=1:15]...) creates an array of 15 identity matrices,
# and then stacks them vertically to account for the priodicity.
"""


"""
# Least squares polynomial fit:
# -----------------------------
# We consider the polynomial fitting problem on page 255 in VMLS and the results
# simlar to those shown in Figure 13.6. We first generate a training set of
# 100 points in the interval [-1, 1] for plotting:
"""
m = 100;
t = -1 .+ 2*rand(m,1);   # Simulated raw data
y = t.^3 - t + 0.4 ./ (1 .+ 25*t.^2) + 0.10*randn(m,1); # Cubic transformation with random noise
pltsim = scatter(t, y, legend=true, label = "Raw data", title = "Simulated raw data")
display(pltsim)
"""
# Next we define a function that fits the polynomial coefficients using least squares.
# We apply the function "polyfit" to fit polynomials of degree 2, 6, 10, 15
# to our simulated dataset:
"""
polyfit(t, y, p) = vandermonde(t, p+1)\y # Note that vandermonde(t, p+1) generates a coeff. matrix
                                         # for fitting polynomial trends up to the degreee p.
theta2  = polyfit(t,y,2)
theta6  = polyfit(t,y,6)
theta10 = polyfit(t,y,10)
theta15 = polyfit(t,y,15)
"""
Finally, we plot these polynomials. To simplify this, we first define a function
that evaluates a polynomial at all points specified in a vector x:
"""
polyeval(theta, x) = vandermonde(x,length(theta))*theta;
# Matrix of scatter-plots for the diffrent polynomial fits:
t_vals = linspace(-1,1,1000); # 1000 equally spaced points in [-1, 1].
pltpol = plot(layout=4, legend=false, ylim=(-0.7, 0.7), title = "Polynomial fits")
pltpol = scatter!(t, y, subplot=1, markersize = 2)
pltpol = plot!(t_vals, polyeval(theta2,t_vals), subplot=1, title = "Degree 2")
pltpol = scatter!(t, y, subplot=2, markersize = 2)
pltpol = plot!(t_vals, polyeval(theta6,t_vals), subplot=2, title = "Degree 6")
pltpol = scatter!(t, y, subplot=3,markersize = 2)
pltpol = plot!(t_vals, polyeval(theta10,t_vals), subplot=3, title = "Degree 10")
pltpol = scatter!(t, y, subplot=4, markersize = 2)
pltpol = plot!(t_vals, polyeval(theta15,t_vals), subplot=4, title = "Degree 15")
display(pltpol)


"""
# Piecewise-linear fit:
# ---------------------
# In the following code, least squares modelling is used to fit a piecewise
# linear function to 100 points, see page 256-257 in VMLS. It produces a
# figure similar to Figure 13.8 in VMLS.
"""
m = 100;
x = -2 .+ 4*rand(m,1); # Generate noisy random datapoints for piecewise linear modelling
y = 1 .+ 2*(x.-1) - 3*max.(x.+1,0) + 4*max.(x.-1,0)+ 0.3*randn(m,1);

# Show scatter plot of the data:
rndplt = scatter(x, y, legend = true, label = "Random data", title = "Random data for piecewise linear modelling")
display(rndplt)

# Least squares fitting of piecewise linear model
X = [ones(m,1) x max.(x.+1,0) max.(x.-1,0)] # X-columns contains 4 basis function features
θ = X\y;

# Plot resulting model:
t = -2.1:0.1:2.1; #t = [-2.1, -1, 1, 2.1];
n = length(t)
T = [ones(n,1) t max.(t.+1,0) max.(t.-1,0)]
yhat = T*θ        # the fitted values from the piecewise linear model
rndplt = plot!(t, yhat, label = "Picewise lin. mod.")
display(rndplt)

"""
# House price regression:
# -----------------------
# We calculate the simple regression model for predicting house sales price (y) from
# area and number of bedrooms, using the data from 774 house sales in Sacramento.
# See page 258 in VMLS.
"""
D = house_sales_data(); # Extract 3 vectors corresponding to the features: area, beds, price
area  = D["area"];
beds  = D["beds"];
y     = D["price"];
m = length(y);          # The number of samples
A = [ ones(m) area beds ];
x = A\y   # The regression coeffs.
Ax = A*x  # The fitted evaluates
residual = y-Ax
rms_error = sqrt(residual'*residual/m) # = rms(y - A*x)

μ = sum(y)/m  # μ - mean value of y
ys = y.-μ     # ys - the mean centered y
std_prices = sqrt(ys'*ys/m) # = stdev(y) # the standard deviation of y (same as the rms of ys).
EV = 100*(1-(rms_error^2)/(std_prices^2)) # the % variance explained by the linear model is 56, considerably better than guessing the average price.


"""
# Auto-regressive time series model:
# ----------------------------------
# In the following Julia code we fit an autoregressive model to the temperature
# time series discussed on page 259 of VMLS. In the plot (see Figure 13.9 in VMLS)
# we compare the first five days of the model predictions with the actual data.
"""
t = temperature_data(); # import time series of temperatures t
N = length(t)
stdev(t) # Standard deviation of t: 3.05055928562933
rms(t[2:end] - t[1:end-1])   # RMS error for simple predictor zhat_{t+1} = z_t : 1.1602431638206119
rms(t[25:end] - t[1:end-24]) # RMS error for simple predictor zhat_{t+1} = z_{t-23} 1.7338941400468744

# Least squares fit of AR predictor with memory M
M = 8
y = t[M+1:end];
A = hcat([ t[i:i+N-M-1] for i = M:-1:1]...); # A = hcat(ones((N-M),1), [ t[i:i+N-M-1] for i = M:-1:1]...); # ?Also include intercept?
θ = A\y;     # Least squares solution of the system Aθ = y based on M previous temperature
ypred = A*θ; # recordings in A for predicting the subsequent temp. recording in y.
rms(ypred - y) # RMS error of LS AR fit: 1.0129632612687514

# Plot first five days
Nplot = 24*5
tsplt = scatter(1:Nplot, t[1:Nplot], legend =true, label = "Real data", title = "Auto-regressive model with "*string(M)*" coeffs")
tsplt = plot!(M+1:Nplot, ypred[1:Nplot-M], label = "Predictions", xlabel = "hrs", ylabel = "Temp. (∘F)")
display(tsplt)


"""
# 13.2 Validation - Polynomial approximation:
# -------------------------------------------
# We return to the polynomial fitting example above and continue with the data
# vectors t and y in the code as the training set, and generate an additional
# test set of 100 randomly chosen points (generated by the same method as used
# for the training set). We then fit polynomials of degree 0,. . . , 20
# (i.e., with p = 1, . . . , 21 coefficients) and compute the RMS errors on the training
# set and the test set. This produces a figure similar to Figure 13.11 in VMLS.
"""
m = 100;
t = -1 .+ 2*rand(m,1);   # Simulated raw data
y = t.^3 - t + 0.4 ./ (1 .+ 25*t.^2) + 0.10*randn(m,1); # Cubic transformation with random noise
polyfit(t, y, p) = vandermonde(t, p+1)\y # Vandermonde(t, p+1) generates a coeff. matrix
                                         # for fitting polynomial trends up to the degreee p.
# Generate the test set.
m = 100;
t_test = -1 .+ 2*rand(m,1);
y_test = t_test.^3 - t_test + 0.4 ./ (1 .+ 25*t_test.^2)+ 0.10*randn(m,1);
error_train = zeros(21);
error_test = zeros(21);
for p = 0:20
    local A = vandermonde(t,p+1)
    local θ = A \ y
    error_train[p+1] = norm(A*θ - y) / norm(y)
    error_test[p+1] = norm( vandermonde(t_test, p+1)*θ - y_test) / norm(y_test);
end
errplt = plot(0:20, error_train, label = "Train", marker = :circle)
errplt = plot!(0:20, error_test, label = "Test", marker = :square)
errplt = plot!(xlabel="Degree", ylabel = "Relative RMS error")
display(errplt)


"""
# Cross-validated house price regression model:
# ---------------------------------------------
# Above we used a data set of 774 house sales data to fit a simple regression
# model yhat = v + β1x1 + β2x2,# where yhat is the predicted sales price,
# x1 is the area and x2 is the number of bedrooms. Here we apply
# cross-validation to assess the generalization ability of the simple model.
# We use five folds, four of size 155 (Nfold in the code below) and one of size
# 154. To choose the five folds, we make a random permutation of the indices
# 1, . . . , 774. (We can do this by calling the randperm function in the
# Random package.) We choose the data points indexed by the first 155 elements
# in the permuted list as fold 1, the next 155 as fold 2, et cetera. The output
# of the following code outputs is similar to Table 13.1 on page 265 in VMLS
# (with slightly different numbers because of the random choice of folds).
"""
D = house_sales_data();
y = D["price"]; area = D["area"]; beds = D["beds"];
N = length(y);
X = [ones(N) area beds];
rms_train = zeros(5,1);
rms_test = zeros(5,1);
nfold = div(N,5); # size of first four folds
I = Random.randperm(N); # random permutation of numbers 1...N
Coeffs = zeros(5,3); errors = zeros(5,2);
for k = 1:5
    if k == 1
        Itrain = I[nfold+1:end];
        Itest = I[1:nfold];
    elseif k == 5
        Itrain = I[1:4*nfold];
        Itest = I[4*nfold+1:end];
    else
        Itrain = I[ [1:(k-1)*nfold ; k*nfold+1 : N]]
        Itest = I[ [(k-1)*nfold+1 ; k*nfold ]];
    end;
    Ntrain = length(Itrain)
    Ntest = length(Itest)
    local θ = X[Itrain,:] \ y[Itrain];
    Coeffs[k,:] = θ;
    rms_train[k] = rms(X[Itrain,:]*θ - y[Itrain])
    rms_test[k]  = rms(X[Itest,:]*θ - y[Itest])
end

Coeffs # 3 coefficient estimates for each of the five folds
RMS_folds = [rms_train rms_test] # RMS errors for the five folds
RMS_est = sqrt.(sum(RMS_folds.^2,dims =1)./5) # The combined RMS-values for training- and testdata respectively.
