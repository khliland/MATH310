using LinearAlgebra, Statistics, VMLS, Plots
include("solve_mols.jl")

# -----------------------------------------------------
# Multi-objective least squares example with 2 systems:
# -----------------------------------------------------
As = [randn(10,5), randn(10,5)];
bs = [randn(10), randn(10)];
N  = 201;
λs = 10 .^ linspace(-6,6,N);
x  = zeros(5,N); J1 = zeros(N); J2 = zeros(N); # Regression coeffs and J-vectors for storing
for k = 1:N
    x[:,k] = solve_mols(As, bs, [1, λs[k]]) # Puts weight 1 on the first problem and weight λs[k] on the second problem.
    J1[k]  = sum((As[1]*x[:,k] - bs[1]).^2) # norm(As[1]*x[:,k] - bs[1])^2
    J2[k]  = sum((As[2]*x[:,k] - bs[2]).^2) # norm(As[2]*x[:,k] - bs[2])^2
end;

# plot the regression coeffs x versus the λs:
rc_plot = plot(λs, x', xscale = :log10, xlabel = "λ-values", ylabel = "x(λ)-values", label = ["x1(λ)" "x2(λ)" "x3(λ)" "x4(λ)" "x5(λ)"]);
rc_plot = plot!(xlims = (1e-6,1e6),title = string("The regression coeffs as functions of λ"));
display(rc_plot)

# plot two objectives versus λ
ob_plot =  plot(λs, J1, xscale = :log10, label = "J1(λ)", title = "SSE-plot for the mols solutions");
ob_plot =  plot!(λs, J2, label = "J2(λ)", xlabel = "λ-values", xlims = (1e-6,1e6), ylabel = "Sum of Squared Errors - SSE(λ))");
display(ob_plot)

# plot trade-off curve
to_plot = plot(J1, J2, xlabel="J1", ylabel = "J2", legend=false);
# add (single-objective) end points to trade-off curve
E1 = [J1[1], J1[N]] #[norm(As[1]*x[:,1]-bs[1])^2, norm(As[1]*x[:,N]-bs[1])^2]; # End-point coordinates for J1
E2 = [J2[1], J2[N]] #[norm(As[2]*x[:,1]-bs[2])^2, norm(As[2]*x[:,N]-bs[2])^2]; # End-point coordinates for J2
to_plot = scatter!(E1, E2, title = "Optimal trade-off curve for the mols-problem");
display(to_plot)

# --------------------------------------------------------------
# Estimation and inversion
# ------------------------
# We consider the example of Figure 15.4 in VMLS. We start by loading the data,
# as a vector with the hourly ozone levels, from a period of 14 days. Missing
# measurements are assigned NaN-values (for "Not a Number").

# The plot command skips the NaN-values in the following figure:
ozone = ozone_data(); # a vector of length 14*24 = 336
k = 14; N = k*24;
oz_plot = plot(1:N, ozone, yscale = :log10, marker = :circle, legend=false)
display(oz_plot)

# we use the mols_solve function to make a periodic (24hrs) hourly fit, for the
# values λ = 0.1, 1, 10 and 100. The Julia command 'isnan' is used tofind
# and discard the missing measurements. Results are shown in the figures below:
Aall = vcat( [eye(24) for i = 1:k]...)
ind = [k for k in 1:length(ozone) if !isnan(ozone[k])]; # indices of the no-nans

# Excluede the entries of ozone-data with nans:
A = Aall[ind,:]; b1 = log.(ozone[ind]);

# Define periodic difference matrix (24hrs period) to enable smoothness
# in the fitted values:
D  = [diff(eye(24), dims=1); 1 zeros(1,22) -1];
#D  = -eye(24) + [zeros(23,1) eye(23); 1 zeros(1,23)]; # Same as above
b2 = zeros(24);

# The two problems we want to solve simultaneously are:
# A*x = b1; and D*x = b2; where the main objective is to
# obtain a good estimate for in periodic pattern in the
# mols-solution x:
As = [A, D]; bs = [b1, b2]

# The solution for λ = 0.01:
λ = 0.1;
x = solve_mols(As, bs, [1, λ])
smooth_plot1 = scatter(1:N, ozone, yscale = :log10, legend=false, title = string("Periodic smoothing with λ = ", λ))
smooth_plot1 = plot!(1:N, vcat([exp.(x) for i = 1:k]...))
display(smooth_plot1)

# The solution for λ = λ0:
λ0 = 100; # check also with λ0 = 1 and 10.
λ = λ0; x = solve_mols( As, bs, [1, λ])
smooth_plot0 = scatter(1:N, ozone, yscale = :log10, legend=false, title = string("Periodic smoothing with λ = ", λ))
smooth_plot0 = plot!(1:N, vcat([exp.(x) for i = 1:k]...))
display(smooth_plot0)

# --------------------------------------------------------------
# Example of regularized data-fitting:
# --------------------------------------------------------------
# Next we consider the small regularized datafitting example on
# page 329 of VMLS. We import data as vectors xtrain, ytrain, xtest, ytest:
Data   = regularized_fit_data(); # Load data dictionary for the example
xtrain = Data["xtrain"]; ytrain = Data["ytrain"]; Ntrain  = length(ytrain);
xtest  = Data["xtest"];  ytest  = Data["ytest"];  Ntest   = length(ytest);
p      = 5;
omega  = [ 13.69; 3.55; 23.25; 6.03 ];
phi    = [  0.21; 0.02; -1.87; 1.72 ];
A      = hcat(ones(Ntrain), sin.(xtrain*omega' + ones(Ntrain)*phi'));
Atest  = hcat(ones(Ntest),  sin.(xtest*omega'  + ones(Ntest)*phi'));
npts   = 100;
λs     = 10 .^ linspace(-6,6,npts);
err_train = zeros(npts);
err_test  = zeros(npts);
θs        = zeros(p,npts);
for k = 1:npts
θ = mols_solve([ A, [zeros(p-1) eye(p-1)]],[ ytrain, zeros(p-1) ], [1, λs[k]])
err_train[k] = rms(ytrain -     A*θ);
err_test[k]  = rms(ytest  - Atest*θ);
θs[:,k]      = θ;
end;
# Plot RMS errors:
rms_plot = plot(λs, err_train, xscale = :log10, label = "Train", title = "RMS-errors")
rms_plot = plot!(λs, err_test, xscale = :log10, label = "Test")
rms_plot = plot!(xlabel = "λ", ylabel = "RMS error", xlim = (1e-6, 1e6));
display(rms_plot)
# Plot the regression coefficients:
coef_plot = plot(λs, θs', xscale = :log10, title = "Regression coeffs", label = ["θ1(λ)" "θ2(λ)" "θ3(λ)" "θ4(λ)" "θ5(λ)"])
plot!(xlabel = "λ", xlim = (1e-6, 1e6));
display(coef_plot)

# ---------------------------------------------------------------
# An illustration of the kernel trick described in VMLS, §15.5.2:
# ----------------------------------------------------------------------
# * Plot and compare the regression coeffs. obtained by PCR and PP
# * Calculate the norm of each solution and compare with the minimum
#   norm solution obtained by the pseudo-inverse.
m = 100; n = 5000; # Note n >> m
# We generate a random "wide" dataset with m samples of n variables:
X = randn(m,n); y = randn(m);
βdes = randn(n); # we want the solution below to be close to βdes.
λ = 2.0;
# The least squares solution β of the system [X; sqrt(λ)*I]β = [y; sqrt(λ)*βdes]
# i.e. the minimizer of ||Xβ-y||^2 + λ ||β-βdes||^2
@time β1 = [X; sqrt(λ)*eye(n)] \ [y; sqrt(λ)*βdes]; # Slow alternative

# Now we use the much faster "kernel trick"
# to compute the least squares solution:
@time β2 = X' * ((X*X' + λ*eye(m))\(y-X*βdes)) + βdes;

# The kernel trick combined with QR-factorization:
@time begin
    Q, R = qr([X'; sqrt(λ)*eye(m)]); #Q = Matrix(Q);
    β3 = X' * (R \ (R' \ (y-X*βdes))) + βdes;
end;

norm(β1-β2) # β1 and β2 are essentially equal.
norm(β1-β3) # β1 and β3 are essentially equal.
norm(β2-β3) # β2 and β3 are essentially equal.

# ----------------------------------------------------------------------
# We do Ridge-regression and PCR for the wide dataset defined above.
# ----------------------------------------------------------------------
include("my_RR.jl")
λval  = 5e-2;
β0, β = my_RR(X,y; λ = λval)
yhat_RR  = β0 .+ X*β; # The fitted values of the RR-model.
res_rr   = y - yhat;  # The RR residuals.
RMS_rr   = sqrt(res_rr'*res_rr/m); # Corresponding RMS-values
Regcoeff_plot = plot(β, label = string("RR(λ= ",λval,")"))
include("my_PCR.jl")
k = 6;
b0, B = my_PCR(X,y; mc = k)
yhat_PCR  = b0[k] .+ X*B[:,k]; # The fitted values of the PCR-model.
res_pcr   = y - yhat_PCR;      # The PCR residuals.
RMS_pcr   = sqrt(res_pcr'*res_pcr/m); # Corresponding RMS-values
Regcoeff_plot = plot!(B[:,k], title = string("Regression coeffs for RR-model and PCR-model"), label = string("PCR(pc = ", k,")"))

display(Regcoeff_plot)

# ----------------------------------------------------------------------

# Exercises:
# ----------
# Use the "spectra.mat" dataset for fitting different regression models:
# ----------------------------------------------------------------------
# * Fit 15 PCR-models using 1 to 15 PC's using the function my_PCR.jl.
# * Calculate and plot the root mean squared error (RMS-erros).
# ----------------------------------------------------------------------
# * Fit 15 RR-models with λs = 10.^ linspace(-7, 7, 15) using
# * my_RR.jl.
# * Calculate and plot the root mean squared error (RMS-erros) and
#   compare with the RMS-errors obtained by the PCR models.
# ----------------------------------------------------------------------
# * Do leave-one-out crossvalidation for both the PCR-model alternatives
#   and the RR-model alternatives. For each method (PCR and RR)
#   choose the model corresponding to the smallest RMS-prediction error.
# * Compare the two selected models by plotting their regression coeffs
#   togehter (re-calculate the regression coeffs based on the complete
#   dataset).
# * Calculate the norm of each solution and compare with the minimum
#   norm solution of X*b = y0 (obtained by the pseudo-inverse) where
#   X and y0 denote the mean centered versions of the data set.
