using LinearAlgebra, Statistics, VMLS, Plots, MAT
include("my_RR.jl"), include("my_PCR.jl"), include("PLS_nip.jl")
vars = matread("spectra.mat") # read NIR/octane data into dictionary
X    = vars["NIR"]
y    = vars["octane"]
# A description of the dataset:
vars["Description"]

m,n = size(X)

# Fit 15 PCR-models using 1 to 15 PC's using the function my_PCR.jl:
pc = 15;
β0, β = my_PCR(X, y, mc = pc);
# Calculate and plot the root mean squared error (RMS-erros):
yhat_PCR = β0.+X*β;     # The fitted values.
res_PCR  = y.-yhat_PCR; # The residuals.
rms_PCR = sqrt.(mean(res_PCR.^2,dims=1))';

# Fit 15 RR-models with λs = (10).^linspace(-7, 7, 15) using my_RR.jl &
# Calculate and plot the root mean squared error (RMS-erros) and compare
# with the RMS-errors obtained by the PCR models:
nλ = 15; λs = (10).^linspace(-7, 7, nλ) # λ-values evenly spaced on log10-scale
rms_RR = zeros(15,1);
for j = 1:nλ
β0, β = my_RR(X, y; λ = λs[j])
yhat_RR = β0.+X*β; res_RR = y-yhat_RR;
rms_RR[j] = sqrt.(mean(res_RR.^2));
end

# Fit 15 PLS-models using 1 to 15 PC's using the function PLS_nip.jl:
pc = 15;
β0, β = PLS_nip(X, y, mc = pc);
# Calculate and plot the root mean squared error (RMS-erros):
yhat_PLS = β0.+X*β;     # The fitted values.
res_PLS  = y.-yhat_PLS; # The residuals.
rms_PLS = sqrt.(mean(res_PLS.^2,dims=1))';

# RMS-errors obtained by the PCR models:
rms_PCR_plot = plot(rms_PCR, label = "PCR", ylims = (0,1.6), ylabel = "rms-values",markershape = :circle, xlabel = "# PCs", title = string("RMS-values for PCR on NIR/Octane data with up to ", pc, " PCs"))
display(rms_PCR_plot)
# RMS-errors obtained by the RR models
rms_RR_plot = plot(λs, rms_RR, xscale = :log10, xlabel = "λ", xflip = true, ylims = (0,1.6), label = "RR", ylabel = "rms-values",markershape = :circle, title = "RMS-values for RR on NIR/Octane data")
#, xscale = :log10, xlabel = "λ", xflip = true, ylims = (0.2,1.6), ylabel = "rms-values", label = "RR", title = "RMS_CV-plot for NIR/Octane data")
display(rms_RR_plot)
# RMS-errors obtained by the PLS models:
rms_PLS_plot = plot(rms_PLS, label = "PLS", ylims = (0,1.6), ylabel = "rms-values",markershape = :circle, xlabel = "# PCs", title = string("RMS-values for PLS on NIR/Octane data with up to ", pc, " components"))
display(rms_PLS_plot)


# Do leave-one-out crossvalidation for both the PCR-model alternatives
# and the RR-model alternatives.
# ----------------------------------------------
# LooCV for PCR:
# ----------------------------------------------
pc = 15;
yhatCVpcr = zeros(m,pc);
    for k = 1:m
        indin = setdiff(1:m,k)
        b0, B = my_PCR(X[indin,:], y[indin]; mc = pc) # Note: For PCR we compute the reg. coeffs. for all comps up to pc.
        yhatCVpcr[k,:] = b0 + X[k,:]'*B
    end
resid = y.-yhatCVpcr;
RMS_CVpcr = sqrt.(mean(resid.^2, dims = 1))'
RMS_CVplot_pcr = plot((1:pc), RMS_CVpcr, xlabel = "pc", markershape = :circle, ylims = (0.2,1.6), ylabel = "RMS_CV", label = "PCR", title = "RMS_CV-plot for NIR/Octane data")
display(RMS_CVplot_pcr)
# -------------------------------------------------
# Leave-one-out cross validation (LooCV) for RR:
# ----------------------------------------------
nλ = 15;
λs = (10).^linspace(-7, 7, nλ) # λ-values evenly spaced on log10-scale
RMS_CVrr = zeros(nλ,1)
for j = 1:nλ
yhatCVrr = zeros(m,1);
    for k = 1:m
        indin = setdiff(1:m,k)
        β0, β = my_RR(X[indin,:], y[indin]; λ = λs[j])
        yhatCVrr[k] = β0 + X[k,:]'*β
    end
resid = y-yhatCVrr;
RMS_CVrr[j] = sqrt((resid'*resid)/m)[1]
end
RMS_CVplot = plot(λs, RMS_CVrr, xscale = :log10, xlabel = "λ", markershape = :circle, xflip = true, ylims = (0.2,1.6), ylabel = "RMS_CV", label = "RR", title = "RMS_CV-plot for NIR/Octane data")
display(RMS_CVplot)
# ----------------------------------------------
# LooCV for PLS:
# ----------------------------------------------
pc = 15;
yhatCVpls = zeros(m,pc);
    for k = 1:m
        indin = setdiff(1:m,k)
        b0, B = PLS_nip(X[indin,:], y[indin]; mc = pc) # Note: For PLS we compute the reg. coeffs. for all comps up to pc.
        yhatCVpls[k,:] = b0 + X[k,:]'*B
    end
resid = y.-yhatCVpls;
RMS_CVpls = sqrt.(mean(resid.^2, dims = 1))'
RMS_CVplot_pls = plot((1:pc), RMS_CVpls, xlabel = "pc", markershape = :circle, ylims = (0.2,1.6), ylabel = "RMS_CV", label = "PLS", title = "RMS_CV-plot for NIR/Octane data")
display(RMS_CVplot_pls)


# For each method (PCR, RR and PLS) choose the model corresponding to
# the smallest RMS-prediction error:
pcrmin = argmin(RMS_CVpcr)[1];
λmin   = λs[argmin(RMS_CVrr)];
plsmin = argmin(RMS_CVpls)[1];

β0PCR, βPCR = my_PCR(X, y, mc = pcrmin);
β0RR, βRR   = my_RR(X, y;   λ = λmin);
β0PLS, βPLS = PLS_nip(X, y, mc = plsmin);

# Compare the two selected models by plotting their regression coeffs togehter
# (re-calculate the regression coeffs based on the complete dataset).
plt_compare = plot(βPCR[:,pcrmin], label = "PCR", title = string("Regression coeffs for PCR(", pcrmin, ") and RR(",λmin,")"))
plt_compare = plot!(βRR, label = "RR")
plt_compare = plot!(βPLS[:,plsmin], label = "PLS")
display(plt_compare)
nβPCR = norm(βPCR);
nβRR  = norm(βRR);
nβPLS = norm(βPLS);

# The minimum norm solution (β) of the full system X0β = y0:
X0 = X.-mean(X, dims= 1); y0 = y.-mean(y); # We center the data
β = pinv(X0)*y0; # the minimum norm solution
plt_compare = plot!(β, label = "min_norm_solution")
display(plt_compare)
