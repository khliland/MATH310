using LinearAlgebra, Statistics, VMLS, Plots, MAT
include("my_RR.jl"), include("my_PCR.jl")
vars = matread("spectra.mat") # read NIR/octane data into dictionary
X    = vars["NIR"]
y    = vars["octane"]
# A description of the dataset:
vars["Description"]

m,n = size(X)

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
res = y-yhatCVrr;
RMS_CVrr[j] = sqrt((res'*res)/m)[1]
end
RMS_CVplot = plot(λs, RMS_CVrr, xscale = :log10, xlabel = "λ", xflip = true, ylims = (0.2,1.6), ylabel = "RMS_CV", label = "RR", title = "RMS_CV-plot for NIR/Octane data")
display(RMS_CVplot)
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
res = y.-yhatCVpcr;
RMS_CVpcr = sqrt.(sum(res.^2, dims = 1)./m)'
RMS_CVplot_pcr = plot((1:pc), RMS_CVpcr, xlabel = "pc", ylims = (0.2,1.6), ylabel = "RMS_CV", label = "PCR", title = "RMS_CV-plot for NIR/Octane data")
display(RMS_CVplot_pcr)
# ----------------------------------------------
