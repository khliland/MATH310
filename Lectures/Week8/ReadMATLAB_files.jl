# pkg> add MAT  # For reading MATLAB-files, see https://github.com/JuliaIO/MAT.jl

using MAT
vars = matread("spectra.mat") # read data into dictionary
X    = vars["NIR"]
y    = vars["octane"]
# A description of the dataset:
vars["Description"]
