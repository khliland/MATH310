"
Exercise 7 (week 10):
 - Make a Julia-script that generates U, S and V to design a
   60×60 matrix A=USV^t with singular values 2^(-1), ... , 2^(-60).
 - Apply your Julia QR-factorization functions CGS and MGS to A.
 - Apply the Julia-internal QR
 - Compare the 60 diagonal elements of the resulting R-matrices by
   plotting them together (use log10-scale for the plotting).
"
using LinearAlgebra
include("CGS.jl")
include("MGS.jl")
pots = 1:60;
s = 1 ./(2 .^pots); S0 = diagm(s); # the desired singular values.
U, S, V = svd(randn(60,60));       # SVD of random 60×60 matrix.
A = U*S0*V';                       # The desired matrix.

Qc, Rc = CGS(A); # Classical GS
Qm, Rm = MGS(A); # Modified GS
Q, R = qr(A)     # Julia-internal QR

drcl = diag(Rc) # Diagonal of Rc
drmo = diag(Rm) # Diagonal of Rm
drqr = diag(R)  # Diagonal of R
dr_plot = plot(log10.(drcl), seriestype = :scatter, markershape = :circle, markercolor = :white, label = "cgs");
dr_plot = plot!(log10.(drmo), seriestype = :scatter, markershape = :cross, markercolor = :red, label = "mgs", ylabel = "log10(diag(R))");
dr_plot = plot!(log10.(abs.(drqr)), seriestype = :scatter, markershape = :xcross, markercolor = :blue, label = "qr", title = "log10 of R-diagonals for QR-factorization algorithms")
display(dr_plot)
