using OrdinaryDiffEq
using PyPlot
using LinearAlgebra

pygui(true);

const M = 31 #number of clusters (odd)
N0 = 10000 #total number of atoms

κ = 1
g,Γ,κ,R = (0.002κ, 0.001κ, κ, 0.001κ)


σ = 0.1κ
ξ0 = ones(M) #atomic dephasing range
ξ = ξ0*0Γ #zero dephasing

##################################
#Gaussian distribution
dD = 2/(M-1)
Δ0 = [-1:dD:1;] #odd number of clusters with different detunings

#shift = [randn(Integer((M-1)/2)); 0; randn(Integer((M-1)/2))].*dD./10
#Δ0 = Δ0 .+shift

Δ = Δ0.*σ

σ3 = σ
G = 1/(σ3*sqrt(2pi)).*exp.(-1/2 .*Δ.^2 ./σ3^2)
N = G #figure (a)

#N[12]=N[12]+100 #figure (b)

#shift = randn(M).*N0/10M
#N = N .+shift #figure (c)

N=N./sum(N)*N0

########################################

#ind = []
const ind = Tuple{Int,Int}[] # use specific type and avoid non-constant global
for a=1:M
    for b=a:M
        push!(ind,(a,b))
    end
end

function mj(m, j)
    findfirst(isequal((m, j)), ind)  #findall(isequal((m, j)), x)
end

const ind_spsm = length(ind)
const ind_ata = ind_spsm + M
const ind_asp = ind_ata + 1
const ind_end = ind_asp + M
const N_ = copy(N)

function ff(du,u,p,t)
    Δ,g,Γ,κ,R,ξ = p

        #m - the number of cluster
        #dspsm/dt, m not equal j
        for m=1:M
            for j=m:M
                a = mj(m, j)
                du[a] = -(Γ+R+ξ[m]/2+ξ[j]/2)*u[a] - im*g*u[ind_asp+m] + im*g*u[ind_asp+j]' + 2*im*g*u[ind_asp+m]*u[ind_spsm+j] - 2*im*g*u[ind_asp+j]'*u[ind_spsm+m] -im*Δ[m]*u[a] +im*Δ[j]*u[a]
            end
        end

        #dspsm/dt, m=j
        for m=1:M
            du[ind_spsm+m] = im*g*u[ind_asp+m]' - im*g*u[ind_asp+m] - (Γ+R)*u[ind_spsm+m] + R
        end

        #dat*a/dt
        du[ind_ata+1] = -κ*u[ind_ata+1] + im*g*sum(N_[j]*u[ind_asp+j] for j=1:M) - im*g*sum(N_[j]*u[ind_asp+j]' for j=1:M)

        #da*sp/dt
        for m=1:M
            du[ind_asp+m] = -(κ/2+Γ/2+R/2+ξ[m]/2+im*Δ[m])*u[ind_asp+m] + im*g*u[ind_ata+1] - im*g*u[ind_spsm+m] - 2*im*g*u[ind_ata+1]*u[ind_spsm+m] - im*g*(N_[m]-1)*u[mj(m,m)]
            for j=1:M
                if j>m   du[ind_asp+m] = du[ind_asp+m] - im*g*N_[j]*u[mj(m, j)] end
                if j<m   du[ind_asp+m] = du[ind_asp+m] - im*g*N_[j]*u[mj(j, m)]' end
            end
        end
end

u0 = zeros(ComplexF64, ind_end)
p0 = (Δ,g,Γ,κ,R,ξ)

prob = ODEProblem(ff,u0,(0.0,10000),p0)
sol = solve(prob,RK4())

n = real.(getindex.(sol.u, ind_ata +1))
pe = real.(getindex.(sol.u, ind_spsm+1))
sz = 2*pe .-1

#=

figure(figsize=(9,3))
subplot(121)
title(L"N=5000, \sigma=0.01\kappa, \Delta \in [-\sigma; \sigma], g=0.005\kappa, \Gamma=0.001\kappa, R=0.01\kappa, \kappa=1")
plot(sol.t, n, label = (L"<n>_{clusters}"))
legend()
xlabel(L"\kappa t")
ylabel(L"\langle n \rangle")

subplot(122)
plot(sol.t, sz, label = (L"<S_z>_{clusters}"))
legend()
xlabel(L"\kappa t")
ylabel(L"\langle S_z \rangle")

tight_layout()
=#


###################################################
# Spectrum Laplace transform
pe0=zeros(M)
for m=1:M
    pe0[m] = real.(getindex.(sol.u, ind_spsm+m))[end]
end

u0_spec = zeros(ComplexF64, M+1)
u0_spec[1] = real.(getindex.(sol.u, ind_ata+1))[end]
for m=1:M
    u0_spec[m+1] = getindex.(sol.u, ind_asp+m)[end]
end

a = (0, [-im*g*N[m] for m=1:M]...)
b = (0, [-im*g*(1-2*pe0[m]) for m=1:M]...)
d = [κ/2, [(Γ+R+ξ[m])/2+im*Δ[m] for m=1:M]...]

#A = Matrix{ComplexF64}(I, M+1, M+1).*d
A = zeros(ComplexF64, M+1, M+1)
A[1,:].= a
A[:,1].=b

A = A + Diagonal(d)

w = [-0.11:0.0001:0.11;]

spec2 = zeros(ComplexF64, length(w))

for i=1:length(w)
s = im*w[i]
S = Diagonal(s*ones(M+1))
B = S+A
spec2[i] = (inv(B)*u0_spec)[1]
end

spec2 = real.(spec2)
spec2 = spec2./maximum(spec2)

rc("font", size = 9)
rc("legend", fontsize = 7)
rc("text", usetex = true)
rc("lines", lw = 1.0)

figure(figsize=(5,3))
title(L"N=10^4, M=31, \sigma = 0.1\kappa, \Delta \in [-\sigma:\sigma], g=0.002\kappa, \Gamma=0.001\kappa")
plot(w, spec2, linewidth = 1,label = (L"R=0.001\kappa"), color = "silver")
legend()
xlabel(L"(\omega - \omega_c) / \kappa")
ylabel(L"Intensity")
tight_layout()
