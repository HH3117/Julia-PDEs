using ApproxFun, OrdinaryDiffEq, Sundials, BenchmarkTools
using LinearAlgebra
using Plots; gr()
# Fourier Space
S = Fourier()
n = 100
x = points(S, n)
D2 = Derivative(S,2)[1:n,1:n]
D  = (Derivative(S) → S)[1:n,1:n]
T = ApproxFun.plan_transform(S, n)
Ti = ApproxFun.plan_itransform(S, n)

# Burger's equation
û₀ = T*cos.(cos.(x.-0.1))
A = 0.01*D2
tmp = similar(û₀)
p = (D,D2,T,Ti,tmp,similar(tmp))
function burgers(dû,û,p,t)
    D,D2,T,Ti,u,tmp = p
    mul!(tmp, D, û)
    mul!(u, Ti, tmp)
    mul!(tmp, Ti, û)
    @. tmp = tmp*u
    mul!(u, T, tmp)
    mul!(tmp, A, û)
    @. dû = tmp - u
end

prob = ODEProblem(burgers, û₀, (0.0,5.0), p)
@time û  = solve(prob, CVODE_BDF(); reltol=1e-5,abstol=1e-5)

plot(x, Ti*û(0.0))
plot!(x, Ti*û(1.0))
plot!(x, Ti*û(2.0))
plot!(x, Ti*û(3.0))
plot!(x, Ti*û(5.0))


# Kuramoto-Sivashinsky
D4 = Derivative(S,4)[1:n,1:n]
û₀ = T*(cos.(x./16).*(1 .+ sin.(x./2.04)))
plot(x,Ti*û₀)
tmp=similar(û₀)
q = (D,D2,T,Ti,D4,tmp,similar(tmp),similar(tmp))
function kuramoto_sivashinsky(dû,û,q,t)
    D,D2,T,Ti,D4,tmp,u,uc = q
    mul!(u, D, û)
    mul!(tmp, Ti, u)
    mul!(u, Ti, û)
    @. tmp=tmp*u
    mul!(u,T, tmp)
    mul!(tmp, D4, û)
    mul!(uc, D2, û)
    @. dû = - tmp - uc - u
end

prob = ODEProblem(kuramoto_sivashinsky, û₀, (0.0,300.0), q)
@time û  = solve(prob, CVODE_BDF(); reltol=1e-5,abstol=1e-5)
for t in 0:50:300
    plt=(plot(x, Ti*û(t)))
    ylims!(1., 2.3)
    display(plt)
    sleep(0.01)
end


# Allen-Cahn
û₀ = T*cos.(cos.(x.-0.1))
ϵ=1/3
tmp = similar(û₀)
w = (D2,T,Ti,tmp,similar(tmp),similar(tmp))
function allen_cahn(dû,û,w,t)
    D2,T,Ti,tmp,u,uc = w
    mul!(u,Ti,û)
    @.uc = u^3
    mul!(u,T,uc)
    mul!(tmp,D2,û)
    @.dû = 10*tmp + (u-û)/(ϵ^2)
end

prob = ODEProblem(allen_cahn, û₀, (0.0,5.0), w)
@time û  = solve(prob, CVODE_BDF(); reltol=1e-5,abstol=1e-5)


plot(x, Ti*û(0.0))
plot!(x, Ti*û(0.01))
plot!(x, Ti*û(0.02))
plot!(x, Ti*û(0.03))
plot!(x, Ti*û(0.04))
plot!(x, Ti*û(0.05))
plot!(x, Ti*û(0.06))
plot!(x, Ti*û(0.07))
plot!(x, Ti*û(0.08))

for t in 0:0.01:0.2
    plt=(plot(x, Ti*û(t)))
    ylims!(0.4, 1.1)
    display(plt)
    sleep(0.05)
end

#Korteweg–de Vries (kdv) equation
D3  = (Derivative(S,3) → S)[1:n,1:n]
û₀ = T*cos.(x)
δ=0.022
tmp = similar(û₀)
p = (D,D3,T,Ti,similar(tmp),tmp)
function kdv(dû,û,p,t)
    D,D3,T,Ti,u,tmp = p
    mul!(u,D,û)
    mul!(tmp,Ti,u)
    mul!(u,Ti,û)
    @. tmp=u*tmp
    mul!(u,T,tmp)
    mul!(tmp,D3,û)
    @.dû = -tmp + 6*u
end

prob = ODEProblem(kdv, û₀, (0.0,1.0), p)
@time û  = solve(prob, CVODE_BDF(); reltol=1e-5,abstol=1e-5) #59s

plot(x, Ti*û(0.0))
plot!(x, Ti*û(0.2))
plot!(x, Ti*û(0.4))
plot!(x, Ti*û(0.6))
plot!(x, Ti*û(0.7))

for t in 0:0.05:0.7
    plt=(plot(x, Ti*û(t)))
    ylims!(-1.7, 1.)
    display(plt)
    sleep(0.05)
end
