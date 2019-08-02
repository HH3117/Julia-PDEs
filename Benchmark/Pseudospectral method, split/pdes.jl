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
function cheb(N)
    N==0 && return (0,1)
    x = cos.(pi*(0:N)/N)
    c = [2; ones(N-1,1); 2].*(-1).^(0:N)
    X = hcat([x for i in 1:N+1]...)
    dX = X-X'
    D  = (c*(1 ./c)')./(dX+I)      # off-diagonal entries
    D  = D .- Diagonal(vec(sum(D,dims=2)))                 # diagonal entries
    D,x
end
N = 128
ChebD2,x = cheb(N)
xx = x
x = x[2:N]
w = .53*x + .47*sin.(-1.5*pi*x) - x # use w = u-x to make BCs homogeneous
u = [1;w+x;-1]

ϵ=0.01
p = (ϵ*(ChebD2^2)[2:N, 2:N], x)
function allen_cahn(du,u,p,t)
    D2, x = p
    mul!(du,D2,u)
    @. du = du + (u + x) - (u + x)^3
end

prob = ODEProblem(allen_cahn, w, (0.0,70), p)
@time sol  = solve(prob, CVODE_BDF(); reltol=1e-5,abstol=1e-5)

plot(xx, [1;x.+sol(0.0);-1])
plot!(xx, [1;x.+sol(1);-1])
plot!(xx, [1;x.+sol(2);-1])
plot!(xx, [1;x.+sol(3);-1])
plot!(xx, [1;x.+sol(4);-1])
plot!(xx, [1;x.+sol(5);-1])
plot!(xx, [1;x.+sol(6);-1])
plot!(xx, [1;x.+sol(7);-1])
plot!(xx, [1;x.+sol(8);-1])
plot!(xx, [1;x.+sol(50);-1])

for t in 0:1:70
    plt=plot(xx, [1;x.+sol(t);-1])
    ylims!(-1.1, 1.1)
    display(plt)
    sleep(0.01)
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
