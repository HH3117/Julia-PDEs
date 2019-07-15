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
p = (D,D2,T,Ti)
û₀ = T*cos.(cos.(x.-0.1))
A = 0.01*D2
function burgers(dû,û,p,t)
    D,D2,T,Ti = p
    u = Ti*û
    up = Ti*(D*û)
    dû .= A*û .- T*(u.*up)
    #x = T*(u.*up)
    #@. dû .= -x
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
q = (D,D2,T,Ti,D4)
Len=2*pi/0.025
η1=min(x/Len,0.1.-x/Len)
η2=20*(x/Len.-0.2).*(0.3.-x/Len)
η3=min(x/Len.-0.6,0.7.-x/Len)
η4=min(x/Len.-0.9,1 .-x/Len)
u₀=zeros(n)
for i =1:n
     u₀[i] = 16*max(0,η1[i],η2[i],η3[i],η4[i])
end
û₀=T*u₀

function kuramoto_sivashinsky(dû,û,q,t)
    D,D2,T,Ti,D4 = q
    u = Ti*û
    up =T*(u.*u)
    dû .= .-D4*û .-D2*û .-(D*up)/2
end


prob = ODEProblem(kuramoto_sivashinsky, û₀, (0.0,5.0), q)
@time û  = solve(prob, CVODE_BDF(); reltol=1e-5,abstol=1e-5)
plot(x,u₀)
for t in 0:0.5:5
    plt=(plot(x, Ti*û(t)))
    ylims!(0.0, 0.4)
    display(plt)
    sleep(0.02)
end


# Allen-Cahn
w = (D2,T,Ti)
û₀ = T*cos.(cos.(x.-0.1))
ϵ=1/3
function allen_cahn(dû,û,w,t)
    D2,T,Ti = w
    u = Ti*û
    uc=u.^3
    dû .= 10*D2*û .+ (T*(uc)-û)/(ϵ^2)
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
plot(x,Ti*û₀)

#Korteweg–de Vries (kdv) equation
D3  = (Derivative(S,3) → S)[1:n,1:n]
p = (D,D3,T,Ti)
#û₀ = T*cos.(pi .*x)
û₀ = T*cos.(x)
δ=0.022
function kdv(dû,û,p,t)
    D,D3,T,Ti = p
    u = Ti*û
    up = Ti*(D*û)
    dû .= .-(D3*û) .+6 .* (T*(u.*up))
    #dû .= .-((δ^2) .*(D3*û) .+ T*(u.*up))
end

prob = ODEProblem(kdv, û₀, (0.0,1.0), p)
@time û  = solve(prob, CVODE_BDF(); reltol=1e-5,abstol=1e-5) #59s

plot(x,Ti*û₀)

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
