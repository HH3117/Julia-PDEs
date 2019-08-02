#psedospectral with u(t,0)=0,u(t,1)=0, works
using ApproxFun, OrdinaryDiffEq, Sundials, BenchmarkTools, DiffEqOperators
using DiffEqDevTools
using LinearAlgebra
using Plots; gr()

# diffusion advection
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
ChebD,x = cheb(N)
xx = x
x = x[2:N]
w = @. sin(3*pi*x)
u = [0;w;0]
#plot(xx,u)

D1=ChebD[2:N, 2:N]
ϵ=0.01
D2=ϵ*(ChebD^2)[2:N, 2:N]

function diffusion_advection(du,u,x,t)
    du .= D1 * u
end


prob = SplitODEProblem(DiffEqArrayOperator(D2), diffusion_advection, w, (0.0,5.0), x)
sol  = solve(prob, RadauIIA5(autodiff=false); reltol=1e-14,abstol=1e-14)

plot((xx.+1)./2,[0;sol(0.0);0])
plot!((xx.+1)./2,[0;sol(0.2);0])
plot!((xx.+1)./2,[0;sol(0.4);0])
plot!((xx.+1)./2,[0;sol(0.6);0])
plot!((xx.+1)./2,[0;sol(0.8);0])
plot!((xx.+1)./2,[0;sol(1.0);0])
