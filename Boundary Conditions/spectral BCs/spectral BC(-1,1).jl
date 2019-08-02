#psedospectral with u(t,0)=-1,u(t,1)=1, works
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

N = 256
ChebD,x = cheb(N)
xx = x
x = x[2:N]
w = @. .53*x + .47*sin(-1.5*pi*x) - x
u = [1;w+x;-1]
plot(xx,u)
plot(x,w)

D1=ChebD[2:N, 2:N]
ϵ=0.01
D2=ϵ*(ChebD^2)[2:N, 2:N]

function diffusion_advection(du,u,x,t)
    du .= D1 * (u .+ x)
end


prob = SplitODEProblem(DiffEqArrayOperator(D2), diffusion_advection, w, (0.0,1.0), x)
sol  = solve(prob, RadauIIA5(autodiff=false); reltol=1e-14,abstol=1e-14)

plot((xx.+1)./2,[1;x+sol(0.0);-1])
plot!((xx.+1)./2,[1;x+sol(0.2);-1])
plot!((xx.+1)./2,[1;x+sol(0.4);-1])
plot!((xx.+1)./2,[1;x+sol(0.6);-1])
plot!((xx.+1)./2,[1;x+sol(0.8);-1])
plot!((xx.+1)./2,[1;x+sol(1.0);-1])
