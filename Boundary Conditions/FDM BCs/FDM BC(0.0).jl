#FDM with u(t,0)=0, u(t,1)=0,works
using ApproxFun,OrdinaryDiffEq, Sundials
using LinearAlgebra
using Plots; gr()
N=100
dx = 1/(N)
#D2
d = ones(N-2) # diagonal
dl = ones(N-3) # super/lower diagonal
ϵ=0.05
D2=diagm(-1 => dl, 0 => -2*d, 1 => dl)
zv=zeros(N-2)
zh=zeros(1,N)
D2new=hcat(zv,D2,zv)
D2new[1,1]=D2new[end,end]=1
Q=Matrix{Int64}(I,N-2,N-2)
QQ=vcat(zeros(1,N-2),Q,zeros(1,N-2))
D2now=(ϵ/(dx^2)) .*D2new
#D1 first order upwind
D1=diagm(-1 => -dl, 0 => d)
kk=zeros(N-2)
D1new=hcat(kk,D1,zeros(N-2))
D1new[1,1]=-1
D1now=(0.5)*(1/dx) .* D1new
#Initial conditions
xs = (1:N) * dx
x = xs[2:N-1]
f0 = x -> exp(-200*(x-0.75)^2)
w = f0.(xs)
tmp=similar(w)
w=w[2:N-1]
p =D1now,D2now,QQ,tmp
plot(x,w)

r=zeros(N-2)
r=[0;r;0]

function bur(du,u,p,t)
    D1now,D2now,QQ,tmp = p
    D2now = D1now + D2now
    du .= D2now * (QQ * u + r)
    #tmp = QQ * u
    #tmp = tmp + r
    #mul!(du,D2now,tmp)
    #mul!(du,D2now,tmp)
end

prob = ODEProblem(bur, w, (0.0,1.0), p)
sol  = solve(prob, TRBDF2(); reltol=1e-5,abstol=1e-5)
plot(x,sol(0.0))
plot!(x,sol(0.2))
plot!(x,sol(0.4))
plot!(x,sol(0.6))
plot!(x,sol(0.8))
plot!(x,sol(1.0))
