using ApproxFun, OrdinaryDiffEq, Sundials, BenchmarkTools, DiffEqOperators
using DiffEqDevTools
using LinearAlgebra
using Plots; gr()
# Fourier Space
S = Fourier()
n = 512
x = points(S, n)
D2 = Derivative(S,2)[1:n,1:n]
D  = (Derivative(S) → S)[1:n,1:n]
T = ApproxFun.plan_transform(S, n)
Ti = ApproxFun.plan_itransform(S, n)

# Burger's equation
û₀ = T*cos.(cos.(x.-0.1))
A = 0.03*D2
tmp = similar(û₀)
p = (D,D2,T,Ti,tmp,similar(tmp))
function burgers_nl(dû,û,p,t)
    D,D2,T,Ti,u,tmp = p
    mul!(tmp, D, û)
    mul!(u, Ti, tmp)
    mul!(tmp, Ti, û)
    @. tmp = tmp*u
    mul!(u, T, tmp)
    @. dû = - u
end

#Reference solution using CVODE_BDF
prob = SplitODEProblem(DiffEqArrayOperator(Diagonal(A)), burgers_nl, û₀, (0.0,5.0), p)
sol  = solve(prob, Rodas5(autodiff=false); reltol=1e-12,abstol=1e-12)
test_sol = TestSolution(sol)

tslices=[0.0 1.0 2.0 3.0 5.0]
ys=hcat((Ti*sol(t) for t in tslices)...)
labels=["t=$t" for t in tslices]
plot(x,ys,label=labels)

#High tolerances
diag_linsolve=LinSolveFactorize(W->let tmp = tmp
    for i in 1:size(W, 1)
        tmp[i] = W[i, i]
    end
    Diagonal(tmp)
end)

#In-family comparisons
#1.IMEX methods (dense linear solver)
abstols = 0.1 .^ (5:8)
reltols = 0.1 .^ (1:4)
multipliers =  0.5 .^ (0:3)
setups = [Dict(:alg => IMEXEuler(linsolve=diag_linsolve), :dts => 1e-3 * multipliers),
          Dict(:alg => CNAB2(linsolve=diag_linsolve), :dts => 5e-3 * multipliers),
          Dict(:alg => CNLF2(linsolve=diag_linsolve), :dts => 5e-3 * multipliers),
          Dict(:alg => SBDF2(linsolve=diag_linsolve), :dts => 1e-3 * multipliers)]
labels = ["IMEXEuler" "CNAB2" "CNLF2" "SBDF2"]
@time wp1 = WorkPrecisionSet(prob,abstols,reltols,setups;
                            print_names=true,names=labels,
                            numruns=5,seconds=5,
                            save_everystop=false,appxsol=test_sol,maxiters=Int(1e5));#52s

plot(wp1,label=labels,markershape=:auto,title="IMEX methods, diagonal linsolve, low order")

# 2. ExpRK methods
abstols = 0.1 .^ (5:8) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (1:4)
multipliers = 0.5 .^ (0:3)
setups = [Dict(:alg => NorsettEuler(), :dts => 1e-3 * multipliers),
          Dict(:alg => ETDRK2(), :dts => 1e-2 * multipliers)]
labels = hcat("NorsettEuler",
              "ETDRK2 (caching)")
@time wp2 = WorkPrecisionSet(prob,abstols,reltols,setups;
                            print_names=true, names=labels,
                            numruns=5,
                            save_everystep=false, appxsol=test_sol, maxiters=Int(1e5)); #9.8s

plot(wp2, label=labels, markershape=:auto, title="ExpRK methods, low order")

#Between family comparisons
abstols = 0.1 .^ (5:8) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (1:4)
multipliers = 0.5 .^ (0:3)
setups = [Dict(:alg => CNAB2(linsolve=diag_linsolve), :dts => 5e-3 * multipliers)]
          #Dict(:alg => CNAB2(linsolve=LS_GMRES), :dts => 5e-3 * multipliers),
          Dict(:alg => ETDRK2(), :dts => 1e-2 * multipliers)]
labels = ["CNAB2 (dense linsolve)" "ETDRK2"]#"CNAB2 (Krylov linsolve)" "ETDRK2 (m=5)"]
@time wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                            print_names=true, names=labels,
                            numruns=5, error_estimate=:l2,
                            save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));#3.8s

plot(wp, label=labels, markershape=:auto, title="Between family, low orders")

#Low tolerances
#In-family comparisons
#1.IMEX methods (dense linear solver)
abstols = 0.1 .^ (7:13)
reltols = 0.1 .^ (4:10)
setups = [#Dict(:alg => KenCarp3(linsolve=diag_linsolve)),
          #Dict(:alg => KenCarp4(linsolve=diag_linsolve)),
          #Dict(:alg => KenCarp5(linsolve=diag_linsolve)),
          Dict(:alg => ARKODE(Sundials.Implicit(), order=3, linear_solver=:Band, jac_upper=1, jac_lower=1)),
          Dict(:alg => ARKODE(Sundials.Implicit(), order=4, linear_solver=:Band, jac_upper=1, jac_lower=1)),
          Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:Band, jac_upper=1, jac_lower=1))]
labels = hcat(#"KenCarp3", "KenCarp4", "KenCarp5",
              "ARKODE3", "ARKODE4", "ARKODE5")
@time wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                            print_names=true, names=labels,
                            numruns=5, error_estimate=:l2,
                            save_everystep=false, appxsol=test_sol, maxiters=Int(1e5)); #996s

plot(wp, label=labels, markershape=:auto, title="IMEX methods, dense linsolve, medium order")

#2.ExpRK methods
abstols = 0.1 .^ (7:11) # all fixed dt methods so these don't matter much
reltols = 0.1 .^ (4:8)
multipliers = 0.5 .^ (0:4)
setups = [Dict(:alg => ETDRK3(), :dts => 1e-2 * multipliers),
          Dict(:alg => ETDRK4(), :dts => 1e-2 * multipliers),
          Dict(:alg => HochOst4(), :dts => 1e-2 * multipliers)]
labels = hcat("ETDRK3 (caching)", "ETDRK4 (caching)",
              "HochOst4 (caching)"]#,"ETDRK4 (m=5)" "ETDRK3 (m=5)" "HochOst4 (m=5)")
@time wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                            print_names=true, names=labels,
                            numruns=5, error_estimate=:l2,
                            save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));

plot(wp, label=labels, markershape=:auto, title="ExpRK methods, medium order") #42s

#Between family comparisons
abstols = 0.1 .^ (7:11)
reltols = 0.1 .^ (4:8)
multipliers = 0.5 .^ (0:4)
setups = [#Dict(:alg => KenCarp5(linsolve=diag_linsolve)),
          Dict(:alg => ARKODE(Sundials.Implicit(), order=5, linear_solver=:Diagonal))]
          #Dict(:alg => KenCarp5(linsolve=LS_GMRES)),
          Dict(:alg => ETDRK3(), :dts => 1e-2 * multipliers),
          Dict(:alg => ETDRK4(), :dts => 1e-2 * multipliers)]
labels = hcat("ARKODE (dense linsolve)")#,"KenCarp5 (dense linsolve)" "KenCarp5 (Krylov linsolve)",
                        #"ARKODE (Krylov linsolve)", "ETDRK3 (m=5)", "ETDRK4 (m=5)")
@time wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                            print_names=true, names=labels,
                            numruns=5, error_estimate=:l2,
                            save_everystep=false, appxsol=test_sol, maxiters=Int(1e5));#178s

plot(wp, label=labels, markershape=:auto, title="Between family, medium order")


#Environment information
versioninfo()
using Pkg
println("OrdinaryDiffEq: ", Pkg.installed()["OrdinaryDiffEq"])
println("Sundials: ", Pkg.installed()["Sundials"])
