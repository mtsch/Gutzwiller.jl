using Gutzwiller
using Rimu
using Optim

include("setup.jl")
H = setup_H(3, t=0.1)

if false
    r_det = gutz_optimize(H, 0.5)
    r_qmc = gutz_optimize(H, 0.5; qmc=true, warmup=1e5, steps=1e5)
    r_opt = optimize(GutzwillerEvaluator(H), [0.5])
end
