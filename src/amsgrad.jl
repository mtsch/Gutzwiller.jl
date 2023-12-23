# m[i] = f(grad)
# n[i] = g(grad)
# delta_p[j] = -αm[j]/√n[j] # update to params
using DataFrames, StaticArrays

# TODO: _continue_ computations to prevent issues with warmup.
#

function amsgrad(f, params; kwargs...)
    params_svec = SVector{length(params)}(params)
    return amsgrad(f, params_svec; kwargs...)
end
function amsgrad(
    f, params::SVector{N,T};
    maxiter=100, α=0.1, β1=0.5, β2=β1^2, verbose=true,
    gtol=√eps(Float64),
    ptol=√eps(Float64),
) where {N,T}
    df = DataFrame()

    first_moment = zeros(SVector{N,T})
    second_moment = zeros(SVector{N,T})


    val, grad = val_and_grad(f, params)
    first_moment = grad

    for iter in 1:maxiter
        val, grad = val_and_grad(f, params)
        first_moment = (1 - β1) * first_moment + β1 * grad
        second_moment = max.((1 - β2) * second_moment + β2 * grad.^2, second_moment)
        Δparams = α .* first_moment ./ sqrt.(second_moment)

        if verbose
            print(stderr, "Step $iter:")
            print(stderr, " y: ", round(val, sigdigits=5))
            print(stderr, ", x: ", round.(params, sigdigits=5))
            print(stderr, ", Δx: ", round.(Δparams, sigdigits=5))
            println(stderr, ", ∇: ", round.(grad, sigdigits=5))
        end


        push!(df, (; iter, val, grad, params, Δparams))
        params -= Δparams

        if norm(Δparams) < ptol
            break
        end
        if norm(grad) < gtol
            break
        end
    end
    return df
end


function find_worst_t(N=10, M=10)
    mingap = Inf
    mint = Inf

    for t in 0.01:0.01:0.5
        addr = near_uniform(BoseFS{N,M})
        H = HubbardReal1D(addr; t)
        res = eigsolve(sparse(H), 2, :SR)
        gap = res[1][2] - res[1][1]
        println("t: $t, gap: $gap")

        if gap < mingap
            mingap = gap
            mint = t
        end
    end
    return mint, mingap
end

if false
    tw = 100_000

    addr = near_uniform(BoseFS{20,20})
    H = HubbardReal1D(addr; t=0.2)
    s_strat = DoubleLogUpdate(targetwalkers=tw)

    df_reg, _ = lomc!(H; laststep=10_000, s_strat)


    qmc = KineticVQMC(H, GutzwillerAnsatz(H); samples=5e6)
    ams = amsgrad(qmc, [0.5]; maxiter=1000, ptol=1e-5)
    par = ams.params[end]
    res = kinetic_vqmc(H, GutzwillerAnsatz(H), par; samples=1e7)
    #pv = map!(sqrt, values(PDVec(res; style=IsDynamicSemistochastic())))
    pv = PDVec(res; style=IsDynamicSemistochastic())
    VectorInterface.scale!(pv, tw / walkernumber(pv))
    compress!(pv)
    val, _ = val_and_grad(res)

    params = RunTillLastStep(shift=val)
    df_fancy, _ = lomc!(GutzwillerSampling(H, par[1]), pv; laststep=10_000, s_strat, params)


    k1 = 1
    k2 = 10000
    f = lines(df_reg.shift[k1:k2]; label="reg").figure
    lines!(df_fancy.shift[k1:k2]; label="fancy")
    f

    k1 = 1
    k2 = 10000
    f = lines(df_reg.len[k1:k2]; label="reg").figure
    lines!(df_fancy.len[k1:k2]; label="fancy")
    f

end

if false

    addr = near_uniform(BoseFS{8,8})
    H = HubbardReal1D(addr; t=0.2)

    best = optimize(GutzwillerEvaluator(H), [0.5]).minimizer

    G = GutzwillerSampling(H, 1.259)

    resH = eigsolve(H, PDVec(addr => 1.0), 1, :SR)
    resG = eigsolve(G, PDVec(addr => 1.0), 1, :SR)
end
