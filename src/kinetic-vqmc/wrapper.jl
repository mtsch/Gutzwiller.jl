using DiffResults
using ForwardDiff
using StatsBase

struct KineticVQMC{H,A}
    hamiltonian::H
    ansatz::A
    steps::Int
    walkers::Int
    var::Bool
end
function KineticVQMC(
    hamiltonian::H, ansatz::A;
    samples=1e6,
    walkers=Threads.nthreads() > 1 ? Threads.nthreads() * 2 : 1,
    steps=round(Int, samples / walkers),
    var=false,
) where {H,A}
    return KineticVQMC{H,A}(hamiltonian, ansatz, steps, walkers, var)
end

function (ke::KineticVQMC)(params)
    g = params[1]
    ansatz = set_params(ke.ansatz, g)

    res = kinetic_vqmc(ke.hamiltonian, ansatz; steps=ke.steps, walkers=ke.walkers)
    if ke.var
        return var(res)
    else
        return mean(res)
    end
end
