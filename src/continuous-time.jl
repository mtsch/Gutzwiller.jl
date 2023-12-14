using Rimu, Gutzwiller, ForwardDiff, Tables

using StatsBase
using Parameters

function mean_energy(state; skip=0)
    w = FrequencyWeights(state.residence_times[skip+1:end])
    μ = mean(state.local_energies[skip+1:end], w)
    σ = √var(state.local_energies[skip+1:end], w; mean=μ)
    return μ, σ
end

function _to_vec!(vec, state::ContinuousTimeVQMCWalkerState)
    for (k, v) in zip(state.addresses, state.residence_times)
        deposit!(vec, k, v, nothing)
    end
    return vec
end
function Rimu.DVec(state::ContinuousTimeVQMCWalkerState{A}; kwargs...) where {A}
    return _to_vec!(DVec{A,Float64}(; kwargs...), state)
end
function Rimu.PDVec(state::ContinuousTimeVQMCWalkerState{A}; kwargs...) where {A}
    return _to_vec!(PDVec{A,Float64}(; kwargs...), state)
end

struct CTVQMCGutzOptimizer{H}
    hamiltonian::H
    steps::Int
    walkers::Int
end
function CTVQMCGutzOptimizer(ham; steps=1e6, walkers=Threads.nthreads())
    return CTVQMCGutzOptimizer(ham, Int(steps), walkers)
end

function Base.show(io::IO, o::CTVQMCGutzOptimizer)
    print(io, "CTVQMCGutzOptimizer(", o.hamiltonian, "; ", " steps = ", o.steps, ")")
end

function (o::CTVQMCGutzOptimizer)(params; skip=0)
    g = params[1]
    ham = o.hamiltonian
    states = [
        ContinuousTimeVQMCWalkerState(ham, GutzwillerAnsatz(ham, g)) for _ in 1:o.walkers
    ]
    continuous_time_vqmc!(states; steps=o.steps)
    return mean(mean_energy(st)[1] for st in states)
end


function series(o::CTVQMCGutzOptimizer, params)
    g = params[1]
    ham = o.hamiltonian
    state = [
        ContinuousTimeVQMCWalkerState(ham, GutzwillerAnsatz(ham, g))
    ]
    continuous_time_vqmc!(state; steps=o.steps)
    return state[1].residence_times, state[1].local_energies
end

function series(o::MHVQMCGutzOptimizer, params)
    g = params[1]
    ham = o.hamiltonian
    collector = LocalEnergyCollector(ham)

    res = metropolis_hastings(collector, ham, GutzwillerAnsatz(ham, g); steps=o.steps, walkers=1, verbose=false, epochs=1)
    @show res.accepted[1] / length(res.samples[1])

    return res.samples[1]
end
