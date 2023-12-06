using Rimu, Gutzwiller, ForwardDiff

using StatsBase
using Parameters

"""

https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.8b00780
"""
function continuous_time_sample!(prob_buffer, hamiltonian, ansatz, addr1)
    offdiags = offdiagonals(hamiltonian, addr1)

    resize!(prob_buffer, length(offdiags))

    val1 = ansatz[addr1]
    local_energy_num = diagonal_element(hamiltonian, addr1) * val1
    residence_time_denom = 0.0

    for (i, (addr2, melem)) in enumerate(offdiags)
        val2 = ansatz[addr2]
        residence_time_denom += val2
        local_energy_num += melem * val2

        prob_buffer[i] = residence_time_denom
    end

    residence_time = val1 / residence_time_denom
    local_energy = local_energy_num / val1

    chosen = pick_random(prob_buffer)
    new_addr, _ = offdiags[chosen]

    return new_addr, residence_time, local_energy
end

@inline function pick_random(cumsum)
    chosen = rand() * last(cumsum)
    i = 1
    @inbounds while true
        if chosen < cumsum[i]
            return i
        end
        i += 1
    end
end

mutable struct ContinuousTimeVQMCWalkerState{A,H,V,T<:Real}
    hamiltonian::H
    ansatz::V
    curr_address::A
    residence_times::Vector{T}
    local_energies::Vector{T}
    addresses::Vector{A}
    prob_buffer::Vector{T}
end
function ContinuousTimeVQMCWalkerState(hamiltonian::H, ansatz::V) where {H,V}
    A = keytype(ansatz)
    T = promote_type(valtype(ansatz), eltype(hamiltonian))
    return ContinuousTimeVQMCWalkerState{A,H,V,T}(
        hamiltonian, ansatz, starting_address(hamiltonian), T[], T[], A[], T[],
    )
end
function Base.empty!(st::ContinuousTimeVQMCWalkerState)
    empty!(st.residence_times)
    empty!(st.local_energies)
    empty!(st.addresses)
    curr_address = starting_address(st.hamiltonian)
    return st
end

function continuous_time_vqmc!(st::ContinuousTimeVQMCWalkerState; steps)
    curr_addr = st.curr_address
    @unpack hamiltonian, ansatz, residence_times, local_energies, addresses, prob_buffer = st

    fst_step = length(residence_times) + 1
    lst_step = length(residence_times) + steps

    resize!(residence_times, lst_step)
    resize!(local_energies, lst_step)
    resize!(addresses, lst_step)

    for k in fst_step:lst_step
        curr_addr, residence_time, local_energy = continuous_time_sample!(
            prob_buffer, hamiltonian, ansatz, curr_addr
        )
        residence_times[k] = residence_time
        local_energies[k] = local_energy
        addresses[k] = curr_addr
    end
    st.curr_address = curr_addr
    return st
end
function continuous_time_vqmc!(sts::Vector{<:ContinuousTimeVQMCWalkerState}; steps)
    Threads.@threads for walker in eachindex(sts)
        continuous_time_vqmc!(sts[walker]; steps)
    end
    return sts
end

function continuous_time_vqmc(
    hamiltonian, ansatz;
    steps=1e6,
    walkers=Threads.nthreads() * 2,
    )

    if walkers > 1
        state = [ContinuousTimeVQMCWalkerState(hamiltonian, ansatz) for _ in 1:walkers]
    else
        state = ContinuousTimeVQMCWalkerState(hamiltonian, ansatz)
    end
    continuous_time_vqmc!(state; steps)

    return state
end

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

    return res.samples[1]
end

function len_norepeat(x)
    len = 1
    prev = x[1]
    for i in 2:length(x)
        if x[i] ≠ prev
            prev = x[i]
            len += 1
        end
    end
    return len
end

using LaTeXStrings
function plotsct(x_ct, y_ct, y_mh)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel="?", ylabel=L"E_L")

    actual_len = len_norepeat(y_mh)
    x_ct = x_ct[1:actual_len]
    y_ct = y_ct[1:actual_len]

    x_ct = circshift(cumsum(x_ct), 1)
    x_ct[1] = 0
    x_ct ./= x_ct[end]


    lines!(ax, x_ct, y_ct; label="CT")
    lines!(ax, range(0, 1; length=length(y_mh)), y_mh; label="MH")
    Legend(fig[:,2], ax)

    return fig
end
