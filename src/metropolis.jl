"""
    local_energy(H, vector, addr)
    local_energy(H, vector)

Compute the local energy of address `addr` in `vector` with respect to `H`. If `addr` is not
given, compute the local energy across the whole vector.
"""
function local_energy(H, vector, addr1)
    bot = vector[addr1]
    top = sum(offdiagonals(H, addr1)) do (addr2, melem)
        melem * vector[addr2]
    end
    top += diagonal_element(H, addr1) * bot
    return top / bot
end

function local_energy(H, vector)
    top = sum(pairs(vector)) do (k, v)
        local_energy(H, vector, k) * v^2
    end
    return top / sum(abs2, vector)
end

"""
    GutzVector{A,H}

Placeholder
"""
struct GutzVector{A,H}
    H::H
    g::Float64
end
function GutzVector(H, g)
    A = typeof(starting_address(H))
    return GutzVector{A,typeof(H)}(H,g)
end
function Base.getindex(gv::GutzVector{A}, addr::A) where {A}
    exp(-gv.g * diagonal_element(gv.H, addr))
end

struct MetropolisHastingsResult
    val::Float64
    err::Float64
    accepted::Float64
    val_replicas::Vector{Vector{Float64}}
    accepted_replicas::Vector{Float64}
end
function Base.show(io::IO, res::MetropolisHastingsResult)
    print(io, "MetropolisHastingsResult(val=$(res.val), err=$(res.err), accepted=$(res.accepted))")
end

"""
    walk(H, vec, warmup, steps, report_every)

Self-contained Metropolis-Hastings random walk. Performs `warmup + steps` steps, but only
records data after `warmup` steps. Print a report every `report_every` steps.
"""
function walk(H, vec, warmup, steps, report_every)
    a_curr = starting_address(H)
    v_curr = abs2(vec[a_curr])
    neighbours = num_offdiagonals(H, a_curr)

    accepted = 0
    local_energies = Float64[]

    last_accepted = true
    for k in 1:warmup + steps
        if report_every > 0 && k % report_every == 0
            if k > warmup
                blocking = blocking_analysis(local_energies)
                μ = blocking.mean
                σ = blocking.err
                acceptance = round(100 * accepted / (k - warmup), digits=2)
                println("step $k: E_v = $μ ± $σ, $acceptance% acceptance")
            else
                println("step $k: still warming up...")
            end
        end

        a_next, _ = offdiagonals(H, a_curr)[rand(1:neighbours)]
        invprob_there = neighbours
        invprob_back = num_offdiagonals(H, a_next)
        v_next = abs2(vec[a_next])

        acceptance_prob = (v_next * invprob_there) / (v_curr * invprob_back)
        if acceptance_prob > rand()
            a_curr = a_next
            v_curr = v_next
            neighbours = invprob_back
            accepted += k > warmup
            last_accepted = true
        end
        if k > warmup
            if last_accepted
                push!(local_energies, local_energy(H, vec, a_curr))
            else
                # sample is repeated - no need to recompute
                push!(local_energies, local_energies[end])
            end
            last_accepted = false
        end
    end
    return local_energies, accepted / steps
end

function sequential_walk(H, vec; warmup=1e6, steps=1e6, report_every=steps/10)
    local_energies, accepted = walk(H, vec, Int(warmup), Int(steps), Int(report_every))
    blocking = blocking_analysis(local_energies)
    return MetropolisHastingsResult(
        blocking.mean, blocking.err, accepted, [local_energies], [accepted],
    )
end

function parallel_walk(H, vec; warmup=1e6, steps=1e6, replicas=Threads.nthreads())
    return parallel_walk(H, vec, Int(warmup), Int(steps), replicas)
end
function parallel_walk(H, vec, warmup, steps, replicas)
    accepted = zeros(replicas)
    local_energies = [Float64[] for _ in 1:replicas]

    Threads.@threads for i in 1:replicas
        loc_en, acc = walk(H, vec, warmup, steps, 0)
        accepted[i] = acc
        local_energies[i] = loc_en
    end

    blocks = map(blocking_analysis, local_energies)
    val = mean(b.mean for b in blocks)
    err = √mean(abs2, b.err for b in blocks)

    return MetropolisHastingsResult(
        val,
        err,
        mean(accepted),
        local_energies,
        accepted,
    )
end

"""
    gutzwiller_energy(H, g; warmup=1e6, steps=1e6, replicas=Threads.nthreads())

Use the Metropolis-Hastings algorithm to compute an estiamte of the energy of the Gutzwiller
ansatz

```math
φ_i = e^{-g H_ii}
```

for a given value of `g`. The algorithm runs for `warmup + steps` steps, and the first
`warmup` steps are discarded. It runs `replicas` independent runs in parallel.
"""
function gutzwiller_energy(H, g; replicas=Threads.nthreads(), report_every=0, kwargs...)
    gutzy = GutzVector(H, g)
    if replicas > 1
        if report_every ≠ 0
            throw(ArgumentError("Reporting in parallel no bueno"))
        end
        return parallel_walk(H, gutzy; replicas, kwargs...)
    else
        return sequential_walk(H, gutzy; report_every, kwargs...)
    end
end
