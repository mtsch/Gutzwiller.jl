###
### METROPOLIS-HASTINGS
###
function metropolis_sample(H, vec, a_curr, v_curr)
    neighbours = num_offdiagonals(H, a_curr)
    a_next, _ = offdiagonals(H, a_curr)[rand(1:neighbours)]
    invprob_there = neighbours
    invprob_back = num_offdiagonals(H, a_next)
    v_next = abs2(vec[a_next])

    acceptance_prob = (v_next * invprob_there) / (v_curr * invprob_back)

    if acceptance_prob > rand()
        return a_next, v_next, true
    else
        return a_curr, v_curr, false
    end
end

function metropolis_hastings!(accum, progress, H, vec, warmup, steps)
    t0 = time()
    a_curr = starting_address(H)
    v_curr = abs2(vec[a_curr])

    num_accepted = 0
    for k in 1:warmup
        a_curr, v_curr, _ = metropolis_sample(H, vec, a_curr, v_curr)
        if !isnothing(progress)
            next!(progress)
        end
    end
    for k in 1:steps
        a_curr, v_curr, accepted = metropolis_sample(H, vec, a_curr, v_curr)
        num_accepted += accepted

        # Do stuff here
        accumulate!(accum, a_curr, accepted, k)
        if !isnothing(progress)
            next!(progress)
        end
    end
    elapsed = time() - t0
    return finalize!(accum, num_accepted / steps, warmup, steps, elapsed)
end

function metropolis_hastings(
    ::Type{A}, H, vec;
    warmup=1e6, steps=1e6, tasks=2 * Tasks.ntasks(), progress=isinteractive(),
) where {A}
    if tasks == 1
        accum = A(H, vec)
        if progress
            prog = Progress(warmup + steps; showspeed=true)
        else
            prog = nothing
        end
        result = metropolis_hastings!(accum, prog, H, vec, Int(warmup), Int(steps))
        if progress
            finish!(prog)
        end
        return result
    else
        results = Vector{result_type(A, H, vec)}(undef, tasks)
        if progress
            prog = Progress((warmup + steps) * tasks; showspeed=true)
        else
            prog = nothing
        end
        Threads.@threads for i in 1:tasks
            accum = A(H, vec)
            results[i] = metropolis_hastings!(accum, prog, H, vec, Int(warmup), Int(steps))
        end
        if progress
            finish!(prog)
        end
        return merge(results)
    end
end

###
### VECTOR ACCUMULATOR
###
struct VectorAccumulator{D<:AbstractDVec}
    vector::D
end
function VectorAccumulator(H::AbstractHamiltonian, _)
    return VectorAccumulator(DVec{typeof(starting_address(H)), Float64}())
end
function result_type(::Type{VectorAccumulator}, H::AbstractHamiltonian, _)
    return VectorAccumulatorResult{DVec{typeof(starting_address(H)),Float64}}
end
function accumulate!(va::VectorAccumulator, addr, args...)
    va.vector[addr] += 1
end
function finalize!(va::VectorAccumulator, args...)
    return VectorAccumulatorResult(normalize!(va.vector))
end

struct VectorAccumulatorResult{D}
    vector::D
end
function Base.merge(rs::Vector{VectorAccumulatorResult})
    return VectorAccumulatorResult(sum(r.vector for r in rs))
end

###
### VARIATIONAL ENERGY
###
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

struct VariationalEnergyAccumulator{H,V}
    hamiltonian::H
    vector::V
    local_energies::Vector{Float64}
end
function VariationalEnergyAccumulator(H, v)
    return VariationalEnergyAccumulator(H, v, Float64[])
end
function result_type(VariationalEnergyAccumulator, _, _)
    VariationalEnergyResult{Vector{Float64}}
end
function accumulate!(lea::VariationalEnergyAccumulator, addr, accepted, _)
    if accepted || length(lea.local_energies) == 0
        push!(lea.local_energies, local_energy(lea.hamiltonian, lea.vector, addr))
    else
        push!(lea.local_energies, lea.local_energies[end])
    end
end
function finalize!(lea::VariationalEnergyAccumulator, acceptance, warmup, steps, elapsed)
    blocking = blocking_analysis(lea.local_energies)
    return VariationalEnergyResult(
        blocking.mean, blocking.err, acceptance, lea.local_energies, warmup, steps, warmup+steps, elapsed
    )
end

struct VariationalEnergyResult{V,T}
    mean::Float64
    err::Float64
    acceptance::Float64
    local_energies::V
    warmup::Int
    steps::Int
    total::Int
    times::T
end
function Base.show(io::IO, res::VariationalEnergyResult)
    μ = lpad(round(res.mean, sigdigits=5), 7)
    σ = lpad(round(res.err, sigdigits=5), 7)
    acc = lpad(round(res.acceptance * 100, digits=3), 7)
    print(io, "$μ ± $σ, $acc% acceptance")
end
function Base.merge(rs::Vector{<:VariationalEnergyResult})
    μ = mean(r.mean for r in rs)
    σ = √mean(r.err^2 for r in rs)
    acceptance = mean(r.acceptance for r in rs)
    local_energies = [r.local_energies for r in rs]
    times = [r.times for r in rs]
    warmup = rs[1].warmup
    steps = rs[1].steps
    total = (warmup + steps) * length(rs)
    return VariationalEnergyResult(
        μ, σ, acceptance, local_energies, warmup, steps, total, times
    )
end
