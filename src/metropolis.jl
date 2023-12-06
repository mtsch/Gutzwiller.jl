"""
    abstract type AbstractVQMCCollector

A supertype for structs to be passed to `metropolis_sample`. Subtypes of this type must
define the following interface:

* [`sample_type`](@ref)
* [`process_address`](@ref)
* [`print_stats`](@ref)
"""
abstract type AbstractVQMCCollector end

"""
    sample_type(::AbstractVQMCCollector, ansatz)

Return the result type corresponding to the collector.

See [`AbstractVQMCCollector`](@ref).
"""
sample_type


"""
    process_address(collector::AbstractVQMCCollector, ansatz, addr)

Do something with an address before storing it. This function should return a value of type
[`sample_type`](@ref)`(collector, ansatz)`.

See [`AbstractVQMCCollector`](@ref).
"""
process_address


"""
    print_stats(::IO, ::AbstractVQMCCollector, samples, accepted, elapsed)

Print stats that are to be shown at the end of each VQMC epoch. `samples` contain the
samples collected so far, `accepted` the numbers of acceptes samples, and `elapsed` is the
time spent so far.

Note that the number of samples collected, time spent and acceptance are printed by
[`metropolis_hasings`](@ref) by default.
"""
print_stats

"""
    AddressCollector() <: AbstractVQMCCollector

The address collector simply collects the addresses to vectors without processing them in
any way.

See [`AbstractVQMCCollector`](@ref), [`LocalEnergyCollector`](@ref).
"""
struct AddressCollector <: AbstractVQMCCollector end

sample_type(::AddressCollector, ansatz) = keytype(ansatz)
process_address(::AddressCollector, _, addr) = addr
function print_stats(io, ::AddressCollector, samples, accepted, elapsed)
    println(io)
end

"""
    LocalEnergyCollector(H) <: AbstractVQMCCollector

Collects local energies for each sample. At reporting intervals, the current value of
variational energy is shown.

See [`AbstractVQMCCollector`](@ref), [`LocalEnergyCollector`](@ref).
"""
struct LocalEnergyCollector{H} <: AbstractVQMCCollector
    hamiltonian::H
end
sample_type(::LocalEnergyCollector, _) = Float64
function process_address(lec::LocalEnergyCollector, ansatz, addr)
    return local_energy(lec.hamiltonian, ansatz, addr)
end
function print_stats(io, ::LocalEnergyCollector, samples, accepted, elapsed)
    μ_sum, σ_sq_sum = Folds.mapreduce(add, samples; init=(0.0, 0.0)) do sample
        b = blocking_analysis(sample)
        b.mean, b.err^2
    end
    μ = round(μ_sum / length(samples); sigdigits=4)
    σ = round(√(σ_sq_sum / length(samples)); sigdigits=4)
    println(io, "E_v = $μ±$σ")
end

###
### METROPOLIS-HASTINGS
###
"""
    metropolis_sample(sampler, ansatz, a_curr, v_curr) -> a_next, v_next, accepted

Sample a new address from `ansatz`, treating the aboslute square of the `ansatz` as a
probability density function. The `sample` is used to select a new candidate address,
`a_curr` is the previous address and `v_curr` is the absolute square of the ansatz evaluated
at `a_curr`, i.e. `v_curr = abs2(ansatz[a_curr])`.

Returns new address, new aboslute squared valuem and a boolean signaling whether the sample
was accepted or not.
"""
function metropolis_sample(sampler, ansatz, a_curr, v_curr)
    neighbours = num_offdiagonals(sampler, a_curr)
    a_next, _ = offdiagonals(sampler, a_curr)[rand(1:neighbours)]
    invprob_there = neighbours
    invprob_back = num_offdiagonals(sampler, a_next)
    v_next = abs2(ansatz[a_next])

    acceptance_prob = (v_next * invprob_there) / (v_curr * invprob_back)

    if acceptance_prob > rand()
        return a_next, v_next, true
    else
        return a_curr, v_curr, false
    end
end

"""
    metropolis_hastings!(collector, accum, address, sampler, ansatz, steps, warmup)
    metropolis_hastings!(collector, accums, addresses, accepted, sampler, ansatz, steps, warmup)

Using the Metropolis-Hastings algorithm, sample addresses from `ansatz`, process them
according to the `collector` (see [`AbstractVQMCCollector`](@ref)), and store the results
into `accum`. The first form is the sequential algorithm, while the second runs several runs
in parallel.

If warmup is set to `true`, no data is stored.

See [`metropolis_hastings`](@ref) for more information.
"""
function metropolis_hastings!(
    collector, accum::Union{Vector,Nothing}, a_curr, sampler, ansatz, steps, warmup
)
    if !warmup
        offset = length(accum)
        resize!(accum, length(accum) + steps)
        res_prev = process_address(collector, ansatz, a_curr)
    end
    v_curr = abs2(ansatz[a_curr])


    num_accepted = 0
    for k in 1:steps
        a_curr, v_curr, accepted = metropolis_sample(sampler, ansatz, a_curr, v_curr)
        num_accepted += accepted

        if !warmup
            if accepted
                res_prev = process_address(collector, ansatz, a_curr)
            end
            accum[k + offset] = res_prev
        end
    end
    return a_curr, num_accepted
end

function metropolis_hastings!(
    collector, accums, addrs, accepted, sampler, ansatz, steps, warmup
)
    Threads.@threads for i in 1:length(addrs)
        addr, acc = metropolis_hastings!(
            collector, accums[i], addrs[i], sampler, ansatz, steps, warmup
        )
        addrs[i] = addr
        if !warmup
            accepted[i] += acc
        end
    end
    return nothing
end

"""
    MetropolisResult{A,T}

Holds the results of a variational quantum Monte Carlo computation.

# Fileds
* `samples`: the samples collected
* `addresses`: the last addresses encountered. Can be used to continue a computation.
* `accepted`: the total number of accepted spawns.
* `elapsed`: the total time elapsed.
"""
mutable struct MetropolisResult{T,A}
    samples::Vector{Vector{T}}
    addresses::Vector{A}
    accepted::Vector{Int}
    elapsed::Float64
end
function Base.show(io::IO, res::MetropolisResult{T}) where {T}
    print(io, "MetropolisResult{$T} with $(sum(length, res.samples)) samples")
end
function collect_to_vec!(dv, res::MetropolisResult)
    for k in Iterators.flatten(res.samples)
        dv[k] += 1
    end
    return dv
end

function Rimu.DVec(res::MetropolisResult{A,A}; kwargs...) where {A}
    collect_to_vec!(DVec{A,Float64}(; kwargs...), res)
end
function Rimu.PDVec(res::MetropolisResult{A,A}; kwargs...) where {A}
    collect_to_vec!(PDVec{A,Float64}(; kwargs...), res)
end

"""
    metropolis_hastings(collector, sampler, ansatz; kwargs...)

Returns [`MetropolisResult`](@ref).

# Interfaces

* The `collector` must implement the [`AbstractVQMCCollector`](@ref) interface.

* The `sampler` must implement `offdiagonals` and `num_offdiagonals`.

* The `ansatz` must implement `Base.getindex` and `Base.keytype` and must be indexable with
  addresses the sampler supports.

# Keyword arguments

* `samples = 1e7`: the total number of samples to collect. The actual number of samples will
  be rounded to fit neatly into `walkers` and `epochs`.

* `walkers_per_thread = 2`: number of walkers to use per thread. Is set to 1 if threading is
  unavailable.

* `walkers = walkers_per_thread * Threads.nthreads()`: the total number of walkers to use.

* `epochs = 10`: the number of epochs. If verbose is set to true, statistics are printed
  after each epoch.

* `steps = round(Int, samples / walkers / epochs)`: number of steps per walker per epoch.

* `warmup = 1e6`: the number of steps to perform at the beginning (without collecting
  statistics).

* `verbose = true`: if true, print statistic after each epoch.

* `continue_from`: optional [`MetropolisResult`](@ref). If given, the computation will be
  continued from that result.

"""
function metropolis_hastings(
    collector::AbstractVQMCCollector, sampler, ansatz;
    samples=1e7,
    walkers_per_thread=Threads.nthreads() == 1 ? 1 : 2,
    walkers=walkers_per_thread * Threads.nthreads(),
    epochs=10,
    steps=round(Int, samples / walkers / epochs),
    warmup=1e6,
    verbose=true,
    continue_from=nothing,
)
    start_time = time()

    if isnothing(continue_from)
        addresses = [starting_address(sampler) for _ in 1:walkers]
        results = [sample_type(collector, ansatz)[] for _ in 1:walkers]
        accepted = zeros(Int, walkers)
    else
        addresses = continue_from.addresses
        results = continue_from.samples
        accepted = continue_from.accepted
    end

    # warmup
    metropolis_hastings!(collector, results, addresses, accepted, sampler, ansatz, Int(warmup), true)

    # measure
    for epoch in 1:epochs
        metropolis_hastings!(collector, results, addresses, accepted, sampler, ansatz, steps, false)
        if verbose
            elapsed = time() - start_time
            N = epoch * steps * walkers
            acceptance = round(sum(accepted) / N * 100, digits=3)
            rate = round(N / elapsed, sigdigits=4)

            print(stderr, "Epoch ", lpad(epoch, length(string(epochs))), "/", epochs, ": ")
            print(stderr, float(N), " samples,")
            print(stderr, " $rate steps/s, acceptance = $acceptance%, ")
            print_stats(stderr, collector, results, accepted, elapsed)
        end
    end

    elapsed = time() - start_time
    if verbose
        print(stderr, "DONE in ")
        time_format(stderr, elapsed)
        println(stderr)
    end
    if !isnothing(continue_from)
        elapsed += continue_from.elapsed
    end
    return MetropolisResult(results, addresses, accepted, elapsed)
end

export MHVQMCGutzOptimizer
struct MHVQMCGutzOptimizer{H}
    hamiltonian::H
    steps::Int
    walkers::Int
end

function MHVQMCGutzOptimizer(ham; steps=1e6, walkers=Threads.nthreads())
    return MHVQMCGutzOptimizer(ham, Int(steps), walkers)
end
function Base.show(io::IO, o::MHVQMCGutzOptimizer)
    print(io, "MHVQMCGutzOptimizer(", o.hamiltonian, "; ", " steps = ", o.steps, ")")
end

function (o::MHVQMCGutzOptimizer)(params)
    g = params[1]
    ham = o.hamiltonian
    collector = LocalEnergyCollector(ham)

    res = metropolis_hastings(collector, ham, GutzwillerAnsatz(ham, g); steps=o.steps, walkers=o.walkers, verbose=false, epochs=1)

    return mean(mean(s) for s in res.samples)
end
