mutable struct GutzwillerVQMC{A,H}
    hamiltonian::H
    warmup::Int
    steps::Int
    substeps::Int
    walkers::Int
    verbose::Bool

    is_warm::Bool
    samples::Vector{Vector{A}}
    addresses::Vector{A}
    accepted::Vector{Int}
end
function GutzwillerVQMC(
    hamiltonian;
    warmup=1e6,
    samples=1e7,
    walkers_per_thread=Threads.nthreads() == 1 ? 1 : 2,
    walkers=Threads.nthreads() * walkers_per_thread,
    steps=round(samples / walkers),
    substeps=1,
    verbose=true,
)
    addresses = [starting_address(hamiltonian) for _ in 1:walkers]
    samples = [eltype(addresses)[] for _ in 1:walkers]
    accepted = zeros(Int, walkers)
    return GutzwillerVQMC(
        hamiltonian, Int(warmup), Int(steps), Int(substeps), Int(walkers), verbose,
        false, samples, addresses, accepted,
    )
end
function Base.show(io::IO, ge::GutzwillerVQMC)
    println(io, "GutzwillerVQMC(")
    for property in (:hamiltonian, :warmup, :steps, :substeps, :walkers, :verbose, :is_warm)
        val = getproperty(ge, property)
        println(io, "$property: $val")
    end
    print(io, "total samples: ", sum(length, ge.samples))
end

function _warmup!(ge::GutzwillerVQMC, g)
    if ge.verbose
        print(stderr, "Warming up... ")
    end
    elapsed = @elapsed begin
        ansatz = GutzwillerAnsatz(ge.hamiltonian, g)
        metropolis_hastings!(
            nothing, ge.addresses, ge.accepted, ge.hamiltonian, ansatz, ge.warmup, 1
        )
        ge.is_warm = true
    end
    if ge.verbose
        print(stderr, "done in ")
        time_format(stderr, elapsed)
        rate = round(ge.warmup * ge.walkers / elapsed, sigdigits=3)
        println(stderr, " ($rate steps/s)")
    end
    return ge
end

function collect_samples!(ge::GutzwillerVQMC, g; restart=false)
    if restart
        foreach(empty!, ge.samples)
        ge.addresses .= starting_address(ge.H)
        ge.accepted .= 0
        _warmup!(ge, g)
    elseif !ge.is_warm
        _warmup!(ge, g)
    end
    _collect_samples!(ge, g)
end

function _collect_samples!(ge, g)
    if ge.verbose
        print(stderr, "Running...    ")
    end
    elapsed = @elapsed begin
        ansatz = GutzwillerAnsatz(ge.hamiltonian, g)
        metropolis_hastings!(
            ge.samples, ge.addresses, ge.accepted,
            ge.hamiltonian, ansatz, ge.steps, ge.substeps,
        )
    end
    if ge.verbose
        print(stderr, "done in ")
        time_format(stderr, elapsed)
        rate = round(ge.steps * ge.walkers * ge.substeps / elapsed, sigdigits=3)
        println(stderr, " ($rate steps/s)")
    end
    return ge
end

function (ge::GutzwillerVQMC)(g)
    if isempty(ge.samples[1])
        throw(ArgumentError("no samples collected! Use `collect_samples!` first"))
    end

    if ge.verbose
        print(stderr, "Evaluating... ")
    end
    elapsed = @elapsed begin
        T = promote_type(typeof(g), eltype(ge.hamiltonian))
        series = [zeros(T, length(ge.samples[w])) for w in 1:ge.walkers]
        result = Folds.sum(zip(series, ge.samples)) do (res, smp)
            gvec = GutzwillerAnsatz(ge.hamiltonian, g)
            local_energy!(res, smp, ge.hamiltonian, gvec)
            mean(res)
        end / ge.walkers
    end
    if ge.verbose
        print(stderr, "done in ")
        time_format(stderr, elapsed)
        rate = round(sum(length, ge.samples) / elapsed, sigdigits=3)
        println(stderr, " ($rate steps/s)")
    end
    return result
end
