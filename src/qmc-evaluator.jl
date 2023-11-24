"""
    GutzVector{A,H}

Placeholder TODO: make a proper DVec
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
function to_vec!(dst, gv::GutzVector)
    basis = build_basis(gv.H)
    for k in basis
        dst[k] = gv[k]
    end
    return dst
end
function Rimu.DVec(gv::GutzVector; kwargs...)
    A = typeof(starting_address(gv.H))
    result = DVec{A,Float64}(; kwargs...)
    return to_vec!(result, gv)
end
function Rimu.PDVec(gv::GutzVector; kwargs...)
    A = typeof(starting_address(gv.H))
    result = PDVec{A,Float64}(; kwargs...)
    return to_vec!(result, gv)
end

struct GutzwillerQMCEvaluator{H}
    hamiltonian::H
    warmup::Int
    steps::Int
    tasks::Int
end

function GutzwillerQMCEvaluator(H; warmup=1e6, steps=1e6, tasks=2Threads.nthreads())
    return GutzwillerQMCEvaluator(H, Int(warmup), Int(steps), Int(tasks))
end
function Base.show(io::IO, ge::GutzwillerQMCEvaluator)
    print(io, "GutzwillerQMCEvaluator(")
    print(io, ge.hamiltonian)
    print(io, ", warmup=", ge.warmup)
    print(io, ", steps=", ge.steps)
    print(io, ", tasks=", ge.tasks)
    print(io, ")")
end

function (ge::GutzwillerQMCEvaluator)(input)
    g = only(input)
    H = ge.hamiltonian
    return metropolis_hastings(
        VariationalEnergyAccumulator,
        H, GutzVector(H, g); warmup=ge.warmup, steps=ge.steps, tasks=ge.tasks,
    )
end
