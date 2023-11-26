struct VQMCAccumulator{A}
    substeps::Int
    vec_sample::Vector{A}
end
result_type(::VQMCAccumulator{A}) where {A} = VQMCResult{A}

struct VQMCResult{A}
    vec_samples::Vector{Vector{A}}
end

function accumulate!(va::VQMCAccumulator, addr, _, step)
    if step % substeps == 0
        push!(va.vec_sample, addr)
    end
end
function merge!(vas::Vector{VQMCAccumulator})
    return VQMCResult(vas)
end

function GutzwillerQMCEvaluator2{H}
    hamiltonian::H
    blocks::Int
    warmup::Int
    steps::Int
    substeps::Int
end

function gutzwiller2(
    H, g; walkers_per_thread=2, blocks=100, steps=1e5, substeps=1, warmup=1e6, verbose=true
)
    walkers = walkers_per_thread * Threads.nthreads()
    vec_samples = [typeof(starting_address(H))[] for _ in 1:walkers]
    Ev_samples = [Float64[] for _ in 1:walkers]
    accums = [VQMCAccumulator(substeps, vec_samples[i]) for i in 1:walkers]

    steps_per_block = steps * substeps

    for iter in 1:blocks
        if iter > 1
            warmup = 0
        end
        Threads.@threads for walker in 1:walkers
            metropolis_hastings!(
                accums[walker], false, H, GutzVector(H, g), warmup, steps_per_block
            )
            resize!(Ev_samples[w], length(vec_samples[w]))
        end
        Threads.@threads for i in 1:walkers
            evaluator = GutzwillerEvaluator(H, vec_samples[i])
            for j in eachindex(vec_samples[i])
            end
        end
    end
end

# Inputs:
#
# walkers_per_thread, blocks, steps, substeps, warmup
