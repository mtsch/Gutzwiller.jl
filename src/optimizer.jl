struct GutzwillerOptimizer{H,P,W}
    hamiltonian::H
    diags::P
    buffer::P
    working_memory::W
end

function GutzwillerOptimizer(ham)
    basis = build_basis(ham)
    return GutzwillerOptimizer(ham)
end

function GutzwillerOptimizer(ham, basis)
    diags = PDVec(zip(basis, map(a -> diagonal_element(ham, a), basis)))
    buffer = copy(diags)
    working_memory = PDWorkingMemory(buffer)

    return GutzwillerOptimizer(ham, diags, buffer, working_memory)
end

function (go::GutzwillerOptimizer)(params)
    g = params[1]
    φ = map!(go.buffer, values(go.diags)) do d
        exp(-d * g)
    end
    return dot(φ, go.hamiltonian, φ, go.working_memory) / norm(φ)^2
end

function gutz_optimize(ham, basis=nothing; verbose=true)
    verbose && @info "Building optimizer..."
    if isnothing(basis)
        el = @elapsed go = GutzwillerOptimizer(ham)
    else
        el = @elapsed go = GutzwillerOptimizer(ham, basis)
    end
    verbose && @info "Done in $el seconds. Optimizing..."
    optimize(go, [0.5])
end
