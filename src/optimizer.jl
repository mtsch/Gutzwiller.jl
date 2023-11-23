struct GutzwillerOptimizer{H,P,W}
    hamiltonian::H
    diags::P
    buffer::P
    working_memory::W
end

function GutzwillerOptimizer(ham)
    basis = build_basis(ham)
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

function gutz_optimize(ham; verbose=true)
    verbose && @info "Building optimizer..."
    el = @elapsed go = GutzwillerOptimizer(ham)
    verbose && @info "Done in $el seconds."

    optimize(go, [0.5]).minimizer[1]
end
