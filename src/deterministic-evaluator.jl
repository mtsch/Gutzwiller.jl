"""
     GutzwillerEvaluator(hamiltonian; basis=build_basis(hamiltonian))

Evaluates the energy of the Gutzwiller ansatz for a given value of `g`.

```jldoctest
julia> H = HubbardReal1D(near_uniform(BoseFS{3,3}))
julia> ge = GutzwillerEvaluator(H)
julia> ge(0.5)
```
"""
struct GutzwillerEvaluator{H,A}
    hamiltonian::H
    basis::Vector{A}
end
function Base.show(io::IO, ge::GutzwillerEvaluator)
    print(io, "GutzwillerEvaluator(", ge.hamiltonian, ")")
end

function GutzwillerEvaluator(ham; basis=build_basis(ham))
    return GutzwillerEvaluator(ham, basis)
end

function (ge::GutzwillerEvaluator)(params)
    g = only(params)

    t, b = Folds.mapreduce(add, ge.basis; init=(0.0, 0.0)) do k
        H = ge.hamiltonian
        diag = diagonal_element(H, k)
        k_val = exp(-diag * g)
        k_val_sq = abs2(k_val)

        bot = k_val_sq
        top = k_val_sq * diag
        for (l, melem) in offdiagonals(H, k)
            l_val = exp(-diagonal_element(H, l) * g)
            top += l_val' * melem * k_val
        end
        (top, bot)
    end
    return t / b
end
