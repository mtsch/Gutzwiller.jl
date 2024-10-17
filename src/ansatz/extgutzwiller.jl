"""
    ExtendedGutzwillerAnsatz(hamiltonian) <: AbstractAnsatz

The Extended Gutzwiller ansatz:

```math
G(|f⟩; 𝐠) = exp(-g_1 ⟨f|H|f⟩ - g_2 ⟨f| ∑_{<i,j>} n_i n_j |f⟩),
```

where ``H`` is an ExtendedHubbardReal1D Hamiltonian. The additional term accounts for the strength of nearest-neighbour interactions.

It takes two parameters, `g_1` and `g_2`.
"""
struct ExtendedGutzwillerAnsatz{A,T<:Real,H} <: AbstractAnsatz{A,T,2}
    hamiltonian::H
end
function ExtendedGutzwillerAnsatz(hamiltonian)
    A = typeof(starting_address(hamiltonian))
    T = eltype(hamiltonian)
    return ExtendedGutzwillerAnsatz{A,T,typeof(hamiltonian)}(hamiltonian)
end

Rimu.build_basis(gv::ExtendedGutzwillerAnsatz) = build_basis(gv.hamiltonian)

function val_and_grad(gv::ExtendedGutzwillerAnsatz, addr, params)
    g1, g2 = params
    ebh_interaction, diag = ebh(addr)

    val = exp(-g1 * diag + -g2*ebh_interaction)
    der_g1 = -diag * val
    der_g2 = -ebh_interaction * val

    return val, SVector(der_g1, der_g2)
end

function (gv::ExtendedGutzwillerAnsatz)(addr, params)
    g1,g2 = params
    ebh_int, bh_int = ebh(addr)
    return exp(-g1*bh_int + -g2*ebh_int)
end
