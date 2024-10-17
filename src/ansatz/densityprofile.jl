"""
    DensityProfileAnsatz(hamiltonian) <: AbstractAnsatz

```math
D(|f⟩; p) = exp(-∑_{i=1}^M p_i ⟨f|n_i|f⟩)
```
"""
struct DensityProfileAnsatz{A,N,H} <: AbstractAnsatz{A,Float64,N}
    hamiltonian::H
end

function DensityProfileAnsatz(hamiltonian)
    address = starting_address(hamiltonian)
    @assert address isa SingleComponentFockAddress
    N = num_modes(address)
    return DensityProfileAnsatz{typeof(address),N,typeof(hamiltonian)}(hamiltonian)
end

Rimu.build_basis(dp::DensityProfileAnsatz) = build_basis(dp.hamiltonian)

function val_and_grad(dp::DensityProfileAnsatz, addr, params)
    o = onr(addr)
    exponent = dot(o, params)
    val = exp(-exponent)
    grad = -val * o
    return val, grad
end
(dp::DensityProfileAnsatz)(addr, params) = exp(-dot(onr(addr), params))
