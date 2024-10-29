using Rimu.Hamiltonians: circshift_dot
# TODO: geometry for relative version, multicomponent support for both.

"""
    JastrowAnsatz(hamiltonian) <: AbstractAnsatz

```math
J(|f⟩; 𝐩) = exp(-∑_{k=1}^M ∑_{l=k}^M p_{k,l} ⟨f| n_k n_l |f⟩)
```

With translationally invariant Hamiltonians, use [`RelativeJastrowAnsatz`](@ref) instead.
"""
struct JastrowAnsatz{A,N,H} <: AbstractAnsatz{A,Float64,N}
    hamiltonian::H
end

function JastrowAnsatz(hamiltonian)
    address = starting_address(hamiltonian)
    @assert address isa SingleComponentFockAddress
    N = num_modes(address) * (num_modes(address) + 1) ÷ 2
    return JastrowAnsatz{typeof(address),N,typeof(hamiltonian)}(hamiltonian)
end

Rimu.build_basis(j::JastrowAnsatz) = build_basis(j.hamiltonian)

function val_and_grad(j::JastrowAnsatz{A,N}, addr::A, params) where {A,N}
    o = onr(addr)
    M = num_modes(addr)

    val = 0.0
    grad = zeros(SVector{N,Float64})

    p = 0
    @inbounds for i in 1:M, j in i:M
        p += 1
        onproduct = o[i] * o[j]
        local_val = params[p] * onproduct
        val += local_val
        grad = setindex(grad, -onproduct, p)
    end
    val = exp(-val)
    grad = grad .* val

    return val, grad
end
function (j::JastrowAnsatz)(addr, params)
    o = onr(addr)
    M = num_modes(addr)
    val = 0.0
    p = 0

    for i in 1:M, j in i:M
        p += 1

        onproduct = o[i] * o[j]
        val += params[p] * onproduct
    end
    return exp(-val)
end

"""
    RelativeJastrowAnsatz(hamiltonian) <: AbstractAnsatz

For a translationally invariant Hamiltonian, this is equivalent to [`JastrowAnsatz`](@ref),
but has fewer parameters.

```math
R(|f⟩; 𝐩) = exp(-∑_{d=0}^{M/2} p_d ∑_{k=1}^M ⟨f| n_k n_{k + d} |f⟩)
```
"""
struct RelativeJastrowAnsatz{A,N,H} <: AbstractAnsatz{A,Float64,N}
    hamiltonian::H
end

function RelativeJastrowAnsatz(hamiltonian)
    address = starting_address(hamiltonian)
    @assert address isa SingleComponentFockAddress
    N = cld(num_modes(address), 2)
    return RelativeJastrowAnsatz{typeof(address),N,typeof(hamiltonian)}(hamiltonian)
end

Rimu.build_basis(rj::RelativeJastrowAnsatz) = build_basis(rj.hamiltonian)

function val_and_grad(rj::RelativeJastrowAnsatz, addr, params)
    o = onr(addr)
    M = num_parameters(rj)

    products = ntuple(i -> circshift_dot(o, o, i - 1), Val(M))
    exponent = dot(params, products)
    val = exp(-exponent)
    grad = -SVector{M,Float64}(products .* val)

    return val, grad
end
function (rj::RelativeJastrowAnsatz)(addr, params)
    o = onr(addr)
    M = num_parameters(rj)

    val = sum(1:M) do i
        params[i] * circshift_dot(o, o, i - 1)
    end
    return exp(-val)
end
