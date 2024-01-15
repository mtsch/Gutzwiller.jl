"""
    abstract type AbstractAnsatz{K,V,N}

An ansatz behaves similar to an [`AbstractDVec`](@ref) with `keytype` `K` and `valtype` `V`
with `N` parameters.

It must provide the following:

* `anstaz(key::K, params)::V`: Get the value of the ansatz for a given address `key` with
  specified parameters.
* `val_and_grad(ansatz, key, params)`: Get the value and gradient (w.r.t. the paramters) of
  the ansatz.
* [`build_basis`](@ref): for collecting the vector to `DVec/PDVec` (optional).
"""
abstract type AbstractAnsatz{K,V,N} end

Base.keytype(::AbstractAnsatz{K}) where {K} = K
Base.valtype(::AbstractAnsatz{<:Any,V}) where {V} = V

function collect_to_vec!(dst, ans::AbstractAnsatz, params)
    basis = build_basis(ans)
    for k in basis
        dst[k] = ans(k, params)
    end
    return dst
end
function Rimu.DVec(ans::AbstractAnsatz{K,V,N}, params; kwargs...) where {K,V,N}
    result = DVec{K,V}(; kwargs...)
    return collect_to_vec!(result, ans, SVector{N,V}(params))
end
function Rimu.PDVec(ans::AbstractAnsatz{K,V,N}, params; kwargs...) where {K,V,N}
    result = PDVec{K,V}(; kwargs...)
    return collect_to_vec!(result, ans, SVector{N,V}(params))
end

"""
    val_and_grad(::AbstractAnsatz, addr, params)

Return ansatz value at `addr` and its gradient w.r.t. `params`.
"""
val_and_grad

"""
    GutzwillerAnsatz(hamiltonian) <: AbstractAnsatz

The Gutzwiller ansatz:

```math
G_i = exp(-g H_{i,i}),
```

where ``H`` is the `hamiltonian` passed to the struct.

It takes a single parameter, `g`.
"""
struct GutzwillerAnsatz{A,T<:Real,H} <: AbstractAnsatz{A,T,1}
    hamiltonian::H
end
function GutzwillerAnsatz(hamiltonian)
    A = typeof(starting_address(hamiltonian))
    T = eltype(hamiltonian)
    return GutzwillerAnsatz{A,T,typeof(hamiltonian)}(hamiltonian)
end

Rimu.build_basis(gv::GutzwillerAnsatz) = build_basis(gv.hamiltonian)

function val_and_grad(gv::GutzwillerAnsatz, addr, params)
    g = only(params)
    diag = diagonal_element(gv.hamiltonian, addr)

    val = exp(-g * diag)
    der = -diag * val

    return val, SVector(der)
end
function (gv::GutzwillerAnsatz)(addr, params)
    return exp(-only(params) * diagonal_element(gv.hamiltonian, addr))
end

"""
    VectorAnsatz(vector::AbstractDVec) <: AbstractAnsatz

Use `vector` as 0-parameter ansatz.
"""
struct VectorAnsatz{A,T,D<:AbstractDVec{A,T}} <: AbstractAnsatz{A,T,0}
    vector::D
end

Rimu.build_basis(va::VectorAnsatz) = collect(keys(va.vector))

function val_and_grad(va::VectorAnsatz, addr, ::SVector{0})
    return va.vector[addr], SVector{0,valtype(va)}()
end
(va::VectorAnsatz)(addr, ::SVector{0}) = va.vector[addr]
