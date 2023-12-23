"""
    abstract type AbstractAnsatz{K,V,N}

An ansatz behaves similar to an [`AbstractDVec`](@ref) with `keytype` `K` and `valtype` `V`
with `N` parameters.

It must provide the following:

* `Base.getindex(anstaz, key::K)::V`: Get the value of the ansatz for a given address `key`.
* `set_params(ansatz, params::Vector{T})`: Change the parameters in the ansatz.
* [`build_basis`](@ref): for collecting the vector to `DVec/PDVec` (optional).
"""
abstract type AbstractAnsatz{K,V,N} end

Base.keytype(::AbstractAnsatz{K}) where {K} = K
Base.valtype(::AbstractAnsatz{<:Any,V}) where {V} = V

function collect_to_dvec!(dst, ans::AbstractAnsatz)
    basis = build_basis(ans)
    for k in basis
        dst[k] = ans[k]
    end
    return dst
end
function Rimu.DVec(ans::AbstractAnsatz{K,V}; kwargs...) where {K,V}
    result = DVec{K,V}(; kwargs...)
    return collect_to_vec!(result, ans)
end
function Rimu.PDVec(ans::AbstractAnsatz{K,V}; kwargs...) where {K,V}
    result = PDVec{K,V}(; kwargs...)
    return collect_to_vec!(result, ans)
end

"""
    val_and_grad(::AbstractAnsatz, addr, params)

Return ansatz value at `addr` and its gradient wrt `params`.
"""
val_and_grad

"""
    GutzwillerAnsatz{A,T}

Placeholder TODO: make a proper DVec
"""
struct GutzwillerAnsatz{A,T<:Real,H} <: AbstractAnsatz{A,T,1}
    hamiltonian::H
end
function GutzwillerAnsatz(hamiltonian)
    A = typeof(starting_address(hamiltonian))
    T = eltype(hamiltonian)
    return GutzwillerAnsatz{A,T,typeof(hamiltonian)}(hamiltonian)
end

function Base.getindex(gv::GutzwillerAnsatz{A}, addr::A) where {A}
    return exp(-gv.g * diagonal_element(gv.hamiltonian, addr))
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
function val_and_grad(va::VectorAnsatz, addr, ::SVector{0})
    return va.vector[add], SVector{0,valtype(va)}()
end
(va::VectorAnsatz)(addr, ::SVector{0}) = va.vector[addr]
