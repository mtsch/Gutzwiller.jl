"""
    abstract type AbstractAnsatz{K,V}

An ansatz behaves similar to an [`AbstractDVec`](@ref) with `keytype` `K` and `valtype` `V`.

It must provide the following:

* `Base.getindex(anstaz, key::K)::V`: Get the value of the ansatz for a given address `key`.
* `set_params(ansatz, params::Vector{T})`: Change the parameters in the ansatz.
* [`build_basis`](@ref): for collecting the vector to `DVec/PDVec` (optional).
"""
abstract type AbstractAnsatz{K,V} end

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
    set_params(::AbstractAnsatz, params::Vector)

Return new ansatz of the same type with new parameters. Used in optimization functions.
"""
set_params

"""
    GutzwillerAnsatz{A,T}

Placeholder TODO: make a proper DVec
"""
struct GutzwillerAnsatz{A,T<:Real,H} <: AbstractAnsatz{A,T}
    hamiltonian::H
    g::T
end
function GutzwillerAnsatz(hamiltonian, g)
    A = typeof(starting_address(hamiltonian))
    T = promote_type(eltype(hamiltonian), typeof(g))
    return GutzwillerAnsatz{A,T,typeof(hamiltonian)}(hamiltonian, T(g))
end

function Base.getindex(gv::GutzwillerAnsatz{A}, addr::A) where {A}
    return exp(-gv.g * diagonal_element(gv.hamiltonian, addr))
end
Rimu.build_basis(gv::GutzwillerAnsatz) = build_basis(gv.hamiltonian)

function set_params(gv::GutzwillerAnsatz, params)
    g = only(params)
    return GutzwillerAnsatz(gv.hamiltonian, g)
end
