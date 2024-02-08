using SpecialFunctions

"""
    abstract type AbstractAnsatz{K,V,N}

An ansatz behaves similar to an [`AbstractDVec`](@ref) with `keytype` `K` and `valtype` `V`
with `N` parameters.

It must provide the following:

* `ansatz(key::K, params)::V`: Get the value of the ansatz for a given address `key` with
  specified parameters.
* `val_and_grad(ansatz, key, params)`: Get the value and gradient (w.r.t. the parameters) of
  the ansatz.
* [`build_basis`](@ref): for collecting the vector to `DVec/PDVec` (optional).
"""
abstract type AbstractAnsatz{K,V,N} end

Base.keytype(::AbstractAnsatz{K}) where {K} = K
Base.keytype(::Type{<:AbstractAnsatz{K}}) where {K} = K
Base.valtype(::AbstractAnsatz{<:Any,V}) where {V} = V
Base.valtype(::Type{<:AbstractAnsatz{<:Any,V}}) where {V} = V
num_parameters(::Type{<:AbstractAnsatz{<:Any,<:Any,N}}) where {N} = N
num_parameters(::AbstractAnsatz{<:Any,<:Any,N}) where {N} = N

function collect_to_vec!(dst, ans::AbstractAnsatz, params, basis)
    for k in basis
        dst[k] = ans(k, params)
    end
    return dst
end
function Rimu.DVec(
    ans::AbstractAnsatz{K,V,N}, params; basis=build_basis(ans), kwargs...
) where {K,V,N}
    result = DVec{K,V}(; kwargs...)
    return collect_to_vec!(result, ans, SVector{N,V}(params), basis)
end
function Rimu.PDVec(
    ans::AbstractAnsatz{K,V,N}, params; basis=build_basis(ans), kwargs...
) where {K,V,N}
    result = PDVec{K,V}(; kwargs...)
    return collect_to_vec!(result, ans, SVector{N,V}(params), basis)
end

"""
    val_and_grad(::AbstractAnsatz, addr, params)

Return ansatz value at `addr` and its gradient w.r.t. `params`.
"""
val_and_grad

function val_and_grad(a::AbstractAnsatz{K,<:Any,0}, addr::K, _) where {K}
    return a(addr, SVector{0,valtype(a)}()), SVector{0,valtype(a)}()
end

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

(va::VectorAnsatz)(addr, _) = va.vector[addr]

"""
    ExtendedGutzwillerAnsatz(hamiltonian) <: AbstractAnsatz

The Extended Gutzwiller ansatz:

```math
G_i = exp(-g_1 H_{i,i}) exp(-g_2 \\sum_{<i,j>} n_i n_j ),
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

"""

"""
struct BinomialAnsatz{K} <: Gutzwiller.AbstractAnsatz{K,Float64,1}
    normalisation::Float64
end
function BinomialAnsatz(H::AbstractHamiltonian)
    K = typeof(starting_address(H))
    N = float(num_particles(K))
    normalisation = gamma(num_particles(K) + 1) / float(num_modes(K))^num_particles(K)
    return BinomialAnsatz{K}(normalisation)
end

function binomial_weight(addr, occ=OccupiedModeMap(addr))
    M = float(num_modes(addr))
    remaining = float(num_particles(addr))
    result = 1.0
    for idx in occ
        num = idx.occnum
        result *= 1/gamma(num + 1)
    end
    return result
end

(b::BinomialAnsatz)(addr, p) = (binomial_weight(addr) * b.normalisation)^p[1]

function val_and_grad(b::BinomialAnsatz, addr, param)
    p = only(param)
    y = binomial_weight(addr) * b.normalisation
    val = y^p
    val, SVector{1,Float64}(val * log(y))
end

"""
    CombinationAnsatz

Superposition of `ansatz1` and `ansatz2`. Has one more parameter than each ansatz controling
the mixing between them.
"""
struct CombinationAnsatz{
    K,V,N,A<:AbstractAnsatz{K,V},B<:AbstractAnsatz{K,V}
} <: AbstractAnsatz{K,V,N}
    ansatz1::A
    ansatz2::B
end

function Base.:+(a1::AbstractAnsatz{K,V,N}, a2::AbstractAnsatz{K,V,M}) where {K,V,N,M}
    return CombinationAnsatz{K,V,N + M + 1,typeof(a1),typeof(a2)}(a1, a2)
end
@generated function (ca::CombinationAnsatz{K,V,<:Any,A,B})(addr::K, params) where {K,V,A,B}
    # Generated to ensure params are handled without allocations
    N = num_parameters(A)
    M = num_parameters(B)
    range1 = SVector{N,Int}(1:N)
    range2 = SVector{M,Int}((N+1):(N+M))
    quote
        p1 = params[$range1]
        p2 = params[$range2]
        α = params[end]

        return α * ca.ansatz1(addr, p1) + (1 - α) * ca.ansatz2(addr, p2)
    end
end

@generated function val_and_grad(
    ca::CombinationAnsatz{K,V,NP,A,B}, addr::K, params
) where {K,V,NP,A,B}
    # Generated to ensure params are handled without allocations
    N = num_parameters(A)
    M = num_parameters(B)
    range1 = SVector{N,Int}(1:N)
    range2 = SVector{M,Int}((N+1):(N+M))
    quote
        p1 = params[$range1]
        p2 = params[$range2]
        α = params[end]
        val1, grad1 = val_and_grad(ca.ansatz1, addr, p1)
        val2, grad2 = val_and_grad(ca.ansatz2, addr, p2)
        grad1 *= α
        grad2 *= (1 - α)

        return α * val1 + (1 - α) * val2, SVector{NP,V}(grad1..., grad2..., val1 - val2)
    end
end
