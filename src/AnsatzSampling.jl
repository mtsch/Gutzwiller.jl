struct AnsatzSampling{T,N,A<:AbstractAnsatz{<:Any,T,N},H} <: AbstractHamiltonian{T}
    hamiltonian::H
    ansatz::A
    params::SVector{N,T}
end

function AnsatzSampling(
    h, ansatz::AbstractAnsatz{<:Any,<:Any,N}, params::Vararg{Number,N}
) where {N}
    return AnsatzSampling(h, ansatz, params)
end

function AnsatzSampling(h, ansatz::AbstractAnsatz{K,T,N}, params) where {K,T,N}
    # sanity checks
    if typeof(starting_address(h)) ≢ K
        throw(ArgumentError("Ansatz keytype does not match Hamiltonian starting_address"))
    end
    if T ≢ promote_type(T, eltype(h))
        throw(ArgumentError("Hamiltonian and Ansatz eltypes don't match"))
    end

    params = SVector{N,T}(params)

    return AnsatzSampling(h, ansatz, params)
end

Rimu.starting_address(h::AnsatzSampling) = starting_address(h.hamiltonian)
Rimu.LOStructure(::Type{<:AnsatzSampling{<:Any,<:Any,<:Any,H}}) where {H} = AdjointUnknown()

Rimu.dimension(h::AnsatzSampling, addr) = dimension(h.hamiltonian, addr)
Rimu.num_offdiagonals(h::AnsatzSampling, add) = num_offdiagonals(h.hamiltonian, add)
Rimu.diagonal_element(h::AnsatzSampling, add) = diagonal_element(h.hamiltonian, add)

function ansatz_modify(matrix_element, add1_ansatz, add2_ansatz)
    return matrix_element * (add2_ansatz/add1_ansatz)
end

function Rimu.get_offdiagonal(h::AnsatzSampling, add1, chosen)
    add2, matrix_element = get_offdiagonal(h.hamiltonian, add1, chosen)
    add1_ansatz = (h.ansatz)(add1, h.params)
    add2_ansatz = (h.ansatz)(add2, h.params)
    return add2, ansatz_modify(matrix_element, add1_ansatz, add2_ansatz)
end

###
### TransformUndoer
###
const AnsatzTransformUndoer{O} = Rimu.Hamiltonians.TransformUndoer{<:Any,<:AnsatzSampling,O}

function Rimu.Hamiltonians.TransformUndoer(
    k::AnsatzSampling, op::Union{Nothing,AbstractHamiltonian}
)
    if isnothing(op)
        T = eltype(k)
    else
        T = promote_type(eltype(k), eltype(op))
    end
    return Rimu.Hamiltonians.TransformUndoer{T,typeof(k),typeof(op)}(k, op)
end

# methods for general operator `f^{-1} A f^{-1}`
Rimu.LOStructure(::Type{<:AnsatzTransformUndoer{A}}) where {A} = LOStructure(A)

function Rimu.diagonal_element(s::AnsatzTransformUndoer, addr)
    ansatz = s.transform.ansatz
    params = s.transform.params
    ansatz1 = ansatz(addr, params)

    diag = diagonal_element(s.op, addr)
    return diag / ansatz1^2 # Apply diagonal `f^{-1}` twice
end

function Rimu.num_offdiagonals(s::AnsatzTransformUndoer, addr)
    return num_offdiagonals(s.op, addr)
end

function Rimu.get_offdiagonal(s::AnsatzTransformUndoer, addr1, chosen)
    addr2, offd = get_offdiagonal(s.op, addr1, chosen)
    # Get the ansatz from the TransformUndoer
    ansatz = s.transform.ansatz
    params = s.transform.params

    ansatz1 = ansatz(addr1, params)
    ansatz2 = ansatz(addr2, params)
    return addr2, offd / (ansatz1 * ansatz2)
end

# Methods for special case `f^{-2}`
Rimu.LOStructure(::Type{<:AnsatzTransformUndoer{Nothing}}) = IsDiagonal()

function Rimu.diagonal_element(s::AnsatzTransformUndoer{Nothing}, addr)
    ansatz = s.transform.ansatz
    params = s.transform.params
    ansatz1 = ansatz(addr, params)

    return 1/(ansatz1^2)
end

Rimu.num_offdiagonals(s::AnsatzTransformUndoer{Nothing}, _) = 0
