struct AnsatzSampling{T,N,A<:AbstractAnsatz{<:Any,T,N},H} <: AbstractHamiltonian{T}
    hamiltonian::H
    ansatz::A
    params::SVector{N,T}
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


"""
    TransformUndoer(k::AnsatzSampling, op::AbstractHamiltonian)
    TransformUndoer(k::AnsatzSampling)

For a general similarity transformation ``\\hat{G} = f \\hat{H} f^{-1}`` under an ansatz,
define the operator ``f^{-1} \\hat{A} f^{-1}``, and special case ``f^{-2}``, in order
to calculate observables. Here ``f`` is a diagonal operator whose entries are
the components of the ansatz vector, i.e.``f_{ii} = v_i``.

See [`AllOverlaps`](@ref), [`GuidingVectorSampling`](@ref).
"""
function TransformUndoer(k::AnsatzSampling, op::Union{Nothing,AbstractHamiltonian})
    if isnothing(op)
        T = eltype(k)
    else
        T = promote_type(eltype(k), eltype(op))
    end
    return TransformUndoer{T,typeof(k),typeof(op)}(k, op)
end

# methods for general operator `f^{-1} A f^{-1}`
LOStructure(::Type{<:TransformUndoer{<:Any,<:AnsatzSampling,A}}) where {A} = LOStructure(A)

function Rimu.diagonal_element(s::TransformUndoer{<:Any,<:GutzwillerSampling,<:AbstractHamiltonian}, addr)
    diagH = diagonal_element(s.transform.hamiltonian, addr)
    diagA = diagonal_element(s.op, addr)
    return ansatz_modify(diagA, 2 * diagH, 1.0) # Apply diagonal `f^{-1}` twice
end

function Rimu.num_offdiagonals(s::TransformUndoer{<:Any,<:AnsatzSampling,<:Any}, addr)
    return num_offdiagonals(s.op, addr)
end

function get_offdiagonal(s::TransformUndoer{<:Any,<:AnsatzSampling,<:Any}, add1, chosen)
    add2, offd = get_offdiagonal(s.op, add1, chosen)
    # Guiding vector `v` is represented as a diagonal operator `f`
    diagH1 = diagonal_element(s.transform.hamiltonian, add)
    diagH2 = diagonal_element(s.transform.hamiltonian, newadd)
    return add2, guided_vector_modify(offd,diagH1 + diagH2,1.0)
end

# methods for special case `f^{-2}`
LOStructure(::Type{<:TransformUndoer{<:Any,<:AnsatzSampling,Nothing}}) = IsDiagonal()

function diagonal_element(s::TransformUndoer{<:Any,<:AnsatzSampling,Nothing}, add)
    diagH = diagonal_element(s.transform.hamiltonian, add)
    return guided_vector_modify(1., true, s.transform.eps, 2 * guide, 1.0)
end

num_offdiagonals(s::TransformUndoer{<:Any,<:AnsatzSampling,Nothing}, add) = 0