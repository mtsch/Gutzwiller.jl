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

    return AnsatzSampling(hamiltonian, ansatz, params)
end

starting_address(h::AnsatzSampling) = starting_address(h.hamiltonian)
LOStructure(::Type{<:AnsatzSampling{<:Any,<:Any,<:Any,H}}) where {H} = AdjointUnknown()

dimension(h::AnsatzSampling, addr) = dimension(h.hamiltonian, addr)
num_offdiagonals(h::AnsatzSampling, add) = num_offdiagonals(h.hamiltonian, add)
diagonal_element(h::AnsatzSampling, add) = diagonal_element(h.hamiltonian, add)

function ansatz_modify(matrix_element, A, add1_ansatz, add2_ansatz)
    return matrix_element * (add1_ansatz/add2_ansatz)
end

function get_offdiagonal(h::AnsatzSampling, add1, chosen)
    add2, matrix_element = get_offdiagonal(h.hamiltonian, add1, chosen)
    add1_ansatz = (h.ansatz)(add1, h.params)
    add2_ansatz = (h.ansatz)(add2, h.params)
    return add2, ansatz_modify(matrix_element, A, add1_ansatz, add2_ansatz)
end
