"""
    VectorAnsatz(vector::AbstractDVec) <: AbstractAnsatz

Use `vector` as 0-parameter ansatz.
"""
struct VectorAnsatz{A,T,D<:AbstractDVec{A,T}} <: AbstractAnsatz{A,T,0}
    vector::D
end

Rimu.build_basis(va::VectorAnsatz) = collect(keys(va.vector))

(va::VectorAnsatz)(addr, _) = va.vector[addr]
