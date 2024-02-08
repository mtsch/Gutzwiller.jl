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
