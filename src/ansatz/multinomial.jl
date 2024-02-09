"""


```math
b
```
"""
struct MultinomialAnsatz{K} <: Gutzwiller.AbstractAnsatz{K,Float64,1}
    normalization::Float64
end
function MultinomialAnsatz(H::AbstractHamiltonian; normalize=false)
    K = typeof(starting_address(H))
    N = float(num_particles(K))
    if normalize
        normalization = gamma(num_particles(K) + 1) / float(num_modes(K))^num_particles(K)
    else
        normalization = 1.0
    end
    return MultinomialAnsatz{K}(normalization)
end

function multinomial_weight(addr, occ=OccupiedModeMap(addr))
    return prod(idx -> 1/gamma(idx.occnum + 1), occ)
end

(b::MultinomialAnsatz)(addr, p) = (multinomial_weight(addr) * b.normalization)^p[1]

function val_and_grad(b::MultinomialAnsatz, addr, param)
    p = only(param)
    y = multinomial_weight(addr) * b.normalization
    val = y^p
    val, SVector{1,Float64}(val * log(y))
end
