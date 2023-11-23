module Gutzwiller

using Rimu
using VectorInterface
using Optim

include("metropolis.jl")
export local_energy, gutzwiller_energy

include("optimizer.jl")
export GutzwillerOptimizer, gutz_optimize

end
