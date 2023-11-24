module Gutzwiller

using Rimu
using VectorInterface
using Folds
using ProgressMeter

include("deterministic-evaluator.jl")
export GutzwillerEvaluator

include("metropolis.jl")
export VariationalEnergyAccumulator, VectorAccumulator, metropolis_hastings

include("qmc-evaluator.jl")
export GutzwillerQMCEvaluator


include("optimization.jl")
export gutz_optimize

end
