module Gutzwiller

using Rimu
using VectorInterface
using Folds
using ProgressMeter

include("utils.jl")
export GutzwillerAnsatz

include("deterministic-evaluator.jl")
export GutzwillerEvaluator

include("metropolis.jl")
export metropolis_hastings, LocalEnergyCollector, AddressCollector

include("qmc-evaluator.jl")
export GutzwillerVQMC, collect_samples!


include("optimization.jl")
export gutz_optimize

end
