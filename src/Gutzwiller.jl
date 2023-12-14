module Gutzwiller

using Rimu
using VectorInterface
using Folds
using ProgressMeter
using Parameters
using Measurements
using Statistics

using Tables

include("utils.jl")
include("ansatz.jl")
export GutzwillerAnsatz

include("deterministic-evaluator.jl")
export GutzwillerEvaluator

include("optimization.jl")
export gutz_optimize

include("kinetic-vqmc/state.jl")
include("kinetic-vqmc/qmc.jl")
include("kinetic-vqmc/wrapper.jl")
export kinetic_vqmc, kinetic_vqmc!, local_energy_estimator, KineticVQMC

#include("metropolis.jl")
#export metropolis_hastings, LocalEnergyCollector, AddressCollector

#include("qmc-evaluator.jl")
#export GutzwillerVQMC, collect_samples!



end
