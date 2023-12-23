module Gutzwiller

using Rimu
using VectorInterface
using Folds
using ProgressMeter
using Parameters
using Measurements
using Statistics
using StatsBase
using StaticArrays

using Tables

export GutzwillerAnsatz, LocalEnergyEvaluator, val_and_grad, local_energy
export kinetic_vqmc, kinetic_vqmc!, local_energy_estimator, KineticVQMC

include("utils.jl")
include("ansatz.jl")
export GutzwillerAnsatz

include("localenergy.jl")
export LocalEnergy

include("kinetic-vqmc/state.jl")
include("kinetic-vqmc/qmc.jl")
include("kinetic-vqmc/wrapper.jl")

end
