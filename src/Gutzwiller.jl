module Gutzwiller

using DataFrames
using Folds
using Measurements
using ProgressMeter
using Parameters
using Rimu
using Statistics
using StatsBase
using StaticArrays
using Tables
using VectorInterface

export GutzwillerAnsatz, LocalEnergyEvaluator, val_and_grad, local_energy
export kinetic_vqmc, kinetic_vqmc!, local_energy_estimator, KineticVQMC

include("utils.jl")
include("ansatz.jl")
include("localenergy.jl")

include("kinetic-vqmc/state.jl")
include("kinetic-vqmc/qmc.jl")
include("kinetic-vqmc/wrapper.jl")

include("amsgrad.jl")

end
