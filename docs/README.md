# Gutzwiller
_variational monte carlo and importance sampling for
[Rimu.jl](https://github.com/joachimbrand/Rimu.jl)_

## Installation

Gutzwiller.jl is not yet registered. To install it, run

```julia
import Pkg; Pkg.add("https://github.com/mtsch/Gutzwiller.jl")
```

## Usage guide

````julia
using Rimu
using Gutzwiller
using CairoMakie
using LaTeXStrings
````

First, we set up a starting address and a Hamiltonian

````julia
addr = near_uniform(BoseFS{10,10})
H = HubbardMom1D(addr)

ansatz = GutzwillerAnsatz(H)
````

````
GutzwillerAnsatz{BoseFS{10, 10, BitString{19, 1, UInt32}}, Float64, HubbardMom1D{Float64, 10, BoseFS{10, 10, BitString{19, 1, UInt32}}, 1.0, 1.0}}(HubbardMom1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=1.0, t=1.0))
````

An ansatz is a struct that given a set of parameters and an address, produces a value

````julia
ansatz(addr, [1.0])
````

````
0.00012340980408667956
````

In addition, the function `val_and_grad` can be used to compute both the value and its
gradient with respect to the parameters.

````julia
val_and_grad(ansatz, addr, [1.0])
````

````
(0.00012340980408667956, [-0.001110688236780116])
````

we want to optimize the parameter

if the basis of the Hamiltonian is small enough to fit into memory, it's best to use the
`LocalEnergyEvaluator`:

````julia
le = LocalEnergyEvaluator(H, ansatz)
````

````
LocalEnergyEvaluator(HubbardMom1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=1.0, t=1.0), GutzwillerAnsatz{BoseFS{10, 10, BitString{19, 1, UInt32}}, Float64, HubbardMom1D{Float64, 10, BoseFS{10, 10, BitString{19, 1, UInt32}}, 1.0, 1.0}}(HubbardMom1D(fs"|1 1 1 1 1 1 1 1 1 1⟩"; u=1.0, t=1.0)))
````

It can be used to evaulate the value of the Rayleigh quotient (or its gradient) for a
given set of parameters

````julia
le([1.0])
````

````
-8.055957336940313
````

````julia
val_and_grad(le, [1.0])
````

````
(-8.055957336940313, [-3.82925738245772])
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

