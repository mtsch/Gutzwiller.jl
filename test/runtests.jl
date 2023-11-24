using Gutzwiller
using KrylovKit
using Rimu
using Test

@testset "metropolis-hastings" begin
    @testset "vector resampling" begin
        # Check that the samples are taken from the correct distribution.
        addr = near_uniform(BoseFS{5,5})
        H = HubbardReal1D(addr; t=0.1)
        groundstate = eigsolve(H, PDVec(addr => 1.0), 1, :SR)[2][1]
        groundstate_sq = normalize!(map(abs2, values(groundstate)))

        sampled1e6 = metropolis_hastings(VectorAccumulator, H, groundstate; steps=1e6)
        sampled1e7 = metropolis_hastings(VectorAccumulator, H, groundstate; steps=1e7)

        Δ1e6 = maximum(abs, values(groundstate_sq - sampled1e6))
        Δ1e7 = maximum(abs, values(groundstate_sq - sampled1e7))

        @test Δ1e6 > Δ1e7
        @test dot(sampled1e6, groundstate_sq) ≈ 1 atol=1e-4
        @test dot(sampled1e7, groundstate_sq) ≈ 1 atol=1e-4
    end
    @testset "variational energy" begin
        addr = near_uniform(BoseFS{4,6})
        H = HubbardReal1D(addr; u=4)
        vector = eigsolve(H, PDVec(addr => 1.0), 3, :SR)[2][1]
        map!(x -> x^2 + rand(), values(vector))


        result = metropolis_hastings(VariationalEnergyAccumulator, H, vector; warmup=1e7)
        Ev = rayleigh_quotient(H, vector)

        @test result.mean - result.err < Ev < result.mean + result.err
   end
end
