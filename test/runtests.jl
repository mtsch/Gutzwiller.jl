using Gutzwiller
using KrylovKit
using Rimu
using Test

@testset "qmc" begin
    for H in (
        HubbardReal1D(near_uniform(BoseFS{5,5}); t=0.1),
        HubbardMom1D(BoseFS(10, 5 => 3); u=4),
        HubbardRealSpace(FermiFS2C((1,0,0,1), (1,1,0,0)); geometry=PeriodicBoundaries(2,2)),
    )
        @testset "$H" begin
            # Check that the samples are taken from the correct distribution.
            res = eigsolve(H, PDVec(starting_address(H) => 1.0), 1, :SR)
            E0 = res[1][1]
            groundstate = res[2][1]
            groundstate_sq = normalize!(map(abs2, values(groundstate)))

            res1e6 = kinetic_vqmc(H, groundstate; samples=1e6)
            res1e7 = kinetic_vqmc(H, groundstate; samples=1e7)

            sampled1e6 = DVec(res1e6)
            sampled1e7 = DVec(res1e7)

            Δ1e6 = maximum(abs, values(groundstate_sq - sampled1e6))
            Δ1e7 = maximum(abs, values(groundstate_sq - sampled1e7))

            @test Δ1e6 > Δ1e7
            @test dot(sampled1e6, groundstate_sq) ≈ 1 rtol=1e-4
            @test dot(sampled1e7, groundstate_sq) ≈ 1 rtol=5e-5

            @test local_energy_estimator(res1e6).mean ≈ E0 rtol=1e-6
            @test local_energy_estimator(res1e7).mean ≈ E0 rtol=5e-7
        end
    end
end

@testset "deterministic" begin
end

@testset "optimisation" begin
end

function count_appearances(res::Gutzwiller.KineticVQMCResult{A}) where {A}
    app = DVec{A,Int}()
    for r in res.states
        for addr in r.addresses
            app[addr] += 1
        end
    end
    return app
end
