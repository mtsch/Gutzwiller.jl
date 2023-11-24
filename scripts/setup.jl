using Rimu

function setup_H(m; N=m*m, t=0.1, u_ib=0.0, g=0.0)
    M = m * m
    geometry = PeriodicBoundaries(m, m)
    if u_ib ≠ 0
        u = [1 u_ib; u_ib 0]
        t = [t, t]
        addr = CompositeFS(
            near_uniform(BoseFS{N,M}),
            BoseFS(M, 1 => 1),
        )
        H = HubbardRealSpace(addr; geometry, u, t)
    else
        addr = near_uniform(BoseFS{N,M})
        H = HubbardRealSpace(addr; geometry, u=[1.0], t=[t])
    end

    if g ≠ 0
        H = GutzwillerSampling(H, g)
    end
    return H
end
