using Gutzwiller
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

function scan_ts(m; ts=0.01:0.01:0.1, u_ibs=(0.0,0.2))
    df = DataFrame()
    for t in ts
        for u_ib in u_ibs
            @info "Optimizing t=$t u_ib=$u_ib"
            H = setup_H(m; t, u_ib)
            go = gutz_optimize(H)

            push!(df, (
                ; m, t, u_ib,
                minimizer=go.minimizer[1],
                minimum=go.minimum,
                f_calls=go.f_calls,
                success=a.ls_success,
            ))
        end
    end
    filename = "g_optimized_m$m.arrow"
    @info "Done. Saving to $filename"
    save_df(filename, df)
    return df
end
