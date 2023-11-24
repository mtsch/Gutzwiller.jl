using Gutzwiller
using Rimu

include("setup.jl")

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
