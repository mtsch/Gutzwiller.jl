using Gutzwiller
using DataFrames
using Rimu
using CairoMakie, LaTeXStrings

function compare_gutzies(N, t, gs; kwargs...)
    addr = near_uniform(BoseFS{N,N})
    H = HubbardReal1D(addr; t)

    df = DataFrame()
    ge_det = GutzwillerEvaluator(H)
    ge_qmc = GutzwillerQMCEvaluator(H; kwargs...)

    for g in gs
        res_det = ge_det(g)
        res_qmc = ge_qmc(g)

        push!(df, (
            ; N, t, g, det=res_det,
            qmc=res_qmc.mean, qmc_err=res_qmc.err, qmc_acc=res_qmc.acceptance,
        ))
    end

    return df
end

function plot_qmcdet_comparison(df)
    fig = Figure()
    ax = Axis(fig[1,1]; xlabel=L"g", ylabel=L"E_v")

    scatter!(ax, df.g, df.det; label="det")
    errorbars!(ax, df.g, df.qmc, df.qmc_err; label="qmc")
    scatter!(ax, df.g, df.qmc; label="qmc")

    Legend(fig[1,2], ax; merge=true)
    return fig
end

if false
    addr = near_uniform(BoseFS{6,6})
    H = HubbardReal1D(addr; t=0.1)
    geq = GutzwillerQMCEvaluator(H; warmup=0, tasks=1)
    ged = GutzwillerEvaluator(H)

    res = geq(0.5)

end
