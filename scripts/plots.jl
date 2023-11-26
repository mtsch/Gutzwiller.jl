using Rimu
using DataFramesMeta
using CairoMakie

function plot_tscan(df)
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel=L"t", ylabel=L"g")

    for d in groupby(df, [:u_ib, :m])
        u_ib = d.u_ib[1]
        m = d.m[1]
        label = L"%$m × %$m, U_{IB}=%$u_ib"

        scatter!(ax, d.t, d.minimizer; label)
    end
    return fig
end
