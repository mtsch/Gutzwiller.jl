using Gutzwiller
using Rimu
using CairoMakie
using Optim
using LaTeXStrings

plot_multinomial(args...; kwargs...) = plot_multinomial!(Figure(), args...; kwargs...)
function plot_multinomial!(fig, H; title)
    eigres = eigsolve(H, PDVec(starting_address(H) => 1.0), 1, :SR)
    E0 = round(eigres[1][1]; sigdigits=4)
    v0 = eigres[2][1]
    v0 *= sign(first(values(v0)))
    basis = build_basis(H)

    ansatz = BinomialAnsatz(H)
    evaluator = LocalEnergyEvaluator(H, ansatz; basis)
    optres = optimize(Optim.only_fg!(evaluator), [0.5])

    ps = sort!(collect(pairs(v0)); by=last, rev=true)
    vs = last.(ps)
    vs ./= vs[1]
    ks = first.(ps)

    if H.u ≠ 0
        v_opt = PDVec(ansatz, optres.minimizer; basis)
        Ev_opt = round(optres.minimum; sigdigits=4)
        vs_opt = [ansatz(k, optres.minimizer) for k in ks]
        vs_opt ./= vs_opt[1]
    end

    v_sqrt = PDVec(ansatz, [0.5]; basis)
    Ev_sqrt = round(evaluator([0.5]); sigdigits=4)
    vs_sqrt = [ansatz(k, [0.5]) for k in ks]
    vs_sqrt ./= vs_sqrt[1]

    ax = Axis(fig[1:3,1:3]; xlabel=L"i", ylabel=L"c_i", title, yscale=log10)

    if H.u ≠ 0
        p = round(optres.minimizer[1]; sigdigits=3)
        scatter!(
            ax, vs_opt; label=L"multi, $p=%$(p)$, $E_v=%$Ev_opt$",
            color=Cycled(3), markersize=5,
        )
    end
    scatter!(
        ax, vs_sqrt; label=L"multi, $p=1/2$, $E_v=%$Ev_sqrt$",
        color=Cycled(2), markersize=5,
    )
    lines!(ax, vs; label=L"exact, $E_0=%$E0$", color=Cycled(1))

    axislegend(ax; position=:lb)
    return fig
end

if !isinteractive()
    begin
        fig = Figure(size=(800, 1000))
        for (k, u) in enumerate((0.0, 0.1, 5.0, 0.01, 1.0, 10.0))
            idx = CartesianIndices((3, 2))[k]
            H = HubbardReal1D(BoseFS((1,1,1,1)); u)
            plot_multinomial!(fig[idx[1],idx[2]], H, title=L"Bose-Hubbard $N=M=4\,,u=%$u$")
        end
        save(joinpath(@__DIR__, "example.pdf"), fig)
    end
end
