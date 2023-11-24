function compare_with_err(y1, y2)
    μ1 = y1.mean
    σ1 = y1.err
    μ2 = y2.mean
    σ2 = y2.err

    delta_sq = (μ1 - μ2)^2
    inter_sq = σ1^2 + σ2^2
    if delta_sq > inter_sq
        return argmin((μ1, μ2))
    else
        return 0
    end
end

function select_smallest(values::NTuple{3,Float64})
    return argmin(values)
end
function select_smallest(values::NTuple{3,<:Any})
    c12 = compare_with_err(values[1], values[2])
    if c12 == 2
        res = compare_with_err(values[2], values[3])
        res += (res ≠ 0)
        return res
    else
        return c12
    end

end

struct GutzOptimizerResult{T}
    minimizer::Float64
    step::Float64
    minimum::T
    converged::Bool
    reason::Symbol
    evals::Int
    iters::Int
end
function Base.show(io::IO, res::GutzOptimizerResult)
    println(io, "GutzOptimizerResult")
    println(io, "  minimizer: ", res.minimizer)
    println(io, "  minimum:   ", res.minimum)
    println(io, "  converged: ", res.converged)
    println(io, "  reason:    ", res.reason)
    println(io, "  evals:     ", res.evals)
    print(io, "  iters:     ", res.iters)
end

function _gutz_optimize(f; x0, step, x_tol, maxiter=100, verbose=false)
    xs = (x0 - step, x0, x0 + step)
    ys = (f(xs[1]), f(xs[2]), f(xs[3]))
    evals = 3
    converged = false
    reason = :steps
    iters = 0
    x_curr = xs[2]
    y_curr = ys[2]

    for iter in 1:maxiter
        x_curr = xs[2]
        y_curr = ys[2]

        iters += 1
        if verbose
            print(stderr, "Iter ", lpad(iters, 4))
            print(stderr, ": g = ", lpad(round(x_curr, sigdigits=5), 7))
            print(stderr, ", step = ", lpad(round(step, sigdigits=5), 7))
            print(stderr, ", E_v = ", y_curr)
            println(stderr)
        end

        next_index = select_smallest(ys)

        if next_index == 1
            # Minimum is to the left - move over
            xs = (xs[1] - step, xs[1], xs[2])
            ys = (f(xs[1]), ys[1], ys[2])
            evals += 1
            x_curr = x_curr - step
        elseif next_index == 3
            # Minimum is to the right - move over
            xs = (xs[2], xs[3], xs[3] + step)
            ys = (ys[2], ys[3], f(xs[3]))
            evals += 1
            x_curr = x_curr + step
        elseif next_index == 2
            # Minimum is in the middle - shrink the range and go to the next step
            step /= 2

            xs = (xs[2] - step, xs[2], xs[2] + step)
            ys = (f(xs[1]), ys[2], f(xs[3]))
            evals += 2
        else
            converged = true
            reason = :errorbars
            break
        end

        if step < x_tol
            converged = true
            reason = :x_tol
            break
        end
    end

    return GutzOptimizerResult(x_curr, step, y_curr, converged, reason, evals, iters)
end

function gutz_optimize(
    H, g0;
    step=0.4, g_tol=1e-7, maxiter=100, qmc=false, verbose=true,
    kwargs...
)

    if qmc
        evaluator = GutzwillerQMCEvaluator(H; progress=false, kwargs...)
    else
        verbose && println(stderr, "Building evaluator")
        el = @elapsed evaluator = GutzwillerEvaluator(H)
        verbose && print(stderr, "done in")
        verbose && Base.time_print(stderr, el * 1e9)
        verbose && println()
    end
    verbose && println(stderr, evaluator)

    res = _gutz_optimize(evaluator; x0=g0, step, x_tol=g_tol, maxiter, verbose)
    if res.converged
        println(
            stderr,
            "Converged (", res.reason, ") in ",
            res.iters, " iterations and ",
            res.evals, " evals."
        )
    else
        println(
            stderr,
            "Failed to converge in $maxiter iterations"
        )
    end
    return res
end
