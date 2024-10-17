"""
    GradientDescentResult

Return type of [`adaptive_gradient_descent`](@ref) variants such as
[`gradient_descent`](@ref) and [`amsgrad`](@ref).

Can be used as a `Table` from Tables.jl.
"""
struct GradientDescentResult{T,P<:SVector{<:Any,T},F}
    fun::F
    hyperparameters::NamedTuple
    initial_params::P

    iterations::Int
    converged::Bool
    reason::String

    param::Vector{P}
    value::Vector{T}
    error::Vector{T}
    gradient::Vector{P}
    first_moment::Vector{P}
    second_moment::Vector{P}
    param_delta::Vector{P}
end

function Base.show(io::IO, r::GradientDescentResult)
    println(io, "GradientDescentResult")
    println(io, "  iterations: ", r.iterations)
    println(io, "  converged: ", r.converged, " (", r.reason, ")")
    println(io, "  last value: ", r.value[end], " ± ", r.error[end])
    print(io, "  last params: ", r.param[end])
end

Tables.istable(::Type{<:GradientDescentResult}) = true
Tables.rowaccess(::Type{<:GradientDescentResult}) = true
Tables.rows(r::GradientDescentResult) = GradientDescentResultRows(r)
function Tables.Schema(r::GradientDescentResult{T,P}) where {T,P}
    hyper_keys = keys(r.hyperparameters)
    hyper_types = typeof.(value.(r.hyperparameters))

    return Tables.Schema(
        (hyper_keys..., :iter, :param, :value, :error, :gradient,
         :first_moment, :second_moment, :param_delta),
        (hyper_types..., Int, Int, P, T, T, P, P, P, P)
    )
end

struct GradientDescentResultRows{T,P,F} <: AbstractVector{NamedTuple}
    result::GradientDescentResult{T,P,F}
end
function Base.getindex(rows::GradientDescentResultRows, i)
    r = rows.result

    return (; r.hyperparameters...,
            iter=i,
            param=r.param[i],
            value=r.value[i],
            error=r.error[i],
            gradient=r.gradient[i],
            first_moment=r.first_moment[i],
            second_moment=r.second_moment[i],
            param_delta=r.param_delta[i],
            )
end
function Base.size(rows::GradientDescentResultRows)
    return (length(rows.result.param), 1)
end

"""
    adaptive_gradient_descent(
        φ, ψ, f, p_init::P;
        maxiter=100, verbose=true, α=0.01,
        grad_tol=√eps(T), param_tol=√eps(T), val_tol=√(eps(T)),
        kwargs...
    )

For immediately useful versions of this function, see [`gradient_descent`] and [`amsgrad`].
Returns [`GradientDescentResult`](@ref).
"""
function adaptive_gradient_descent(φ, ψ, f, params; kwargs...)
    params_svec = SVector{length(params)}(params)
    return adaptive_gradient_descent(φ, ψ, f, params_svec; kwargs...)
end
function adaptive_gradient_descent(
    φ, ψ, f, p_init::P;
    maxiter=100, verbose=true, α=0.01,
    grad_tol=√eps(T), param_tol=√eps(T), val_tol=√(eps(T)),
    kwargs...
) where {N,T,P<:SVector{N,T}}

    first_moment = zeros(SVector{N,T})
    second_moment = zeros(SVector{N,T})

    val, grad = val_and_grad(f, p_init)
    first_moment = grad
    old_val = Inf

    iter = 0

    params = P[]
    values = T[]
    errors = T[]
    gradients = P[]
    first_moments = P[]
    second_moments = P[]
    param_deltas = P[]
    converged = false
    reason = "iterations"

    p = p_init

    verbose && (prog = Progress(maxiter))

    while iter ≤ maxiter
        iter += 1
        val, err, grad = val_err_and_grad(f, p)
        first_moment = φ(first_moment, grad; kwargs...)
        second_moment = ψ(second_moment, grad; kwargs...)

        δp = -α * first_moment ./ .√second_moment
        δval = old_val - val
        old_val = val

        push!(params, p)
        push!(values, val)
        push!(errors, err)
        push!(gradients, grad)
        push!(first_moments, first_moment)
        push!(second_moments, second_moment)
        push!(param_deltas, δp)

        if norm(δp) < param_tol
            verbose && @info "Converged (params)"
            reason = "params"
            converged = true
            break
        end
        if norm(grad) < grad_tol
            verbose && @info "Converged (grad)"
            reason = "gradient"
            converged = true
            break
        end
        if abs(δval) < val_tol
            verbose && @info "Converged (value)"
            reason = "value"
            converged = true
            break
        end

        p = p + δp

        verbose && next!(
            prog; showvalues=(((:iter, iter), (:value, val), (:param, p)))
        )
    end
    iter == maxiter && verbose && @info "Aborted (maxiter)"

    verbose && finish!(prog)

    return GradientDescentResult(
        f, (; α, kwargs...), p_init, length(params), converged, reason,
        params, values, errors, gradients, first_moments, second_moments, param_deltas,
    )
end

"""
    gradient_descent(
        f, p0;
        maxiter=100, verbose=true, α=0.01,
        grad_tol=√eps(T), param_tol=√eps(T), val_tol=√(eps(T)),
    )

Vanilla gradient descent on function `f` with initial parameters `p0`.

See also [`adaptive_gradient_descent`](@ref). Returns [`GradientDescentResult`](@ref).
"""
function gradient_descent(f, params; kwargs...)
    φ(m1, g; _...) = g
    ψ(m2, g; _...) = ones(typeof(g))
    return adaptive_gradient_descent(φ, ψ, f, params; kwargs...)
end

"""
    amsgrad(
        f, p0;
        maxiter=100, verbose=true, α=0.01, β1=0.1, β2=0.01,
        grad_tol=√eps(T), param_tol=√eps(T), val_tol=√(eps(T)),
    )

[AMSGrad](https://paperswithcode.com/method/amsgrad) on function `f` with initial parameters
`p0`.

See also [`adaptive_gradient_descent`](@ref). Returns [`GradientDescentResult`](@ref).
"""
function amsgrad(f, params; β1=0.1, β2=0.01, kwargs...)
    φ(m1, g; β1, _...) = (1 - β1) * m1 + β1 * g
    ψ(m2, g; β2, _...) = max.((1 - β2) * m2 + β2 * g.^2, m2)
    return adaptive_gradient_descent(φ, ψ, f, params; β1, β2, kwargs...)
end
