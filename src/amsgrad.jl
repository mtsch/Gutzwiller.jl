function amsgrad(f, params; kwargs...)
    params_svec = SVector{length(params)}(params)
    return amsgrad(f, params_svec; kwargs...)
end
function amsgrad(
    f, params::SVector{N,T};
    maxiter=100, α=0.01, β1=0.1, β2=β1^2, verbose=true,
    gtol=√eps(Float64),
    ptol=√eps(Float64),
) where {N,T}
    df = DataFrame()

    first_moment = zeros(SVector{N,T})
    second_moment = zeros(SVector{N,T})


    val, grad = val_and_grad(f, params)
    first_moment = grad

    for iter in 1:maxiter
        val, grad = val_and_grad(f, params)
        first_moment = (1 - β1) * first_moment + β1 * grad
        second_moment = max.((1 - β2) * second_moment + β2 * grad.^2, second_moment)
        Δparams = α .* first_moment ./ sqrt.(second_moment)

        if verbose
            print(stderr, "Step $iter:")
            print(stderr, " y: ", round(val, sigdigits=5))
            print(stderr, ", x: ", round.(params, sigdigits=5))
            print(stderr, ", Δx: ", round.(Δparams, sigdigits=5))
            println(stderr, ", ∇: ", round.(grad, sigdigits=5))
        end


        push!(df, (; iter, val, grad, params, Δparams))
        params -= Δparams

        if norm(Δparams) < ptol
            break
        end
        if norm(grad) < gtol
            break
        end
    end
    return df
end
