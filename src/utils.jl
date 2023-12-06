function time_format(io, x)
    minutes, seconds = divrem(x, 60)
    hours, minutes = divrem(minutes, 60)

    if hours > 0
        print(io, hours, "h ", minutes, "m ", seconds, "s")
    elseif minutes > 0
        print(io, minutes, "m ", seconds, "s")
    else
        print(io, round(seconds, digits=3), "s")
    end
end

"""
    local_energy(H, vector, addr)
    local_energy(H, vector)

Compute the local energy of address `addr` in `vector` with respect to `H`. If `addr` is not
given, compute the local energy across the whole vector.
"""
function local_energy(H, vector, addr1)
    bot = vector[addr1]
    top = sum(offdiagonals(H, addr1)) do (addr2, melem)
        melem * vector[addr2]
    end
    top += diagonal_element(H, addr1) * bot
    return top / bot
end

function local_energy(H, vector)
    top = sum(pairs(vector)) do (k, v)
        local_energy(H, vector, k) * v^2
    end
    return top / sum(abs2, vector)
end

function local_energy!(result, addrs, H, vector)
    if length(result) != length(addrs)
        throw(ArgumentError("lengths of `result` and `addrs` don't match!"))
    end
    if length(result) == 0
        return result
    end
    curr_addr = first(addrs)
    result[1] = local_energy(H, vector, curr_addr)
    for i in 2:length(result)
        prev_addr = curr_addr
        curr_addr = addrs[i]
        if curr_addr ≠ prev_addr
            result[i] = local_energy(H, vector, curr_addr)
        else
            result[i] = result[i - 1]
        end
    end
    return result
end

"""
    GutzwillerAnsatz{A,H}

Placeholder TODO: make a proper DVec
"""
struct GutzwillerAnsatz{A,H,T<:Real}
    H::H
    g::T
end
function GutzwillerAnsatz(H, g)
    A = typeof(starting_address(H))
    return GutzwillerAnsatz{A,typeof(H),typeof(g)}(H,g)
end

Base.keytype(::GutzwillerAnsatz{A}) where {A} = A
Base.valtype(ga::GutzwillerAnsatz{<:Any,<:Any,T}) where {T} = promote_type(eltype(ga.H), T)

function Base.getindex(gv::GutzwillerAnsatz{A}, addr::A) where {A}
    exp(-gv.g * diagonal_element(gv.H, addr))
end

function collect_to_dvec!(dst, gv::GutzwillerAnsatz)
    basis = build_basis(gv.H)
    for k in basis
        dst[k] = gv[k]
    end
    return dst
end
function Rimu.DVec(gv::GutzwillerAnsatz; kwargs...)
    A = typeof(starting_address(gv.H))
    result = DVec{A,Float64}(; kwargs...)
    return collect_to_vec!(result, gv)
end
function Rimu.PDVec(gv::GutzwillerAnsatz; kwargs...)
    A = typeof(starting_address(gv.H))
    result = PDVec{A,Float64}(; kwargs...)
    return collect_to_vec!(result, gv)
end
