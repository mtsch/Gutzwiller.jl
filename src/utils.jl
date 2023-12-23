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
