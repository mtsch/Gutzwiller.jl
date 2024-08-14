for fn in EXAMPLES_FILES
    fnmd_full = Literate.markdown(
        joinpath(EXAMPLES_INPUT, fn), EXAMPLES_OUTPUT;
        documenter = true, execute = true
        )
    ex_num, margintitle = parse_header(fnmd_full)
    push!(EXAMPLES_NUMS, ex_num)
    fnmd = fn[1:end-2]*"md"     # full path does not work
    push!(EXAMPLES_PAIRS, margintitle => joinpath("generated", fnmd))
end
