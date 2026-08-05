
ghex_compile_test(user_concepts)
ghex_reg_parallel_test(user_concepts 4 false)
ghex_reg_parallel_test(user_concepts 2 true)

ghex_compile_test(inline_unpack_race)
ghex_reg_parallel_test(inline_unpack_race 2 false)

