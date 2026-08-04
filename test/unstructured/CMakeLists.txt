
ghex_compile_test(user_concepts)
ghex_reg_parallel_test(user_concepts 4 false)
ghex_reg_parallel_test(user_concepts 2 true)

ghex_compile_test(large_halo)
ghex_reg_parallel_test(large_halo 2 false)

