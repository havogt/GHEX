/*
 * ghex-org
 *
 * Copyright (c) 2014-2026, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// The unstructured GPU pack/unpack kernels place ceil(num_indices / blockDim.y)
// in gridDim.y, which CUDA limits to 65535. A halo with more than
// blockDim.y * 65535 indices per neighbour makes the kernel launch fail with an
// invalid configuration; the missing launch-error check lets the exchange
// complete silently with the halo left untouched.

#include <gtest/gtest.h>
#include "../mpi_runner/mpi_test_fixture.hpp"

#include <ghex/config.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/communication_object.hpp>
#include "../util/memory.hpp"

#include <numeric>
#include <set>
#include <vector>

using domain_id_type = int;
using global_index_type = int;
using domain_descriptor_type =
    ghex::unstructured::domain_descriptor<domain_id_type, global_index_type>;
using halo_generator_type = ghex::unstructured::halo_generator<domain_id_type, global_index_type>;
using grid_type = ghex::unstructured::grid;

#ifdef GHEX_CUDACC

using data_descriptor_gpu_type =
    ghex::unstructured::data_descriptor<ghex::gpu, domain_id_type, global_index_type, int>;

namespace
{
// smallest halo for which the pack/unpack kernels exceed the CUDA gridDim.y
// limit of 65535
constexpr int halo_size = GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK_Y * 65535 + 1;
constexpr int inner_size = halo_size;

// Each rank owns gids [rank*inner_size, (rank+1)*inner_size); its halo is the
// first halo_size gids of the other rank, stored after the inner elements.
domain_descriptor_type
make_two_rank_domain(int rank)
{
    const int                      other = 1 - rank;
    std::vector<global_index_type> gids(inner_size + halo_size);
    std::iota(gids.begin(), gids.begin() + inner_size, rank * inner_size);
    std::iota(gids.begin() + inner_size, gids.end(), other * inner_size);
    std::vector<domain_descriptor_type::local_index_type> halo_lids(halo_size);
    std::iota(halo_lids.begin(), halo_lids.end(), inner_size);
    return {rank, gids.begin(), gids.end(), halo_lids.begin(), halo_lids.end()};
}
} // namespace

TEST_F(mpi_test_fixture, large_halo)
{
    if (world_size != 2) GTEST_SKIP() << "test requires exactly 2 ranks";

    ghex::context ctxt{MPI_COMM_WORLD, false};

    std::vector<domain_descriptor_type> local_domains{make_two_rank_domain(ctxt.rank())};
    const auto&                         d = local_domains[0];

    std::set<global_index_type> halo_set(d.outer_gids().begin(), d.outer_gids().end());
    halo_generator_type         hg{halo_set.begin(), halo_set.end()};

    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);
    auto co = ghex::make_communication_object<decltype(patterns)>(ctxt);

    ghex::test::util::memory<int> field(d.size(), 0);
    for (unsigned int lid = 0; lid < d.size(); ++lid)
        field[lid] = lid < inner_size ? static_cast<int>(d.global_index(lid).value()) + 1 : 0;
    field.clone_to_device();
    data_descriptor_gpu_type data{d, field.device_data(), 1, true, 0, 0};

    co.exchange(patterns(data)).wait();

    field.clone_to_host();
    int num_bad = 0;
    for (unsigned int lid = inner_size; lid < d.size(); ++lid)
        if (field[lid] != static_cast<int>(d.global_index(lid).value()) + 1) ++num_bad;
    EXPECT_EQ(num_bad, 0) << "halo elements were not exchanged correctly";
}

#else // GHEX_CUDACC

TEST_F(mpi_test_fixture, large_halo) { GTEST_SKIP() << "GPU-only test"; }

#endif
