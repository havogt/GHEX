/*
 * ghex-org
 *
 * Copyright (c) 2014-2026, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Reproducer for a suspected race: when the transport completes a GPU receive
// at Irecv-post time (oomph mpi backend, inline callback), the unpack kernel is
// launched on a non-blocking stream while kernels submitted to the user stream
// before schedule_exchange() may still be running and reading the halo.
//
// Rank 1 launches a long-running kernel on the user stream that repeatedly sums
// the halo, then delays so that rank 0's (small, eager) message has already
// arrived when it posts its receives. If unpacking respects the user stream, no
// sum may contain post-exchange values.

#include <gtest/gtest.h>
#include "../mpi_runner/mpi_test_fixture.hpp"

#include <ghex/config.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/communication_object.hpp>
#include "../util/memory.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <vector>

using domain_id_type = int;
using global_index_type = int;
using domain_descriptor_type =
    ghex::unstructured::domain_descriptor<domain_id_type, global_index_type>;
using halo_generator_type = ghex::unstructured::halo_generator<domain_id_type, global_index_type>;
using grid_type = ghex::unstructured::grid;

#ifdef GHEX_CUDACC

#include <ghex/device/cuda/runtime.hpp>

using data_descriptor_gpu_type =
    ghex::unstructured::data_descriptor<ghex::gpu, domain_id_type, global_index_type, int>;

namespace
{
long
env_or(const char* name, long def)
{
    const char* v = std::getenv(name);
    return v ? std::atol(v) : def;
}

// Each rank owns gids [rank*inner_size, (rank+1)*inner_size); its halo is the
// first halo_size gids of the other rank, stored after the inner elements.
domain_descriptor_type
make_two_rank_domain(int rank, int inner_size, int halo_size)
{
    const int                      other = 1 - rank;
    std::vector<global_index_type> gids(inner_size + halo_size);
    std::iota(gids.begin(), gids.begin() + inner_size, rank * inner_size);
    std::iota(gids.begin() + inner_size, gids.end(), other * inner_size);
    std::vector<domain_descriptor_type::local_index_type> halo_lids(halo_size);
    std::iota(halo_lids.begin(), halo_lids.end(), inner_size);
    return {rank, gids.begin(), gids.end(), halo_lids.begin(), halo_lids.end()};
}

__global__ void
reader_kernel(const int* field, int offset, int n, long long spin_cycles, int num_passes,
    unsigned long long* sums)
{
    for (int p = 0; p < num_passes; ++p)
    {
        unsigned long long s = 0;
        for (int i = 0; i < n; ++i) s += static_cast<const volatile int*>(field)[offset + i];
        sums[p] = s;
        const long long start = clock64();
        while (clock64() - start < spin_cycles) {}
    }
}

struct pass_stats
{
    int old_passes = 0;
    int new_passes = 0;
    int torn_passes = 0;
    int first_overlap = -1;
};

pass_stats
analyze(const unsigned long long* sums, int num_passes, unsigned long long expected_new)
{
    pass_stats st;
    for (int p = 0; p < num_passes; ++p)
    {
        if (sums[p] == 0ull) ++st.old_passes;
        else
        {
            if (st.first_overlap < 0) st.first_overlap = p;
            if (sums[p] == expected_new) ++st.new_passes;
            else
                ++st.torn_passes;
        }
    }
    return st;
}

struct case_params
{
    int  halo_size;
    int  num_passes;
    long spin_cycles;
    long sleep_ms;
    int  num_control_trials;
    int  num_trials;
};

// returns the number of skewed trials in which the unpack overlapped the
// pre-exchange reader kernel (only meaningful on rank 1)
int
run_case(ghex::context& ctxt, MPI_Comm world, const case_params& p)
{
    const int halo_size = p.halo_size;
    const int inner_size = std::max(4096, halo_size);

    std::vector<domain_descriptor_type> local_domains{
        make_two_rank_domain(ctxt.rank(), inner_size, halo_size)};
    const auto& d = local_domains[0];

    std::set<global_index_type> halo_set(d.outer_gids().begin(), d.outer_gids().end());
    halo_generator_type         hg{halo_set.begin(), halo_set.end()};

    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);
    auto co = ghex::make_communication_object<decltype(patterns)>(ctxt);

    ghex::test::util::memory<int>                field(d.size(), 0);
    ghex::test::util::memory<unsigned long long> sums(p.num_passes, 0);
    data_descriptor_gpu_type                     data_gpu{d, field.device_data(), 1, true, 0, 0};

    // inner element with gid g holds g+1; halo starts at 0, so after unpacking
    // each halo element is nonzero
    unsigned long long expected_new = 0;
    for (int i = 0; i < halo_size; ++i) expected_new += (1 - ctxt.rank()) * inner_size + i + 1;

    cudaStream_t stream;
    GHEX_CHECK_CUDA_RESULT(cudaStreamCreate(&stream));

    const bool slow_rank = (ctxt.rank() == 1);
    int        trials_with_overlap = 0;
    int        trials_with_torn = 0;

    for (int mode = 0; mode < 2; ++mode)
    {
        const bool sync_before_exchange = (mode == 0);
        const int  n_trials = sync_before_exchange ? p.num_control_trials : p.num_trials;
        for (int trial = 0; trial < n_trials; ++trial)
        {
            for (unsigned int lid = 0; lid < d.size(); ++lid)
                field[lid] = lid < static_cast<unsigned int>(inner_size)
                                 ? static_cast<int>(d.global_index(lid).value()) + 1
                                 : 0;
            field.clone_to_device();
            for (int i = 0; i < p.num_passes; ++i) sums[i] = 0xdeadbeefull;
            sums.clone_to_device();
            GHEX_CHECK_CUDA_RESULT(cudaDeviceSynchronize());
            MPI_Barrier(world);

            if (slow_rank)
            {
                reader_kernel<<<1, 1, 0, stream>>>(field.device_data(), inner_size, halo_size,
                    p.spin_cycles, p.num_passes, sums.device_data());
                GHEX_CHECK_CUDA_RESULT(cudaGetLastError());
                if (sync_before_exchange) { GHEX_CHECK_CUDA_RESULT(cudaDeviceSynchronize()); }
                else
                {
                    // progress MPI while waiting, like a busy application rank would,
                    // so the neighbour's unexpected message gets staged before Irecv;
                    // stagger the delay across trials to sample different
                    // arrival/progress interleavings
                    const long this_sleep_ms = p.sleep_ms + trial * 50;
                    for (long t = 0; t < this_sleep_ms; ++t)
                    {
                        int flag;
                        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, world, &flag, MPI_STATUS_IGNORE);
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
            }

            auto h = co.schedule_exchange(stream, patterns(data_gpu));
            h.schedule_wait(stream);
            GHEX_CHECK_CUDA_RESULT(cudaStreamSynchronize(stream));

            field.clone_to_host();
            {
                int first_bad = -1, num_bad = 0;
                for (unsigned int lid = inner_size; lid < d.size(); ++lid)
                    if (field[lid] != static_cast<int>(d.global_index(lid).value()) + 1)
                    {
                        if (first_bad < 0) first_bad = static_cast<int>(lid);
                        ++num_bad;
                    }
                if (num_bad > 0)
                    std::fprintf(stderr,
                        "[rank %d] %s trial %2d: halo mismatches %d/%d, first at lid %d "
                        "(value %d, expected %d)\n",
                        ctxt.rank(), sync_before_exchange ? "control" : "race   ", trial, num_bad,
                        halo_size, first_bad, field[first_bad],
                        static_cast<int>(d.global_index(first_bad).value()) + 1);
                EXPECT_EQ(num_bad, 0) << "halo content wrong";
            }

            if (slow_rank)
            {
                sums.clone_to_host();
                const auto st = analyze(sums.host_data(), p.num_passes, expected_new);
                std::fprintf(stderr,
                    "[rank 1] %s trial %2d: passes old=%d new=%d torn=%d first_overlap=%d\n",
                    sync_before_exchange ? "control" : "race   ", trial, st.old_passes,
                    st.new_passes, st.torn_passes, st.first_overlap);
                if (sync_before_exchange)
                {
                    EXPECT_EQ(st.new_passes + st.torn_passes, 0)
                        << "control run must not observe post-exchange halo values";
                }
                else
                {
                    trials_with_overlap += (st.new_passes + st.torn_passes) > 0 ? 1 : 0;
                    trials_with_torn += st.torn_passes > 0 ? 1 : 0;
                }
            }
            MPI_Barrier(world);
        }
    }

    if (slow_rank)
        std::fprintf(stderr,
            "[rank 1] summary (halo=%d ints, %zu bytes): %d/%d skewed trials with unpack "
            "overlapping the pre-exchange kernel (%d with torn reads)\n",
            halo_size, halo_size * sizeof(int), trials_with_overlap, p.num_trials,
            trials_with_torn);

    GHEX_CHECK_CUDA_RESULT(cudaStreamDestroy(stream));
    return trials_with_overlap;
}
} // namespace

TEST_F(mpi_test_fixture, inline_unpack_race)
{
    if (world_size != 2) GTEST_SKIP() << "test requires exactly 2 ranks";

    case_params p;
    p.num_passes = static_cast<int>(env_or("RACE_PASSES", 4000));
    p.spin_cycles = env_or("RACE_SPIN_CYCLES", 500000);
    p.sleep_ms = env_or("RACE_SLEEP_MS", 150);
    p.num_control_trials = static_cast<int>(env_or("RACE_CONTROL_TRIALS", 2));
    p.num_trials = static_cast<int>(env_or("RACE_TRIALS", 8));

    ghex::context ctxt{MPI_COMM_WORLD, false};

    // whether the inline completion fires depends on transport, message size and
    // placement; cover several eager-sized messages unless RACE_HALO selects one
    std::vector<int> halo_sizes = {128, 512, 4096};
    if (const long h = env_or("RACE_HALO", 0)) halo_sizes = {static_cast<int>(h)};

    int total_overlap = 0;
    for (const int halo_size : halo_sizes)
    {
        p.halo_size = halo_size;
        total_overlap += run_case(ctxt, world, p);
    }

    if (ctxt.rank() == 1)
        EXPECT_EQ(total_overlap, 0)
            << "unpack kernel ran concurrently with work submitted to the user stream before "
               "schedule_exchange()";
}

#else // GHEX_CUDACC

TEST_F(mpi_test_fixture, inline_unpack_race) { GTEST_SKIP() << "GPU-only test"; }

#endif
