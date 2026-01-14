#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

#include <benchmark/benchmark.h>

import alp;

namespace
{

    /// Generate a vector of random integers for benchmark use
    std::vector<int64_t> generateRandomInts(size_t count, uint64_t seed = 42)
    {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int64_t> dist;
        std::vector<int64_t> result;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            result.push_back(dist(rng));
        }
        return result;
    }

    /// Generate a vector of sequential integers starting from 0
    std::vector<int64_t> generateSequentialInts(size_t count)
    {
        std::vector<int64_t> result;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            result.push_back(static_cast<int64_t>(i));
        }
        return result;
    }

    /// Benchmark inserting sequential integers into an empty set
    void bmSetInsertSequential(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateSequentialInts(count);

        for (auto _ : state)
        {
            alp::Set<int64_t> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetInsertSequential)->Range(8, 1 << 17);

    /// Benchmark inserting random integers (causes more collisions)
    void bmSetInsertRandom(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);

        for (auto _ : state)
        {
            alp::Set<int64_t> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetInsertRandom)->Range(8, 1 << 17);

    /// Benchmark successful lookups (all elements exist)
    void bmSetLookupHit(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);
        alp::Set<int64_t> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            for (auto val : data)
            {
                benchmark::DoNotOptimize(set.contains(val));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetLookupHit)->Range(8, 1 << 17);

    /// Benchmark failed lookups (no elements exist)
    void bmSetLookupMiss(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count, 42);
        auto const missData = generateRandomInts(count, 12345);  // Different seed
        alp::Set<int64_t> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            for (auto val : missData)
            {
                benchmark::DoNotOptimize(set.contains(val));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetLookupMiss)->Range(8, 1 << 17);

    /// Benchmark erasing elements from the set
    void bmSetErase(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);

        for (auto _ : state)
        {
            state.PauseTiming();
            alp::Set<int64_t> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            state.ResumeTiming();

            for (auto val : data)
            {
                set.erase(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetErase)->Range(8, 1 << 17);

    /// Benchmark iterating over all elements
    void bmSetIterate(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);
        alp::Set<int64_t> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            int64_t sum = 0;
            for (auto const& val : set)
            {
                sum += val;
            }
            benchmark::DoNotOptimize(sum);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetIterate)->Range(8, 1 << 17);

    /// Benchmark copy construction
    void bmSetCopy(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);
        alp::Set<int64_t> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            alp::Set<int64_t> copy(set);
            benchmark::DoNotOptimize(copy);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetCopy)->Range(8, 1 << 17);

    /// Baseline: std::unordered_set insert
    void bmStdSetInsertSequential(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateSequentialInts(count);

        for (auto _ : state)
        {
            std::unordered_set<int64_t> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmStdSetInsertSequential)->Range(8, 1 << 17);

    /// Baseline: std::unordered_set random insert
    void bmStdSetInsertRandom(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);

        for (auto _ : state)
        {
            std::unordered_set<int64_t> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmStdSetInsertRandom)->Range(8, 1 << 17);

    /// Baseline: std::unordered_set lookup hit
    void bmStdSetLookupHit(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);
        std::unordered_set<int64_t> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            for (auto val : data)
            {
                benchmark::DoNotOptimize(set.contains(val));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmStdSetLookupHit)->Range(8, 1 << 17);

    /// Baseline: std::unordered_set lookup miss
    void bmStdSetLookupMiss(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count, 42);
        auto const missData = generateRandomInts(count, 12345);
        std::unordered_set<int64_t> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            for (auto val : missData)
            {
                benchmark::DoNotOptimize(set.contains(val));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmStdSetLookupMiss)->Range(8, 1 << 17);

    /// Baseline: std::unordered_set iterate
    void bmStdSetIterate(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);
        std::unordered_set<int64_t> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            int64_t sum = 0;
            for (auto const& val : set)
            {
                sum += val;
            }
            benchmark::DoNotOptimize(sum);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmStdSetIterate)->Range(8, 1 << 17);

    /// Baseline: std::unordered_set erase
    void bmStdSetErase(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);

        for (auto _ : state)
        {
            state.PauseTiming();
            std::unordered_set<int64_t> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            state.ResumeTiming();

            for (auto val : data)
            {
                set.erase(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmStdSetErase)->Range(8, 1 << 17);

    /// Baseline: std::unordered_set copy
    void bmStdSetCopy(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);
        std::unordered_set<int64_t> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            std::unordered_set<int64_t> copy(set);
            benchmark::DoNotOptimize(copy);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmStdSetCopy)->Range(8, 1 << 17);

    /// Template benchmark for insert with configurable hash policy
    template<typename Policy>
    void bmSetInsertRandomPolicy(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);

        for (auto _ : state)
        {
            alp::Set<int64_t, std::hash<int64_t>, std::equal_to<int64_t>, Policy> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetInsertRandomPolicy<alp::MixHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetInsertRandom/MixHashPolicy");
    BENCHMARK(bmSetInsertRandomPolicy<alp::IdentityHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetInsertRandom/IdentityHashPolicy");

    /// Template benchmark for sequential insert with configurable hash policy
    template<typename Policy>
    void bmSetInsertSequentialPolicy(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateSequentialInts(count);

        for (auto _ : state)
        {
            alp::Set<int64_t, std::hash<int64_t>, std::equal_to<int64_t>, Policy> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetInsertSequentialPolicy<alp::MixHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetInsertSequential/MixHashPolicy");
    BENCHMARK(bmSetInsertSequentialPolicy<alp::IdentityHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetInsertSequential/IdentityHashPolicy");

    /// Template benchmark for lookup hit with configurable hash policy
    template<typename Policy>
    void bmSetLookupHitPolicy(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);
        alp::Set<int64_t, std::hash<int64_t>, std::equal_to<int64_t>, Policy> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            for (auto val : data)
            {
                benchmark::DoNotOptimize(set.contains(val));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetLookupHitPolicy<alp::MixHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetLookupHit/MixHashPolicy");
    BENCHMARK(bmSetLookupHitPolicy<alp::IdentityHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetLookupHit/IdentityHashPolicy");

    /// Template benchmark for lookup miss with configurable hash policy
    template<typename Policy>
    void bmSetLookupMissPolicy(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count, 42);
        auto const missData = generateRandomInts(count, 12345);
        alp::Set<int64_t, std::hash<int64_t>, std::equal_to<int64_t>, Policy> set;
        for (auto val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            for (auto val : missData)
            {
                benchmark::DoNotOptimize(set.contains(val));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetLookupMissPolicy<alp::MixHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetLookupMiss/MixHashPolicy");
    BENCHMARK(bmSetLookupMissPolicy<alp::IdentityHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetLookupMiss/IdentityHashPolicy");

    /// Template benchmark for erase with configurable hash policy
    template<typename Policy>
    void bmSetErasePolicy(benchmark::State& state)
    {
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = generateRandomInts(count);

        for (auto _ : state)
        {
            state.PauseTiming();
            alp::Set<int64_t, std::hash<int64_t>, std::equal_to<int64_t>, Policy> set;
            for (auto val : data)
            {
                set.insert(val);
            }
            state.ResumeTiming();

            for (auto val : data)
            {
                set.erase(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }
    BENCHMARK(bmSetErasePolicy<alp::MixHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetErase/MixHashPolicy");
    BENCHMARK(bmSetErasePolicy<alp::IdentityHashPolicy>)
        ->Range(8, 1 << 17)
        ->Name("bmSetErase/IdentityHashPolicy");

}  // namespace
