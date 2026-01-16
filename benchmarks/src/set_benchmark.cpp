#include <algorithm>
#include <cstdint>
#include <random>
#include <ratio>
#include <string>
#include <unordered_set>
#include <vector>

#include <absl/container/flat_hash_set.h>
#include <benchmark/benchmark.h>

import alp;

namespace
{
    template<typename T>
    struct DataGenerator;

    template<>
    struct DataGenerator<int64_t>
    {
        static std::vector<int64_t> generate(size_t count, uint64_t seed = 42)
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
    };

    template<>
    struct DataGenerator<std::string>
    {
        static std::vector<std::string> generate(size_t count, uint64_t seed = 42)
        {
            std::mt19937_64 rng(seed);
            std::uniform_int_distribution<int> charDist(0, 61);
            char const charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

            std::vector<std::string> result;
            result.reserve(count);

            size_t const length = 32;

            for (size_t i = 0; i < count; ++i)
            {
                std::string s;
                s.reserve(length);
                for (size_t j = 0; j < length; ++j)
                {
                    s += charset[charDist(rng)];
                }
                result.push_back(std::move(s));
            }
            return result;
        }
    };

    template<typename Container>
    void bmInsert(benchmark::State& state)
    {
        using T = Container::value_type;
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = DataGenerator<T>::generate(count);

        for (auto _ : state)
        {
            Container set;
            for (auto const& val : data)
            {
                set.insert(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }

    template<typename Container>
    void bmLookupHit(benchmark::State& state)
    {
        using T = Container::value_type;
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = DataGenerator<T>::generate(count);

        Container set;
        set.reserve(count);
        for (auto const& val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            for (auto const& val : data)
            {
                benchmark::DoNotOptimize(set.contains(val));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }

    template<typename Container>
    void bmLookupMiss(benchmark::State& state)
    {
        using T = typename Container::value_type;
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = DataGenerator<T>::generate(count, 42);
        auto const missData = DataGenerator<T>::generate(count, 1337);

        Container set;
        set.reserve(count);
        for (auto const& val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            for (auto const& val : missData)
            {
                benchmark::DoNotOptimize(set.contains(val));
            }
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }

    template<typename Container>
    void bmErase(benchmark::State& state)
    {
        using T = typename Container::value_type;
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = DataGenerator<T>::generate(count);

        for (auto _ : state)
        {
            state.PauseTiming();
            Container set;
            set.reserve(count);
            for (auto const& val : data)
            {
                set.insert(val);
            }
            state.ResumeTiming();

            for (auto const& val : data)
            {
                set.erase(val);
            }
            benchmark::DoNotOptimize(set);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }

    template<typename Container>
    void bmIterate(benchmark::State& state)
    {
        using T = typename Container::value_type;
        auto const count = static_cast<size_t>(state.range(0));
        auto const data = DataGenerator<T>::generate(count);

        Container set;
        set.reserve(count);
        for (auto const& val : data)
        {
            set.insert(val);
        }

        for (auto _ : state)
        {
            size_t items = 0;
            for (auto const& val : set)
            {
                benchmark::DoNotOptimize(val);
                items++;
            }
            benchmark::DoNotOptimize(items);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(count));
    }

    template<typename Container>
    void registerSuite(std::string const& suiteName)
    {
        auto registerWithRange = [&](std::string testName, auto func)
        {
            benchmark::RegisterBenchmark((suiteName + "/" + testName).c_str(), func)
                ->Range(8, 1 << 22);
        };

        registerWithRange("Insert", bmInsert<Container>);
        registerWithRange("LookupHit", bmLookupHit<Container>);
        registerWithRange("LookupMiss", bmLookupMiss<Container>);
        registerWithRange("Erase", bmErase<Container>);
        registerWithRange("Iterate", bmIterate<Container>);
    }

    template<template<typename...> typename Container>
    void registerSuites(std::string const& suiteName)
    {
        registerSuite<Container<int64_t>>(suiteName + "_Int64");
        registerSuite<Container<std::string>>(suiteName + "_String");
    }

    template<typename Hash,
             typename LoadFactorRatio,
             typename ProbingScheme>
    struct AlpSetBinder
    {
        template<typename T>
        using type = alp::Set<T,
                              Hash,
                              std::equal_to<T>,
                              typename alp::HashPolicySelector<T, Hash>::type,
                              alp::DefaultBackend,
                              std::allocator<std::byte>,
                              LoadFactorRatio,
                              alp::HashStorageSelector<T>,
                              ProbingScheme>;
    };

    template<typename Hash = alp::RapidHasher,
             typename LoadFactorRatio = alp::DefaultLoadFactor>
    void registerProbingSuites(std::string const& suiteName)
    {
        registerSuites<AlpSetBinder<Hash, LoadFactorRatio, alp::LinearProbing>::
                           template type>(suiteName + "_Linear");

        registerSuites<
            AlpSetBinder<Hash, LoadFactorRatio, alp::QuadraticProbing>::
                template type>(suiteName + "_Quadratic");
    }

}  // namespace

// Helper to adjust load factor by a delta (in units of 0.001)
template<typename Ratio, int DeltaMillis>
struct AdjustRatio
{
    static constexpr long long newNum = static_cast<long long>(Ratio::num) * 1000
        + static_cast<long long>(Ratio::den) * DeltaMillis;
    static constexpr long long newDen = static_cast<long long>(Ratio::den) * 1000;

    static constexpr long long gcd_val = std::gcd(newNum > 0 ? newNum : -newNum, newDen);

    using type = std::ratio<newNum / gcd_val, newDen / gcd_val>;
};

template<typename Ratio, int DeltaMillis>
using AdjustRatio_t = typename AdjustRatio<Ratio, DeltaMillis>::type;

int main(int argc, char** argv)
{
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    registerSuite<std::unordered_set<int64_t>>("Std_UnorderedSet_Int64");
    registerSuite<std::unordered_set<std::string>>("Std_UnorderedSet_String");

    registerSuite<absl::flat_hash_set<int64_t>>("Absl_FlatHashSet_Int64");
    registerSuite<absl::flat_hash_set<std::string>>("Absl_FlatHashSet_String");

    using DefaultBackendLF = alp::DynamicLoadFactorSelector<alp::DefaultBackend>::type;
    using DefaultLF_Minus = AdjustRatio_t<DefaultBackendLF, -25>;  // Default - 0.025
    using DefaultLF_Plus = AdjustRatio_t<DefaultBackendLF, 25>;  // Default + 0.025

    registerProbingSuites<alp::RapidHasher, DefaultBackendLF>("Alp_Rapid_LF_Default");
    registerProbingSuites<alp::RapidHasher, DefaultLF_Minus>("Alp_Rapid_LF_Minus025");
    registerProbingSuites<alp::RapidHasher, DefaultLF_Plus>("Alp_Rapid_LF_Plus025");

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
