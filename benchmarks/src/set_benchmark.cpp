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

    template<typename T,
             typename Hash = alp::RapidHasher,
             typename Policy = alp::HashPolicySelector<T, Hash>::type,
             typename LoadFactorRatio = alp::DefaultLoadFactor,
             typename HashStoragePolicy = alp::DefaultHashStoragePolicy>
    void registerProbingSuites(std::string const& suiteName)
    {
        registerSuite<alp::Set<T,
                               Hash,
                               std::equal_to<T>,
                               Policy,
                               alp::DefaultBackend,
                               std::allocator<std::byte>,
                               LoadFactorRatio,
                               HashStoragePolicy,
                               alp::LinearProbing>>(suiteName + "_Linear");
        registerSuite<alp::Set<T,
                               Hash,
                               std::equal_to<T>,
                               Policy,
                               alp::DefaultBackend,
                               std::allocator<std::byte>,
                               LoadFactorRatio,
                               HashStoragePolicy,
                               alp::QuadraticProbing>>(suiteName + "_Quadratic");
    }

}  // namespace

int main(int argc, char** argv)
{
    benchmark::MaybeReenterWithoutASLR(argc, argv);

    registerProbingSuites<int64_t>("Alp_Int64_Rapid");
    registerProbingSuites<std::string>("Alp_String_Rapid");

    registerProbingSuites<int64_t, std::hash<int64_t>>("Alp_Int64_StdHash");
    registerProbingSuites<std::string, std::hash<std::string>>("Alp_String_StdHash");

    registerSuite<std::unordered_set<int64_t>>("Std_UnorderedSet_Int64");
    registerSuite<std::unordered_set<std::string>>("Std_UnorderedSet_String");

    registerSuite<absl::flat_hash_set<int64_t>>("Absl_FlatHashSet_Int64");
    registerSuite<absl::flat_hash_set<std::string>>("Absl_FlatHashSet_String");

    // Load Factor Benchmarks (87.5%, 85%, 90%)
    using LF875 = std::ratio<7, 8>;
    using LF85 = std::ratio<17, 20>;
    using LF90 = std::ratio<9, 10>;

    registerProbingSuites<std::string,
                          alp::RapidHasher,
                          alp::IdentityHashPolicy,
                          LF875,
                          alp::StoreHashTag>("Alp_String_Rapid_StoreHash_LF875");
    registerProbingSuites<std::string,
                          alp::RapidHasher,
                          alp::IdentityHashPolicy,
                          LF85,
                          alp::StoreHashTag>("Alp_String_Rapid_StoreHash_LF85");
    registerProbingSuites<std::string,
                          alp::RapidHasher,
                          alp::IdentityHashPolicy,
                          LF90,
                          alp::StoreHashTag>("Alp_String_Rapid_StoreHash_LF90");

    registerProbingSuites<std::string,
                          alp::RapidHasher,
                          alp::IdentityHashPolicy,
                          LF875,
                          alp::NoStoreHashTag>("Alp_String_Rapid_NoStoreHash_LF875");
    registerProbingSuites<std::string,
                          alp::RapidHasher,
                          alp::IdentityHashPolicy,
                          LF85,
                          alp::NoStoreHashTag>("Alp_String_Rapid_NoStoreHash_LF85");
    registerProbingSuites<std::string,
                          alp::RapidHasher,
                          alp::IdentityHashPolicy,
                          LF90,
                          alp::NoStoreHashTag>("Alp_String_Rapid_NoStoreHash_LF90");

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
