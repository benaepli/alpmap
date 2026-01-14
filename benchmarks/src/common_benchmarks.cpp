#include <cstdint>
#include <random>
#include <vector>

namespace benchmarkUtil
{

    /// Generate a vector of random integers for benchmark use
    inline std::vector<int64_t> generateRandomInts(size_t count, uint64_t seed = 42)
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
    inline std::vector<int64_t> generateSequentialInts(size_t count)
    {
        std::vector<int64_t> result;
        result.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            result.push_back(static_cast<int64_t>(i));
        }
        return result;
    }

}  // namespace benchmarkUtil
