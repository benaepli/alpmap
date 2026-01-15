module;

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "rapidhash.h"

export module alp:rapid_hash;

namespace alp
{
    /// A hash policy that mixes bits to protect against poor std::hash implementations.
    /// Uses MurmurHash3's 64-bit finalizer for high-quality bit distribution.
    export struct MixHashPolicy
    {
        static constexpr size_t apply(size_t h)
        {
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return h;
        }
    };

    /// A hash policy for when the user provides a high-quality hash.
    /// Skips the mixing step for maximum performance.
    export struct IdentityHashPolicy
    {
        static constexpr size_t apply(size_t h) { return h; }
    };

    export struct RapidHasher
    {
        using is_transparent = void;
        static constexpr std::uint64_t SEED = 0xbdd89aa982704029ULL;

        template<typename T>
            requires std::is_trivially_copyable_v<T>
        std::uint64_t operator()(T const& key) const noexcept
        {
            return rapidhash_withSeed(&key, sizeof(T), SEED);
        }

        template<typename T>
            requires requires(T t) {
                { t.data() } -> std::convertible_to<void const*>;
                t.size();
            }
        std::uint64_t operator()(T const& key) const noexcept
        {
            return rapidhash_withSeed(key.data(), key.size() * sizeof(T::value_type), SEED);
        }
    };

    /// Trait to select the appropriate hash policy based on the hasher type.
    /// Default: Assume the hash is weak (like std::hash) and needs mixing.
    export template<typename T, typename Hash>
    struct HashPolicySelector
    {
        using type = MixHashPolicy;
    };

    /// Specialization: If we are using RapidHasher, we don't need mixing.
    template<typename T>
    struct HashPolicySelector<T, RapidHasher>
    {
        using type = IdentityHashPolicy;
    };
}  // namespace alp