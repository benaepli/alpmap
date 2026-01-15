module;

#include <cstdint>
#include <optional>

#include <experimental/simd>

export module alp:backend_std_simd;

namespace alp
{
    namespace simd = std::experimental;

    /// SIMD backend implementation using std::experimental::simd.
    /// Operates on 16-byte groups (Standard Swiss Table size).
    export struct StdSimdBackend
    {
        using Register = simd::simd<std::uint8_t>;
        using Mask = Register::mask_type;

        static constexpr std::size_t GroupSize = Register::size();

        /// Iterator that works directly on the SIMD mask without converting to integer bitmask.
        struct Iterable
        {
            Mask mask;

            struct Iterator
            {
                Mask mask;
                int current_idx;

                int operator*() const { return current_idx; }

                // Finds the next set bit by clearing the current one
                Iterator& operator++()
                {
                    // Clear the current bit in the mask
                    mask[current_idx] = false;
                    if (simd::any_of(mask))
                    {
                        current_idx = simd::find_first_set(mask);
                    }
                    else
                    {
                        current_idx = GroupSize;
                    }
                    return *this;
                }

                bool operator!=(Iterator const& other) const
                {
                    return current_idx != other.current_idx;
                }
            };

            Iterator begin() const
            {
                if (simd::none_of(mask))
                {
                    return {mask, static_cast<int>(GroupSize)};
                }
                return {mask, simd::find_first_set(mask)};
            }

            Iterator end() const { return {mask, static_cast<int>(GroupSize)}; }
        };

        /// Load control bytes from memory into a SIMD register.
        static Register load(std::uint8_t const* ptr)
        {
            Register reg;
            reg.copy_from(ptr, simd::vector_aligned_tag {});
            return reg;
        }

        /// Match all lanes equal to the given value.
        static Mask match(Register reg, std::uint8_t val) { return reg == val; }

        /// Match all lanes marked as empty (0x80).
        static Mask matchEmpty(Register reg) { return reg == static_cast<std::uint8_t>(0x80); }

        /// Match all lanes that contain a value (not empty, deleted, or sentinel).
        static Mask matchFull(Register reg)
        {
            // We use bitwise operators on masks
            return (reg != 0x80)  // not Empty
                && (reg != 0xFE)  // not Deleted
                && (reg != 0xFF);  // not Sentinel
        }

        /// Check if any lane in the mask is true.
        static bool any(Mask mask) { return simd::any_of(mask); }

        /// Find the index of the first true lane.
        static std::optional<int> firstTrue(Mask mask)
        {
            if (simd::none_of(mask))
            {
                return std::nullopt;
            }
            return simd::find_first_set(mask);
        }

        /// Create an iterable object from the mask.
        static Iterable iterate(Mask mask) { return Iterable {mask}; }

        /// Find the next true bit starting search at `nextIndex` (inclusive).
        /// Used for skipping slots.
        static std::optional<int> nextTrue(Mask mask, std::size_t nextIndex)
        {
            for (std::size_t i = 0; i < nextIndex; ++i)
            {
                mask[i] = false;
            }

            if (simd::none_of(mask))
            {
                return std::nullopt;
            }
            return simd::find_first_set(mask);
        }
    };
}  // namespace alp