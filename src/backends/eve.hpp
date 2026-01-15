#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include <eve/eve.hpp>

namespace alp
{
    /// SIMD backend implementation using the EVE library.
    /// Provides zero-cost abstraction over EVE SIMD operations.
    struct EveBackend
    {
        // Determine lane count (capped at 64 to fit in uint64_t bitmask)
        static constexpr std::size_t max_size = 64;
        static constexpr auto native_size = eve::expected_cardinal_v<std::uint8_t>;
        static constexpr auto capped_size =
            std::min(native_size, static_cast<std::ptrdiff_t>(max_size));

        // SIMD types
        using ctrl_simd = eve::wide<std::uint8_t, eve::fixed<capped_size>>;
        using ctrl_mask = eve::logical<ctrl_simd>;

        // Backend interface types
        static constexpr std::size_t GroupSize = ctrl_simd::size();
        using Register = ctrl_simd;
        using Mask = ctrl_mask;
        using BitMask = std::uint64_t;

        /// Load control bytes from memory into a SIMD register.
        static Register load(std::uint8_t const* ptr) { return Register {ptr}; }

        /// Match all lanes equal to the given value.
        static Mask match(Register reg, std::uint8_t val) { return reg == ctrl_simd(val); }

        /// Match all lanes marked as empty (0x80).
        static Mask matchEmpty(Register reg) { return reg == static_cast<std::uint8_t>(0x80); }

        /// Match all lanes that contain a value (not empty, deleted, or sentinel).
        static Mask matchFull(Register reg)
        {
            return (reg != static_cast<std::uint8_t>(0x80))  // not Empty
                && (reg != static_cast<std::uint8_t>(0xFE))  // not Deleted
                && (reg != static_cast<std::uint8_t>(0xFF));  // not Sentinel
        }

        /// Check if any lane in the mask is true.
        static bool any(Mask mask) { return eve::any(mask); }

        /// Find the index of the first true lane.
        static std::optional<int> firstTrue(Mask mask) { return eve::first_true(mask); }

        /// Convert a SIMD mask to a scalar bitmask for iteration.
        static BitMask toBits(Mask mask) { return eve::top_bits {mask}.as_int(); }
    };
}  // namespace alp
