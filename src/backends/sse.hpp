#pragma once

#include <bit>
#include <cstdint>
#include <optional>

#include <emmintrin.h>

namespace alp
{
    struct SseBackend
    {
        static constexpr std::size_t GroupSize = 16;

        using Register = __m128i;
        using BitMask = std::uint32_t;

        using Mask = BitMask;

        static Register load(std::uint8_t const* ptr)
        {
            return _mm_loadu_si128(reinterpret_cast<Register const*>(ptr));
        }

        static Mask match(Register reg, std::uint8_t val)
        {
            auto matched = _mm_cmpeq_epi8(reg, _mm_set1_epi8(val));
            return static_cast<Mask>(_mm_movemask_epi8(matched));
        }

        static Mask matchEmpty(Register reg)
        {
            auto matched = _mm_cmpeq_epi8(reg, _mm_set1_epi8(static_cast<char>(0x80)));
            return static_cast<Mask>(_mm_movemask_epi8(matched));
        }

        static Mask matchFull(Register reg)
        {
            return (~static_cast<Mask>(_mm_movemask_epi8(reg))) & 0xFFFF;
        }

        static bool any(Mask mask) { return mask != 0; }

        static std::optional<int> firstTrue(Mask mask)
        {
            if (mask == 0)
            {
                return std::nullopt;
            }
            return std::countr_zero(mask);
        }

        static BitMask toBits(Mask mask) { return mask; }
    };
}  // namespace alp