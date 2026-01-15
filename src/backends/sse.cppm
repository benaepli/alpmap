module;

#include <bit>
#include <cstdint>
#include <optional>

#include <emmintrin.h>

export module alp:backend_sse;

namespace alp
{
    export struct SseBackend
    {
        static constexpr std::size_t GroupSize = 16;

        using Register = __m128i;
        using BitMask = std::uint32_t;

        using Mask = BitMask;

        struct Iterable
        {
            BitMask bits;

            struct Iterator
            {
                BitMask bits;
                int operator*() const { return std::countr_zero(bits); }
                Iterator& operator++()
                {
                    bits &= (bits - 1);
                    return *this;
                }
                bool operator!=(Iterator const& other) const { return bits != other.bits; }
            };

            Iterator begin() const { return {bits}; }
            Iterator end() const { return {0}; }
        };

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

        static Iterable iterate(Mask mask) { return Iterable{mask}; }

        static std::optional<int> nextTrue(Mask mask, size_t offset)
        {
            BitMask bits = mask & (~0U << offset);

            if (bits == 0)
            {
                return std::nullopt;
            }
            return std::countr_zero(bits);
        }

        static BitMask toBits(Mask mask) { return mask; }
    };
}  // namespace alp
