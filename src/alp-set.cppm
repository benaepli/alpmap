module;

#include <cmath>
#include <cstdint>
#include <expected>
#include <functional>

#include <emmintrin.h>

export module alp:set;

namespace alp
{
    using ctrl_t = uint8_t;

    export enum class Error : uint8_t
    {
        NotFound,
    };

    enum class Ctrl : ctrl_t
    {
        Empty = 0b10000000,
        Deleted = 0b11111110,
        Sentinel = 0b11111111,
    };

    template<typename T>
    struct Slot
    {
        alignas(T) uint8_t storage[sizeof(T)];

        T* element() { return reinterpret_cast<T*>(storage); }
        T const* element() const { return reinterpret_cast<T const*>(storage); }
    };

    class BitMaskIterable
    {
      public:
        explicit BitMaskIterable(uint32_t mask)
            : mask_(mask)
        {
        }

        struct Iterator
        {
            uint32_t mask;

            int operator*() const { return std::countr_zero(mask); }

            Iterator& operator++()
            {
                mask &= (mask - 1);
                return *this;
            }

            bool operator!=(Iterator const& other) const { return mask != other.mask; }
        };

        Iterator begin() const { return Iterator {mask_}; }
        Iterator end() const { return Iterator {0}; }

        explicit operator bool() const { return mask_ != 0; }

      private:
        uint32_t mask_;
    };

    struct Group
    {
        ctrl_t const* ctrl;

        BitMaskIterable match(uint8_t h2) const { return BitMaskIterable(matchAgainst(h2)); }

        uint32_t matchFull() const { return (~matchAllEmpty()) & 0xFFFF; }

        bool matchEmpty() const { return matchAgainst(static_cast<uint8_t>(Ctrl::Empty)) != 0; }

        bool atEnd() const { return ctrl[0] == static_cast<ctrl_t>(Ctrl::Sentinel); }

        bool nonEmpty() const { return matchAllEmpty() != 0xFFFF; }

      private:
        uint32_t matchAllEmpty() const
        {
            auto data = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ctrl));
            // High bit is 1 for empty, deleted, and sentinel
            return static_cast<uint32_t>(_mm_movemask_epi8(data));
        }

        auto mask(uint8_t val) const
        {
            auto match = _mm_set1_epi8(val);
            auto data = _mm_loadu_si128(reinterpret_cast<__m128i const*>(ctrl));
            return _mm_cmpeq_epi8(match, data);
        }

        uint32_t matchAgainst(uint8_t val) const
        {
            return static_cast<uint32_t>(_mm_movemask_epi8(mask(val)));
        }
    };

    constexpr size_t h1(size_t hash)
    {
        return hash >> 7;
    }
    constexpr ctrl_t h2(size_t hash)
    {
        return hash & 0x7F;
    }

    export template<typename T>
    struct SetIterator
    {
        ctrl_t const* ctrl;
        Slot<T>* slot;

        bool operator==(SetIterator const& other) const { return ctrl == other.ctrl; }

        Slot<T>& operator*() const { return *slot; }
        Slot<T>* operator->() const { return slot; }

        SetIterator& operator++()
        {
            ++ctrl;
            ++slot;

            if (isFull(*ctrl) || atEnd())
            {
                return *this;
            }

            return skipEmptySlots();
        }

      private:
        static bool isFull(ctrl_t ctrl) { return (ctrl & 0x80) == 0; }
        bool atEnd() const
        {
            // Sentinel is always the 0th byte.
            return *ctrl == static_cast<uint8_t>(Ctrl::Sentinel);
        }

        SetIterator& skipEmptySlots()
        {
            // First pass: possibly unaligned.
            auto addr = reinterpret_cast<uintptr_t>(ctrl);
            auto alignedAddr = addr & ~static_cast<uintptr_t>(15);
            if (alignedAddr == addr)
            {
                return skipEmptySlotsAligned();
            }
            auto* groupPtr = reinterpret_cast<ctrl_t*>(alignedAddr);
            int offset = static_cast<int>(addr & 15);
            Group g {groupPtr};
            uint32_t mask = g.matchFull();
            mask &= ~((1U << (offset + 1)) - 1);
            if (mask != 0)
            {
                int nextIndex = std::countr_zero(mask);
                int jump = nextIndex - offset;
                ctrl += jump;
                slot += jump;
                return *this;
            }
            int jumpToNext = 16 - offset;
            ctrl += jumpToNext;
            slot += jumpToNext;
            if (atEnd())
            {
                return *this;
            }
            return skipEmptySlotsAligned();
        }

        SetIterator& skipEmptySlotsAligned()
        {
            while (true)
            {
                Group g {ctrl};
                uint32_t mask = g.matchFull();

                if (mask != 0)
                {
                    int nextIndex = std::countr_zero(mask);
                    ctrl += nextIndex;
                    slot += nextIndex;
                    return *this;
                }

                if (atEnd())
                {
                    return *this;
                }
                ctrl += 16;
                slot += 16;
            }
        }

        friend class Set;
    };

    struct alignas(16) Block
    {
        uint8_t ctrl[16];
    };

    constexpr auto MAX_LOAD_FACTOR = 0.875;

    export template<typename T,
                    typename Hash = std::hash<std::remove_cvref_t<T>>,
                    typename Equal = std::equal_to<T>>
    class Set
    {
      public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using hasher = Hash;
        using key_equal = Equal;
        using reference = value_type&;
        using const_reference = value_type const&;
        using pointer = value_type*;
        using const_pointer = value_type const*;

        using iterator = SetIterator<T>;
        using const_iterator = SetIterator<T const>;

        Set() = default;
        explicit Set(size_type capacity) { reserve(capacity); }

        Set(Set const&);
        Set(Set&&) noexcept;
        Set& operator=(Set const&);
        Set& operator=(Set&&) noexcept;

        ~Set();

        iterator begin()
        {
            auto it = iteratorAt(0);
            it.skipEmptySlots();
            return it;
        }
        iterator end() { return iteratorAt(capacity_); }
        const_iterator begin() const
        {
            auto it = iteratorAt(0);
            it.skipEmptySlots();
            return it;
        }
        const_iterator end() const { return iteratorAt(capacity_); }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend() const { return end(); }

        [[nodiscard]] bool empty() const { return size_ == 0; }
        [[nodiscard]] size_t size() const { return size_; }

        template<typename Self, typename K>
            requires std::is_same_v<std::remove_cvref_t<K>, T>
            || requires { typename Hash::is_transparent; }
        [[nodiscard]] auto find(this Self&& self, K const& key)
        {
            if (self.size_ == 0)
            {
                return end();
            }

            auto hash = self.hasher(key);
            auto group = self.h1(hash) % self.numGroups();
            auto h2 = self.h2(hash);

            while (true)
            {
                Group g {self.ctrl_ + group * 16};
                for (int i : g.match(h2))
                {
                    if (key == self.slots_[group * 16 + i])
                    {
                        return self.iteratorAt(group * 16 + i);
                    }
                }
                if (g.matchEmpty())
                    return self.end();
                group = (group + 1) % self.numGroups();
            }
        }

        void clear() noexcept
        {
            size_ = 0;
            used_ = 0;
            capacity_ = 0;

            blocks_.clear();
            slots_.clear();
            ctrl_ = nullptr;
        }

        template<typename K>
            requires std::is_same_v<std::remove_cvref_t<K>, T>
            || requires { typename Hash::is_transparent; }
        bool contains(K const& key) const
        {
            return find(key) != end();
        }

        void reserve(size_type count)
        {
            if (count <= capacity_)
            {
                return;
            }

            // We find the smallest power of n such that 16 * 2^n >= ceil(count / MAX_LOAD_FACTOR).
            size_t min_cap = static_cast<size_t>(std::ceil(count / MAX_LOAD_FACTOR));
            min_cap = std::max(min_cap, size_t {16});
            size_t buckets = std::bit_ceil(min_cap);
            int n = std::countr_zero(buckets) - 4;
            rehashImpl(n);
        }

        void rehash(size_type count);

        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args);

        iterator erase(const_iterator pos);
        size_type erase(T const& key)
        {
            auto it = find(key);
            if (it == end())
            {
                return 0;
            }
            erase(it);
            return 1;
        }

        [[nodiscard]]
        std::expected<std::reference_wrapper<T>, Error> get(T const& key)
        {
            if (auto it = find(key); it != end())
                return std::ref(*it);
            return std::unexpected(Error::NotFound);
        }

        std::expected<void, Error> tryErase(T const& key)
        {
            if (auto it = find(key); it != end())
            {
                erase(it);
                return {};
            }
            return std::unexpected(Error::NotFound);
        }

        void swap(Set& other) noexcept;
        friend void swap(Set& lhs, Set& rhs) { lhs.swap(rhs); }

      private:
        bool isFull() const { return used_ + size_ == capacity_; }
        size_t numGroups() const { return blocks_.size(); }

        iterator iteratorAt(size_t offset) { return {ctrl_ + offset, slots_.data() + offset}; }
        const_iterator iteratorAt(size_t offset) const
        {
            return {ctrl_ + offset, slots_.data() + offset};
        }

        void rehashImpl(int newGroupCount) {}

        size_t size_;
        size_t used_;
        size_t capacity_;

        std::vector<Block> blocks_;
        std::vector<Slot<T>> slots_;
        ctrl_t const* ctrl_;
    };
}  // namespace alp