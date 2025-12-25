module;

#include <cmath>
#include <cstdint>
#include <cstring>
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

        bool atEnd() const { return matchAgainst(static_cast<uint8_t>(Ctrl::Sentinel) != 0); }

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

    export template<typename T,
                    typename Hash = std::hash<std::remove_cvref_t<T>>,
                    typename Equal = std::equal_to<T>>
    class Set;

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

            if (isFull(*ctrl))
            {
                return *this;
            }

            return skipEmptySlots();
        }

      private:
        static bool isFull(ctrl_t ctrl) { return (ctrl & 0x80) == 0; }

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
            if (g.atEnd())
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

                ctrl += 16;
                slot += 16;
                if (g.atEnd())
                {
                    return *this;
                }
            }
        }

        template<typename U, typename Hash, typename Equal>
        friend class Set;
    };

    constexpr auto MAX_LOAD_FACTOR = 0.875;

    template<typename T, typename Hash, typename Equal>
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
        Set(Set&& other) noexcept
            : size_(0)
            , used_(0)
            , capacity_(0)
            , ctrl_(nullptr)
        {
            swap(*this, other);
        }

        Set& operator=(Set const&);
        Set& operator=(Set&& other) noexcept
        {
            swap(*this, other);
            return *this;
        }

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

            auto h = hasher {};
            auto hash = h(key);
            auto group = self.h1(hash) % self.groups_;
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
                group = (group + 1) % self.groups_;
            }
        }

        void clear() noexcept
        {
            if (ctrl_ != nullptr)
            {
                for (size_t i = 0; i < capacity_; ++i)
                {
                    // TODO: speed up.
                    if ((ctrl_[i] & 0x80) == 0)
                    {
                        slots_[i].element()->~T();
                    }
                }
                std::free(ctrl_);
            }

            size_ = 0;
            used_ = 0;
            capacity_ = 0;
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

            rehash(static_cast<size_t>(std::ceil(count / MAX_LOAD_FACTOR)));
        }

        void rehash(size_type count)
        {
            size_t groupCount = findSmallestN(count);
            return rehashImpl(groupCount);
        }

        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            Slot<T> val(std::forward<Args>(args)...);
 auto h = hasher {};
            auto hash = h(val);
            auto h1Val = h1(hash);
            auto h2Val = h2(hash);
        }

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
        friend void swap(Set& lhs, Set& rhs) noexcept { lhs.swap(rhs); }

      private:
        bool isFull() const { return used_ == capacity_; }

        iterator iteratorAt(size_t offset) { return {ctrl_ + offset, slots_.data() + offset}; }
        const_iterator iteratorAt(size_t offset) const
        {
            return {ctrl_ + offset, slots_.data() + offset};
        }

        // Finds the smallest n such that  16 * n >= ceil(count / MAX_LOAD_FACTOR) + 1
        // with n being a power of 2.
        static size_t findSmallestN(size_t count)
        {
            size_t min_capacity = count + 1;
            return std::bit_ceil((min_capacity + 15) >> 4);
        }

        void rehashImpl(size_t newGroupCount)
        {
            auto count = 16 * newGroupCount;
            auto newCapacity = count - 1;
            auto* newCtrl = allocate(count);

            std::vector<Slot<T>> newSlots(newCapacity);
            std::memset(newCtrl, static_cast<ctrl_t>(Ctrl::Empty), newCapacity);
            newCtrl[newCapacity] = static_cast<ctrl_t>(Ctrl::Sentinel);

            auto insertNew = [&](Slot<T> const& value)
            {
                auto hash = hasher {}(*value.element());
                auto h1Val = h1(hash);
                auto h2Val = h2(hash);

                size_t mask = newGroupCount - 1;
                size_t group = h1Val & mask;

                while (true)
                {
                    Group g {newCtrl + group * 16};
                    // ~matchFull gives us Empty + Sentinel candidates.
                    uint32_t candidates = ~g.matchFull() & 0xFFFF;

                    while (candidates != 0)
                    {
                        int offset = std::countr_zero(candidates);
                        size_t idx = group * 16 + offset;

                        if (idx == newCapacity)
                        {
                            candidates &= ~(1U << offset);
                            continue;
                        }

                        // TODO: handle non trivially-relocatable types.
                        newSlots[idx] = value;
                        newCtrl[idx] = h2Val;
                        return;
                    }

                    group = (group + 1) & mask;
                }
            };

            if (capacity_ > 0)
            {
                ctrl_t* current = ctrl_;
                while (true)
                {
                    Group g(current);
                    for (auto i : BitMaskIterable(g.matchFull()))
                    {
                        auto& old = slots_[current - ctrl_ + i];
                        insertNew(old);
                    }
                    if (g.atEnd())
                    {
                        break;
                    }
                    current += 16;
                }
                std::free(ctrl_);
            }

            ctrl_ = newCtrl;
            slots_ = std::move(newSlots);
            capacity_ = newCapacity;
            groups_ = newGroupCount;
        }

        static ctrl_t* allocate(size_t count)
        {
            constexpr size_t alignment = 16;
            return static_cast<ctrl_t*>(std::aligned_alloc(alignment, sizeof(ctrl_t) * count));
        }

        size_t size_;
        size_t used_;
        size_t capacity_;
        size_t groups_;

        std::vector<Slot<T>> slots_;
        ctrl_t* ctrl_;
    };
}  // namespace alp