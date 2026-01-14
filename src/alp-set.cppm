module;

#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <functional>
#include <memory>

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
        // The sentinel value indicates that we have reached the end of the array.
        // It is located at the fifteenth (last byte) of each group.
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

        /// Returns an iterator over all slots in the group of sixteen slots
        /// whose control hash matches.
        BitMaskIterable match(uint8_t h2) const { return BitMaskIterable(matchAgainst(h2)); }

        /// Returns a bit mask of all slots in the group that contain a value.
        uint32_t matchFull() const { return (~matchNoValue()) & 0xFFFF; }

        uint32_t matchEmpty() const { return matchAgainst(static_cast<uint8_t>(Ctrl::Empty)); }

        /// Returns true if and only if there exists a slot in the group that has Ctrl::Empty.
        bool anyEmpty() const { return matchAgainst(static_cast<uint8_t>(Ctrl::Empty)) != 0; }

        bool atEnd() const { return matchAgainst(static_cast<uint8_t>(Ctrl::Sentinel)) != 0; }

        /// Returns true if and only if there exists a slot that isn't empty, deleted, or sentinel.
        bool hasValue() const { return matchNoValue() != 0xFFFF; }

      private:
        uint32_t matchNoValue() const
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

    // A policy that mixes bits to protect against poor std::hash implementations
    struct MixHashPolicy
    {
        static constexpr size_t apply(size_t h)
        {
            // MurmurHash3 64-bit finalizer
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return h;
        }
    };

    // A policy for when the user provides a high-quality hash and wants to skip mixing
    export struct IdentityHashPolicy
    {
        static constexpr size_t apply(size_t h) { return h; }
    };

    export template<typename T,
                    typename Hash = std::hash<std::remove_cvref_t<T>>,
                    typename Equal = std::equal_to<T>,
                    typename Policy = MixHashPolicy>
        requires std::move_constructible<T>
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
        /// Returns true if the current slot contains an element.
        static bool isFull(ctrl_t ctrl) { return (ctrl & 0x80) == 0; }

        [[nodiscard]] ctrl_t* alignedGroup() const
        {
            auto addr = reinterpret_cast<uintptr_t>(ctrl);
            auto alignedAddr = addr & ~static_cast<uintptr_t>(15);
            return reinterpret_cast<ctrl_t*>(alignedAddr);
        }

        [[nodiscard]] int groupOffset() const
        {
            auto addr = reinterpret_cast<uintptr_t>(ctrl);
            return static_cast<int>(addr & 15);
        }

        SetIterator& skipEmptySlots()
        {
            // First pass: possibly unaligned.
            auto* groupPtr = alignedGroup();
            if (groupPtr == ctrl)
            {
                return skipEmptySlots();
            }
            auto const offset = groupOffset();
            Group g {groupPtr};
            uint32_t mask = g.matchFull();
            // Zero out all in our mask before the current element.
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
            // The group `g` references our old pointer, so skipping to the next
            // would mean that we're past the end of the control block.
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

        template<typename U, typename Hash, typename Equal, typename Policy>
            requires std::move_constructible<U>
        friend class Set;
    };

    constexpr auto MAX_LOAD_FACTOR = 0.875;

    template<typename T, typename Hash, typename Equal, typename Policy>
        requires std::move_constructible<T>
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

        Set(Set const& other)
            requires std::copy_constructible<T>
            : size_(other.size_)
            , used_(other.used_)
            , capacity_(other.capacity_)
            , groups_(other.groups_)
            , slots_(other.capacity_)
            , ctrl_(nullptr)
        {
            if (other.ctrl_ == nullptr)
            {
                return;
            }
            size_t allocSize = 16 * groups_;
            ctrl_ = allocate(allocSize);
            std::memset(ctrl_, static_cast<ctrl_t>(Ctrl::Empty), allocSize * sizeof(ctrl_t));

            if (capacity_ > 0)
            {
                try
                {
                    for (auto it = other.begin(); it != other.end(); ++it)
                    {
                        size_t offset = it.ctrl - other.ctrl_;
                        std::construct_at(slots_[offset].element(), *it->element());

                        ctrl_[offset] = other.ctrl_[offset];
                    }
                    ctrl_[capacity_] = static_cast<ctrl_t>(Ctrl::Sentinel);
                }
                catch (...)
                {
                    // Clean up: destroy all successfully constructed elements
                    for (size_t i = 0; i < capacity_; ++i)
                    {
                        if ((ctrl_[i] & 0x80) == 0)  // isFull check
                        {
                            slots_[i].element()->~T();
                        }
                    }
                    std::free(ctrl_);
                    throw;
                }
            }
        }

        Set(Set&& other) noexcept
            : size_(0)
            , used_(0)
            , capacity_(0)
            , groups_(0)
            , ctrl_(nullptr)
        {
            swap(*this, other);
        }

        Set& operator=(Set const& other)
            requires std::copy_constructible<T>
        {
            {
                if (this != &other)
                {
                    Set temp(other);
                    swap(temp);
                }
                return *this;
            }
        }
        Set& operator=(Set&& other) noexcept
        {
            swap(*this, other);
            return *this;
        }

        ~Set() { clear(); }

        iterator begin()
        {
            auto it = iteratorAt(0);
            if (it != end())
            {
                // We need to perform this check since skipNextSlots reads from the control block.
                it.skipEmptySlots();
            }
            return it;
        }
        iterator end() { return iteratorAt(capacity_); }
        const_iterator begin() const
        {
            auto it = iteratorAt(0);
            if (it != end())
            {
                it.skipEmptySlots();
            }
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
                return self.end();
            }

            auto h = hasher {};
            auto hash = h(key);
            auto group = self.h1(hash) & (self.groups_ - 1);  // Since self.groups_ is a power of 2
            auto h2 = self.h2(hash);

            while (true)
            {
                Group g {self.ctrl_ + group * 16};
                for (int i : g.match(h2))
                {
                    if (Equal {}(key, *self.slots_[group * 16 + i].element()))
                    {
                        return self.iteratorAt(group * 16 + i);
                    }
                }
                if (g.anyEmpty())
                {
                    return self.end();
                }
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
            // We first find the smallest number of groups
            // that satisfies size and power of 2 requirements.
            size_t groupCount = findSmallestN(count);
            return rehashImpl(groupCount);
        }

        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            alignas(T) uint8_t tempStorage[sizeof(T)];
            T* temp =
                std::construct_at(reinterpret_cast<T*>(tempStorage), std::forward<Args>(args)...);

            auto h = hasher {};
            auto hash = h(*temp);
            auto h1Val = h1(hash);
            auto h2Val = h2(hash);

            auto result = emplace_internal(*temp, h1Val, h2Val);

            temp->~T();

            return result;
        }

        void erase(const_iterator pos)
        {
            pos.slot->element()->~T();
            --size_;

            auto groupPtr = pos.alignedGroup();
            Group g {groupPtr};
            if (g.anyEmpty())
            {
                *pos.ctrl = static_cast<ctrl_t>(Ctrl::Empty);
            }
            else
            {
                *pos.ctrl = static_cast<ctrl_t>(Ctrl::Deleted);
            }
        }

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
                return std::ref(*it->element());
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

        void swap(Set& other) noexcept
        {
            using std::swap;
            swap(size_, other.size_);
            swap(used_, other.used_);
            swap(capacity_, other.capacity_);
            swap(groups_, other.groups_);
            swap(slots_, other.slots_);
            swap(ctrl_, other.ctrl_);
        }
        friend void swap(Set& lhs, Set& rhs) noexcept { lhs.swap(rhs); }

      private:
        std::pair<iterator, bool> emplace_internal(T& value, size_t h1Val, ctrl_t h2Val)
        {
            size_t mask = groups_ - 1;
            size_t group = h1Val & mask;

            while (true)
            {
                auto baseSlot = group * 16;
                Group g {ctrl_ + baseSlot};

                BitMaskIterable candidates = g.match(h2Val);
                for (auto i : candidates)
                {
                    auto slotNumber = baseSlot + i;
                    T const& result = *slots_[slotNumber].element();

                    if (Equal {}(result, value))
                    {
                        return {iteratorAt(slotNumber), false};
                    }
                }

                auto empty = g.matchEmpty();
                if (empty != 0)
                {
                    if (size_ + 1 > capacity_ * MAX_LOAD_FACTOR)
                    {
                        reserve(size_ + 1);
                        return emplace_internal(value, h1Val, h2Val);
                    }

                    int offset = std::countr_zero(empty);
                    size_t idx = baseSlot + offset;

                    ctrl_[idx] = h2Val;
                    std::construct_at(slots_[idx].element(), std::move(value));
                    size_++;

                    return {iteratorAt(idx), true};
                }

                group = (group + 1) & mask;
            }
        }

        // Incorporate the policy mix directly into these helpers
        static constexpr size_t h1(size_t hash)
        {
            // Apply policy, then shift for group index
            return Policy::apply(hash) >> 7;
        }

        static constexpr ctrl_t h2(size_t hash)
        {
            // Apply policy, then mask for control byte
            // Even if 'hash' is aligned (e.g. 128), the mix ensures entropy here.
            return Policy::apply(hash) & 0x7F;
        }

        bool isFull() const { return used_ == capacity_; }

        iterator iteratorAt(size_t offset) { return {ctrl_ + offset, slots_.data() + offset}; }
        const_iterator iteratorAt(size_t offset) const
        {
            return {const_cast<ctrl_t*>(ctrl_ + offset), const_cast<Slot<T>*>(slots_.data() + offset)};
        }

        /// Finds the smallest n such that  16 * n >= count + 1
        /// with n being a power of 2.
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

            auto insertNew = [&](Slot<T>& value)
            {
                auto hash = hasher {}(*value.element());
                auto h1Val = h1(hash);
                auto h2Val = h2(hash);

                size_t mask = newGroupCount - 1;
                // We calculate the original desired group. Equivalent to self.h1(hash) %
                // self.groups_ by power of 2.
                size_t group = h1Val & mask;

                while (true)
                {
                    Group g {newCtrl + group * 16};
                    uint32_t candidates = g.matchEmpty();

                    if (candidates != 0)
                    {
                        int offset = std::countr_zero(candidates);
                        size_t idx = group * 16 + offset;

                        std::construct_at(newSlots[idx].element(), std::move(*value.element()));
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

                // Destroy moved-from elements before freeing memory
                current = ctrl_;
                while (true)
                {
                    Group g(current);
                    for (auto i : BitMaskIterable(g.matchFull()))
                    {
                        auto& old = slots_[current - ctrl_ + i];
                        old.element()->~T();
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