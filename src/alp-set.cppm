module;

#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

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

    /// A slot stores an element of type T using aligned raw storage.
    /// This allows us to manually control construction and destruction.
    template<typename T>
    struct Slot
    {
        alignas(T) uint8_t storage[sizeof(T)];

        T* element() { return reinterpret_cast<T*>(storage); }
        T const* element() const { return reinterpret_cast<T const*>(storage); }
    };

    /// A helper class for iterating over the set bits in a 16-bit mask.
    /// Used to iterate over matching slots in a SIMD group.
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

    /// A group of 16 control bytes, the fundamental unit of Swiss Table probing.
    /// Uses SSE/SIMD instructions for fast parallel matching.
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

    /// A hash policy that mixes bits to protect against poor std::hash implementations.
    /// Uses MurmurHash3's 64-bit finalizer for high-quality bit distribution.
    export struct MixHashPolicy
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

    /// A hash policy for when the user provides a high-quality hash.
    /// Skips the mixing step for maximum performance.
    export struct IdentityHashPolicy
    {
        static constexpr size_t apply(size_t h) { return h; }
    };

    /// Forward declaration of the Table base class.
    export template<typename T, typename Hash, typename Equal, typename Policy>
    class Table;

    /// Iterator for traversing elements in a Swiss Table.
    /// Uses the control byte array to efficiently skip empty/deleted slots.
    export template<typename T>
    struct SetIterator
    {
        template<typename U>
        friend struct SetIterator;

        /// Pointer to the current control byte.
        ctrl_t const* ctrl;
        /// Pointer to the current slot.
        Slot<T>* slot;

        SetIterator(ctrl_t const* c, Slot<T>* s)
            : ctrl(c)
            , slot(s)
        {
        }

        // Converting constructor: allows iterator to convert to const_iterator
        SetIterator(SetIterator<std::remove_const_t<T>> const& other)
            requires std::is_const_v<T> && (!std::is_same_v<T, std::remove_const_t<T>>)
            : ctrl(other.ctrl)
            , slot(reinterpret_cast<Slot<T>*>(other.slot))
        {
        }

        SetIterator(SetIterator const&) = default;
        SetIterator& operator=(SetIterator const&) = default;

        // Update comparison to work across iterator/const_iterator
        template<typename U>
        friend bool operator==(SetIterator<T> const& lhs, SetIterator<U> const& rhs)
        {
            return lhs.ctrl == rhs.ctrl;
        }

        T& operator*() const { return *slot->element(); }
        T* operator->() const { return slot->element(); }

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
                // Already aligned, skip directly to aligned version
                return skipEmptySlotsAligned();
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
        friend class Table;
    };

    /// Maximum load factor before rehashing is triggered.
    /// 7/8 = 0.875 provides a good balance between memory usage and probe length.
    constexpr auto MAX_LOAD_FACTOR = 0.875;

    /// Base class for Swiss Table hash containers.
    /// Implements the core Swiss Table algorithm with SIMD-accelerated probing.
    /// Uses open addressing with linear probing and a control byte array.
    template<typename T, typename Hash, typename Equal, typename Policy>
    class Table
    {
      protected:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator = SetIterator<T>;
        using const_iterator = SetIterator<T const>;

        Table() = default;

        explicit Table(size_type capacity) { reserve(capacity); }

        Table(Table const& other)
            requires std::copy_constructible<T>
            : size_(other.size_)
            , used_(other.used_)
            , capacity_(other.capacity_)
            , ctrlLen_(other.ctrlLen_)
            , groups_(other.groups_)
            , slots_(other.capacity_ > 0 ? new Slot<T>[other.capacity_] : nullptr)
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
                        std::construct_at(slots_[offset].element(), *it);

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

        Table(Table&& other) noexcept { this->swap(other); }

        Table& operator=(Table const& other)
            requires std::copy_constructible<T>
        {
            {
                if (this != &other)
                {
                    Table temp(other);
                    swap(temp);
                }
                return *this;
            }
        }

        Table& operator=(Table&& other) noexcept
        {
            this->swap(other);
            return *this;
        }

        ~Table() { clear(); }

      public:
        iterator begin()
        {
            auto it = iteratorAt(0);
            if (it != end())
            {
                it.skipEmptySlots();
            }
            return it;
        }
        iterator end() { return iteratorAt(ctrlLen_); }
        const_iterator begin() const
        {
            auto it = iteratorAt(0);
            if (it != end())
            {
                it.skipEmptySlots();
            }
            return it;
        }
        const_iterator end() const { return iteratorAt(ctrlLen_); }
        const_iterator cbegin() const { return begin(); }
        const_iterator cend() const { return end(); }

        [[nodiscard]] bool empty() const { return size_ == 0; }
        [[nodiscard]] size_t size() const { return size_; }

        void clear() noexcept
        {
            if (ctrl_ != nullptr)
            {
                for (size_t i = 0; i < capacity_; ++i)
                {
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
            ctrlLen_ = 0;
            slots_.reset();
            ctrl_ = nullptr;
        }

        void swap(Table& other) noexcept
        {
            using std::swap;
            swap(size_, other.size_);
            swap(used_, other.used_);
            swap(capacity_, other.capacity_);
            swap(ctrlLen_, other.ctrlLen_);
            swap(groups_, other.groups_);
            swap(slots_, other.slots_);
            swap(ctrl_, other.ctrl_);
        }

      protected:
        /// Finds the index of the slot containing the given key.
        /// Returns ctrlLen_ if not found.
        template<typename K>
        [[nodiscard]] auto find_internal(K const& key) const -> size_t
        {
            if (size_ == 0)
            {
                return ctrlLen_;
            }

            auto h = Hash {};
            auto hash = h(key);
            auto group = h1(hash) & (groups_ - 1);  // Since groups_ is a power of 2
            auto h2Val = h2(hash);

            while (true)
            {
                Group g {ctrl_ + group * 16};
                for (int i : g.match(h2Val))
                {
                    if (Equal {}(key, *slots_[group * 16 + i].element()))
                    {
                        return group * 16 + i;
                    }
                }
                if (g.anyEmpty())
                {
                    return ctrlLen_;
                }
                group = (group + 1) & (groups_ - 1);
            }
        }

        /// Core insertion logic. Checks for duplicates using SIMD probing,
        /// triggers rehash if needed, and inserts into an empty slot.
        /// Returns the index and whether insertion occurred.
        std::pair<size_t, bool> emplace_internal(T& value, size_t h1Val, ctrl_t h2Val)
        {
            if (capacity_ == 0)
            {
                reserve(1);
            }
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
                        return {slotNumber, false};
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

                    return {idx, true};
                }

                group = (group + 1) & mask;
            }
        }

        /// Wrapper that constructs a temporary element and computes the hash
        /// before delegating to the core insertion logic.
        template<typename... Args>
        std::pair<size_t, bool> emplace_wrapper(Args&&... args)
        {
            alignas(T) uint8_t tempStorage[sizeof(T)];
            T* temp =
                std::construct_at(reinterpret_cast<T*>(tempStorage), std::forward<Args>(args)...);

            auto h = Hash {};
            auto hash = h(*temp);
            auto h1Val = h1(hash);
            auto h2Val = h2(hash);

            auto result = emplace_internal(*temp, h1Val, h2Val);

            temp->~T();

            return result;
        }

        void reserve(size_type count)
        {
            auto desired = static_cast<size_t>(std::ceil(count / MAX_LOAD_FACTOR));
            if (desired <= capacity_)
            {
                return;
            }
            rehash(desired);
        }

        void rehash(size_type count)
        {
            // We first find the smallest number of groups
            // that satisfies size and power of 2 requirements.
            size_t groupCount = findSmallestN(count);
            return rehashImpl(groupCount);
        }

        /// Erases the element at the given slot index.
        /// Marks the slot as deleted or empty based on group state.
        void erase_slot(size_t offset)
        {
            slots_[offset].element()->~T();
            --size_;

            auto addr = reinterpret_cast<uintptr_t>(ctrl_ + offset);
            auto alignedAddr = addr & ~static_cast<uintptr_t>(15);
            auto groupPtr = reinterpret_cast<ctrl_t*>(alignedAddr);

            Group g {groupPtr};
            if (g.anyEmpty())
            {
                ctrl_[offset] = static_cast<ctrl_t>(Ctrl::Empty);
            }
            else
            {
                ctrl_[offset] = static_cast<ctrl_t>(Ctrl::Deleted);
            }
        }

        /// Extracts the upper bits for group selection.
        static constexpr size_t h1(size_t hash) { return Policy::apply(hash) >> 7; }
        /// Extracts the lower 7 bits for control byte matching.
        static constexpr ctrl_t h2(size_t hash) { return Policy::apply(hash) & 0x7F; }

        iterator iteratorAt(size_t offset) { return {ctrl_ + offset, slots_.get() + offset}; }
        const_iterator iteratorAt(size_t offset) const
        {
            return iterator {(ctrl_ + offset), const_cast<Slot<T>*>(slots_.get() + offset)};
        }

        size_t size_ = 0;
        size_t used_ = 0;
        size_t capacity_ = 0;
        size_t ctrlLen_ = 0;
        size_t groups_ = 0;
        std::unique_ptr<Slot<T>[]> slots_;
        ctrl_t* ctrl_ = nullptr;

      private:
        /// Finds the smallest n such that 16 * n >= count + 1
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

            // Allocate raw memory without initialization
            std::unique_ptr<Slot<T>[]> newSlots(new Slot<T>[newCapacity]);
            std::memset(newCtrl, static_cast<ctrl_t>(Ctrl::Empty), newCapacity);
            newCtrl[newCapacity] = static_cast<ctrl_t>(Ctrl::Sentinel);

            // Lambda to insert an element into the new table, moving and destroying in place
            auto insertNew = [&](Slot<T>& value)
            {
                auto hash = Hash {}(*value.element());
                auto h1Val = h1(hash);
                auto h2Val = h2(hash);

                size_t mask = newGroupCount - 1;
                // Calculate the desired group. Equivalent to h1Val % newGroupCount
                // since newGroupCount is a power of 2.
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
                        // Destroy the moved-from element immediately
                        value.element()->~T();
                        return;
                    }

                    group = (group + 1) & mask;
                }
            };

            // Single pass: move elements to new table and destroy originals
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
            ctrlLen_ = count;
            capacity_ = newCapacity;
            groups_ = newGroupCount;
        }

        static ctrl_t* allocate(size_t count)
        {
            constexpr size_t alignment = 16;
            return static_cast<ctrl_t*>(std::aligned_alloc(alignment, sizeof(ctrl_t) * count));
        }
    };

    /// A hash set based on Swiss Tables.
    /// Uses SIMD-accelerated probing for efficient lookup, insertion, and deletion.
    /// Elements are stored immutably; iterator always provides const access.
    export template<typename T,
                    typename Hash = std::hash<std::remove_cvref_t<T>>,
                    typename Equal = std::equal_to<T>,
                    typename Policy = MixHashPolicy>
        requires std::move_constructible<T>
    class Set : private Table<T, Hash, Equal, Policy>
    {
        using Base = Table<T, Hash, Equal, Policy>;

      public:
        using value_type = T;
        using size_type = typename Base::size_type;
        using difference_type = typename Base::difference_type;
        using hasher = Hash;
        using key_equal = Equal;
        using reference = value_type&;
        using const_reference = value_type const&;
        using pointer = value_type*;
        using const_pointer = value_type const*;

        using iterator = SetIterator<T const>;
        using const_iterator = SetIterator<T const>;

        using Base::Base;
        using Base::clear;
        using Base::empty;
        using Base::reserve;
        using Base::size;
        using Base::swap;

        Set() = default;
        explicit Set(size_type capacity)
            : Base(capacity)
        {
        }

        iterator begin() { return Base::begin(); }
        iterator end() { return Base::end(); }
        const_iterator begin() const { return Base::begin(); }
        const_iterator end() const { return Base::end(); }
        const_iterator cbegin() const { return Base::cbegin(); }
        const_iterator cend() const { return Base::cend(); }

        template<typename K>
            requires std::is_same_v<std::remove_cvref_t<K>, T>
            || requires { typename Hash::is_transparent; }
        [[nodiscard]] iterator find(K const& key)
        {
            size_t idx = Base::find_internal(key);
            if (idx == this->ctrlLen_)
                return end();
            return Base::iteratorAt(idx);
        }

        template<typename K>
            requires std::is_same_v<std::remove_cvref_t<K>, T>
            || requires { typename Hash::is_transparent; }
        [[nodiscard]] const_iterator find(K const& key) const
        {
            size_t idx = Base::find_internal(key);
            if (idx == this->ctrlLen_)
                return end();
            return Base::iteratorAt(idx);
        }

        template<typename K>
            requires std::is_same_v<std::remove_cvref_t<K>, T>
            || requires { typename Hash::is_transparent; }
        bool contains(K const& key) const
        {
            return find(key) != end();
        }

        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            auto [idx, success] = Base::emplace_wrapper(std::forward<Args>(args)...);
            return {Base::iteratorAt(idx), success};
        }

        /// Inserts the given value into the set.
        /// Returns a pair of an iterator to the element and a bool indicating
        /// whether insertion took place (true) or the element already existed (false).
        std::pair<iterator, bool> insert(T const& value)
        {
            return emplace(value);
        }

        /// Inserts the given value into the set by moving it.
        /// Returns a pair of an iterator to the element and a bool indicating
        /// whether insertion took place (true) or the element already existed (false).
        std::pair<iterator, bool> insert(T&& value)
        {
            return emplace(std::move(value));
        }

        void erase(const_iterator pos)
        {
            size_t offset = pos.ctrl - this->ctrl_;
            Base::erase_slot(offset);
        }

        size_type erase(T const& key)
        {
            size_t idx = Base::find_internal(key);
            if (idx == this->ctrlLen_)
                return 0;
            Base::erase_slot(idx);
            return 1;
        }

        [[nodiscard]]
        std::expected<std::reference_wrapper<T const>, Error> get(T const& key)
        {
            auto it = find(key);
            if (it != end())
                return std::ref(*it);
            return std::unexpected(Error::NotFound);
        }

        std::expected<void, Error> tryErase(T const& key)
        {
            auto it = find(key);
            if (it != end())
            {
                erase(it);
                return {};
            }
            return std::unexpected(Error::NotFound);
        }

        friend void swap(Set& lhs, Set& rhs) noexcept { lhs.swap(rhs); }
    };
}  // namespace alp
