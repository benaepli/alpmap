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

#include <experimental/simd>

export module alp:set;

namespace alp
{
    namespace simd = std::experimental;

    using ctrl_simd = simd::fixed_size_simd<uint8_t, 32>;
    constexpr size_t LANE_COUNT = ctrl_simd::size();
    using ctrl_mask = ctrl_simd::mask_type;

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
        // It is located at the last byte of the control array.
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

    struct MatchIterable
    {
        ctrl_mask mask;

        struct Iterator
        {
            ctrl_mask mask;
            size_t idx;

            int operator*() const { return static_cast<int>(idx); }

            Iterator& operator++()
            {
                mask[idx] = false;
                if (simd::any_of(mask))
                {
                    idx = simd::find_first_set(mask);
                }
                else
                {
                    idx = mask.size();
                }
                return *this;
            }

            bool operator!=(Iterator const& other) const { return idx != other.idx; }
        };

        Iterator begin() const
        {
            return {
                mask,
                simd::any_of(mask) ? static_cast<size_t>(simd::find_first_set(mask)) : mask.size()};
        }
        Iterator end() const { return {mask, mask.size()}; }

        explicit operator bool() const { return simd::any_of(mask); }
    };

    /// A group of control bytes, the fundamental unit of Swiss Table probing.
    /// Uses SIMD instructions for fast parallel matching.
    struct Group
    {
        ctrl_simd data;

        explicit Group(ctrl_t const* ctrl) { data.copy_from(ctrl, simd::element_aligned_tag {}); }

        /// Returns an iterator over all slots in the group of slots
        /// whose control hash matches.
        MatchIterable match(uint8_t h2) const { return MatchIterable {data == h2}; }

        /// Returns a mask of all slots in the group that contain a value.
        /// (Not empty, deleted, or sentinel)
        auto matchFull() const
        {
            return (data != static_cast<ctrl_t>(Ctrl::Empty))
                && (data != static_cast<ctrl_t>(Ctrl::Deleted))
                && (data != static_cast<ctrl_t>(Ctrl::Sentinel));
        }

        auto matchEmpty() const { return data == static_cast<ctrl_t>(Ctrl::Empty); }

        /// Returns true if and only if there exists a slot in the group that has Ctrl::Empty.
        bool anyEmpty() const { return simd::any_of(matchEmpty()); }

        bool atEnd() const { return simd::any_of(data == static_cast<ctrl_t>(Ctrl::Sentinel)); }

        /// Returns true if and only if there exists a slot that isn't empty, deleted, or sentinel.
        bool hasValue() const { return simd::any_of(matchFull()); }
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
            // We align down to LANE_COUNT boundary.
            // Note: ctrl_t is 1 byte, so address arithmetic works directly.
            auto addr = reinterpret_cast<uintptr_t>(ctrl);
            auto alignedAddr = addr & ~static_cast<uintptr_t>(LANE_COUNT - 1);
            return reinterpret_cast<ctrl_t*>(alignedAddr);
        }

        [[nodiscard]] int groupOffset() const
        {
            auto addr = reinterpret_cast<uintptr_t>(ctrl);
            return static_cast<int>(addr & (LANE_COUNT - 1));
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
            auto mask = g.matchFull();
            for (int i = 0; i < offset; ++i)
            {
                mask[i] = false;
            }

            if (simd::any_of(mask))
            {
                int idx = simd::find_first_set(mask);
                int jump = idx - offset;
                ctrl += jump;
                slot += jump;
                return *this;
            }

            int jumpToNext = LANE_COUNT - offset;
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
                auto mask = g.matchFull();

                if (simd::any_of(mask))
                {
                    int nextIndex = simd::find_first_set(mask);
                    ctrl += nextIndex;
                    slot += nextIndex;
                    return *this;
                }

                ctrl += LANE_COUNT;
                slot += LANE_COUNT;
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
            size_t allocSize = LANE_COUNT * groups_;
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
                Group g {ctrl_ + group * LANE_COUNT};
                for (int i : g.match(h2Val))
                {
                    if (Equal {}(key, *slots_[group * LANE_COUNT + i].element()))
                    {
                        return group * LANE_COUNT + i;
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
                auto baseSlot = group * LANE_COUNT;
                Group g {ctrl_ + baseSlot};

                MatchIterable candidates = g.match(h2Val);
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
                if (simd::any_of(empty))
                {
                    if (size_ + 1 > capacity_ * MAX_LOAD_FACTOR)
                    {
                        reserve(size_ + 1);
                        return emplace_internal(value, h1Val, h2Val);
                    }

                    int offset = simd::find_first_set(empty);
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
            auto alignedAddr = addr & ~static_cast<uintptr_t>(LANE_COUNT - 1);
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
            return std::bit_ceil((min_capacity + LANE_COUNT - 1) / LANE_COUNT);
        }

        void rehashImpl(size_t newGroupCount)
        {
            auto count = LANE_COUNT * newGroupCount;
            auto newCapacity = count - 1;
            auto* newCtrl = allocate(count);

            std::unique_ptr<Slot<T>[]> newSlots(new Slot<T>[newCapacity]);
            std::memset(newCtrl, static_cast<ctrl_t>(Ctrl::Empty), newCapacity);
            newCtrl[newCapacity] = static_cast<ctrl_t>(Ctrl::Sentinel);

            size_t mask = newGroupCount - 1;

            auto insertUnchecked = [&](Slot<T>& oldSlot, size_t h1Val, ctrl_t h2Val)
            {
                size_t group = h1Val & mask;

                while (true)
                {
                    Group g {newCtrl + group * LANE_COUNT};
                    auto emptyMask = g.matchEmpty();

                    if (simd::any_of(emptyMask))
                    {
                        int offset = simd::find_first_set(emptyMask);
                        size_t idx = group * LANE_COUNT + offset;

                        if constexpr (std::is_trivially_copyable_v<T>)
                        {
                            // Fast path: direct memcpy for trivially copyable types
                            std::memcpy(newSlots[idx].storage, oldSlot.storage, sizeof(T));
                        }
                        else
                        {
                            // Standard path: move construct and destroy
                            std::construct_at(newSlots[idx].element(),
                                              std::move(*oldSlot.element()));
                            oldSlot.element()->~T();
                        }
                        newCtrl[idx] = h2Val;
                        return;
                    }

                    group = (group + 1) & mask;
                }
            };

            if (capacity_ > 0)
            {
                for (size_t gIdx = 0; gIdx < groups_; ++gIdx)
                {
                    Group g {ctrl_ + gIdx * LANE_COUNT};
                    auto fullMask = g.matchFull();

                    for (int i : MatchIterable(fullMask))
                    {
                        size_t oldIdx = gIdx * LANE_COUNT + i;
                        auto& oldSlot = slots_[oldIdx];

                        auto hash = Hash {}(*oldSlot.element());
                        auto h1Val = h1(hash);
                        auto h2Val = h2(hash);

                        insertUnchecked(oldSlot, h1Val, h2Val);
                    }
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
            constexpr size_t alignment = simd::memory_alignment_v<ctrl_simd>;
            return static_cast<ctrl_t*>(std::aligned_alloc(alignment, sizeof(ctrl_t) * count));
        }
    };

    /// A hash set based on Swiss Tables.
    /// Uses SIMD-accelerated probing for efficient lookup, insertion, and deletion.
    export template<typename T,
                    typename Hash = std::hash<std::remove_cvref_t<T>>,
                    typename Equal = std::equal_to<T>,
                    typename Policy = MixHashPolicy>
        requires std::move_constructible<T>
    class Set : Table<T, Hash, Equal, Policy>
    {
        using Base = Table<T, Hash, Equal, Policy>;

      public:
        using value_type = T;
        using size_type = Base::size_type;
        using difference_type = Base::difference_type;
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
        std::pair<iterator, bool> insert(T const& value) { return emplace(value); }

        /// Inserts the given value into the set by moving it.
        /// Returns a pair of an iterator to the element and a bool indicating
        /// whether insertion took place (true) or the element already existed (false).
        std::pair<iterator, bool> insert(T&& value) { return emplace(std::move(value)); }

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
