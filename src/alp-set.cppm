module;

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <ratio>
#include <tuple>
#include <utility>
#include <vector>

export module alp:set;

#if defined(ALP_USE_EVE)
import :backend_eve;
#endif
import :backend_sse;
import :rapid_hash;

namespace alp
{
    export template<typename B>
    concept SimdBackend = requires(std::uint8_t const* ptr, std::uint8_t val) {
        { B::GroupSize } -> std::convertible_to<std::size_t>;

        typename B::Register;  // SIMD register type (e.g., eve::wide<uint8_t>)
        typename B::Mask;  // SIMD mask type (e.g., eve::logical<...>)

        typename B::Iterable;  // Must provide begin()/end()

        // Load operation
        { B::load(ptr) } -> std::same_as<typename B::Register>;

        // Matching operations return SIMD masks
        { B::match(B::load(ptr), val) } -> std::same_as<typename B::Mask>;
        { B::matchEmpty(B::load(ptr)) } -> std::same_as<typename B::Mask>;
        { B::matchFull(B::load(ptr)) } -> std::same_as<typename B::Mask>;

        // Mask query operations
        { B::any(typename B::Mask {}) } -> std::convertible_to<bool>;
        { B::firstTrue(typename B::Mask {}) } -> std::convertible_to<std::optional<int>>;

        { B::iterate(typename B::Mask {}) } -> std::same_as<typename B::Iterable>;
        // Get the next set bit for some offset greater than zero
        { B::nextTrue(typename B::Mask {}, size_t {}) } -> std::convertible_to<std::optional<int>>;
    };

#if defined(ALP_USE_EVE)
    export using DefaultBackend = EveBackend;
#else
    export using DefaultBackend = SseBackend;
#endif

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

    struct StoreHashTag
    {
    };
    struct NoStoreHashTag
    {
    };

    using DefaultHashStoragePolicy = NoStoreHashTag;

    /// A slot stores an element of type T using aligned raw storage.
    /// This allows us to manually control construction and destruction.
    /// Primary template: stores the full hash to avoid recomputation during rehash.
    template<typename T, typename HashStoragePolicy = StoreHashTag>
    struct Slot
    {
        alignas(T) uint8_t storage[sizeof(T)];
        size_t hash;  // Cached hash for fast rehashing

        T* element() { return reinterpret_cast<T*>(storage); }
        T const* element() const { return reinterpret_cast<T const*>(storage); }
    };

    /// Specialization: without hash storage (memory-saving mode)
    template<typename T>
    struct Slot<T, NoStoreHashTag>
    {
        alignas(T) uint8_t storage[sizeof(T)];
        // No hash field - saves sizeof(size_t) per slot

        T* element() { return reinterpret_cast<T*>(storage); }
        T const* element() const { return reinterpret_cast<T const*>(storage); }
    };

    /// A byte wrapper that carries alignment in the type system.
    /// Allocators respecting alignof(value_type) will automatically align.
    template<size_t Alignment>
    struct alignas(Alignment) AlignedByte
    {
        std::byte value;
    };

    /// Allocator adapter that guarantees alignment by over-allocating.
    /// Stores original pointer in header before aligned region for deallocation.
    template<typename Alloc, size_t Alignment>
    struct AlignedAllocatorAdapter
    {
        using value_type = std::byte;
        using InnerAllocTraits = std::allocator_traits<Alloc>;

        [[no_unique_address]] Alloc inner_;

        AlignedAllocatorAdapter() = default;

        template<typename U>
        explicit AlignedAllocatorAdapter(U const& alloc)
            : inner_(alloc)
        {
        }

        std::byte* allocate(size_t n)
        {
            // Over-allocate: alignment for adjustment + pointer storage
            size_t extra = Alignment - 1 + sizeof(void*);
            size_t totalUnits =
                (n + extra + sizeof(AlignedByte<Alignment>) - 1) / sizeof(AlignedByte<Alignment>);
            auto* raw = InnerAllocTraits::allocate(inner_, totalUnits);
            auto* rawBytes = reinterpret_cast<std::byte*>(raw);

            // Align: leave room for pointer storage before aligned region
            uintptr_t rawAddr = reinterpret_cast<uintptr_t>(rawBytes) + sizeof(void*);
            uintptr_t alignedAddr = (rawAddr + Alignment - 1) & ~(Alignment - 1);
            auto* aligned = reinterpret_cast<std::byte*>(alignedAddr);

            // Store original pointer just before aligned region
            auto* header = reinterpret_cast<void**>(aligned) - 1;
            *header = raw;

            return aligned;
        }

        void deallocate(std::byte* p, size_t n)
        {
            // Retrieve original pointer from header
            auto* header = reinterpret_cast<void**>(p) - 1;
            auto* raw = static_cast<AlignedByte<Alignment>*>(*header);

            size_t extra = Alignment - 1 + sizeof(void*);
            size_t totalUnits =
                (n + extra + sizeof(AlignedByte<Alignment>) - 1) / sizeof(AlignedByte<Alignment>);
            InnerAllocTraits::deallocate(inner_, raw, totalUnits);
        }

        // Propagation traits: forward from inner allocator
        using propagate_on_container_copy_assignment =
            typename InnerAllocTraits::propagate_on_container_copy_assignment;
        using propagate_on_container_move_assignment =
            typename InnerAllocTraits::propagate_on_container_move_assignment;
        using propagate_on_container_swap = typename InnerAllocTraits::propagate_on_container_swap;
        using is_always_equal = typename InnerAllocTraits::is_always_equal;

        AlignedAllocatorAdapter select_on_container_copy_construction() const
        {
            return AlignedAllocatorAdapter {
                InnerAllocTraits::select_on_container_copy_construction(inner_)};
        }

        bool operator==(AlignedAllocatorAdapter const& other) const = default;
    };

    /// Helper for computing co-located memory layout.
    /// Memory layout: [ctrl bytes][padding][slots...]
    template<typename T, size_t GroupSize, typename HashStoragePolicy = StoreHashTag>
    struct TableLayout
    {
        static constexpr size_t ctrlAlignment = GroupSize;
        static constexpr size_t slotAlignment = alignof(Slot<T, HashStoragePolicy>);

        /// Computes the offset from buffer start to the slots array.
        static constexpr size_t slotsOffset(size_t ctrlLen)
        {
            size_t ctrlSize = ctrlLen * sizeof(ctrl_t);
            // Round up to slot alignment
            return (ctrlSize + slotAlignment - 1) & ~(slotAlignment - 1);
        }

        /// Computes total buffer size for given control length and capacity.
        static constexpr size_t bufferSize(size_t ctrlLen, size_t capacity)
        {
            return slotsOffset(ctrlLen) + capacity * sizeof(Slot<T, HashStoragePolicy>);
        }

        /// Maximum alignment requirement for the buffer.
        static constexpr size_t bufferAlignment() { return std::max(ctrlAlignment, slotAlignment); }
    };

    /// A group of control bytes, the fundamental unit of Swiss Table probing.
    /// Uses SIMD instructions for fast parallel matching.
    template<SimdBackend Backend>
    struct Group
    {
        using Register = Backend::Register;
        using Mask = Backend::Mask;

        Register data;

        /// Constructs a Group by loading control bytes from memory.
        explicit Group(ctrl_t const* ctrl)
            : data(Backend::load(ctrl))
        {
        }

        /// Returns an iterator over all slots in the group of slots
        /// whose control hash matches.
        Backend::Iterable match(std::uint8_t h2) const
        {
            return Backend::iterate(Backend::match(data, h2));
        }

        /// Returns a mask of all slots in the group that contain a value.
        /// (Not empty, deleted, or sentinel)
        Mask matchFull() const { return Backend::matchFull(data); }

        Mask matchEmpty() const { return Backend::matchEmpty(data); }

        /// Returns true if and only if there exists a slot in the group that has Ctrl::Empty.
        bool anyEmpty() const { return Backend::any(matchEmpty()); }

        /// Returns true if we've reached the sentinel at the end of the control array.
        bool atEnd() const
        {
            return Backend::any(Backend::match(data, static_cast<ctrl_t>(Ctrl::Sentinel)));
        }

        /// Returns true if and only if there exists a slot that isn't empty, deleted, or sentinel.
        bool hasValue() const { return Backend::any(matchFull()); }
    };

    // Export hash storage policy tags
    export using StoreHashTag = StoreHashTag;
    export using NoStoreHashTag = NoStoreHashTag;
    export using DefaultHashStoragePolicy = DefaultHashStoragePolicy;

    /// Forward declaration of the Table base class.
    export template<typename T,
                    typename Hash,
                    typename Equal,
                    typename Policy,
                    SimdBackend Backend,
                    typename Allocator,
                    typename LoadFactorRatio,
                    typename HashStoragePolicy>
    class Table;

    /// Iterator for traversing elements in a Swiss Table.
    /// Uses the control byte array to efficiently skip empty/deleted slots.
    export template<typename T, SimdBackend Backend, typename HashStoragePolicy = StoreHashTag>
    struct SetIterator
    {
        static constexpr size_t LANE_COUNT = Backend::GroupSize;

        template<typename U, SimdBackend B, typename HSP>
        friend struct SetIterator;

        /// Pointer to the current control byte.
        ctrl_t const* ctrl;
        /// Pointer to the current slot.
        Slot<std::remove_const_t<T>, HashStoragePolicy>* slot;

        SetIterator(ctrl_t const* c, Slot<std::remove_const_t<T>, HashStoragePolicy>* s)
            : ctrl(c)
            , slot(s)
        {
        }

        // Converting constructor: allows iterator to convert to const_iterator
        SetIterator(SetIterator<std::remove_const_t<T>, Backend, HashStoragePolicy> const& other)
            requires std::is_const_v<T> && (!std::is_same_v<T, std::remove_const_t<T>>)
            : ctrl(other.ctrl)
            , slot(reinterpret_cast<Slot<std::remove_const_t<T>, HashStoragePolicy>*>(other.slot))
        {
        }

        SetIterator(SetIterator const&) = default;
        SetIterator& operator=(SetIterator const&) = default;

        // Update comparison to work across iterator/const_iterator
        template<typename U>
        friend bool operator==(SetIterator<T, Backend, HashStoragePolicy> const& lhs,
                               SetIterator<U, Backend, HashStoragePolicy> const& rhs)
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
            Group<Backend> g {groupPtr};
            auto mask = g.matchFull();

            auto idx = Backend::nextTrue(g.matchFull(), offset);
            if (idx)
            {
                int jump = *idx - offset;
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
                Group<Backend> g {ctrl};
                auto mask = g.matchFull();

                auto idx = Backend::firstTrue(mask);
                if (idx)
                {
                    ctrl += *idx;
                    slot += *idx;
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

        template<typename U,
                 typename Hash,
                 typename Equal,
                 typename Policy,
                 SimdBackend B,
                 typename Allocator,
                 typename LoadFactorRatio,
                 typename H>
        friend class Table;
    };

    /// Default load factor before rehashing is triggered.
    /// 7/8 = 0.875 provides a good balance between memory usage and probe length.
    using DEFAULT_LOAD_FACTOR = std::ratio<7, 8>;

    /// Concept: allocator that can be safely swapped without causing UB.
    /// Swap is safe if allocators propagate on swap or are always equal.
    template<typename Alloc>
    concept SafeSwappableAllocator =
        std::allocator_traits<Alloc>::propagate_on_container_swap::value
        || std::allocator_traits<Alloc>::is_always_equal::value;

    /// Base class for Swiss Table hash containers.
    /// Implements the core Swiss Table algorithm with SIMD-accelerated probing.
    /// Uses open addressing with linear probing and a control byte array.
    template<typename T,
             typename Hash,
             typename Equal,
             typename Policy,
             SimdBackend Backend,
             typename Allocator = std::allocator<std::byte>,
             typename LoadFactorRatio = DEFAULT_LOAD_FACTOR,
             typename HashStoragePolicy = DefaultHashStoragePolicy>
    class Table
    {
      protected:
        static constexpr size_t LANE_COUNT = Backend::GroupSize;
        static constexpr double loadFactor =
            static_cast<double>(LoadFactorRatio::num) / static_cast<double>(LoadFactorRatio::den);

        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator = SetIterator<T, Backend, HashStoragePolicy>;
        using const_iterator = SetIterator<T const, Backend, HashStoragePolicy>;

        using allocator_type = Allocator;
        using AllocTraits = std::allocator_traits<Allocator>;
        using AlignedByteType = AlignedByte<LANE_COUNT>;
        using InnerAlloc = typename AllocTraits::template rebind_alloc<AlignedByteType>;
        using ByteAlloc = AlignedAllocatorAdapter<InnerAlloc, LANE_COUNT>;
        using ByteAllocTraits = std::allocator_traits<ByteAlloc>;

        Table()
            : alloc_()
            , byte_alloc_(alloc_)
        {
        }

        explicit Table(Allocator const& alloc)
            : alloc_(alloc)
            , byte_alloc_(alloc_)
        {
        }

        explicit Table(size_type capacity, Allocator const& alloc = Allocator())
            : alloc_(alloc)
            , byte_alloc_(alloc_)
        {
            reserve(capacity);
        }

        Table(Table const& other)
            requires std::copy_constructible<T>
            : size_(other.size_)
            , used_(other.used_)
            , capacity_(other.capacity_)
            , ctrlLen_(other.ctrlLen_)
            , groups_(other.groups_)
            , alloc_(AllocTraits::select_on_container_copy_construction(other.alloc_))
            , byte_alloc_(alloc_)
        {
            if (other.buffer_ == nullptr)
            {
                return;
            }

            // Allocate co-located buffer
            buffer_ = allocateBuffer(ctrlLen_, capacity_);
            ctrl_ = reinterpret_cast<ctrl_t*>(buffer_);
            slots_ = reinterpret_cast<Slot<T, HashStoragePolicy>*>(buffer_
                                                                   + Layout::slotsOffset(ctrlLen_));

            std::memset(ctrl_, static_cast<ctrl_t>(Ctrl::Empty), ctrlLen_ * sizeof(ctrl_t));

            if (capacity_ > 0)
            {
                try
                {
                    for (auto it = other.begin(); it != other.end(); ++it)
                    {
                        size_t offset = it.ctrl - other.ctrl_;
                        AllocTraits::construct(alloc_, slots_[offset].element(), *it);

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
                            AllocTraits::destroy(alloc_, slots_[i].element());
                        }
                    }
                    deallocateBuffer(buffer_, ctrlLen_, capacity_);
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
            if (buffer_ != nullptr)
            {
                for (size_t i = 0; i < capacity_; ++i)
                {
                    if ((ctrl_[i] & 0x80) == 0)
                    {
                        AllocTraits::destroy(alloc_, slots_[i].element());
                    }
                }
                deallocateBuffer(buffer_, ctrlLen_, capacity_);
            }

            size_ = 0;
            used_ = 0;
            capacity_ = 0;
            ctrlLen_ = 0;
            buffer_ = nullptr;
            ctrl_ = nullptr;
            slots_ = nullptr;
        }

        void swap(Table& other) noexcept
            requires SafeSwappableAllocator<Allocator>
        {
            using std::swap;
            swap(size_, other.size_);
            swap(used_, other.used_);
            swap(capacity_, other.capacity_);
            swap(ctrlLen_, other.ctrlLen_);
            swap(groups_, other.groups_);
            if constexpr (AllocTraits::propagate_on_container_swap::value)
            {
                swap(alloc_, other.alloc_);
                swap(byte_alloc_, other.byte_alloc_);
            }
            swap(buffer_, other.buffer_);
            swap(ctrl_, other.ctrl_);
            swap(slots_, other.slots_);
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
            auto hash = Policy::apply(h(key));
            auto group = h1(hash) & (groups_ - 1);  // Since groups_ is a power of 2
            auto h2Val = h2(hash);

            while (true)
            {
                Group<Backend> g {ctrl_ + group * LANE_COUNT};
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
        std::pair<size_t, bool> emplace_internal(T& value, size_t hash)
        {
            if (capacity_ == 0)
            {
                reserve(1);
            }
            auto h1Val = h1(hash);
            auto h2Val = h2(hash);
            size_t mask = groups_ - 1;
            size_t group = h1Val & mask;

            while (true)
            {
                auto baseSlot = group * LANE_COUNT;
                Group<Backend> g {ctrl_ + baseSlot};

                typename Backend::Iterable candidates = g.match(h2Val);
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
                auto emptyIdx = Backend::firstTrue(empty);
                if (emptyIdx)
                {
                    if (size_ + 1 > capacity_ * loadFactor)
                    {
                        reserve(size_ + 1);
                        return emplace_internal(value, hash);
                    }

                    int offset = static_cast<int>(*emptyIdx);
                    size_t idx = baseSlot + offset;

                    ctrl_[idx] = h2Val;
                    setSlotHash(slots_[idx],
                                hash);  // Store full hash for fast rehashing if policy requires
                    AllocTraits::construct(alloc_, slots_[idx].element(), std::move(value));
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

            auto hash = Policy::apply(Hash {}(*temp));
            auto result = emplace_internal(*temp, hash);

            temp->~T();

            return result;
        }

        void reserve(size_type count)
        {
            auto desired = static_cast<size_t>(std::ceil(count / loadFactor));
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
            AllocTraits::destroy(alloc_, slots_[offset].element());
            --size_;

            auto addr = reinterpret_cast<uintptr_t>(ctrl_ + offset);
            auto alignedAddr = addr & ~static_cast<uintptr_t>(LANE_COUNT - 1);
            auto groupPtr = reinterpret_cast<ctrl_t*>(alignedAddr);

            Group<Backend> g {groupPtr};
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
        static constexpr size_t h1(size_t hash) { return hash >> 7; }
        /// Extracts the lower 7 bits for control byte matching.
        static constexpr ctrl_t h2(size_t hash) { return hash & 0x7F; }

        /// Get hash value (stored or recomputed)
        [[nodiscard]] size_t getSlotHash(Slot<T, HashStoragePolicy> const& slot) const
        {
            if constexpr (std::is_same_v<HashStoragePolicy, StoreHashTag>)
            {
                return slot.hash;
            }
            else
            {
                return Policy::apply(Hash {}(*slot.element()));
            }
        }

        /// Store hash if policy requires
        void setSlotHash(Slot<T, HashStoragePolicy>& slot, size_t hash)
        {
            if constexpr (std::is_same_v<HashStoragePolicy, StoreHashTag>)
            {
                slot.hash = hash;
            }
            // NoStoreHashTag: no-op, optimized away by compiler
        }

        iterator iteratorAt(size_t offset) { return {ctrl_ + offset, slots_ + offset}; }
        const_iterator iteratorAt(size_t offset) const
        {
            return iterator {(ctrl_ + offset),
                             const_cast<Slot<T, HashStoragePolicy>*>(slots_ + offset)};
        }

        using Layout = TableLayout<T, LANE_COUNT, HashStoragePolicy>;

        size_t size_ = 0;
        size_t used_ = 0;
        size_t capacity_ = 0;
        size_t ctrlLen_ = 0;
        size_t groups_ = 0;
        [[no_unique_address]] Allocator alloc_;
        [[no_unique_address]] ByteAlloc byte_alloc_;  // Rebound allocator for buffer
        std::byte* buffer_ = nullptr;  // Single co-located allocation
        ctrl_t* ctrl_ = nullptr;  // Points into buffer_
        Slot<T, HashStoragePolicy>* slots_ = nullptr;  // Points into buffer_ after ctrl

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

            // Allocate new co-located buffer
            auto* newBuffer = allocateBuffer(count, newCapacity);
            auto* newCtrl = reinterpret_cast<ctrl_t*>(newBuffer);
            auto* newSlots = reinterpret_cast<Slot<T, HashStoragePolicy>*>(
                newBuffer + Layout::slotsOffset(count));

            std::memset(newCtrl, static_cast<ctrl_t>(Ctrl::Empty), newCapacity);
            newCtrl[newCapacity] = static_cast<ctrl_t>(Ctrl::Sentinel);

            size_t mask = newGroupCount - 1;

            auto insertUnchecked = [&](Slot<T, HashStoragePolicy>& oldSlot)
            {
                // Get hash (stored or recomputed)
                size_t fullHash = getSlotHash(oldSlot);
                auto h1Val = h1(fullHash);
                auto h2Val = h2(fullHash);
                size_t group = h1Val & mask;

                while (true)
                {
                    Group<Backend> g {newCtrl + group * LANE_COUNT};
                    auto emptyMask = g.matchEmpty();

                    auto emptyIdx = Backend::firstTrue(emptyMask);
                    if (emptyIdx)
                    {
                        int offset = static_cast<int>(*emptyIdx);
                        size_t idx = group * LANE_COUNT + offset;

                        if constexpr (std::is_trivially_copyable_v<T>)
                        {
                            // Fast path: direct memcpy for trivially copyable types
                            std::memcpy(newSlots[idx].storage, oldSlot.storage, sizeof(T));
                        }
                        else
                        {
                            // Standard path: move construct and destroy
                            AllocTraits::construct(
                                alloc_, newSlots[idx].element(), std::move(*oldSlot.element()));
                            AllocTraits::destroy(alloc_, oldSlot.element());
                        }
                        setSlotHash(newSlots[idx], fullHash);  // Store hash if policy requires
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
                    Group<Backend> g {ctrl_ + gIdx * LANE_COUNT};
                    auto fullMask = g.matchFull();

                    for (int i : Backend::iterate(fullMask))
                    {
                        size_t oldIdx = gIdx * LANE_COUNT + i;
                        insertUnchecked(slots_[oldIdx]);
                    }
                }

                deallocateBuffer(buffer_, ctrlLen_, capacity_);
            }

            buffer_ = newBuffer;
            ctrl_ = newCtrl;
            slots_ = newSlots;
            ctrlLen_ = count;
            capacity_ = newCapacity;
            groups_ = newGroupCount;
        }

        /// Allocates a combined buffer for ctrl + slots using the allocator.
        std::byte* allocateBuffer(size_t ctrlLen, size_t capacity)
        {
            auto size = Layout::bufferSize(ctrlLen, capacity);
            return ByteAllocTraits::allocate(byte_alloc_, size);
        }

        /// Deallocates the combined buffer.
        void deallocateBuffer(std::byte* buffer, size_t ctrlLen, size_t capacity)
        {
            if (buffer)
            {
                auto size = Layout::bufferSize(ctrlLen, capacity);
                ByteAllocTraits::deallocate(byte_alloc_, buffer, size);
            }
        }
    };

    /// A hash set based on Swiss Tables.
    /// Uses SIMD-accelerated probing for efficient lookup, insertion, and deletion.
    export template<typename T,
                    typename Hash = std::hash<std::remove_cvref_t<T>>,
                    typename Equal = std::equal_to<T>,
                    typename Policy = HashPolicySelector<T, Hash>::type,
                    SimdBackend Backend = DefaultBackend,
                    typename Allocator = std::allocator<std::byte>,
                    typename LoadFactorRatio = DEFAULT_LOAD_FACTOR,
                    typename HashStoragePolicy = DefaultHashStoragePolicy>
        requires std::move_constructible<T>
    class Set
        : Table<T, Hash, Equal, Policy, Backend, Allocator, LoadFactorRatio, HashStoragePolicy>
    {
        using Base =
            Table<T, Hash, Equal, Policy, Backend, Allocator, LoadFactorRatio, HashStoragePolicy>;

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

        using iterator = SetIterator<T const, Backend, HashStoragePolicy>;
        using const_iterator = SetIterator<T const, Backend, HashStoragePolicy>;

        using Base::Base;
        using Base::clear;
        using Base::empty;
        using Base::reserve;
        using Base::size;
        using Base::swap;

        using allocator_type = typename Base::allocator_type;

        Set() = default;

        explicit Set(Allocator const& alloc)
            : Base(alloc)
        {
        }

        explicit Set(size_type capacity, Allocator const& alloc = Allocator())
            : Base(capacity, alloc)
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
