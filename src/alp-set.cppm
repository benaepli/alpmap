module;

#include <cstdint>
#include <expected>
#include <functional>

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

    template<std::size_t N>
    constexpr std::array<ctrl_t, N> toCtrlArray(Ctrl const (&arr)[N])
    {
        std::array<ctrl_t, N> result {};
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = static_cast<ctrl_t>(arr[i]);
        }
        return result;
    }

    alignas(16) static constexpr auto EMPTY_GROUP = toCtrlArray({Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Empty,
                                                                 Ctrl::Sentinel});

    template<typename T>
    struct Slot
    {
        alignas(T) uint8_t storage[sizeof(T)];

        T* element() { return reinterpret_cast<T*>(storage); }
        T const* element() const { return reinterpret_cast<T const*>(storage); }
    };

    export template<typename T>
    struct SetIterator
    {
        ctrl_t* ctrl;
        Slot<T>* slot;

        bool operator==(SetIterator const& other) const { return ctrl == other.ctrl; }

        SetIterator operator++();
    };

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
        explicit Set(size_type capacity);

        Set(Set const&);
        Set(Set&&) noexcept;
        Set& operator=(Set const&);
        Set& operator=(Set&&) noexcept;

        ~Set();

        iterator begin();
        iterator end();
        const_iterator begin() const;
        const_iterator end() const;
        const_iterator cbegin() const { return begin(); }
        const_iterator cend() const { return end(); }

        [[nodiscard]] bool empty() const;
        [[nodiscard]] size_t size() const;

        iterator find(T const& key);
        const_iterator find(T const& key) const;

        template<typename K>
            requires requires { typename Hash::is_transparent; }
        [[nodiscard]] auto find(K const& key) -> iterator;

        void clear() noexcept;
        bool contains(T const& key) const;

        template<typename K>
            requires requires { typename Hash::is_transparent; }
        bool contains(K const& key) const;

        void reserve(size_type count);

        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args);

        iterator erase(const_iterator pos);
        size_type erase(T const& key);

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
        bool isFull() const;
    };
}  // namespace alp