module;

#include <expected>
#include <functional>
#include <tuple>

export module alp:map;

import :set;

namespace alp
{
    template<typename Key, typename Hash>
    struct MapHashAdapter
    {
        using is_transparent = void;
        [[no_unique_address]] Hash hasher;

        auto operator()(Key const& k) const noexcept { return hasher(k); }
        template<typename V>
        auto operator()(std::pair<Key const, V> const& p) const noexcept
        {
            return hasher(p.first);
        }

        template<typename T>
            requires requires { typename Hash::is_transparent; }
        auto operator()(T const& t) const noexcept
        {
            return hasher(t);
        }
    };

    template<typename Key, typename Equal>
    struct MapEqualAdapter
    {
        using is_transparent = void;
        [[no_unique_address]] Equal eq;

        template<typename V>
        bool operator()(std::pair<Key const, V> const& lhs,
                        std::pair<Key const, V> const& rhs) const
        {
            return eq(lhs.first, rhs.first);
        }

        template<typename V>
        bool operator()(std::pair<Key const, V> const& lhs, Key const& rhs) const
        {
            return eq(lhs.first, rhs);
        }

        template<typename V>
        bool operator()(Key const& lhs, std::pair<Key const, V> const& rhs) const
        {
            return eq(lhs, rhs.first);
        }

        template<typename A, typename B>
            requires requires { typename Equal::is_transparent; }
        bool operator()(A const& a, B const& b) const
        {
            return eq(a, b);
        }
    };

    export template<typename Key,
                    typename Value,
                    typename Hash = std::hash<Key>,
                    typename Equal = std::equal_to<Key>>
    class Map
        : Set<std::pair<Key const, Value>, MapHashAdapter<Key, Hash>, MapEqualAdapter<Key, Equal>>
    {
        using PairType = std::pair<Key const, Value>;
        using Base = Set<PairType, MapHashAdapter<Key, Hash>, MapEqualAdapter<Key, Equal>>;

      public:
        using key_type = Key;
        using mapped_type = Value;
        using value_type = PairType;
        using size_type = Base::size_type;
        using iterator = Base::iterator;
        using const_iterator = Base::const_iterator;

        using Base::Base;

        using Base::begin;
        using Base::cbegin;
        using Base::cend;
        using Base::clear;
        using Base::empty;
        using Base::end;
        using Base::reserve;
        using Base::size;
        using Base::swap;

        iterator find(Key const& key) { return Base::find(key); }
        const_iterator find(Key const& key) const { return Base::find(key); }

        bool contains(Key const& key) const { return Base::contains(key); }

        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            return Base::emplace(std::forward<Args>(args)...);
        }

        std::pair<iterator, bool> insert(value_type const& value) { return Base::emplace(value); }

        std::pair<iterator, bool> insert(value_type&& value)
        {
            return Base::emplace(std::move(value));
        }

        template<typename M>
        std::pair<iterator, bool> insert_or_assign(Key const& k, M&& obj)
        {
            auto it = find(k);
            if (it != end())
            {
                it->second = std::forward<M>(obj);
                return {it, false};
            }
            return emplace(std::piecewise_construct,
                           std::forward_as_tuple(k),
                           std::forward_as_tuple(std::forward<M>(obj)));
        }

        Value& operator[](Key const& key)
        {
            auto it = find(key);
            if (it != end())
                return it->second;

            auto [new_it, success] = Base::emplace(
                std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple());
            return new_it->second;
        }

        [[nodiscard]]
        std::expected<std::reference_wrapper<Value>, Error> get(Key const& key)
        {
            auto it = find(key);
            if (it != end())
            {
                return std::ref(it->second);
            }
            return std::unexpected(Error::NotFound);
        }

        using Base::erase;

        size_type erase(Key const& key)
        {
            auto it = find(key);
            if (it != end())
            {
                Base::erase(it);
                return 1;
            }
            return 0;
        }

        std::expected<void, Error> tryErase(Key const& key)
        {
            auto it = find(key);
            if (it != end())
            {
                Base::erase(it);
                return {};
            }
            return std::unexpected(Error::NotFound);
        }
    };
}  // namespace alp