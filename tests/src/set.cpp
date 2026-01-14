#include <algorithm>
#include <cstdint>
#include <memory>
#include <ranges>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

import alp;

namespace
{
    int g_destruction_count = 0;
    struct DestructorCounter
    {
        int value;
        explicit DestructorCounter(int v)
            : value(v)
        {
        }
        DestructorCounter(DestructorCounter const& other)
            : value(other.value)
        {
        }
        DestructorCounter(DestructorCounter&& other) noexcept
            : value(other.value)
        {
        }
        DestructorCounter& operator=(DestructorCounter const&) = default;
        DestructorCounter& operator=(DestructorCounter&&) noexcept = default;
        ~DestructorCounter() { ++g_destruction_count; }
        bool operator==(DestructorCounter const& other) const { return value == other.value; }
    };

    // Identity hash for predictable collision testing
    struct IdentityHash
    {
        size_t operator()(int x) const noexcept { return static_cast<size_t>(x); }
    };

    // Type that throws on copy (for exception safety tests)
    struct ThrowsOnCopy
    {
        int value;
        static int copy_count;
        static int throw_after;
        explicit ThrowsOnCopy(int v)
            : value(v)
        {
        }
        ThrowsOnCopy(ThrowsOnCopy&& other) noexcept
            : value(other.value)
        {
        }
        ThrowsOnCopy(ThrowsOnCopy const& other)
            : value(other.value)
        {
            if (++copy_count >= throw_after)
            {
                throw std::runtime_error("Copy failed");
            }
        }
        ThrowsOnCopy& operator=(ThrowsOnCopy const&) = default;
        ThrowsOnCopy& operator=(ThrowsOnCopy&&) noexcept = default;
        ~ThrowsOnCopy() = default;
        bool operator==(ThrowsOnCopy const& other) const { return value == other.value; }
    };
    int ThrowsOnCopy::copy_count = 0;
    int ThrowsOnCopy::throw_after = 100;
}  // namespace

template<>
struct std::hash<DestructorCounter>
{
    size_t operator()(DestructorCounter const& dc) const noexcept
    {
        return std::hash<int> {}(dc.value);
    }
};
template<>
struct std::hash<ThrowsOnCopy>
{
    size_t operator()(ThrowsOnCopy const& t) const noexcept { return std::hash<int> {}(t.value); }
};

TEST(SetCore, BasicOperationsInt)
{
    alp::Set<int> s;
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.size(), 0);
    auto [it1, inserted1] = s.emplace(42);
    EXPECT_TRUE(inserted1);
    EXPECT_EQ(s.size(), 1);
    EXPECT_TRUE(s.contains(42));
    auto [it2, inserted2] = s.emplace(42);
    EXPECT_FALSE(inserted2);
    EXPECT_EQ(s.size(), 1);
    s.emplace(10);
    s.emplace(20);
    EXPECT_EQ(s.size(), 3);
    EXPECT_TRUE(s.contains(10));
    EXPECT_TRUE(s.contains(20));
    EXPECT_FALSE(s.contains(999));
}
TEST(SetCore, BasicOperationsString)
{
    alp::Set<std::string> s;
    s.emplace("hello");
    s.emplace("world");
    s.emplace("test");
    EXPECT_EQ(s.size(), 3);
    EXPECT_TRUE(s.contains(std::string("hello")));
    EXPECT_TRUE(s.contains(std::string("world")));
    EXPECT_TRUE(s.contains(std::string("test")));
    EXPECT_FALSE(s.contains(std::string("missing")));
}

TEST(SetCore, EraseByIterator)
{
    alp::Set<int> s;
    s.emplace(1);
    s.emplace(2);
    s.emplace(3);
    EXPECT_EQ(s.size(), 3);
    auto it = s.find(2);
    EXPECT_NE(it, s.end());
    s.erase(it);
    EXPECT_EQ(s.size(), 2);
    EXPECT_FALSE(s.contains(2));
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(3));
}

TEST(SetCore, EraseByKey)
{
    alp::Set<int> s;
    s.emplace(10);
    s.emplace(20);
    s.emplace(30);
    size_t erased = s.erase(20);
    EXPECT_EQ(erased, 1);
    EXPECT_EQ(s.size(), 2);
    EXPECT_FALSE(s.contains(20));
    erased = s.erase(999);
    EXPECT_EQ(erased, 0);
    EXPECT_EQ(s.size(), 2);
}
TEST(SetCore, TryEraseSuccess)
{
    alp::Set<int> s;
    s.emplace(42);
    auto result = s.tryErase(42);
    EXPECT_TRUE(result.has_value());
    EXPECT_FALSE(s.contains(42));
}
TEST(SetCore, TryEraseFail)
{
    alp::Set<int> s;
    s.emplace(42);
    auto result = s.tryErase(999);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), alp::Error::NotFound);
    EXPECT_TRUE(s.contains(42));
}
TEST(SetCore, ClearAndEmpty)
{
    alp::Set<int> s;
    for (int i = 0; i < 100; ++i)
    {
        s.emplace(i);
    }
    EXPECT_EQ(s.size(), 100);
    EXPECT_FALSE(s.empty());
    s.clear();
    EXPECT_TRUE(s.empty());
    EXPECT_EQ(s.size(), 0);
}
TEST(SetCore, DestructorCounting)
{
    g_destruction_count = 0;
    {
        alp::Set<DestructorCounter> s;
        s.emplace(1);
        s.emplace(2);
        s.emplace(3);
        // Destructor not called yet for elements in set
    }
    // After set goes out of scope, destructors should be called
    // Note: emplace creates a temp object that gets destroyed, so count may be higher
    EXPECT_GE(g_destruction_count, 3);
}
TEST(SetCore, ClearCallsDestructors)
{
    g_destruction_count = 0;
    alp::Set<DestructorCounter> s;
    s.emplace(1);
    s.emplace(2);
    s.emplace(3);
    int count_before_clear = g_destruction_count;
    s.clear();
    int count_after_clear = g_destruction_count;
    // Clear should destroy exactly the 3 elements
    EXPECT_EQ(count_after_clear - count_before_clear, 3);
}

TEST(SetGroup, ExactlyOneGroup)
{
    // 16 elements should fit in exactly one group
    alp::Set<int> s;
    for (int i = 0; i < 16; ++i)
    {
        s.emplace(i);
    }
    EXPECT_EQ(s.size(), 16);
    // Verify all elements are present
    for (int i = 0; i < 16; ++i)
    {
        EXPECT_TRUE(s.contains(i)) << "Missing element: " << i;
    }
    // Verify iteration finds all elements
    std::vector<int> found;
    for (auto it = s.begin(); it != s.end(); ++it)
    {
        found.push_back(*it);
    }
    EXPECT_EQ(found.size(), 16);
}
TEST(SetGroup, GroupBoundaryCross)
{
    // 17 elements forces crossing into a second group
    alp::Set<int> s;
    for (int i = 0; i < 17; ++i)
    {
        s.emplace(i);
    }
    EXPECT_EQ(s.size(), 17);
    // Verify all elements are present
    for (int i = 0; i < 17; ++i)
    {
        EXPECT_TRUE(s.contains(i)) << "Missing element: " << i;
    }
    // Verify iteration correctly crosses group boundary
    std::vector<int> found;
    for (auto it = s.begin(); it != s.end(); ++it)
    {
        found.push_back(*it);
    }
    EXPECT_EQ(found.size(), 17);
}
TEST(SetGroup, SentinelIteration)
{
    // Fill a set close to capacity and iterate
    alp::Set<int> s(32);  // Reserve 32 slots
    // Fill with elements (staying under load factor)
    for (int i = 0; i < 25; ++i)
    {
        s.emplace(i);
    }
    // Iterate and verify no out-of-bounds read
    int count = 0;
    for (auto it = s.begin(); it != s.end(); ++it)
    {
        ++count;
    }
    EXPECT_EQ(count, 25);
}

TEST(SetCollision, ForcedCollisionSameGroup)
{
    // Use IdentityHashPolicy with IdentityHash to force collisions
    using CollisionSet = alp::Set<int, IdentityHash, std::equal_to<int>, alp::IdentityHashPolicy>;
    CollisionSet s;
    // Keys 0, 128, 256 should all hash to group 0 with identity policy
    // (h1 = hash >> 7, so 0>>7=0, 128>>7=1, but we need to find values that collide)
    // Actually, values that differ by 128 will hash to consecutive groups
    // Let's use values within the same group: 0, 1, 2, etc.
    // Insert multiple elements that will hash to overlapping locations
    s.emplace(0);
    s.emplace(1);
    s.emplace(2);
    s.emplace(128);  // h1=1 with identity
    s.emplace(256);  // h1=2 with identity
    // Verify all elements are findable
    EXPECT_TRUE(s.contains(0));
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(2));
    EXPECT_TRUE(s.contains(128));
    EXPECT_TRUE(s.contains(256));
}
TEST(SetCollision, TombstoneReuse)
{
    // Test that deleted slots don't break probing
    using CollisionSet = alp::Set<int, IdentityHash, std::equal_to<int>, alp::IdentityHashPolicy>;
    CollisionSet s;
    // Insert A and B which may collide
    s.emplace(0);  // A
    s.emplace(128);  // B - may probe to different slot
    EXPECT_TRUE(s.contains(0));
    EXPECT_TRUE(s.contains(128));
    // Erase A (creates tombstone)
    s.erase(0);
    EXPECT_FALSE(s.contains(0));
    // B should still be findable (probing must not stop at tombstone)
    EXPECT_TRUE(s.contains(128));
    // Insert C which may reuse A's slot or probe past it
    s.emplace(256);
    EXPECT_TRUE(s.contains(256));
    // All remaining elements should be findable
    EXPECT_TRUE(s.contains(128));
    EXPECT_TRUE(s.contains(256));
}
TEST(SetCollision, MultipleDeletesAndInserts)
{
    alp::Set<int> s;
    // Insert 20 elements
    for (int i = 0; i < 20; ++i)
    {
        s.emplace(i);
    }
    // Delete every other element
    for (int i = 0; i < 20; i += 2)
    {
        s.erase(i);
    }
    // Insert new elements
    for (int i = 100; i < 110; ++i)
    {
        s.emplace(i);
    }
    // Verify odd originals and new elements are present
    for (int i = 1; i < 20; i += 2)
    {
        EXPECT_TRUE(s.contains(i)) << "Missing odd: " << i;
    }
    for (int i = 100; i < 110; ++i)
    {
        EXPECT_TRUE(s.contains(i)) << "Missing new: " << i;
    }
    // Verify deleted elements are gone
    for (int i = 0; i < 20; i += 2)
    {
        EXPECT_FALSE(s.contains(i)) << "Should be deleted: " << i;
    }
}

TEST(SetIterator, SparseIteration)
{
    alp::Set<int> s;
    // Insert 0-19
    for (int i = 0; i < 20; ++i)
    {
        s.emplace(i);
    }
    // Delete all even elements
    for (int i = 0; i < 20; i += 2)
    {
        s.erase(i);
    }
    // Iterate and collect
    std::vector<int> found;
    for (auto it = s.begin(); it != s.end(); ++it)
    {
        found.push_back(*it);
    }
    // Should have exactly 10 odd elements
    EXPECT_EQ(found.size(), 10);
    std::sort(found.begin(), found.end());
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(found[i], 2 * i + 1);  // 1, 3, 5, 7, ...
    }
}
TEST(SetIterator, EmptySetIteration)
{
    alp::Set<int> s;
    EXPECT_EQ(s.begin(), s.end());
    // After clear
    s.emplace(1);
    s.clear();
    EXPECT_EQ(s.begin(), s.end());
}
TEST(SetIterator, RangesCompatibility)
{
    alp::Set<int> s;
    for (int i = 0; i < 10; ++i)
    {
        s.emplace(i);
    }
    // Use range-based for loop
    int count = 0;
    for (auto& slot : s)
    {
        (void)slot;
        ++count;
    }
    EXPECT_EQ(count, 10);
}
TEST(SetIterator, ConstIteration)
{
    alp::Set<int> s;
    for (int i = 0; i < 5; ++i)
    {
        s.emplace(i);
    }
    alp::Set<int> const& cs = s;
    int count = 0;
    for (auto it = cs.begin(); it != cs.end(); ++it)
    {
        ++count;
    }
    EXPECT_EQ(count, 5);
}

TEST(SetRehash, LoadFactorRehash)
{
    alp::Set<int> s;
    // Start small, then force growth
    // Default construction + first insert will allocate minimal capacity
    for (int i = 0; i < 100; ++i)
    {
        s.emplace(i);
    }
    // All elements should still be present after multiple rehashes
    EXPECT_EQ(s.size(), 100);
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_TRUE(s.contains(i)) << "Missing after rehash: " << i;
    }
}
TEST(SetRehash, AllElementsSurviveRehash)
{
    alp::Set<int> s(16);  // Small initial capacity
    std::vector<int> inserted;
    for (int i = 0; i < 50; ++i)
    {
        s.emplace(i);
        inserted.push_back(i);
    }
    // Verify
    for (int v : inserted)
    {
        EXPECT_TRUE(s.contains(v)) << "Lost element: " << v;
    }
}
TEST(SetRehash, PowerOfTwoCapacity)
{
    // Constructing with 100 should snap to power-of-2 group count
    alp::Set<int> s(100);
    // Insert elements to verify it works
    for (int i = 0; i < 100; ++i)
    {
        s.emplace(i);
    }
    EXPECT_EQ(s.size(), 100);
    // Verify all present
    for (int i = 0; i < 100; ++i)
    {
        EXPECT_TRUE(s.contains(i));
    }
}
TEST(SetRehash, ReserveDoesNotShrink)
{
    alp::Set<int> s;
    // Reserve large
    s.reserve(1000);
    // Insert a few elements
    s.emplace(1);
    s.emplace(2);
    // Reserve smaller should not shrink
    s.reserve(10);
    EXPECT_TRUE(s.contains(1));
    EXPECT_TRUE(s.contains(2));
}

TEST(SetTypes, MoveOnlyType)
{
    alp::Set<std::unique_ptr<int>> s;
    s.emplace(std::make_unique<int>(42));
    s.emplace(std::make_unique<int>(100));
    EXPECT_EQ(s.size(), 2);
    // Iterate and verify values
    std::vector<int> values;
    for (auto& slot : s)
    {
        values.push_back(*slot);
    }
    std::sort(values.begin(), values.end());
    EXPECT_EQ(values.size(), 2);
    EXPECT_EQ(values[0], 42);
    EXPECT_EQ(values[1], 100);
}
TEST(SetTypes, CopyConstruction)
{
    alp::Set<int> s1;
    for (int i = 0; i < 20; ++i)
    {
        s1.emplace(i);
    }
    alp::Set<int> s2(s1);
    EXPECT_EQ(s2.size(), s1.size());
    for (int i = 0; i < 20; ++i)
    {
        EXPECT_TRUE(s2.contains(i));
    }
}
TEST(SetTypes, MoveConstruction)
{
    alp::Set<int> s1;
    for (int i = 0; i < 20; ++i)
    {
        s1.emplace(i);
    }
    alp::Set<int> s2(std::move(s1));
    EXPECT_EQ(s2.size(), 20);
    for (int i = 0; i < 20; ++i)
    {
        EXPECT_TRUE(s2.contains(i));
    }
}
TEST(SetTypes, CopyAssignment)
{
    alp::Set<int> s1;
    for (int i = 0; i < 10; ++i)
    {
        s1.emplace(i);
    }
    alp::Set<int> s2;
    s2.emplace(999);
    s2 = s1;
    EXPECT_EQ(s2.size(), 10);
    EXPECT_FALSE(s2.contains(999));
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(s2.contains(i));
    }
}
TEST(SetTypes, MoveAssignment)
{
    alp::Set<int> s1;
    for (int i = 0; i < 10; ++i)
    {
        s1.emplace(i);
    }
    alp::Set<int> s2;
    s2.emplace(999);
    s2 = std::move(s1);
    EXPECT_EQ(s2.size(), 10);
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_TRUE(s2.contains(i));
    }
}
TEST(SetTypes, Swap)
{
    alp::Set<int> s1;
    alp::Set<int> s2;
    s1.emplace(1);
    s1.emplace(2);
    s2.emplace(10);
    s2.emplace(20);
    s2.emplace(30);
    swap(s1, s2);
    EXPECT_EQ(s1.size(), 3);
    EXPECT_EQ(s2.size(), 2);
    EXPECT_TRUE(s1.contains(10));
    EXPECT_TRUE(s2.contains(1));
}
TEST(SetTypes, ExceptionDuringCopy)
{
    // This test verifies copy constructor exception safety
    ThrowsOnCopy::copy_count = 0;
    ThrowsOnCopy::throw_after = 3;  // Throw on 3rd copy
    alp::Set<ThrowsOnCopy> s1;
    s1.emplace(1);
    s1.emplace(2);
    s1.emplace(3);
    s1.emplace(4);
    s1.emplace(5);
    ThrowsOnCopy::copy_count = 0;  // Reset before copy
    EXPECT_THROW(
        {
            alp::Set<ThrowsOnCopy> s2(s1);
            (void)s2;
        },
        std::runtime_error);
    // Original set should still be valid
    EXPECT_EQ(s1.size(), 5);
    // Reset for cleanup
    ThrowsOnCopy::throw_after = 100;
}

TEST(SetEdge, GetSuccess)
{
    alp::Set<int> s;
    s.emplace(42);
    auto result = s.get(42);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value().get(), 42);
}
TEST(SetEdge, GetFail)
{
    alp::Set<int> s;
    s.emplace(42);
    auto result = s.get(999);
    EXPECT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), alp::Error::NotFound);
}
TEST(SetEdge, LargeScale)
{
    alp::Set<int> s;
    // Insert 10000 elements
    for (int i = 0; i < 10000; ++i)
    {
        s.emplace(i);
    }
    EXPECT_EQ(s.size(), 10000);
    // Verify random access
    for (int i = 0; i < 10000; i += 100)
    {
        EXPECT_TRUE(s.contains(i));
    }
    // Delete half
    for (int i = 0; i < 10000; i += 2)
    {
        s.erase(i);
    }
    EXPECT_EQ(s.size(), 5000);
    // Verify remaining
    for (int i = 1; i < 10000; i += 2)
    {
        EXPECT_TRUE(s.contains(i));
    }
}

TEST(SetEdge, SingleElement)
{
    alp::Set<int> s;
    s.emplace(42);
    EXPECT_EQ(s.size(), 1);
    EXPECT_TRUE(s.contains(42));
    auto it = s.begin();
    EXPECT_NE(it, s.end());
    EXPECT_EQ(*it, 42);
    ++it;
    EXPECT_EQ(it, s.end());
}