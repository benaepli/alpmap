#include <cstdint>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

import alp;

namespace
{
    TEST(AlpmapErrorTest, EnumValues)
    {
        EXPECT_EQ(static_cast<std::uint8_t>(alp::Error::NotFound), 0);
    }
}  // namespace
