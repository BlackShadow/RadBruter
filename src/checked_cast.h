#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace lm {
// BSP fields and worker protocols use fixed-width integers, even on 64-bit hosts.
template <class To, class From> To checked_cast(From value) {
    static_assert(std::is_integral_v<To> && std::is_arithmetic_v<From>);
    bool fits;
    if constexpr (std::is_integral_v<From>) {
        fits = std::in_range<To>(value);
    } else {
        constexpr long double low = static_cast<long double>(std::numeric_limits<To>::lowest());
        constexpr long double high = static_cast<long double>(std::numeric_limits<To>::max()) + 1.0L;
        const long double wide = static_cast<long double>(value);
        fits = std::isfinite(wide) && wide >= low && wide < high;
    }
    if (!fits) throw std::overflow_error("Numeric value exceeds the supported integer range");
    return static_cast<To>(value);
}
} // namespace lm
