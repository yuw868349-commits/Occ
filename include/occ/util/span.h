#pragma once

// Minimal span. Used by the parsers, which take byte ranges and must never
// assume a null terminator exists.

#include <cstddef>
#include <cstdint>

namespace occ {

template <typename T>
class Span {
public:
    constexpr Span() noexcept = default;

    constexpr Span(T* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    [[nodiscard]] constexpr T* data() const noexcept { return data_; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
    [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] constexpr T& operator[](std::size_t i) const noexcept {
        return data_[i];
    }

    [[nodiscard]] constexpr T* begin() const noexcept { return data_; }
    [[nodiscard]] constexpr T* end() const noexcept { return data_ + size_; }

    [[nodiscard]] constexpr Span subspan(std::size_t offset,
                                         std::size_t count) const noexcept {
        if (offset > size_) {
            return Span{};
        }
        const std::size_t avail = size_ - offset;
        return Span{data_ + offset, count < avail ? count : avail};
    }

    [[nodiscard]] constexpr Span subspan(std::size_t offset) const noexcept {
        if (offset > size_) {
            return Span{};
        }
        return Span{data_ + offset, size_ - offset};
    }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

using ByteSpan = Span<const std::uint8_t>;

} // namespace occ
