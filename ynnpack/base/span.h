// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_BASE_SPAN_H_
#define XNNPACK_YNNPACK_BASE_SPAN_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <vector>

namespace ynn {

constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

template <typename T, std::size_t Extent = dynamic_extent>
class span {
public:
    using value_type = std::remove_const_t<T>;

    span(const value_type* data, std::size_t size) : data_(data) { assert(size == Extent); }
    span(const value_type* begin, const value_type* end) : data_(begin) { assert(end - begin == Extent); }
    template <std::size_t N>
    span(const value_type (&x)[N]) : data_(&x[0]) {
        static_assert(N == Extent, "");
    }
    span(const std::array<value_type, Extent>& x) : data_(x.data()) {}
    span(const std::vector<value_type>& c) : data_(c.data()) { assert(c.size() == Extent); }

    span(const span&) = default;
    span(span&&) = default;
    span& operator=(const span&) = default;
    span& operator=(span&&) = default;

    const value_type* data() const { return data_; }
    static constexpr std::size_t size() { return Extent; }
    static constexpr bool empty() { return Extent == 0; }
    const value_type* begin() const { return data_; }
    const value_type* end() const { return data_ + size(); }

    const value_type& operator[](std::size_t i) const { return data_[i]; }
    const value_type& front() const { return data_[0]; }
    const value_type& back() const { return data_[size() - 1]; }

    span<T, dynamic_extent> subspan(std::size_t offset) const {
        return span<T, dynamic_extent>(data_ + offset, size() - offset);
    }
    span<T, dynamic_extent> subspan(std::size_t offset, std::size_t count) const {
        return span<T, dynamic_extent>(data_ + offset, count);
    }

private:
    const value_type* data_;
};

template <typename T>
class span<T, dynamic_extent> {
public:
    using value_type = std::remove_const_t<T>;

    span() : data_(nullptr), size_(0) {}
    span(const value_type* data, std::size_t size) : data_(data), size_(size) {}
    span(const value_type* begin, const value_type* end) : data_(begin), size_(end - begin) {}
    template <std::size_t N>
    span(const value_type (&x)[N]) : data_(&x[0]), size_(N) {}
    span(const value_type (&x)[0]) : data_(nullptr), size_(0) {}
    template <std::size_t N>
    span(const std::array<value_type, N>& x) : data_(x.data()), size_(N) {}
    span(const std::vector<value_type>& c) : data_(c.data()), size_(c.size()) {}

    span(const span&) = default;
    span(span&&) = default;
    span& operator=(const span&) = default;
    span& operator=(span&&) = default;

    const value_type* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    const value_type* begin() const { return data_; }
    const value_type* end() const { return data_ + size(); }
    std::reverse_iterator<const value_type*> rbegin() const {
        return std::reverse_iterator<const value_type*>(end());
    }
    std::reverse_iterator<const value_type*> rend() const {
        return std::reverse_iterator<const value_type*>(begin());
    }

    const value_type& operator[](std::size_t i) const { return data_[i]; }
    const value_type& front() const { return data_[0]; }
    const value_type& back() const { return data_[size() - 1]; }

    span subspan(std::size_t offset) const { return span(data_ + offset, size() - offset); }
    span subspan(std::size_t offset, std::size_t count) const { return span(data_ + offset, count); }

private:
    const value_type* data_;
    std::size_t size_;
};

template <typename T, typename U>
bool operator==(span<T> a, span<U> b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}
template <typename T, typename U>
bool operator!=(span<T> a, span<U> b) {
    return !(a == b);
}

template <typename T, std::size_t Extent = dynamic_extent>
class mutable_span {
public:
    using value_type = std::remove_const_t<T>;

    mutable_span(value_type* data, std::size_t size) : data_(data) { assert(size == Extent); }
    mutable_span(value_type* begin, value_type* end) : data_(begin) { assert(end - begin == Extent); }
    template <std::size_t N>
    mutable_span(value_type (&x)[N]) : data_(&x[0]) {
        static_assert(N == Extent, "");
    }
    mutable_span(std::array<value_type, Extent>& x) : data_(x.data()) {}
    mutable_span(std::vector<value_type>& c) : data_(c.data()) { assert(c.size() == Extent); }

    mutable_span(const mutable_span&) = default;
    mutable_span(mutable_span&&) = default;
    mutable_span& operator=(const mutable_span&) = default;
    mutable_span& operator=(mutable_span&&) = default;

    value_type* data() const { return data_; }
    static constexpr std::size_t size() { return Extent; }
    static constexpr bool empty() { return Extent == 0; }
    value_type* begin() const { return data_; }
    value_type* end() const { return data_ + size(); }

    value_type& operator[](std::size_t i) const { return data_[i]; }
    value_type& front() const { return data_[0]; }
    value_type& back() const { return data_[size() - 1]; }

    mutable_span<T, dynamic_extent> subspan(std::size_t offset) const {
        return mutable_span<T, dynamic_extent>(data_ + offset, size() - offset);
    }
    mutable_span<T, dynamic_extent> subspan(std::size_t offset, std::size_t count) const {
        return mutable_span<T, dynamic_extent>(data_ + offset, count);
    }

private:
    value_type* data_;
};

template <typename T>
class mutable_span<T, dynamic_extent> {
public:
    using value_type = std::remove_const_t<T>;

    mutable_span() : data_(nullptr), size_(0) {}
    mutable_span(value_type* data, std::size_t size) : data_(data), size_(size) {}
    mutable_span(value_type* begin, value_type* end) : data_(begin), size_(end - begin) {}
    template <std::size_t N>
    mutable_span(value_type (&x)[N]) : data_(&x[0]), size_(N) {}
    mutable_span(value_type (&x)[0]) : data_(nullptr), size_(0) {}
    template <std::size_t N>
    mutable_span(std::array<value_type, N>& x) : data_(x.data()), size_(N) {}
    mutable_span(std::vector<value_type>& c) : data_(c.data()), size_(c.size()) {}

    mutable_span(const mutable_span&) = default;
    mutable_span(mutable_span&&) = default;
    mutable_span& operator=(const mutable_span&) = default;
    mutable_span& operator=(mutable_span&&) = default;

    value_type* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size() == 0; }
    value_type* begin() const { return data_; }
    value_type* end() const { return data_ + size(); }

    value_type& operator[](std::size_t i) const { return data_[i]; }
    value_type& front() const { return data_[0]; }
    value_type& back() const { return data_[size() - 1]; }

    mutable_span subspan(std::size_t offset) const { return mutable_span(data_ + offset, size() - offset); }
    mutable_span subspan(std::size_t offset, std::size_t count) const { return mutable_span(data_ + offset, count); }

private:
    value_type* data_;
    std::size_t size_;
};

template <typename T>
std::vector<T> permute(span<int> p, const std::vector<T>& x, T default_value = T()) {
    std::vector<T> result(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        result[i] = static_cast<std::size_t>(p[i]) < x.size() ? x[p[i]] : default_value;
    }
    return result;
}

template <typename T>
std::vector<typename span<T>::value_type> to_vector(span<T> x) {
    return {x.begin(), x.end()};
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_BASE_SPAN_H_
