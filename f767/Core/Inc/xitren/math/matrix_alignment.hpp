#pragma once

#include <xitren/math/gemm_core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

namespace xitren::math {

// The code snippet defines a C++ template class named `matrix_aligned` within the namespace
// `xitren::math`. This class is a matrix representation with aligned memory allocation for
// optimized matrix operations.
template <class Type, std::size_t Rows, std::size_t Columns, xitren::math::optimization Alg>
class matrix_aligned {
    using Core = xitren::math::gemm_core<Rows, Columns, Type, Alg>;

    static_assert(noexcept(Core::template mult<32>(nullptr, nullptr, nullptr)));
    static_assert(noexcept(Core::add(nullptr, nullptr, nullptr)));
    static_assert(noexcept(Core::sub(nullptr, nullptr, nullptr)));
    static_assert(noexcept(Core::transpose(nullptr, nullptr)));
    static_assert(noexcept(Core::trace(nullptr)));
    static_assert(noexcept(Core::min(nullptr)));
    static_assert(noexcept(Core::max(nullptr)));

public:

    template <std::size_t ColumnsOther>
    static void
    mult(matrix_aligned<Type, Rows, ColumnsOther, Alg> const&    a,
         matrix_aligned<Type, ColumnsOther, Columns, Alg> const& b,
         matrix_aligned<Type, Rows, Columns, Alg>&               c)
    {
        Core::template mult<ColumnsOther>(a.data_, b.data_, c.data_);
    }

    auto&
    get(std::size_t row, std::size_t column)
    {
        return data_[(row * Columns) + column];
    }

    void
    get_rand_matrix(double max_val, double min_val)
    {
        std::random_device               rd;
        std::mt19937                     gen(rd());
        std::uniform_real_distribution<> dis(min_val, max_val);
        for (std::uint32_t i = 0; i < Rows * Columns; ++i) {
            dd_[i] = dis(gen);
        }
    }

    void
    get_zeros_matrix()
    {
        for (std::uint32_t i = 0; i < Rows * Columns; ++i) {
            dd_[i] = 0;
        }
    }

private:
    std::array<Type, Rows * Columns> dd_{};

public:
    Type* data_=dd_.data();

};

}    // namespace xitren::math
