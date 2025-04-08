#pragma once
#include <xitren/math/gemm_core.hpp>

namespace xitren::math {

// This code snippet is defining a specialization of the `gemm_core` class template for the
// optimization strategy `optimization::avx256`.
template <std::uint_fast32_t Rows, std::uint_fast32_t Columns>
class gemm_core<Rows, Columns, std::int8_t, optimization::mve>
    : gemm_core<Rows, Columns, std::int8_t, optimization::naive> {
    static constexpr std::uint_fast32_t vectorization = 4;
    static_assert(Columns >= vectorization, "Should be greater or equal to blocksize!");
    static_assert(!(Columns % vectorization), "Should be dividable to blocksize!");

public:
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::add;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::sub;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::transpose;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::trace;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::min;
    using gemm_core<Rows, Columns, std::int8_t, optimization::naive>::max;

    template <std::uint_fast32_t Other>
    static void
    mult(std::int8_t const* a, std::int8_t const* b, std::int8_t* c) noexcept
    {
        for (std::uint_fast32_t i = 0; i < Rows; i++) {
            for (std::uint_fast32_t j = 0; j < Columns; j += vectorization) {
                auto const current = i * Columns + j;
                std::uint32_t    c0      = *((std::uint32_t*)(c + current)); /* c0 = C[i][j] */
                for (std::uint_fast32_t k = 0; k < Other; k++) {
                    auto const var_a = *((std::uint32_t*)(a + i * Other + k)); //broadcast
                    auto const var_b = *((std::uint32_t*)(b + k * Columns + j));

                    auto const var_a0 = __PKHTB(var_a, var_a<<16, 0) & 0x0F0F;
                    auto const var_b0 = __PKHTB(var_b, var_b<<8, 0) & 0x0F0F;
                    auto const var_b1 = __PKHTB(var_b>>16, var_b>>8, 0) & 0x0F0F;

                    auto const result
                        = __SMLAD(var_a0, var_b0, c0);
                    c0 = __SMLAD(var_a0, var_b1, result); /* c0 += A[i][k]*B[k][j] */
                }
                (*((std::uint32_t*)(c + current))) = c0;  /* C[i][j] = c0 */
            }
        }
    }
};

}    // namespace xitren::math
