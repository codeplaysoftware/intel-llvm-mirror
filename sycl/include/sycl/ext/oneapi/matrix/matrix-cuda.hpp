#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental::matrix {

    enum class matrix_type
    {
        a,
        b,
        accumulator
    };

    // packed layout not yet implemented (requires changes to LIBCLC).
    enum class matrix_layout
    {
        row_major,
        col_major,
        packed
    };
    template <typename Group, matrix_type MT, matrix_layout Layout, size_t Rows = sycl::dynamic_extent,
              size_t Cols = sycl::dynamic_extent, typename Cond = void>
    struct joint_matrix
    {
        joint_matrix(Group g) {}
    };

    template <matrix_layout Layout>
    struct joint_matrix<sub_group, matrix_type::a, Layout, 8, 4, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major || Layout == experimental::matrix::matrix_layout::col_major>>
    {
        double data[1];
    };

    template <matrix_layout Layout>
    struct joint_matrix<sub_group, matrix_type::b, Layout, 4, 8, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major || Layout == experimental::matrix::matrix_layout::col_major>>
    {
        double data[1];
    };

    template <matrix_layout Layout>
    struct joint_matrix<sub_group, matrix_type::accumulator, Layout, 8, 8, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major || Layout == experimental::matrix::matrix_layout::col_major>>
    {
        double data[2];
    };

} // namespace experimental::matrix

namespace detail {
    using namespace experimental;

    template <typename Group, matrix::matrix_type MT, size_t NumRows,
              size_t NumCols, matrix::matrix_layout Layout,
              access::address_space Space, typename Cond = void>
    struct joint_matrix_load_impl
    {
        void load(matrix::joint_matrix<Group, MT, Layout, NumRows, NumCols> &res,
                  multi_ptr<double, Space> src, size_t stride);
    };

    // Helper
    template <matrix::matrix_layout Layout>
    constexpr int get_layout_int();

    template <>
    constexpr int get_layout_int<matrix::matrix_layout::row_major>()
    {
        return 0;
    }

    template <>
    constexpr int get_layout_int<matrix::matrix_layout::col_major>()
    {
        return 1;
    }

    // Helper alternative is something like this
    /*template <matrix::matrix_layout Layout, typename Cond = void>
    struct get_layout_int
    {
       int get();
    };

    template <matrix::matrix_layout Layout>
    struct get_layout_int<matrix::matrix_layout Layout, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major>>
    {
        int get(){
            return 0;
        }
    };

    template <matrix::matrix_layout Layout>
    struct get_layout_int<matrix::matrix_layout Layout, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::col_major>>
    {
        int get(){
            return 1;
        }
    };*/

    template <matrix::matrix_layout Layout, access::address_space Space>
    struct joint_matrix_load_impl<sub_group, matrix::matrix_type::a, 8, 4, Layout,
                                  Space, typename std::enable_if_t<Layout == matrix::matrix_layout::row_major || Layout == matrix::matrix_layout::col_major>>
    {
        void load(matrix::joint_matrix<sub_group, matrix::matrix_type::a, Layout, 8, 4> &res,
                  multi_ptr<double, Space> src, size_t stride)
        {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
                 __dmma_m8n8k4_ld_a(res.data, src.get(), stride, get_layout_int<Layout>());
#endif
#endif
        }
    };

    template <matrix::matrix_layout Layout, access::address_space Space>
    struct joint_matrix_load_impl<sub_group, matrix::matrix_type::b, 4, 8, Layout,
                                  Space, typename std::enable_if_t<Layout == matrix::matrix_layout::row_major || Layout == matrix::matrix_layout::col_major>>
    {
        void load(matrix::joint_matrix<sub_group, matrix::matrix_type::b, Layout, 4, 8> &res,
                  multi_ptr<double, Space> src, size_t stride)
        {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
                __dmma_m8n8k4_ld_b(res.data, src.get(), stride, get_layout_int<Layout>());
#endif
#endif
        }
    };

    template <matrix::matrix_layout Layout, access::address_space Space>
    struct joint_matrix_load_impl<sub_group, matrix::matrix_type::accumulator, 8, 8, Layout,
                                  Space, typename std::enable_if_t<Layout == matrix::matrix_layout::row_major || Layout == matrix::matrix_layout::col_major>>
    {
        void load(matrix::joint_matrix<sub_group, matrix::matrix_type::accumulator, Layout, 8, 8> &res,
                  multi_ptr<double, Space> src, size_t stride)
        {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
                __dmma_m8n8k4_ld_c(res.data, src.get(), stride, get_layout_int<Layout>());
#endif
#endif
        }
    };

    template <typename Group, size_t NumRows, size_t NumCols,
              matrix::matrix_layout Layout, access::address_space Space, typename Cond = void>
    struct joint_matrix_store_impl
    {
        void store(matrix::joint_matrix<Group, matrix::matrix_type::accumulator, Layout, NumRows,
                                        NumCols> &src,
                   multi_ptr<double, Space> dst, size_t stride);
    };

    template <matrix::matrix_layout Layout, access::address_space Space>
    struct joint_matrix_store_impl<sub_group, 8, 8, Layout, Space, typename std::enable_if_t<Layout == matrix::matrix_layout::row_major || Layout == matrix::matrix_layout::col_major>>
    {
        void store(matrix::joint_matrix<sub_group, matrix::matrix_type::accumulator, Layout, 8, 8> &src,
                   multi_ptr<double, Space> dst, size_t stride)
        {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
                __dmma_m8n8k4_st_c_f64(dst.get(), src.data, stride, get_layout_int<Layout>());
#endif
#endif
        }
    };
    template <typename Group, std::size_t M, std::size_t K, std::size_t N,
              matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
              matrix::matrix_layout LayoutC, typename Cond = void>
    struct joint_matrix_mma_impl
    {
        matrix::joint_matrix<Group, matrix::matrix_type::accumulator, LayoutC, M, N>
        mma(Group sg,
            matrix::joint_matrix<Group, matrix::matrix_type::a, LayoutA, M, K> A,
            matrix::joint_matrix<Group, matrix::matrix_type::b, LayoutB, K, N> B,
            matrix::joint_matrix<Group, matrix::matrix_type::accumulator, LayoutC, M, N> C);
    };

    template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB>
    constexpr int get_layout_comb_int();

    template <>
    constexpr int get_layout_comb_int<matrix::matrix_layout::row_major, matrix::matrix_layout::row_major>()
    {
        return 0;
    }

    template <>
    constexpr int get_layout_comb_int<matrix::matrix_layout::row_major, matrix::matrix_layout::col_major>()
    {
        return 1;
    }

    template <>
    constexpr int get_layout_comb_int<matrix::matrix_layout::col_major, matrix::matrix_layout::row_major>()
    {
        return 2;
    }

    template <>
    constexpr int get_layout_comb_int<matrix::matrix_layout::col_major, matrix::matrix_layout::col_major>()
    {
        return 3;
    }

    template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
              matrix::matrix_layout LayoutC>
    struct joint_matrix_mma_impl<sub_group, 8, 4, 8, LayoutA, LayoutB, LayoutC, typename std::enable_if_t<(LayoutA == matrix::matrix_layout::row_major || LayoutA == matrix::matrix_layout::col_major) && (LayoutB == experimental::matrix::matrix_layout::row_major || LayoutB == experimental::matrix::matrix_layout::col_major)&& (LayoutC == experimental::matrix::matrix_layout::row_major || LayoutC == experimental::matrix::matrix_layout::col_major)>>
    {
        matrix::joint_matrix<sub_group, matrix::matrix_type::accumulator, LayoutC, 8, 8>
        mma(sub_group sg,
            matrix::joint_matrix<sub_group, matrix::matrix_type::a, LayoutA, 8, 4>
                A,
            matrix::joint_matrix<sub_group, matrix::matrix_type::b, LayoutB, 4, 8>
                B,
            matrix::joint_matrix<sub_group, matrix::matrix_type::accumulator, LayoutC, 8, 8>
                C)
        {
            matrix::joint_matrix<sub_group, matrix::matrix_type::accumulator, LayoutC, 8, 8> D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
    __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, get_layout_comb_int<LayoutA, LayoutB>(), 0);
#endif
#endif

            return D;
        }
    };

} // namespace detail

namespace experimental::matrix {

    template <typename Group, matrix_type MT, size_t NumRows, size_t NumCols,
              matrix_layout Layout, access::address_space Space>
    void joint_matrix_load(Group sg,
                           joint_matrix<Group, MT, Layout, NumRows, NumCols> &res,
                           multi_ptr<double, Space> src, size_t stride)
    {
        detail::joint_matrix_load_impl<Group, MT, NumRows, NumCols, Layout, Space>{}
            .load(res, src, stride);
    }

    template <typename Group, size_t NumRows, size_t NumCols, matrix_layout Layout,
              access::address_space Space>
    void joint_matrix_store(
        Group sg,
        joint_matrix<Group, matrix_type::accumulator, Layout, NumRows, NumCols> &src,
        multi_ptr<double, Space> dst, size_t stride)
    {
        detail::joint_matrix_store_impl<Group, NumRows, NumCols, Layout, Space>{}
            .store(src, dst, stride);
    }

    template <typename Group, std::size_t M, std::size_t K, std::size_t N,
              matrix_layout LayoutA, matrix_layout LayoutB, matrix_layout LayoutC>
    joint_matrix<Group, matrix_type::accumulator, LayoutC, M, N>
    joint_matrix_mad(Group sg, joint_matrix<Group, matrix_type::a, LayoutA, M, K> A,
                     joint_matrix<Group, matrix_type::b, LayoutB, K, N> B,
                     joint_matrix<Group, matrix_type::accumulator, LayoutC, M, N> C)
    {
        return detail::joint_matrix_mma_impl<Group, M, K, N, LayoutA, LayoutB,
                                             LayoutC>{}
            .mma(sg, A, B, C);
    }

} // namespace experimental::matrix
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)