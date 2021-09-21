#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl)
{
    namespace sycl
    {
        namespace ext
        {
            namespace intel
            {

                namespace experimental::matrix
                {

                    enum class matrix_type
                    {
                        a,
                        b,
                        accumulator
                    };

                    // packed layout not yet implemented (requires changes to LIBCLC). // We currently use SFINAE to remove overload if joint matrices use packed layout.
                    enum class matrix_layout
                    {
                        row_major,
                        col_major,
                        packed
                    };
                    // Note that the Group typename is never used here (it is only used in the JIT AMX implementation)
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

                namespace detail
                {
                    using namespace experimental;

                    template <typename Group, matrix::matrix_type MT, size_t NumRows,
                              size_t NumCols, matrix::matrix_layout Layout,
                              access::address_space Space, typename Cond = void>
                    struct joint_matrix_load_impl
                    {
                        void load(matrix::joint_matrix<Group, MT, Layout, NumRows, NumCols> &res,
                                  multi_ptr<double, Space> src, size_t stride);
                    };

                    // Helper (Would be good to use something similar- currently cannot use it!)
                    template <matrix::matrix_layout Layout>
                    const int get_layout_int()
                    {
                        static_assert(Layout == matrix::matrix_layout::row_major ||
                                      Layout == matrix::matrix_layout::col_major);
                        return Layout == matrix::matrix_layout::row_major ? 0 : 1;
                    }

                    template <matrix::matrix_layout Layout, access::address_space Space>
                    struct joint_matrix_load_impl<sub_group, matrix::matrix_type::a, 8, 4, Layout,
                                                  Space, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major || Layout == experimental::matrix::matrix_layout::col_major>>
                    {
                        void load(matrix::joint_matrix<sub_group, matrix::matrix_type::a, Layout, 8, 4> &res,
                                  multi_ptr<double, Space> src, size_t stride)
                        {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
                            if (Layout == matrix::matrix_layout::row_major)
                                __dmma_m8n8k4_ld_a(res.data, src.get(), stride, 0);
                            else
                                __dmma_m8n8k4_ld_a(res.data, src.get(), stride, 1);
#endif
#endif
                        }
                    };

                    template <matrix::matrix_layout Layout, access::address_space Space>
                    struct joint_matrix_load_impl<sub_group, matrix::matrix_type::b, 4, 8, Layout,
                                                  Space, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major || Layout == experimental::matrix::matrix_layout::col_major>>
                    {
                        void load(matrix::joint_matrix<sub_group, matrix::matrix_type::b, Layout, 4, 8> &res,
                                  multi_ptr<double, Space> src, size_t stride)
                        {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
                            if (Layout == matrix::matrix_layout::row_major)
                                __dmma_m8n8k4_ld_b(res.data, src.get(), stride, 0);
                            else
                                __dmma_m8n8k4_ld_b(res.data, src.get(), stride, 1);
#endif
#endif
                        }
                    };

                    template <matrix::matrix_layout Layout, access::address_space Space>
                    struct joint_matrix_load_impl<sub_group, matrix::matrix_type::accumulator, 8, 8, Layout,
                                                  Space, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major || Layout == experimental::matrix::matrix_layout::col_major>>
                    {
                        void load(matrix::joint_matrix<sub_group, matrix::matrix_type::accumulator, Layout, 8, 8> &res,
                                  multi_ptr<double, Space> src, size_t stride)
                        {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
                            if (Layout == matrix::matrix_layout::row_major)
                                __dmma_m8n8k4_ld_c(res.data, src.get(), stride, 0);
                            else
                                __dmma_m8n8k4_ld_c(res.data, src.get(), stride, 1);
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
                    struct joint_matrix_store_impl<sub_group, 8, 8, Layout, Space, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major || Layout == experimental::matrix::matrix_layout::col_major>>
                    {
                        void store(matrix::joint_matrix<sub_group, matrix::matrix_type::accumulator, Layout, 8, 8> &src,
                                   multi_ptr<double, Space> dst, size_t stride)
                        {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
                            if (Layout == matrix::matrix_layout::row_major)
                                __dmma_m8n8k4_st_c_f64(dst.get(), src.data, stride, 0);
                            else
                                __dmma_m8n8k4_st_c_f64(dst.get(), src.data, stride, 1);
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

                    // Helper (currently cannot be used)
                    template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB>
                    int get_layout_comb_int()
                    {
                        static_assert(LayoutA == matrix::matrix_layout::row_major ||
                                      LayoutA == matrix::matrix_layout::col_major);
                        static_assert(LayoutB == matrix::matrix_layout::row_major ||
                                      LayoutB == matrix::matrix_layout::col_major);
                        if (LayoutA == matrix::matrix_layout::row_major)
                            return LayoutB == matrix::matrix_layout::row_major ? 0 : 1;
                        // LayoutA == matrix_layout::col_major must hold here
                        return LayoutB == matrix::matrix_layout::row_major ? 2 : 3;
                    }

                    // TODO: Condition for LayoutC also or is this unnecessary?
                    template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
                              matrix::matrix_layout LayoutC>
                    struct joint_matrix_mma_impl<sub_group, 8, 4, 8, LayoutA, LayoutB, LayoutC, typename std::enable_if_t<(LayoutA == experimental::matrix::matrix_layout::row_major || LayoutA == experimental::matrix::matrix_layout::col_major) && (LayoutB == experimental::matrix::matrix_layout::row_major || LayoutB == experimental::matrix::matrix_layout::col_major)>>
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
                            //__imma_m16n16k16_mma_s8(D.data, A.data, B.data, C.data, 1, 0);
                            if (LayoutA == matrix::matrix_layout::row_major)
                            {
                                if (LayoutB == matrix::matrix_layout::row_major)
                                    __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, 0, 0);
                                else
                                    __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, 1, 0);
                            }
                            else
                            {
                                if (LayoutB == matrix::matrix_layout::row_major)
                                    __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, 2, 0);
                                else
                                    __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data, 3, 0);
                            }
#endif
#endif

                            return D;
                        }
                    };

                } // namespace detail

                namespace experimental::matrix
                {
                    template <typename Group, matrix_type MT, size_t NumRows, size_t NumCols,
                              matrix_layout Layout, access::address_space Space>
                    void joint_matrix_load(Group sg,
                                           joint_matrix<Group, MT, Layout, NumRows, NumCols> &res,
                                           multi_ptr<double, Space> src, size_t stride)
                    {
                        detail::joint_matrix_load_impl<Group, MT, NumRows, NumCols, Layout, Space, typename std::enable_if_t<Layout == experimental::matrix::matrix_layout::row_major || Layout == experimental::matrix::matrix_layout::col_major>>{}
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
        }     // namespace ext
    }         // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
