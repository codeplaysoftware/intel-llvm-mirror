#pragma once

#include <CL/sycl/detail/defines_elementary.hpp>
#include <immintrin.h>

__SYCL_INLINE_NAMESPACE(cl) {
  namespace sycl {
  namespace ext {
  namespace oneapi {
  namespace experimental::matrix {

  enum class matrix_use { a, b, accumulator };

  enum class matrix_layout { row_major, col_major, packed_a, packed_b };

  template <typename T, matrix_use MT, size_t Rows = sycl::dynamic_extent,
            size_t Cols = sycl::dynamic_extent,
            matrix_layout Layout = matrix_layout::row_major,
            typename Group = sycl::sub_group, typename Cond = void>
  struct joint_matrix {
    joint_matrix(Group g) {}
  };

#define __SYCL_JOINT_MATRIX_OVERLOAD(type, use, M, N, frag_type, frag_size)    \
  template <matrix_layout Layout>                                              \
  struct joint_matrix<                                                         \
      type, matrix_use::use, M, N, Layout, sycl::sub_group,                    \
      typename std::enable_if_t<Layout == matrix_layout::row_major ||          \
                                Layout == matrix_layout::col_major>> {         \
    frag_type data[frag_size];                                                 \
  };

  // m8n8k4
  __SYCL_JOINT_MATRIX_OVERLOAD(double, a, 8, 4, double, 1)
  __SYCL_JOINT_MATRIX_OVERLOAD(double, b, 4, 8, double, 1)
  __SYCL_JOINT_MATRIX_OVERLOAD(double, accumulator, 8, 8, double, 2)

  // m8n32k16 //TODO: will there be bf16 object introduced?
  __SYCL_JOINT_MATRIX_OVERLOAD(unsigned short, a, 8, 16, int32_t, 2)
  __SYCL_JOINT_MATRIX_OVERLOAD(unsigned short, b, 16, 32, int32_t, 8)

  /*__SYCL_JOINT_MATRIX_OVERLOAD(fp16, a, 8, 16, int32_t, 2)
  __SYCL_JOINT_MATRIX_OVERLOAD(fp16, b, 16, 32, int32_t, 8)*/

  __SYCL_JOINT_MATRIX_OVERLOAD(float, accumulator, 8, 32, float, 8)

  __SYCL_JOINT_MATRIX_OVERLOAD(int8_t, a, 8, 16, int32_t, 1)
  __SYCL_JOINT_MATRIX_OVERLOAD(int8_t, b, 16, 32, int32_t, 4)
  __SYCL_JOINT_MATRIX_OVERLOAD(uint8_t, a, 8, 16, int32_t, 1)
  __SYCL_JOINT_MATRIX_OVERLOAD(uint8_t, b, 16, 32, int32_t, 4)
  __SYCL_JOINT_MATRIX_OVERLOAD(int32_t, accumulator, 8, 32, int32_t, 8)

  // m16n16k16
  __SYCL_JOINT_MATRIX_OVERLOAD(unsigned short, a, 16, 16, int32_t, 4)
  __SYCL_JOINT_MATRIX_OVERLOAD(unsigned short, b, 16, 16, int32_t, 4)
  __SYCL_JOINT_MATRIX_OVERLOAD(float, accumulator, 16, 16, float, 8)

  __SYCL_JOINT_MATRIX_OVERLOAD(int8_t, a, 16, 16, int32_t, 2)
  __SYCL_JOINT_MATRIX_OVERLOAD(int8_t, b, 16, 16, int32_t, 2)
  __SYCL_JOINT_MATRIX_OVERLOAD(uint8_t, a, 16, 16, int32_t, 2)
  __SYCL_JOINT_MATRIX_OVERLOAD(uint8_t, b, 16, 16, int32_t, 2)
  __SYCL_JOINT_MATRIX_OVERLOAD(int32_t, accumulator, 16, 16, int32_t, 8)

#undef __SYCL_JOINT_MATRIX_OVERLOAD
  } // namespace experimental::matrix

  namespace detail {
  using namespace experimental;

  template <typename T, matrix::matrix_use Use, size_t NumRows, size_t NumCols,
            matrix::matrix_layout Layout, access::address_space Space,
            typename Cond = void>
  struct joint_matrix_load_impl {
    void load(matrix::joint_matrix<T, Use, NumRows, NumCols, Layout> &res,
              multi_ptr<T, Space> src, size_t stride);
  };

  template <matrix::matrix_layout Layout> constexpr int get_layout_id();

  template <> constexpr int get_layout_id<matrix::matrix_layout::row_major>() {
    return 0;
  }

  template <> constexpr int get_layout_id<matrix::matrix_layout::col_major>() {
    return 1;
  }

  template <typename T, matrix::matrix_use Use, size_t NumRows, size_t NumCols,
            matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_load_impl<
      T, Use, NumRows, NumCols, Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void load(matrix::joint_matrix<T, Use, NumRows, NumCols, Layout> &res,
              multi_ptr<T, Space> src, size_t stride) {
#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__

      if constexpr (NumRows == 16 && NumCols == 16) {
        // //All m16n16k16 shape cases

        // A/B matrix cases
        if constexpr (std::is_same<T, unsigned short>::value) {
          int32_t *tileptr = reinterpret_cast<int32_t *>(src.get());
          if constexpr (Use == matrix::matrix_use::a) {
            __mma_bf16_m16n16k16_ld_a(res.data, tileptr, stride,
                                      get_layout_id<Layout>());
          } else if constexpr (Use == matrix::matrix_use::b) {
            __mma_bf16_m16n16k16_ld_b(res.data, tileptr, stride,
                                      get_layout_id<Layout>());
          }
        } else if constexpr (std::is_same<T, uint8_t>::value) {
          int32_t *tileptr = reinterpret_cast<int32_t *>(src.get());
          if constexpr (Use == matrix::matrix_use::a) {
            __imma_m16n16k16_ld_a_u8(res.data, tileptr, stride,
                                     get_layout_id<Layout>());
          } else if constexpr (Use == matrix::matrix_use::b) {
            __imma_m16n16k16_ld_b_u8(res.data, tileptr, stride,
                                     get_layout_id<Layout>());
          }
        } else if constexpr (std::is_same<T, int8_t>::value) {
          int32_t *tileptr = reinterpret_cast<int32_t *>(src.get());
          if constexpr (Use == matrix::matrix_use::a) {
            __imma_m16n16k16_ld_a_s8(res.data, tileptr, stride,
                                     get_layout_id<Layout>());
          } else if constexpr (Use == matrix::matrix_use::b) {
            __imma_m16n16k16_ld_b_s8(res.data, tileptr, stride,
                                     get_layout_id<Layout>());
          }
          /// Accumulator cases
        } else if constexpr (std::is_same<T, int32_t>::value) {
          __imma_m16n16k16_ld_c(res.data, src.get(), stride,
                                get_layout_id<Layout>());
        } else if constexpr (std::is_same<T, float>::value) {
          __hmma_m16n16k16_ld_c_f32(res.data, src.get(), stride,
                                    get_layout_id<Layout>());
        }
      } else if constexpr (NumRows == 8 && NumCols == 16) {
        int32_t *tileptr = reinterpret_cast<int32_t *>(src.get());
        if constexpr (std::is_same<T, unsigned short>::value) {
          __mma_bf16_m8n32k16_ld_a(res.data, tileptr, stride,
                                   get_layout_id<Layout>());
        } else if constexpr (std::is_same<T, uint8_t>::value) {
          __imma_m8n32k16_mma_u8(res.data, tileptr, stride,
                                 get_layout_id<Layout>());
        }

      } else if constexpr (NumRows == 16 && NumCols == 32) {
        int32_t *tileptr = reinterpret_cast<int32_t *>(src.get());
        if constexpr (std::is_same<T, unsigned short>::value) {
          __mma_bf16_m8n32k16_ld_b(res.data, tileptr, stride,
                                   get_layout_id<Layout>());
        }

      } else if constexpr (NumRows == 8 && NumCols == 32) {
        // int32_t* tileptr = reinterpret_cast<int32_t*>(src.get());
        __hmma_m8n32k16_ld_c_f32(res.data, src.get(), stride,
                                 get_layout_id<Layout>());

      } else if constexpr (std::is_same<T, double>::value) {
        if constexpr (Use == matrix::matrix_use::a) {
          __dmma_m8n8k4_ld_a(res.data, src.get(), stride,
                             get_layout_id<Layout>());
        } else if constexpr (Use == matrix::matrix_use::b) {
          __dmma_m8n8k4_ld_b(res.data, src.get(), stride,
                             get_layout_id<Layout>());
        } else if constexpr (Use == matrix::matrix_use::accumulator) {
          __dmma_m8n8k4_ld_c(res.data, src.get(), stride,
                             get_layout_id<Layout>());
        }
      }

#endif
#endif
    }
  };

  template <typename T, size_t NumRows, size_t NumCols,
            matrix::matrix_layout Layout, access::address_space Space,
            typename Cond = void>
  struct joint_matrix_store_impl {
    void store(matrix::joint_matrix<T, matrix::matrix_use::accumulator, NumRows,
                                    NumCols, Layout> &src,
               multi_ptr<T, Space> dst, size_t stride);
  };

  template <typename T, size_t NumRows, size_t NumCols,
            matrix::matrix_layout Layout, access::address_space Space>
  struct joint_matrix_store_impl<
      T, NumRows, NumCols, Layout, Space,
      typename std::enable_if_t<Layout == matrix::matrix_layout::row_major ||
                                Layout == matrix::matrix_layout::col_major>> {
    void store(matrix::joint_matrix<T, matrix::matrix_use::accumulator, NumRows,
                                    NumCols, Layout> &src,
               multi_ptr<T, Space> dst, size_t stride) {

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      if (NumRows == 16 && NumCols == 16) {
        if constexpr (std::is_same<T, float>::value) {
          __hmma_m16n16k16_st_c_f32(dst.get(), src.data, stride,
                                    get_layout_id<Layout>());
        } else if constexpr (std::is_same<T, int32_t>::value) {
          __imma_m16n16k16_st_c_i32(dst.get(), src.data, stride,
                                    get_layout_id<Layout>());
        }
      } else if (NumRows == 8 && NumCols == 32) {
        if constexpr (std::is_same<T, float>::value) {
          __hmma_m8n32k16_st_c_f32(dst.get(), src.data, stride,
                                   get_layout_id<Layout>());
        }

      } else if constexpr (std::is_same<T, double>::value) {
        __dmma_m8n8k4_st_c_f64(dst.get(), src.data, stride,
                               get_layout_id<Layout>());
      }
#endif
#endif
    }
  };

  // todo switch around?
  template <typename T1, typename T2, std::size_t M, std::size_t K,
            std::size_t N, matrix::matrix_layout LayoutA,
            matrix::matrix_layout LayoutB, matrix::matrix_layout LayoutC,
            typename Cond = void>
  struct joint_matrix_mad_impl {
    matrix::joint_matrix<T2, matrix::matrix_use::accumulator, M, N, LayoutC>
    mad(matrix::joint_matrix<T1, matrix::matrix_use::a, M, K, LayoutA> A,
        matrix::joint_matrix<T1, matrix::matrix_use::b, K, N, LayoutB> B,
        matrix::joint_matrix<T2, matrix::matrix_use::accumulator, M, N, LayoutC>
            C);
  };

  template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB>
  constexpr int get_layout_pair_id();

  template <>
  constexpr int get_layout_pair_id<matrix::matrix_layout::row_major,
                                   matrix::matrix_layout::row_major>() {
    return 0;
  }

  template <>
  constexpr int get_layout_pair_id<matrix::matrix_layout::row_major,
                                   matrix::matrix_layout::col_major>() {
    return 1;
  }

  template <>
  constexpr int get_layout_pair_id<matrix::matrix_layout::col_major,
                                   matrix::matrix_layout::row_major>() {
    return 2;
  }

  template <>
  constexpr int get_layout_pair_id<matrix::matrix_layout::col_major,
                                   matrix::matrix_layout::col_major>() {
    return 3;
  }

  // TODO: host error?
  template <typename T1, typename T2, std::size_t M, std::size_t K,
            std::size_t N, matrix::matrix_layout LayoutA,
            matrix::matrix_layout LayoutB, matrix::matrix_layout LayoutC>
  struct joint_matrix_mad_impl<
      T1, T2, M, K, N, LayoutA, LayoutB, LayoutC,
      typename std::enable_if_t<(LayoutA == matrix::matrix_layout::row_major ||
                                 LayoutA == matrix::matrix_layout::col_major) &&
                                (LayoutB == matrix::matrix_layout::row_major ||
                                 LayoutB == matrix::matrix_layout::col_major) &&
                                (LayoutC == matrix::matrix_layout::row_major ||
                                 LayoutC ==
                                     matrix::matrix_layout::col_major)>> {
    matrix::joint_matrix<T2, matrix::matrix_use::accumulator, M, N, LayoutC>
    mad(matrix::joint_matrix<T1, matrix::matrix_use::a, M, K, LayoutA> A,
        matrix::joint_matrix<T1, matrix::matrix_use::b, K, N, LayoutB> B,
        matrix::joint_matrix<T2, matrix::matrix_use::accumulator, M, N, LayoutC>
            C) {
      matrix::joint_matrix<T2, matrix::matrix_use::accumulator, M, N, LayoutC>
          D;

#ifdef __NVPTX__
#ifdef __SYCL_DEVICE_ONLY__
      if constexpr (M == 16 && N == 16 && K == 16) {
        if constexpr (std::is_same<T2, int32_t>::value) {
          if constexpr (std::is_same<T1, int8_t>::value) {
            __imma_m16n16k16_mma_s8(D.data, A.data, B.data, C.data,
                                    get_layout_pair_id<LayoutA, LayoutB>(), 0);
          } else if constexpr (std::is_same<T1, uint8_t>::value) {
            __imma_m16n16k16_mma_u8(D.data, A.data, B.data, C.data,
                                    get_layout_pair_id<LayoutA, LayoutB>(), 0);
          }
        } else if constexpr (std::is_same<T2, float>::value) {
          if constexpr (std::is_same<T1, unsigned short>::value) {
            __mma_bf16_m16n16k16_mma_f32(D.data, A.data, B.data, C.data,
                                         get_layout_pair_id<LayoutA, LayoutB>(),
                                         0);
          }
        }
      } else if constexpr (M == 8 && N == 32 && K == 16) {
        if constexpr (std::is_same<T2, int32_t>::value) {
          if constexpr (std::is_same<T1, int8_t>::value) {
            __imma_m8n32k16_mma_s8(D.data, A.data, B.data, C.data,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
          } else if constexpr (std::is_same<T1, uint8_t>::value) {
            __imma_m8n32k16_mma_u8(D.data, A.data, B.data, C.data,
                                   get_layout_pair_id<LayoutA, LayoutB>(), 0);
          }
        } else if constexpr (std::is_same<T2, float>::value) {
          if constexpr (std::is_same<T1, unsigned short>::value) {
            __mma_bf16_m8n32k16_mma_f32(D.data, A.data, B.data, C.data,
                                        get_layout_pair_id<LayoutA, LayoutB>(),
                                        0);
          }
        }
      } else if constexpr (std::is_same<T1, double>::value) {
        __dmma_m8n8k4_mma_f64(D.data, A.data, B.data, C.data,
                              get_layout_pair_id<LayoutA, LayoutB>(), 0);
      }

#endif
#endif

      return D;
    }
  };

  /* fp16 impl doesnt work
     template <matrix::matrix_layout LayoutA, matrix::matrix_layout LayoutB,
              matrix::matrix_layout LayoutC>
    struct joint_matrix_mad_impl<
        matrix::matrix_type::fp16, matrix::matrix_type::fp32, 8, 16, 32,
  LayoutA, LayoutB, LayoutC, typename std::enable_if_t<(LayoutA ==
  matrix::matrix_layout::row_major || LayoutA ==
  matrix::matrix_layout::col_major) && (LayoutB ==
  matrix::matrix_layout::row_major || LayoutB ==
  matrix::matrix_layout::col_major) && (LayoutC ==
  matrix::matrix_layout::row_major || LayoutC ==
                                       matrix::matrix_layout::col_major)>> {
      matrix::joint_matrix<matrix::matrix_type::fp32,
                           matrix::matrix_use::accumulator, 8, 32, LayoutC>
      mad(matrix::joint_matrix<matrix::matrix_type::fp16, matrix::matrix_use::a,
                               8, 16, LayoutA>
              A,
          matrix::joint_matrix<matrix::matrix_type::fp16, matrix::matrix_use::b,
                               16, 32, LayoutB>
              B,
          matrix::joint_matrix<matrix::matrix_type::fp32,
                               matrix::matrix_use::accumulator, 8, 32, LayoutC>
              C) {
        matrix::joint_matrix<matrix::matrix_type::fp32,
                             matrix::matrix_use::accumulator, 8, 32, LayoutC>
            D;

  #ifdef __NVPTX__
  #ifdef __SYCL_DEVICE_ONLY__
        __hmma_m32n8k16_mma_f32f32(D.data, A.data, B.data, C.data,
                                     get_layout_pair_id<LayoutA, LayoutB>(), 0);
  #endif
  #endif

        return D;
      }
    };*/

  } // namespace detail

  namespace experimental::matrix {

  // todo: asserts here for valid types??

  template <typename Group, typename T, matrix_use MT, size_t NumRows,
            size_t NumCols, matrix_layout Layout, access::address_space Space>
  void
  joint_matrix_load(Group sg,
                    joint_matrix<T, MT, NumRows, NumCols, Layout, Group> &res,
                    multi_ptr<T, Space> src, size_t stride) {
    detail::joint_matrix_load_impl<T, MT, NumRows, NumCols, Layout, Space>{}
        .load(res, src, stride);
  }

  template <typename Group, typename T, size_t NumRows, size_t NumCols,
            matrix_layout Layout, access::address_space Space>
  void joint_matrix_store(Group sg,
                          joint_matrix<T, matrix_use::accumulator, NumRows,
                                       NumCols, Layout, Group> &src,
                          multi_ptr<T, Space> dst, size_t stride) {
    detail::joint_matrix_store_impl<T, NumRows, NumCols, Layout, Space>{}.store(
        src, dst, stride);
  }

  template <typename Group, typename T1, typename T2, std::size_t M,
            std::size_t K, std::size_t N, matrix_layout LayoutA,
            matrix_layout LayoutB, matrix_layout LayoutC>
  joint_matrix<T2, matrix_use::accumulator, M, N, LayoutC, Group>
  joint_matrix_mad(
      Group sg, joint_matrix<T1, matrix_use::a, M, K, LayoutA, Group> A,
      joint_matrix<T1, matrix_use::b, K, N, LayoutB, Group> B,
      joint_matrix<T2, matrix_use::accumulator, M, N, LayoutC, Group> C) {
    return detail::joint_matrix_mad_impl<T1, T2, M, K, N, LayoutA, LayoutB,
                                         LayoutC>{}
        .mad(A, B, C);
  }

  } // namespace experimental::matrix
  } // namespace oneapi
  } // namespace ext
  } // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
