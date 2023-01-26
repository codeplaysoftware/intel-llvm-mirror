// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <type_traits>

using namespace sycl;

static device dev;

// Uncomment to print all test mismatches
// #define VERBOSE_PRINT

// Helpers and utilities
struct util {
  template <typename DType, int NChannels,
            std::enable_if_t<std::is_integral<DType>::value, bool> = true>
  static void fill_rand(std::vector<sycl::vec<DType, NChannels>> &v) {
    std::default_random_engine generator;
    std::uniform_int_distribution<DType> distribution(-100.0, 100.0);
    for (int i = 0; i < v.size(); ++i) {
      v[i] = sycl::vec<DType, NChannels>(distribution(generator));
    }
  }

  template <typename DType, int NChannels,
            std::enable_if_t<std::is_floating_point<DType>::value, bool> = true>
  static void fill_rand(std::vector<sycl::vec<DType, NChannels>> &v) {
    std::default_random_engine generator;
    std::uniform_real_distribution<DType> distribution(0, 100);
    for (int i = 0; i < v.size(); ++i) {
      v[i] = sycl::vec<DType, NChannels>(distribution(generator));
    }
  }

  template <typename DType, int NChannels>
  static void add_host(const std::vector<sycl::vec<DType, NChannels>> &in_0,
                       const std::vector<sycl::vec<DType, NChannels>> &in_1,
                       std::vector<sycl::vec<DType, NChannels>> &out) {
    for (int i = 0; i < out.size(); ++i) {
      for (int j = 0; j < NChannels; ++j) {
        out[i][j] = in_0[i][j] + in_1[i][j];
      }
    }
  }

  template <typename DType, int NChannels,
            typename = std::enable_if_t<NChannels == 1>>
  static DType add_kernel(const DType in_0, const DType in_1) {
    return in_0 + in_1;
  }

  template <typename DType, int NChannels,
            typename = std::enable_if_t<(NChannels > 1)>>
  static sycl::vec<DType, NChannels>
  add_kernel(const sycl::vec<DType, NChannels> &in_0,
             const sycl::vec<DType, NChannels> &in_1) {
    sycl::vec<DType, NChannels> out;
    for (int i = 0; i < NChannels; ++i) {
      out[i] = in_0[i] + in_1[i];
    }
    return out;
  }

  // parallel_for 3D
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 3>>
  static void run_ndim_test(sycl::queue q, sycl::range<3> globalSize,
                            sycl::range<3> localSize,
                            sycl::ext::oneapi::unsampled_image_handle input_0,
                            sycl::ext::oneapi::unsampled_image_handle input_1,
                            sycl::ext::oneapi::unsampled_image_handle output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](handler &cgh) {
        cgh.parallel_for<KernelName>(
            nd_range<NDims>{globalSize, localSize}, [=](nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);
              size_t dim2 = it.get_global_id(2);

              VecType px1 = sycl::ext::oneapi::read_image<VecType>(
                  input_0, int4(dim0, dim1, dim2, 0));
              VecType px2 = sycl::ext::oneapi::read_image<VecType>(
                  input_1, int4(dim0, dim1, dim2, 0));

              auto sum = util::add_kernel<DType, NChannels>(px1, px2);

              sycl::ext::oneapi::write_image<VecType>(
                  output, int4(dim0, dim1, dim2, 0), VecType(sum));
            });
      });
    } catch (...) {
      std::cout << "\tKernel submission failed!" << std::endl;
    }
  }

  // parallel_for 2D
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 2>>
  static void run_ndim_test(sycl::queue q, sycl::range<2> globalSize,
                            sycl::range<2> localSize,
                            sycl::ext::oneapi::unsampled_image_handle input_0,
                            sycl::ext::oneapi::unsampled_image_handle input_1,
                            sycl::ext::oneapi::unsampled_image_handle output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](handler &cgh) {
        cgh.parallel_for<KernelName>(
            nd_range<NDims>{globalSize, localSize}, [=](nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);
              size_t dim1 = it.get_global_id(1);

              VecType px1 = sycl::ext::oneapi::read_image<VecType>(
                  input_0, int2(dim0, dim1));
              VecType px2 = sycl::ext::oneapi::read_image<VecType>(
                  input_1, int2(dim0, dim1));

              auto sum = util::add_kernel<DType, NChannels>(px1, px2);

              sycl::ext::oneapi::write_image<VecType>(output, int2(dim0, dim1),
                                                      VecType(sum));
            });
      });
    } catch (...) {
      std::cout << "\tKernel submission failed!" << std::endl;
    }
  }

  // parallel_for 1D
  template <int NDims, typename DType, int NChannels, typename KernelName,
            typename = std::enable_if_t<NDims == 1>>
  static void run_ndim_test(sycl::queue q, sycl::range<1> globalSize,
                            sycl::range<1> localSize,
                            sycl::ext::oneapi::unsampled_image_handle input_0,
                            sycl::ext::oneapi::unsampled_image_handle input_1,
                            sycl::ext::oneapi::unsampled_image_handle output) {
    using VecType = sycl::vec<DType, NChannels>;
    try {
      q.submit([&](handler &cgh) {
        cgh.parallel_for<KernelName>(
            nd_range<NDims>{globalSize, localSize}, [=](nd_item<NDims> it) {
              size_t dim0 = it.get_global_id(0);

              VecType px1 =
                  sycl::ext::oneapi::read_image<VecType>(input_0, int(dim0));
              VecType px2 =
                  sycl::ext::oneapi::read_image<VecType>(input_1, int(dim0));

              auto sum = util::add_kernel<DType, NChannels>(px1, px2);

              sycl::ext::oneapi::write_image<VecType>(output, int(dim0),
                                                      VecType(sum));
            });
      });
    } catch (...) {
      std::cout << "\tKernel submission failed!" << std::endl;
    }
  }
};

template <int NDims, typename DType, int NChannels, image_channel_type CType,
          image_channel_order COrder, typename KernelName>
bool run_test(sycl::range<NDims> dims, sycl::range<NDims> localSize) {
  queue q(dev);
  auto ctxt = q.get_context();

  using VecType = sycl::vec<DType, NChannels>;

  size_t num_elems = dims[0];
  if (NDims > 1)
    num_elems *= dims[1];
  if (NDims > 2)
    num_elems *= dims[2];

  std::vector<VecType> input_0(num_elems);
  std::vector<VecType> input_1(num_elems);
  std::vector<VecType> expected(num_elems);
  std::vector<VecType> actual(num_elems);

  std::srand(0);
  util::fill_rand(input_0);
  util::fill_rand(input_1);
  util::add_host(input_0, input_1, expected);

  sycl::ext::oneapi::image_descriptor desc(dims, COrder, CType);

  auto device_ptr_0 = sycl::ext::oneapi::allocate_image(ctxt, desc);
  auto device_ptr_1 = sycl::ext::oneapi::allocate_image(ctxt, desc);
  auto device_ptr_2 = sycl::ext::oneapi::allocate_image(ctxt, desc);

  if (device_ptr_0 == nullptr || device_ptr_1 == nullptr ||
      device_ptr_2 == nullptr) {
    std::cout << "Error allocating images!" << std::endl;
    return false;
  }

  sycl::ext::oneapi::copy_image(q, device_ptr_0, input_0.data(), desc,
                                sycl::ext::oneapi::image_copy_flags::HtoD);
  sycl::ext::oneapi::copy_image(q, device_ptr_1, input_1.data(), desc,
                                sycl::ext::oneapi::image_copy_flags::HtoD);

  sycl::ext::oneapi::unsampled_image_handle img_input_0 =
      sycl::ext::oneapi::create_image(ctxt, device_ptr_0);
  sycl::ext::oneapi::unsampled_image_handle img_input_1 =
      sycl::ext::oneapi::create_image(ctxt, device_ptr_1);
  sycl::ext::oneapi::unsampled_image_handle img_output =
      sycl::ext::oneapi::create_image(ctxt, device_ptr_2);

  sycl::range<NDims> globalSize = dims;

  q.wait();
  util::run_ndim_test<NDims, DType, NChannels, KernelName>(
      q, globalSize, localSize, img_input_0, img_input_1, img_output);
  q.wait();

  sycl::ext::oneapi::copy_image(q, actual.data(), device_ptr_2, desc,
                                sycl::ext::oneapi::image_copy_flags::DtoH);

  // Cleanup
  try {
    sycl::ext::oneapi::destroy_image_handle(ctxt, img_input_0);
    sycl::ext::oneapi::destroy_image_handle(ctxt, img_input_1);
    sycl::ext::oneapi::destroy_image_handle(ctxt, img_output);

    sycl::ext::oneapi::free_image(ctxt, device_ptr_0);
    sycl::ext::oneapi::free_image(ctxt, device_ptr_1);
    sycl::ext::oneapi::free_image(ctxt, device_ptr_2);
  } catch (...) {
    std::cout << "Failed to destroy image handle." << std::endl;
    return false;
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < num_elems; i++) {
    for (int j = 0; j < NChannels; ++j) {
      bool mismatch = false;
      if (actual[i][j] != expected[i][j]) {
        mismatch = true;
        validated = false;
      }
      if (mismatch) {
        std::cout << "\tResult mismatch at [" << i << "][" << j
                  << "] Expected: " << expected[i][j]
                  << ", Actual: " << actual[i][j] << std::endl;
#ifndef VERBOSE_PRINT
        return false;
#endif //VERBOSE_PRINT
      }
    }
  }
  if (validated) {
    std::cout << "\tCorrect output!" << std::endl;
  }

  return validated;
}

int main() {

  std::cout << "Running 3D float4\n";
  run_test<3, float, 4, image_channel_type::fp32, image_channel_order::rgba,
           class float4_3d>({1024, 1024, 32}, {16, 16, 4});
  std::cout << "Running 2D float4\n";
  run_test<2, float, 4, image_channel_type::fp32, image_channel_order::rgba,
           class float4_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D float4\n";
  run_test<1, float, 4, image_channel_type::fp32, image_channel_order::rgba,
           class float4_1d>({1024}, {512});

  std::cout << "Running 3D float2\n";
  run_test<3, float, 2, image_channel_type::fp32, image_channel_order::rg,
           class float2_3d>({1024, 1024, 32}, {16, 16, 4});
  std::cout << "Running 2D float2\n";
  run_test<2, float, 2, image_channel_type::fp32, image_channel_order::rg,
           class float2_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D float2\n";
  run_test<1, float, 2, image_channel_type::fp32, image_channel_order::rg,
           class float2_1d>({1024}, {512});

  std::cout << "Running 3D float\n";
  run_test<3, float, 1, image_channel_type::fp32, image_channel_order::r,
           class float_3d>({1024, 1024, 32}, {16, 16, 4});
  std::cout << "Running 2D float\n";
  run_test<2, float, 1, image_channel_type::fp32, image_channel_order::r,
           class float_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D float\n";
  run_test<1, float, 1, image_channel_type::fp32, image_channel_order::r,
           class float_1d>({1024}, {512});

  std::cout << "Running 3D int4\n";
  run_test<3, int32_t, 4, image_channel_type::signed_int32,
           image_channel_order::rgba, class int4_3d>({1024, 1024, 32},
                                                     {16, 16, 4});
  std::cout << "Running 2D int4\n";
  run_test<2, int32_t, 4, image_channel_type::signed_int32,
           image_channel_order::rgba, class int4_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D int4\n";
  run_test<1, int32_t, 4, image_channel_type::signed_int32,
           image_channel_order::rgba, class int4_1d>({1024}, {512});

  std::cout << "Running 3D int2\n";
  run_test<3, int32_t, 2, image_channel_type::signed_int32,
           image_channel_order::rg, class int2_3d>({1024, 1024, 32},
                                                   {16, 16, 4});
  std::cout << "Running 2D int2\n";
  run_test<2, int32_t, 2, image_channel_type::signed_int32,
           image_channel_order::rg, class int2_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D int2\n";
  run_test<1, int32_t, 2, image_channel_type::signed_int32,
           image_channel_order::rg, class int2_1d>({1024}, {512});

  std::cout << "Running 3D int\n";
  run_test<3, int32_t, 1, image_channel_type::signed_int32,
           image_channel_order::r, class int_3d>({1024, 1024, 32}, {16, 16, 4});
  std::cout << "Running 2D int\n";
  run_test<2, int32_t, 1, image_channel_type::signed_int32,
           image_channel_order::r, class int_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D int\n";
  run_test<1, int32_t, 1, image_channel_type::signed_int32,
           image_channel_order::r, class int_1d>({1024}, {512});

  std::cout << "Running 3D uint4\n";
  run_test<3, uint32_t, 4, image_channel_type::unsigned_int32,
           image_channel_order::rgba, class uint4_3d>({1024, 1024, 32},
                                                      {16, 16, 4});
  std::cout << "Running 2D uint4\n";
  run_test<2, uint32_t, 4, image_channel_type::unsigned_int32,
           image_channel_order::rgba, class uint4_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D uint4\n";
  run_test<1, uint32_t, 4, image_channel_type::unsigned_int32,
           image_channel_order::rgba, class uint4_1d>({1024}, {512});

  std::cout << "Running 3D uint2\n";
  run_test<3, uint32_t, 2, image_channel_type::unsigned_int32,
           image_channel_order::rg, class uint2_3d>({1024, 1024, 32},
                                                    {16, 16, 4});
  std::cout << "Running 2D uint2\n";
  run_test<2, uint32_t, 2, image_channel_type::unsigned_int32,
           image_channel_order::rg, class uint2_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D uint2\n";
  run_test<1, uint32_t, 2, image_channel_type::unsigned_int32,
           image_channel_order::rg, class uint2_1d>({1024}, {512});

  std::cout << "Running 3D uint\n";
  run_test<3, uint32_t, 1, image_channel_type::unsigned_int32,
           image_channel_order::r, class uint_3d>({1024, 1024, 32},
                                                  {16, 16, 4});
  std::cout << "Running 2D uint\n";
  run_test<2, uint32_t, 1, image_channel_type::unsigned_int32,
           image_channel_order::r, class uint_2d>({4096, 4096}, {32, 32});
  std::cout << "Running 1D uint\n";
  run_test<1, uint32_t, 1, image_channel_type::unsigned_int32,
           image_channel_order::r, class uint_1d>({1024}, {512});

  return 0;
}
