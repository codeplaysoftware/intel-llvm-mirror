// This test checks that the native macros accurately checks that the
// corresponding built-ins are resolved to their native versions.
//
// The native variants only exists for single precision floating point, so this
// test also ensures the half and double variants of the built-ins can still be
// instantiated with the macros set.
//
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm %s -o - | FileCheck %s

// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_COS %s -o - | FileCheck --check-prefix=CHECK-NATIVE-COS %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_EXP %s -o - | FileCheck --check-prefix=CHECK-NATIVE-EXP %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_EXP2 %s -o - | FileCheck --check-prefix=CHECK-NATIVE-EXP2 %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_EXP10 %s -o - | FileCheck --check-prefix=CHECK-NATIVE-EXP10 %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_LOG %s -o - | FileCheck --check-prefix=CHECK-NATIVE-LOG %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_LOG2 %s -o - | FileCheck --check-prefix=CHECK-NATIVE-LOG2 %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_LOG10 %s -o - | FileCheck --check-prefix=CHECK-NATIVE-LOG10 %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_POWR %s -o - | FileCheck --check-prefix=CHECK-NATIVE-POWR %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_RSQRT %s -o - | FileCheck --check-prefix=CHECK-NATIVE-RSQRT %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_SIN %s -o - | FileCheck --check-prefix=CHECK-NATIVE-SIN %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_SQRT %s -o - | FileCheck --check-prefix=CHECK-NATIVE-SQRT %s
// RUN: %clangxx -fsycl -S -fsycl-device-only -Xclang -emit-llvm -DSYCL_NATIVE_TAN %s -o - | FileCheck --check-prefix=CHECK-NATIVE-TAN %s

#include <CL/sycl.hpp>
#include <sycl/ext/oneapi/experimental/native_macros.hpp>

using namespace cl::sycl::ext::oneapi;

template <typename T>
SYCL_EXTERNAL T builtins(T x) {
  T ret = 0.0f;
  T y = 1.0f;

  // CHECK: call {{.*}}__spirv_ocl_cosf
  // CHECK-NATIVE-COS: call {{.*}}__spirv_ocl_native_cosf
  ret = experimental::cos(x);

  // CHECK: call {{.*}}__spirv_ocl_expf
  // CHECK-NATIVE-EXP: call {{.*}}__spirv_ocl_native_expf
  ret = experimental::exp(ret);

  // CHECK: call {{.*}}__spirv_ocl_exp2f
  // CHECK-NATIVE-EXP2: call {{.*}}__spirv_ocl_native_exp2f
  ret = experimental::exp2(ret);

  // CHECK: call {{.*}}__spirv_ocl_exp10f
  // CHECK-NATIVE-EXP10: call {{.*}}__spirv_ocl_native_exp10f
  ret = experimental::exp10(ret);

  // CHECK: call {{.*}}__spirv_ocl_logf
  // CHECK-NATIVE-LOG: call {{.*}}__spirv_ocl_native_logf
  ret = experimental::log(ret);

  // CHECK: call {{.*}}__spirv_ocl_log2f
  // CHECK-NATIVE-LOG2: call {{.*}}__spirv_ocl_native_log2f
  ret = experimental::log2(ret);

  // CHECK: call {{.*}}__spirv_ocl_log10f
  // CHECK-NATIVE-LOG10: call {{.*}}__spirv_ocl_native_log10f
  ret = experimental::log10(ret);

  // CHECK: call {{.*}}__spirv_ocl_powrf
  // CHECK-NATIVE-POWR: call {{.*}}__spirv_ocl_native_powrf
  ret = experimental::powr(ret, y);

  // CHECK: call {{.*}}__spirv_ocl_rsqrtf
  // CHECK-NATIVE-RSQRT: call {{.*}}__spirv_ocl_native_rsqrtf
  ret = experimental::rsqrt(ret);

  // CHECK: call {{.*}}__spirv_ocl_sinf
  // CHECK-NATIVE-SIN: call {{.*}}__spirv_ocl_native_sinf
  ret = experimental::sin(ret);

  // CHECK: call {{.*}}__spirv_ocl_sqrtf
  // CHECK-NATIVE-SQRT: call {{.*}}__spirv_ocl_native_sqrtf
  ret = experimental::sqrt(ret);

  // CHECK: call {{.*}}__spirv_ocl_tanf
  // CHECK-NATIVE-TAN: call {{.*}}__spirv_ocl_native_tanf
  ret = experimental::tan(ret);

  return ret;
}

SYCL_EXTERNAL void builtins_main() {
  float f = builtins<float>(1.0f);
  sycl::half h = builtins<sycl::half>(1.0f);
  double d = builtins<double>(1.0);
}
