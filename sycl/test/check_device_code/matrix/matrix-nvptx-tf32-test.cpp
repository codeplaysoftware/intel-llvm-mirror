// REQUIRES: gpu, cuda

// RUN: %clangxx -fsycl-device-only -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX=3 -S -Xclang -emit-llvm %s -o -| FileCheck %s

#include <CL/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int stride = 16;

int main() {

  buffer<uint32_t, 1> bufA(nullptr, range<1>(1));
  buffer<uint32_t, 1> bufB(nullptr, range<1>(1));
  buffer<float, 1> bufC(nullptr, range<1>(1));
  buffer<float, 1> bufD(nullptr, range<1>(1));

  queue q;

  q.submit([&](handler &cgh) {

    auto accC = bufC.get_access<access::mode::read_write>(cgh);
    auto accA = bufA.get_access<access::mode::read_write>(cgh);
    auto accB = bufB.get_access<access::mode::read_write>(cgh);
    auto accD = bufD.get_access<access::mode::read_write>(cgh);
   
    cgh.parallel_for<class row_row_m16n16k8>(
        nd_range<2>({1, 32}, {1, 32}), [=
    ](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();


        joint_matrix<float, matrix_use::accumulator, 16, 16,
                       matrix_layout::row_major>
              sub_c;

          joint_matrix<uint32_t, matrix_use::a, 16, 8, matrix_layout::row_major>
              sub_a;

          joint_matrix<uint32_t, matrix_use::b, 8, 16, matrix_layout::row_major>
              sub_b;
        // Note this uses the llvm load c instruction for m16n16k16 which matches this shape's instruction in this case.
        //CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.row.stride.f32.p1f32(float addrspace(1)* %add.ptr.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          //CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.row.stride.tf32.p0i32(i32* %call.ascast.i.i46.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          //CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.b.row.stride.tf32.p0i32(i32* %call.ascast.i.i.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          //CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.row.row.tf32(i32 %18, i32 %19, i32 %20, i32 %21, i32 %23, i32 %24, i32 undef, i32 0, float %9, float %10, float %11, float %12, float %13, float %14, float %15, float %16) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          //CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p1f32(float addrspace(1)* %add.ptr.i76, float %26, float %27, float %28, float %29, float %30, float %31, float %32, float %33, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);

        });

                cgh.parallel_for<class col_col_m16n16k8>(
        nd_range<2>({1, 32}, {1, 32}), [=
    ](nd_item<2> item) [[sycl::reqd_work_group_size(1, 1, 32)]] {
          sycl::sub_group sg = item.get_sub_group();


        joint_matrix<float, matrix_use::accumulator, 16, 16,
                       matrix_layout::col_major>
              sub_c;

          joint_matrix<uint32_t, matrix_use::a, 16, 8, matrix_layout::col_major>
              sub_a;

          joint_matrix<uint32_t, matrix_use::b, 8, 16, matrix_layout::col_major>
              sub_b;

        //CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k16.load.c.col.stride.f32.p1f32(float addrspace(1)* %add.ptr.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_c, accC.get_pointer(), stride);
          //CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.a.col.stride.tf32.p0i32(i32* %call.ascast.i.i46.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_a, accA.get_pointer(), stride);
          //CHECK: tail call { i32, i32, i32, i32 } @llvm.nvvm.wmma.m16n16k8.load.b.col.stride.tf32.p0i32(i32* %call.ascast.i.i.i, i32 16) #{{.*}}
          joint_matrix_load(sg, sub_b, accB.get_pointer(), stride);
          //CHECK: tail call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.m16n16k8.mma.col.col.tf32(i32 %18, i32 %19, i32 %20, i32 %21, i32 %23, i32 %24, i32 undef, i32 0, float %9, float %10, float %11, float %12, float %13, float %14, float %15, float %16) #{{.*}}
          sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
          //CHECK: tail call void @llvm.nvvm.wmma.m16n16k16.store.d.col.stride.f32.p1f32(float addrspace(1)* %add.ptr.i76, float %26, float %27, float %28, float %29, float %30, float %31, float %32, float %33, i32 16) #{{.*}}
          joint_matrix_store(sg, sub_c, accD.get_pointer(), stride);

        });
  });

  return 0;
};
