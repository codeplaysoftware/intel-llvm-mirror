// RUN: %clangxx -fsycl -std=c++17 -fsycl-targets=%sycl_triple %s -o %t.run
// RUN: %GPU_RUN_PLACEHOLDER %t.run
// RUN: %CPU_RUN_PLACEHOLDER %t.run
// RUN: %ACC_RUN_PLACEHOLDER %t.run
// RUN: env SYCL_DEVICE_FILTER=host %t.run
//
// Crashes on AMD
// XFAIL: hip_amd

#include "common.h"
#include <CL/sycl.hpp>

using namespace cl::sycl;

template <typename T, typename G> class TypeHelper;

template <typename T, typename G>
using KernelName = class TypeHelper<
    typename std::conditional<std::is_same<T, std::byte>::value, unsigned char,
                              T>::type,
    G>;

// Define the number of work items to enqueue.
const size_t NElems = 32;
const size_t WorkGroupSize = 8;

template <typename T> void initInputBuffer(buffer<T, 1> &Buf, size_t Stride) {
  auto Acc = Buf.template get_access<access::mode::write>();
  for (size_t I = 0; I < Buf.size(); I += WorkGroupSize) {
    for (size_t J = 0; J < WorkGroupSize; J++)
      Acc[I + J] = static_cast<T>(I + J + ((J % Stride == 0) ? 100 : 0));
  }
}

template <typename T> int checkResults(buffer<T, 1> &OutBuf, size_t Stride) {
  auto Out = OutBuf.template get_access<access::mode::read>();
  int EarlyFailout = 20;

  for (size_t I = 0; I < OutBuf.size(); I += WorkGroupSize) {
    for (size_t J = 0; J < WorkGroupSize; J++) {
      size_t ExpectedVal = (J % Stride == 0) ? (100 + I + J) : 0;
      if (!checkEqual(Out[I + J], ExpectedVal)) {
        std::cerr << std::string(typeid(T).name()) + ": Stride=" << Stride
                  << " : Incorrect value at index " << I + J
                  << " : Expected: " << toString(ExpectedVal)
                  << ", Computed: " << toString(Out[I + J]) << "\n";
        if (--EarlyFailout == 0)
          return 1;
      }
    }
  }
  return EarlyFailout - 20;
}

template <typename T, typename G> int test(size_t Stride) {
  queue Q;

  buffer<T, 1> InBuf(NElems);
  buffer<T, 1> OutBuf(NElems);

  initInputBuffer(InBuf, Stride);
  initOutputBuffer(OutBuf);

  Q.submit([&](handler &CGH) {
     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Out = OutBuf.template get_access<access::mode::write>(CGH);
     accessor<T, 1, access::mode::read_write, access::target::local> Local(
         range<1>{WorkGroupSize}, CGH);

     nd_range<1> NDR{range<1>(NElems), range<1>(WorkGroupSize)};
     CGH.parallel_for<KernelName<T, G>>(NDR, [=](nd_item<1> NDId) {
       auto GrId = NDId.get_group_linear_id();
       size_t NElemsToCopy =
           WorkGroupSize / Stride + ((WorkGroupSize % Stride) ? 1 : 0);
       size_t Offset = GrId * WorkGroupSize;

        auto sg = NDId.get_sub_group();

       if (std::is_same<G, group<1>>::value) {
         auto Group = NDId.get_group();
         if (Stride == 1) { // Check the version without stride arg.
           auto E = NDId.async_work_group_copy(
               Local.get_pointer(), In.get_pointer() + Offset, NElemsToCopy);
            
           E.wait();

         } else {
           auto E = NDId.async_work_group_copy(Local.get_pointer(),
                                               In.get_pointer() + Offset,
                                               NElemsToCopy, Stride);
           E.wait();
         }

         if (Stride == 1) { // Check the version without stride arg.
           auto E = Group.async_work_group_copy(
               Out.get_pointer() + Offset, Local.get_pointer(), NElemsToCopy);
           Group.wait_for(E);

           
         } else {
           auto E = Group.async_work_group_copy(Out.get_pointer() + Offset,
                                                Local.get_pointer(),
                                                NElemsToCopy, Stride);
           Group.wait_for(E);
         }
       } else if (std::is_same<G, sub_group>::value) {
         auto mask_active = NDId.ext_oneapi_active_sub_group_items();
         auto Sub_group = NDId.get_sub_group();
//sycl::ext::oneapi::src_stride{Stride}
         if (Stride == 1) { // Check the version without stride arg.
           
           auto E = sycl::ext::oneapi::async_group_copy(
             Sub_group, mask_active, In.get_pointer(), Local.get_pointer(),
             NElems);

           sycl::ext::oneapi::wait_for(sg, mask_active, E);


//auto E = sycl::ext::oneapi::async_group_copy(
               /*Sub_group, In.get_pointer() + Offset, Local.get_pointer(),
               NElemsToCopy);
           sycl::ext::oneapi::wait_for(Sub_group, E);*/
         } else {

            auto E = sycl::ext::oneapi::async_group_copy(
             Sub_group, mask_active, In.get_pointer(), Local.get_pointer(),
             NElems, Stride);

           sycl::ext::oneapi::wait_for(sg, mask_active, E);
          /* auto E = sycl::ext::oneapi::async_group_copy(
               Sub_group, In.get_pointer() + Offset, Local.get_pointer(),
               NElemsToCopy, Stride);
           sycl::ext::oneapi::wait_for(Sub_group, E);*/
         }

         if (Stride == 1) { // Check the version without stride arg.
         
         auto F = sycl::ext::oneapi::async_group_copy(
             Sub_group, mask_active, Local.get_pointer(), Out.get_pointer(),
             NElems);

         sycl::ext::oneapi::wait_for(sg, mask_active, F);
         /*
           auto E = sycl::ext::oneapi::async_group_copy(
               Sub_group, Local.get_pointer(), Out.get_pointer() + Offset,
               NElemsToCopy);
           sycl::ext::oneapi::wait_for(Sub_group, E);*/
         } else {
           /*auto E = sycl::ext::oneapi::async_group_copy(
               Sub_group, Local.get_pointer(), Out.get_pointer() + Offset,
               NElemsToCopy, Stride);
           sycl::ext::oneapi::wait_for(Sub_group, E);*/
                    auto F = sycl::ext::oneapi::async_group_copy(
             Sub_group, mask_active, Local.get_pointer(), Out.get_pointer(),
             NElems, Stride);

         sycl::ext::oneapi::wait_for(sg, mask_active, F);
         }
       }
     });
   }).wait();

  return checkResults(OutBuf, Stride);
}

int main() {
  /*for (int Stride = 1; Stride < WorkGroupSize; Stride++) {
    if (test<int, group<1>>(Stride))
      return 1;
    if (test<uint, group<1>>(Stride))
      return 1;
    if (test<double, group<1>>(Stride))
      return 1;
    if (test<float, group<1>>(Stride))
      return 1;
    if (test<long, group<1>>(Stride))
      return 1;
    if (test<ulong, group<1>>(Stride))
      return 1;
    if (test<vec<int, 1>, group<1>>(Stride))
      return 1;
    if (test<int4, group<1>>(Stride))
      return 1;
    if (test<bool, group<1>>(Stride))
      return 1;
    if (test<vec<bool, 1>, group<1>>(Stride))
      return 1;
    if (test<vec<bool, 4>, group<1>>(Stride))
      return 1;
    if (test<cl::sycl::cl_bool, group<1>>(Stride))
      return 1;
    if (test<std::byte, group<1>>(Stride))
      return 1;
  }*/

  for (int Stride = 1; Stride < WorkGroupSize; Stride++) {
   /* if (test<int, sub_group>(Stride))
      return 1;*/
    /*if (test<uint, sub_group>(Stride))
      return 1;
    if (test<double, sub_group>(Stride))
      return 1;
    if (test<float, sub_group>(Stride))
      return 1;
    if (test<long, sub_group>(Stride))
      return 1;
    if (test<ulong, sub_group>(Stride))
      return 1;
    if (test<vec<int, 1>, sub_group>(Stride))
      return 1;*/
    if (test<int4, sub_group>(Stride))
      return 1;
      /*
    if (test<bool, sub_group>(Stride))
      return 1;
    if (test<vec<bool, 1>, sub_group>(Stride))
      return 1;
    if (test<vec<bool, 4>, sub_group>(Stride))
      return 1;
    if (test<cl::sycl::cl_bool, sub_group>(Stride))
      return 1;
    if (test<std::byte, sub_group>(Stride))
      return 1;*/
  }

  std::cout << "Test passed.\n";
  return 0;
}
