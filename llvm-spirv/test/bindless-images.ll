; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv -spirv-ext=+SPV_NV_bindless_texture %t.bc -spirv-text -o %t
; RUN: FileCheck --check-prefix=CHECK-SPIRV < %t %s
; RUN: llvm-spirv %t.bc -spirv-ext=+SPV_NV_bindless_texture -o %t.spv
; RUN: spirv-val %t.spv

; CHECK-SPIRV: Capability BindlessTextureNV
; CHECK-SPIRV: Decorate 12 BindlessImageNV
; CHECK-SPIRV: Decorate 16 BindlessSamplerNV
; CHECK-SPIRV: Decorate 20 BindlessImageNV


; CHECK-SPIRV: Decorate 23 BindlessImageNV
; CHECK-SPIRV: Decorate 31 BindlessImageNV
; CHECK-SPIRV: Decorate 45 BindlessImageNV

; CHECK-SPIRV: Decorate 50 BindlessImageNV
; CHECK-SPIRV: Decorate 56 BindlessImageNV
; CHECK-SPIRV: Decorate 67 BindlessImageNV


; CHECK-SPIRV: Constant 9 10 234343 0
; CHECK-SPIRV: Constant 9 14 484670474 10
; CHECK-SPIRV: Constant 9 18 433334553 0

; CHECK-SPIRV: Constant 9 22 1234 0
; CHECK-SPIRV: Constant 9 29 4321 0
; CHECK-SPIRV: Constant 9 43 1357 0

; CHECK-SPIRV: Constant 9 49 6789 0
; CHECK-SPIRV: Constant 9 53 9876 0
; CHECK-SPIRV: Constant 9 65 9753 0


; CHECK-SPIRV: TypeImage 11 2 0 0 0 0 0 0 0
; CHECK-SPIRV: TypeSampler 15
; CHECK-SPIRV: TypeSampledImage 19 11

; CHECK-SPIRV: TypeImage 30 2 1 0 0 0 0 0 0
; CHECK-SPIRV: TypeImage 44 2 2 0 0 0 0 0 1
; CHECK-SPIRV: TypeImage 54 2 2 0 0 0 0 0 0
; CHECK-SPIRV: TypeSampledImage 55 54
; CHECK-SPIRV: TypeImage 66 2 1 0 0 0 0 0 1


; CHECK-SPIRV: ConvertUToImageNV 11 12 10
; CHECK-SPIRV: ConvertImageToUNV 9 13 12

; CHECK-SPIRV: ConvertUToSamplerNV 15 16 14
; CHECK-SPIRV: ConvertSamplerToUNV 9 17 16

; CHECK-SPIRV: ConvertUToSampledImageNV 19 20 18
; CHECK-SPIRV: ConvertSampledImageToUNV 9 21 20


; CHECK-SPIRV: ConvertUToImageNV 11 23 22
; CHECK-SPIRV: ImageRead 25 28 23 27

; CHECK-SPIRV: ConvertUToImageNV 30 31 29
; CHECK-SPIRV: ImageRead 25 36 31 35 

; CHECK-SPIRV: ConvertUToImageNV 44 45 43
; CHECK-SPIRV: ImageWrite 45 48 42

; CHECK-SPIRV: ConvertUToSampledImageNV 19 50 49
; CHECK-SPIRV: ImageSampleExplicitLod 25 52 50 27 2 51

; CHECK-SPIRV: ConvertUToSampledImageNV 55 56 53
; CHECK-SPIRV: ImageSampleExplicitLod 25 59 56 58 2 51

; CHECK-SPIRV: ConvertUToImageNV 66 67 65
; CHECK-SPIRV: ImageWrite 67 35 64

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%opencl.image1d_ro_t = type opaque
%opencl.sampler_t = type opaque
%spirv.SampledImage.image1d_ro_t = type opaque
%opencl.image2d_ro_t = type opaque
%opencl.image3d_wo_t = type opaque
%spirv.SampledImage.image3d_ro_t = type opaque
%opencl.image2d_wo_t = type opaque

$_ZTS11test_kernel = comdat any

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @_ZTS11test_kernel() local_unnamed_addr #0 comdat !srcloc !46 !kernel_arg_buffer_location !47 !sycl_fixed_targets !47 !sycl_kernel_omit_args !47 {
entry:
  call void @__itt_offload_wi_start_wrapper()
  %call.i.i = tail call spir_func %opencl.image1d_ro_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image1d_roET_m(i64 noundef 234343) #2
  %call1.i.i = tail call spir_func noundef i64 @_Z25__spirv_ConvertImageToUNVI14ocl_image1d_roEmT_(%opencl.image1d_ro_t addrspace(1)* %call.i.i) #2
  %call.i22.i = tail call spir_func %opencl.sampler_t addrspace(2)* @_Z27__spirv_ConvertUToSamplerNVI11ocl_samplerET_m(i64 noundef 43434343434) #2
  %call1.i23.i = tail call spir_func noundef i64 @_Z27__spirv_ConvertSamplerToUNVI11ocl_samplerEmT_(%opencl.sampler_t addrspace(2)* %call.i22.i) #2
  %call.i24.i = tail call spir_func %spirv.SampledImage.image1d_ro_t addrspace(1)* @_Z32__spirv_ConvertUToSampledImageNVI32__spirv_SampledImage__image1d_roET_m(i64 noundef 433334553) #2
  %call1.i25.i = tail call spir_func noundef i64 @_Z32__spirv_ConvertSampledImageToUNVI32__spirv_SampledImage__image1d_roEmT_(%spirv.SampledImage.image1d_ro_t addrspace(1)* %call.i24.i) #2
  %call.i26.i = tail call spir_func %opencl.image1d_ro_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image1d_roET_m(i64 noundef 1234) #2, !noalias !48
  %call1.i.i.i = tail call spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4_f14ocl_image1d_roiET_T0_T1_(%opencl.image1d_ro_t addrspace(1)* %call.i26.i, i32 noundef 0) #2, !noalias !51
  %call.i28.i = tail call spir_func %opencl.image2d_ro_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image2d_roET_m(i64 noundef 4321) #2, !noalias !54
  %call1.i.i29.i = tail call spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4_f14ocl_image2d_roDv2_iET_T0_T1_(%opencl.image2d_ro_t addrspace(1)* %call.i28.i, <2 x i32> noundef <i32 3, i32 5>) #2, !noalias !57
  %px1.sroa.0.0.vec.extract.i = extractelement <4 x float> %call1.i.i.i, i64 0
  %px2.sroa.0.0.vec.extract.i = extractelement <4 x float> %call1.i.i29.i, i64 0
  %add.i = fadd float %px1.sroa.0.0.vec.extract.i, %px2.sroa.0.0.vec.extract.i
  %splat.splatinsert.i.i = insertelement <4 x float> poison, float %add.i, i64 0
  %splat.splat.i.i = shufflevector <4 x float> %splat.splatinsert.i.i, <4 x float> poison, <4 x i32> zeroinitializer
  %call.i34.i = tail call spir_func %opencl.image3d_wo_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image3d_woET_m(i64 noundef 1357) #2
  tail call spir_func void @_Z18__spirv_ImageWriteI14ocl_image3d_woDv4_iDv4_fEvT_T0_T1_(%opencl.image3d_wo_t addrspace(1)* %call.i34.i, <4 x i32> noundef <i32 3, i32 5, i32 7, i32 0>, <4 x float> noundef %splat.splat.i.i) #2
  %call.i36.i = tail call spir_func %spirv.SampledImage.image1d_ro_t addrspace(1)* @_Z32__spirv_ConvertUToSampledImageNVI32__spirv_SampledImage__image1d_roET_m(i64 noundef 6789) #2, !noalias !60
  %call1.i.i37.i = tail call spir_func noundef <4 x float> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_fiET0_T_T1_if(%spirv.SampledImage.image1d_ro_t addrspace(1)* %call.i36.i, i32 noundef 0, i32 noundef 2, float noundef 0.000000e+00) #2, !noalias !63
  %call.i41.i = tail call spir_func %spirv.SampledImage.image3d_ro_t addrspace(1)* @_Z32__spirv_ConvertUToSampledImageNVI32__spirv_SampledImage__image3d_roET_m(i64 noundef 9876) #2, !noalias !66
  %call1.i.i44.i = tail call spir_func noundef <4 x float> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image3d_roDv4_fDv4_iET0_T_T1_if(%spirv.SampledImage.image3d_ro_t addrspace(1)* %call.i41.i, <4 x i32> noundef <i32 3, i32 5, i32 9, i32 0>, i32 noundef 2, float noundef 0.000000e+00) #2, !noalias !69
  %px3.sroa.0.0.vec.extract.i = extractelement <4 x float> %call1.i.i37.i, i64 0
  %px4.sroa.0.0.vec.extract.i = extractelement <4 x float> %call1.i.i44.i, i64 0
  %add16.i = fadd float %px3.sroa.0.0.vec.extract.i, %px4.sroa.0.0.vec.extract.i
  %splat.splatinsert.i49.i = insertelement <4 x float> poison, float %add16.i, i64 0
  %splat.splat.i50.i = shufflevector <4 x float> %splat.splatinsert.i49.i, <4 x float> poison, <4 x i32> zeroinitializer
  %call.i53.i = tail call spir_func %opencl.image2d_wo_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image2d_woET_m(i64 noundef 9753) #2
  tail call spir_func void @_Z18__spirv_ImageWriteI14ocl_image2d_woDv2_iDv4_fEvT_T0_T1_(%opencl.image2d_wo_t addrspace(1)* %call.i53.i, <2 x i32> noundef <i32 3, i32 5>, <4 x float> noundef %splat.splat.i50.i) #2
  call void @__itt_offload_wi_finish_wrapper()
  ret void
}

; Function Attrs: convergent
declare dso_local spir_func noundef i64 @_Z25__spirv_ConvertImageToUNVI14ocl_image1d_roEmT_(%opencl.image1d_ro_t addrspace(1)*) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %opencl.image1d_ro_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image1d_roET_m(i64 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func noundef i64 @_Z27__spirv_ConvertSamplerToUNVI11ocl_samplerEmT_(%opencl.sampler_t addrspace(2)*) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %opencl.sampler_t addrspace(2)* @_Z27__spirv_ConvertUToSamplerNVI11ocl_samplerET_m(i64 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func noundef i64 @_Z32__spirv_ConvertSampledImageToUNVI32__spirv_SampledImage__image1d_roEmT_(%spirv.SampledImage.image1d_ro_t addrspace(1)*) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.SampledImage.image1d_ro_t addrspace(1)* @_Z32__spirv_ConvertUToSampledImageNVI32__spirv_SampledImage__image1d_roET_m(i64 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4_f14ocl_image1d_roiET_T0_T1_(%opencl.image1d_ro_t addrspace(1)*, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %opencl.image2d_ro_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image2d_roET_m(i64 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func noundef <4 x float> @_Z17__spirv_ImageReadIDv4_f14ocl_image2d_roDv2_iET_T0_T1_(%opencl.image2d_ro_t addrspace(1)*, <2 x i32> noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %opencl.image3d_wo_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image3d_woET_m(i64 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z18__spirv_ImageWriteI14ocl_image3d_woDv4_iDv4_fEvT_T0_T1_(%opencl.image3d_wo_t addrspace(1)*, <4 x i32> noundef, <4 x float> noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func noundef <4 x float> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image1d_roDv4_fiET0_T_T1_if(%spirv.SampledImage.image1d_ro_t addrspace(1)*, i32 noundef, i32 noundef, float noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %spirv.SampledImage.image3d_ro_t addrspace(1)* @_Z32__spirv_ConvertUToSampledImageNVI32__spirv_SampledImage__image3d_roET_m(i64 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func noundef <4 x float> @_Z30__spirv_ImageSampleExplicitLodI32__spirv_SampledImage__image3d_roDv4_fDv4_iET0_T_T1_if(%spirv.SampledImage.image3d_ro_t addrspace(1)*, <4 x i32> noundef, i32 noundef, float noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func %opencl.image2d_wo_t addrspace(1)* @_Z25__spirv_ConvertUToImageNVI14ocl_image2d_woET_m(i64 noundef) local_unnamed_addr #1

; Function Attrs: convergent
declare dso_local spir_func void @_Z18__spirv_ImageWriteI14ocl_image2d_woDv2_iDv4_fEvT_T0_T1_(%opencl.image2d_wo_t addrspace(1)*, <2 x i32> noundef, <4 x float> noundef) local_unnamed_addr #1

declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

declare void @__itt_offload_wi_start_wrapper()

declare void @__itt_offload_wi_finish_wrapper()

attributes #0 = { convergent norecurse "frame-pointer"="all" "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="/home/duncan/documents/git/dpc/other/intel-llvm-mirror/sycl/test/extensions/bindless_images_SPIRV_inst.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!sycl_aspects = !{!4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"host", i32 0}
!5 = !{!"cpu", i32 1}
!6 = !{!"gpu", i32 2}
!7 = !{!"accelerator", i32 3}
!8 = !{!"custom", i32 4}
!9 = !{!"fp16", i32 5}
!10 = !{!"fp64", i32 6}
!11 = !{!"image", i32 9}
!12 = !{!"online_compiler", i32 10}
!13 = !{!"online_linker", i32 11}
!14 = !{!"queue_profiling", i32 12}
!15 = !{!"usm_device_allocations", i32 13}
!16 = !{!"usm_host_allocations", i32 14}
!17 = !{!"usm_shared_allocations", i32 15}
!18 = !{!"usm_restricted_shared_allocations", i32 16}
!19 = !{!"usm_system_allocations", i32 17}
!20 = !{!"ext_intel_pci_address", i32 18}
!21 = !{!"ext_intel_gpu_eu_count", i32 19}
!22 = !{!"ext_intel_gpu_eu_simd_width", i32 20}
!23 = !{!"ext_intel_gpu_slices", i32 21}
!24 = !{!"ext_intel_gpu_subslices_per_slice", i32 22}
!25 = !{!"ext_intel_gpu_eu_count_per_subslice", i32 23}
!26 = !{!"ext_intel_max_mem_bandwidth", i32 24}
!27 = !{!"ext_intel_mem_channel", i32 25}
!28 = !{!"usm_atomic_host_allocations", i32 26}
!29 = !{!"usm_atomic_shared_allocations", i32 27}
!30 = !{!"atomic64", i32 28}
!31 = !{!"ext_intel_device_info_uuid", i32 29}
!32 = !{!"ext_oneapi_srgb", i32 30}
!33 = !{!"ext_oneapi_native_assert", i32 31}
!34 = !{!"host_debuggable", i32 32}
!35 = !{!"ext_intel_gpu_hw_threads_per_eu", i32 33}
!36 = !{!"ext_oneapi_cuda_async_barrier", i32 34}
!37 = !{!"ext_oneapi_bfloat16_math_functions", i32 35}
!38 = !{!"ext_intel_free_memory", i32 36}
!39 = !{!"ext_intel_device_id", i32 37}
!40 = !{!"ext_intel_memory_clock_rate", i32 38}
!41 = !{!"ext_intel_memory_bus_width", i32 39}
!42 = !{!"int64_base_atomics", i32 7}
!43 = !{!"int64_extended_atomics", i32 8}
!44 = !{!"usm_system_allocator", i32 17}
!46 = !{i32 4342}
!47 = !{}
!48 = !{!49}
!49 = distinct !{!49, !50, !"_ZN4sycl3_V13ext6oneapi10read_imageINS0_3vecIfLi4EEEiEET_RKNS2_22unsampled_image_handleERKT0_: %agg.result"}
!50 = distinct !{!50, !"_ZN4sycl3_V13ext6oneapi10read_imageINS0_3vecIfLi4EEEiEET_RKNS2_22unsampled_image_handleERKT0_"}
!51 = !{!52, !49}
!52 = distinct !{!52, !53, !"_ZL19__invoke__ImageReadIN4sycl3_V13vecIfLi4EEE14ocl_image1d_roiET_T0_T1_: %agg.result"}
!53 = distinct !{!53, !"_ZL19__invoke__ImageReadIN4sycl3_V13vecIfLi4EEE14ocl_image1d_roiET_T0_T1_"}
!54 = !{!55}
!55 = distinct !{!55, !56, !"_ZN4sycl3_V13ext6oneapi10read_imageINS0_3vecIfLi4EEENS4_IiLi2EEEEET_RKNS2_22unsampled_image_handleERKT0_: %agg.result"}
!56 = distinct !{!56, !"_ZN4sycl3_V13ext6oneapi10read_imageINS0_3vecIfLi4EEENS4_IiLi2EEEEET_RKNS2_22unsampled_image_handleERKT0_"}
!57 = !{!58, !55}
!58 = distinct !{!58, !59, !"_ZL19__invoke__ImageReadIN4sycl3_V13vecIfLi4EEE14ocl_image2d_roNS2_IiLi2EEEET_T0_T1_: %agg.result"}
!59 = distinct !{!59, !"_ZL19__invoke__ImageReadIN4sycl3_V13vecIfLi4EEE14ocl_image2d_roNS2_IiLi2EEEET_T0_T1_"}
!60 = !{!61}
!61 = distinct !{!61, !62, !"_ZN4sycl3_V13ext6oneapi10read_imageINS0_3vecIfLi4EEEiEET_RKNS2_20sampled_image_handleERKT0_: %agg.result"}
!62 = distinct !{!62, !"_ZN4sycl3_V13ext6oneapi10read_imageINS0_3vecIfLi4EEEiEET_RKNS2_20sampled_image_handleERKT0_"}
!63 = !{!64, !61}
!64 = distinct !{!64, !65, !"_ZL29__invoke__ImageReadExpSamplerIN4sycl3_V13vecIfLi4EEE32__spirv_SampledImage__image1d_roiET_T0_T1_: %agg.result"}
!65 = distinct !{!65, !"_ZL29__invoke__ImageReadExpSamplerIN4sycl3_V13vecIfLi4EEE32__spirv_SampledImage__image1d_roiET_T0_T1_"}
!66 = !{!67}
!67 = distinct !{!67, !68, !"_ZN4sycl3_V13ext6oneapi10read_imageINS0_3vecIfLi4EEENS4_IiLi4EEEEET_RKNS2_20sampled_image_handleERKT0_: %agg.result"}
!68 = distinct !{!68, !"_ZN4sycl3_V13ext6oneapi10read_imageINS0_3vecIfLi4EEENS4_IiLi4EEEEET_RKNS2_20sampled_image_handleERKT0_"}
!69 = !{!70, !67}
!70 = distinct !{!70, !71, !"_ZL29__invoke__ImageReadExpSamplerIN4sycl3_V13vecIfLi4EEE32__spirv_SampledImage__image3d_roNS2_IiLi4EEEET_T0_T1_: %agg.result"}
!71 = distinct !{!71, !"_ZL29__invoke__ImageReadExpSamplerIN4sycl3_V13vecIfLi4EEE32__spirv_SampledImage__image3d_roNS2_IiLi4EEEET_T0_T1_"}
