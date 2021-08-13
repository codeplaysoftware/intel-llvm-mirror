; RUN: opt -enable-new-pm=0 -kernel-args-const-promotion %s -S -o - | FileCheck %s
; ModuleID = 'basic-transformation.bc'
source_filename = "kernel-args-const-promotion.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda-sycldevice"

; This test checks that the transformation is applied in the basic case.
; TODO: JOE: Add attribute to the list.

; CHECK: %_TK_kacp_struct_ty
; CHECK: @_TK_kacp_struct_data = linkonce_odr addrspace(4) constant
define weak_odr dso_local void @_TK(i32 addrspace(3)* %a, i32 addrspace(1)* %b, i32 %c) local_unnamed_addr #0  {
; CHECK: define weak_odr dso_local void @_TK_kacp() local_unnamed_addr #0 {
entry:
; CHECK: %Arg_GEP_0 = getelementptr
; CHECK: %Arg_Load_0 = load
; CHECK-NO: %a
; CHECK-NO: %b
; CHECK-NO: %c
  %0 = load i32, i32 addrspace(3)* %a
  %1 = load i32, i32 addrspace(1)* %b
  %2 = add i32 %0, %1
  %3 = add i32 %0, %c
  ret void
}

attributes #0 = { noinline norecurse "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="attr_test.cpp" "target-cpu"="sm_50" "target-features"="+ptx72,+sm_50" "uniform-work-group-size"="true" "kernel-const-mem" }

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!nvvmir.version = !{!5}

; CHECK: !0 = distinct !{void ()* @_TK_kacp, !"kernel", i32 1
!0 = distinct !{void (i32 addrspace(3)*, i32 addrspace(1)*, i32)* @_TK, !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{i32 1, i32 4}
