; RUN: llc < %s -march=nvptx -mcpu=sm_70 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_70 | FileCheck %s

; CHECK-LABEL: .func test_atomics_scope(
define void @test_atomics_scope(float* %fp, float %f,
                                double* %dfp, double %df,
                                i32* %ip, i32 %i,
                                i32* %uip, i32 %ui,
                                i64* %llp, i64 %ll) #0 {
entry:


  ; CHECK: atom.acquire.add.s32
  %tmp0 = tail call i32 @llvm.nvvm.atomic.add.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.add.s32
  %tmp1 = tail call i32 @llvm.nvvm.atomic.add.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.add.s32
  %tmp2 = tail call i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.add.s32
  %tmp3 = tail call i32 @llvm.nvvm.atomic.add.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.add.s32
  %tmp4 = tail call i32 @llvm.nvvm.atomic.add.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.add.s32
  %tmp5 = tail call i32 @llvm.nvvm.atomic.add.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.add.s32
  %tmp6 = tail call i32 @llvm.nvvm.atomic.add.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.add.s32
  %tmp7 = tail call i32 @llvm.nvvm.atomic.add.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.add.s32
  %tmp8 = tail call i32 @llvm.nvvm.atomic.add.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.add.u64
  %tmp9 = tail call i64 @llvm.nvvm.atomic.add.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.add.u64
  %tmp10 = tail call i64 @llvm.nvvm.atomic.add.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.add.u64
  %tmp11 = tail call i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.add.u64
  %tmp12 = tail call i64 @llvm.nvvm.atomic.add.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.add.u64
  %tmp13 = tail call i64 @llvm.nvvm.atomic.add.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.add.u64
  %tmp14 = tail call i64 @llvm.nvvm.atomic.add.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.add.u64
  %tmp15 = tail call i64 @llvm.nvvm.atomic.add.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.add.u64
  %tmp16 = tail call i64 @llvm.nvvm.atomic.add.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.add.u64
  %tmp17 = tail call i64 @llvm.nvvm.atomic.add.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.add.f32
  %tmp18 = tail call float @llvm.nvvm.atomic.add.gen.f.acquire.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.release.add.f32
  %tmp19 = tail call float @llvm.nvvm.atomic.add.gen.f.release.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acq_rel.add.f32
  %tmp20 = tail call float @llvm.nvvm.atomic.add.gen.f.acq.rel.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acquire.sys.add.f32
  %tmp21 = tail call float @llvm.nvvm.atomic.add.gen.f.sys.acquire.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.release.sys.add.f32
  %tmp22 = tail call float @llvm.nvvm.atomic.add.gen.f.sys.release.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acq_rel.sys.add.f32
  %tmp23 = tail call float @llvm.nvvm.atomic.add.gen.f.sys.acq.rel.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acquire.cta.add.f32
  %tmp24 = tail call float @llvm.nvvm.atomic.add.gen.f.cta.acquire.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.release.cta.add.f32
  %tmp25 = tail call float @llvm.nvvm.atomic.add.gen.f.cta.release.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acq_rel.cta.add.f32
  %tmp26 = tail call float @llvm.nvvm.atomic.add.gen.f.cta.acq.rel.f32.p0f32(float* %fp, float %f);

  ; CHECK: atom.acquire.add.f64
  %tmp27 = tail call double @llvm.nvvm.atomic.add.gen.f.acquire.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.release.add.f64
  %tmp28 = tail call double @llvm.nvvm.atomic.add.gen.f.release.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acq_rel.add.f64
  %tmp29 = tail call double @llvm.nvvm.atomic.add.gen.f.acq.rel.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acquire.sys.add.f64
  %tmp30 = tail call double @llvm.nvvm.atomic.add.gen.f.sys.acquire.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.release.sys.add.f64
  %tmp31 = tail call double @llvm.nvvm.atomic.add.gen.f.sys.release.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acq_rel.sys.add.f64
  %tmp32 = tail call double @llvm.nvvm.atomic.add.gen.f.sys.acq.rel.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acquire.cta.add.f64
  %tmp33 = tail call double @llvm.nvvm.atomic.add.gen.f.cta.acquire.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.release.cta.add.f64
  %tmp34 = tail call double @llvm.nvvm.atomic.add.gen.f.cta.release.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acq_rel.cta.add.f64
  %tmp35 = tail call double @llvm.nvvm.atomic.add.gen.f.cta.acq.rel.f64.p0f64(double* %dfp, double %df);

  ; CHECK: atom.acquire.exch.b32
  %tmp36 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.exch.b32
  %tmp37 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.exch.b32
  %tmp38 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.exch.b32
  %tmp39 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.exch.b32
  %tmp40 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.exch.b32
  %tmp41 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.exch.b32
  %tmp42 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.exch.b32
  %tmp43 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.exch.b32
  %tmp44 = tail call i32 @llvm.nvvm.atomic.exch.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.exch.b64
  %tmp45 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.exch.b64
  %tmp46 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.exch.b64
  %tmp47 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.exch.b64
  %tmp48 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.exch.b64
  %tmp49 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.exch.b64
  %tmp50 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.exch.b64
  %tmp51 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.exch.b64
  %tmp52 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.exch.b64
  %tmp53 = tail call i64 @llvm.nvvm.atomic.exch.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.max.s32
  %tmp54 = tail call i32 @llvm.nvvm.atomic.max.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.max.s32
  %tmp55 = tail call i32 @llvm.nvvm.atomic.max.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.max.s32
  %tmp56 = tail call i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.max.s32
  %tmp57 = tail call i32 @llvm.nvvm.atomic.max.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.max.s32
  %tmp58 = tail call i32 @llvm.nvvm.atomic.max.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.max.s32
  %tmp59 = tail call i32 @llvm.nvvm.atomic.max.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.max.s32
  %tmp60 = tail call i32 @llvm.nvvm.atomic.max.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.max.s32
  %tmp61 = tail call i32 @llvm.nvvm.atomic.max.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.max.s32
  %tmp62 = tail call i32 @llvm.nvvm.atomic.max.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.max.s64
  %tmp63 = tail call i64 @llvm.nvvm.atomic.max.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.max.s64
  %tmp64 = tail call i64 @llvm.nvvm.atomic.max.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.max.s64
  %tmp65 = tail call i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.max.s64
  %tmp66 = tail call i64 @llvm.nvvm.atomic.max.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.max.s64
  %tmp67 = tail call i64 @llvm.nvvm.atomic.max.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.max.s64
  %tmp68 = tail call i64 @llvm.nvvm.atomic.max.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.max.s64
  %tmp69 = tail call i64 @llvm.nvvm.atomic.max.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.max.s64
  %tmp70 = tail call i64 @llvm.nvvm.atomic.max.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.max.s64
  %tmp71 = tail call i64 @llvm.nvvm.atomic.max.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.max.u32
  %tmp72 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.max.u32
  %tmp73 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.max.u32
  %tmp74 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.max.u32
  %tmp75 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.max.u32
  %tmp76 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.max.u32
  %tmp77 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.max.u32
  %tmp78 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.max.u32
  %tmp79 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.max.u32
  %tmp80 = tail call i32 @llvm.nvvm.atomic.max.gen.ui.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.max.u64
  %tmp81 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.max.u64
  %tmp82 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.max.u64
  %tmp83 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.max.u64
  %tmp84 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.max.u64
  %tmp85 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.max.u64
  %tmp86 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.max.u64
  %tmp87 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.max.u64
  %tmp88 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.max.u64
  %tmp89 = tail call i64 @llvm.nvvm.atomic.max.gen.ui.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.min.s32
  %tmp90 = tail call i32 @llvm.nvvm.atomic.min.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.min.s32
  %tmp91 = tail call i32 @llvm.nvvm.atomic.min.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.min.s32
  %tmp92 = tail call i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.min.s32
  %tmp93 = tail call i32 @llvm.nvvm.atomic.min.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.min.s32
  %tmp94 = tail call i32 @llvm.nvvm.atomic.min.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.min.s32
  %tmp95 = tail call i32 @llvm.nvvm.atomic.min.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.min.s32
  %tmp96 = tail call i32 @llvm.nvvm.atomic.min.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.min.s32
  %tmp97 = tail call i32 @llvm.nvvm.atomic.min.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.min.s32
  %tmp98 = tail call i32 @llvm.nvvm.atomic.min.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.min.s64
  %tmp99 = tail call i64 @llvm.nvvm.atomic.min.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.min.s64
  %tmp100 = tail call i64 @llvm.nvvm.atomic.min.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.min.s64
  %tmp101 = tail call i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.min.s64
  %tmp102 = tail call i64 @llvm.nvvm.atomic.min.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.min.s64
  %tmp103 = tail call i64 @llvm.nvvm.atomic.min.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.min.s64
  %tmp104 = tail call i64 @llvm.nvvm.atomic.min.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.min.s64
  %tmp105 = tail call i64 @llvm.nvvm.atomic.min.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.min.s64
  %tmp106 = tail call i64 @llvm.nvvm.atomic.min.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.min.s64
  %tmp107 = tail call i64 @llvm.nvvm.atomic.min.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.min.u32
  %tmp108 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.min.u32
  %tmp109 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.min.u32
  %tmp110 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.min.u32
  %tmp111 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.min.u32
  %tmp112 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.min.u32
  %tmp113 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.min.u32
  %tmp114 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.min.u32
  %tmp115 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.min.u32
  %tmp116 = tail call i32 @llvm.nvvm.atomic.min.gen.ui.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.min.u64
  %tmp117 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.min.u64
  %tmp118 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.min.u64
  %tmp119 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.min.u64
  %tmp120 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.min.u64
  %tmp121 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.min.u64
  %tmp122 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.min.u64
  %tmp123 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.min.u64
  %tmp124 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.min.u64
  %tmp125 = tail call i64 @llvm.nvvm.atomic.min.gen.ui.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.inc.u32
  %tmp126 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.inc.u32
  %tmp127 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.inc.u32
  %tmp128 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.inc.u32
  %tmp129 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.inc.u32
  %tmp130 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.inc.u32
  %tmp131 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.inc.u32
  %tmp132 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.inc.u32
  %tmp133 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.inc.u32
  %tmp134 = tail call i32 @llvm.nvvm.atomic.inc.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.inc.u64
  %tmp135 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.inc.u64
  %tmp136 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.inc.u64
  %tmp137 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.inc.u64
  %tmp138 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.inc.u64
  %tmp139 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.inc.u64
  %tmp140 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.inc.u64
  %tmp141 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.inc.u64
  %tmp142 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.inc.u64
  %tmp143 = tail call i64 @llvm.nvvm.atomic.inc.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.dec.u32
  %tmp144 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.dec.u32
  %tmp145 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.dec.u32
  %tmp146 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.dec.u32
  %tmp147 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.dec.u32
  %tmp148 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.dec.u32
  %tmp149 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.dec.u32
  %tmp150 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.dec.u32
  %tmp151 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.dec.u32
  %tmp152 = tail call i32 @llvm.nvvm.atomic.dec.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.dec.u64
  %tmp153 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.dec.u64
  %tmp154 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.dec.u64
  %tmp155 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.dec.u64
  %tmp156 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.dec.u64
  %tmp157 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.dec.u64
  %tmp158 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.dec.u64
  %tmp159 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.dec.u64
  %tmp160 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.dec.u64
  %tmp161 = tail call i64 @llvm.nvvm.atomic.dec.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.and.b32
  %tmp162 = tail call i32 @llvm.nvvm.atomic.and.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.and.b32
  %tmp163 = tail call i32 @llvm.nvvm.atomic.and.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.and.b32
  %tmp164 = tail call i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.and.b32
  %tmp165 = tail call i32 @llvm.nvvm.atomic.and.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.and.b32
  %tmp166 = tail call i32 @llvm.nvvm.atomic.and.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.and.b32
  %tmp167 = tail call i32 @llvm.nvvm.atomic.and.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.and.b32
  %tmp168 = tail call i32 @llvm.nvvm.atomic.and.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.and.b32
  %tmp169 = tail call i32 @llvm.nvvm.atomic.and.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.and.b32
  %tmp170 = tail call i32 @llvm.nvvm.atomic.and.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.and.b64
  %tmp171 = tail call i64 @llvm.nvvm.atomic.and.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.and.b64
  %tmp172 = tail call i64 @llvm.nvvm.atomic.and.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.and.b64
  %tmp173 = tail call i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.and.b64
  %tmp174 = tail call i64 @llvm.nvvm.atomic.and.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.and.b64
  %tmp175 = tail call i64 @llvm.nvvm.atomic.and.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.and.b64
  %tmp176 = tail call i64 @llvm.nvvm.atomic.and.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.and.b64
  %tmp177 = tail call i64 @llvm.nvvm.atomic.and.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.and.b64
  %tmp178 = tail call i64 @llvm.nvvm.atomic.and.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.and.b64
  %tmp179 = tail call i64 @llvm.nvvm.atomic.and.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.or.b32
  %tmp180 = tail call i32 @llvm.nvvm.atomic.or.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.or.b32
  %tmp181 = tail call i32 @llvm.nvvm.atomic.or.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.or.b32
  %tmp182 = tail call i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.or.b32
  %tmp183 = tail call i32 @llvm.nvvm.atomic.or.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.or.b32
  %tmp184 = tail call i32 @llvm.nvvm.atomic.or.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.or.b32
  %tmp185 = tail call i32 @llvm.nvvm.atomic.or.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.or.b32
  %tmp186 = tail call i32 @llvm.nvvm.atomic.or.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.or.b32
  %tmp187 = tail call i32 @llvm.nvvm.atomic.or.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.or.b32
  %tmp188 = tail call i32 @llvm.nvvm.atomic.or.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.or.b64
  %tmp189 = tail call i64 @llvm.nvvm.atomic.or.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.or.b64
  %tmp190 = tail call i64 @llvm.nvvm.atomic.or.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.or.b64
  %tmp191 = tail call i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.or.b64
  %tmp192 = tail call i64 @llvm.nvvm.atomic.or.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.or.b64
  %tmp193 = tail call i64 @llvm.nvvm.atomic.or.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.or.b64
  %tmp194 = tail call i64 @llvm.nvvm.atomic.or.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.or.b64
  %tmp195 = tail call i64 @llvm.nvvm.atomic.or.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.or.b64
  %tmp196 = tail call i64 @llvm.nvvm.atomic.or.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.or.b64
  %tmp197 = tail call i64 @llvm.nvvm.atomic.or.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.xor.b32
  %tmp198 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.xor.b32
  %tmp199 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.xor.b32
  %tmp200 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.sys.xor.b32
  %tmp201 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.sys.xor.b32
  %tmp202 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.sys.xor.b32
  %tmp203 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.cta.xor.b32
  %tmp204 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.release.cta.xor.b32
  %tmp205 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acq_rel.cta.xor.b32
  %tmp206 = tail call i32 @llvm.nvvm.atomic.xor.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i);

  ; CHECK: atom.acquire.xor.b64
  %tmp207 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.xor.b64
  %tmp208 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.xor.b64
  %tmp209 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.sys.xor.b64
  %tmp210 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.sys.xor.b64
  %tmp211 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.sys.xor.b64
  %tmp212 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cta.xor.b64
  %tmp213 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.release.cta.xor.b64
  %tmp214 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acq_rel.cta.xor.b64
  %tmp215 = tail call i64 @llvm.nvvm.atomic.xor.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll);

  ; CHECK: atom.acquire.cas.b32
  %tmp216 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.acquire.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.release.cas.b32
  %tmp217 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.release.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.cas.b32
  %tmp218 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acquire.sys.cas.b32
  %tmp219 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.sys.acquire.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.release.sys.cas.b32
  %tmp220 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.sys.release.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.sys.cas.b32
  %tmp221 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.sys.acq.rel.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acquire.cta.cas.b32
  %tmp222 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.cta.acquire.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.release.cta.cas.b32
  %tmp223 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.cta.release.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acq_rel.cta.cas.b32
  %tmp224 = tail call i32 @llvm.nvvm.atomic.cas.gen.i.cta.acq.rel.i32.p0i32(i32* %ip, i32 %i, i32 %i);

  ; CHECK: atom.acquire.cas.b64
  %tmp225 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.acquire.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.release.cas.b64
  %tmp226 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.release.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.cas.b64
  %tmp227 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.sys.cas.b64
  %tmp228 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.sys.acquire.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.release.sys.cas.b64
  %tmp229 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.sys.release.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.sys.cas.b64
  %tmp230 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.sys.acq.rel.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acquire.cta.cas.b64
  %tmp231 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.cta.acquire.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.release.cta.cas.b64
  %tmp232 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.cta.release.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);

  ; CHECK: atom.acq_rel.cta.cas.b64
  %tmp233 = tail call i64 @llvm.nvvm.atomic.cas.gen.i.cta.acq.rel.i64.p0i64(i64* %llp, i64 %ll, i64 %ll);


  ; CHECK: ret
  ret void
}

declare i32 @llvm.nvvm.atomic.add.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.add.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.add.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare float @llvm.nvvm.atomic.add.gen.f.acquire.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.release.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.acq.rel.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.sys.acquire.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.sys.release.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.sys.acq.rel.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.cta.acquire.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.cta.release.f32.p0f32(float* nocapture, float) #1
declare float @llvm.nvvm.atomic.add.gen.f.cta.acq.rel.f32.p0f32(float* nocapture, float) #1
declare double @llvm.nvvm.atomic.add.gen.f.acquire.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.release.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.acq.rel.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.sys.acquire.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.sys.release.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.sys.acq.rel.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.cta.acquire.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.cta.release.f64.p0f64(double* nocapture, double) #1
declare double @llvm.nvvm.atomic.add.gen.f.cta.acq.rel.f64.p0f64(double* nocapture, double) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.exch.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.exch.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.max.gen.ui.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.max.gen.ui.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.min.gen.ui.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.min.gen.ui.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.inc.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.inc.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.dec.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.dec.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.and.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.and.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.or.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.or.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.sys.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.cta.release.i32.p0i32(i32* nocapture, i32) #1
declare i32 @llvm.nvvm.atomic.xor.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.sys.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.cta.release.i64.p0i64(i64* nocapture, i64) #1
declare i64 @llvm.nvvm.atomic.xor.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.acquire.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.release.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.sys.acquire.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.sys.release.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.sys.acq.rel.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.cta.acquire.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.cta.release.i32.p0i32(i32* nocapture, i32, i32) #1
declare i32 @llvm.nvvm.atomic.cas.gen.i.cta.acq.rel.i32.p0i32(i32* nocapture, i32, i32) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.acquire.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.release.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.acq.rel.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.sys.acquire.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.sys.release.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.sys.acq.rel.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.cta.acquire.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.cta.release.i64.p0i64(i64* nocapture, i64, i64) #1
declare i64 @llvm.nvvm.atomic.cas.gen.i.cta.acq.rel.i64.p0i64(i64* nocapture, i64, i64) #1

attributes #1 = { argmemonly nounwind }