define i64 @__clc__sampled_image_unpack_image(i64 %img, i32 %sampl) nounwind alwaysinline {
entry:
  ret i64 %img
}

define i32 @__clc__sampled_image_unpack_sampler(i64 %img, i32 %sampl) nounwind alwaysinline {
entry:
  ret i32 %sampl
}

define {i64, i32} @__clc__sampled_image_pack(i64 %img, i32 %sampl) nounwind alwaysinline {
entry:
  %0 = insertvalue {i64, i32} undef, i64 %img, 0
  %1 = insertvalue {i64, i32} %0, i32 %sampl, 1
  ret {i64, i32} %1
}

define i32 @__clc__sampler_extract_normalized_coords_prop(i32 %sampl) nounwind alwaysinline {
entry:
  %0 = and i32 %sampl, 1
  ret i32 %0
}

define i32 @__clc__sampler_extract_filter_mode_prop(i32 %sampl) nounwind alwaysinline {
entry:
  %0 = lshr i32 %sampl, 1
  %1 = and i32 %0, 1
  ret i32 %1
}

define i32 @__clc__sampler_extract_addressing_mode_prop(i32 %sampl) nounwind alwaysinline {
entry:
  %0 = lshr i32 %sampl, 2
  ret i32 %0
}

define <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %s) nounwind alwaysinline {
  %a = alloca {i32,i32,i32,i32}
  store {i32,i32,i32,i32} %s, {i32,i32,i32,i32}* %a
  %bc = bitcast {i32,i32,i32,i32} * %a to <4 x i32> *
  %v = load <4 x i32>, <4 x i32> * %bc, align 128
  ret <4 x i32> %v
}

define <4 x float> @__clc_structf32_to_vector({float,float,float,float} %s) nounwind alwaysinline {
  %a = alloca {float,float,float,float}
  store {float,float,float,float} %s, {float,float,float,float}* %a
  %bc = bitcast {float,float,float,float} * %a to <4 x float> *
  %v = load <4 x float>, <4 x float> * %bc, align 128
  ret <4 x float> %v
}

define <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %s) nounwind alwaysinline {
  %a = alloca {i16,i16,i16,i16}
  store {i16,i16,i16,i16} %s, {i16,i16,i16,i16}* %a
  %bc = bitcast {i16,i16,i16,i16} * %a to <4 x i16> *
  %v = load <4 x i16>, <4 x i16> * %bc, align 128
  ret <4 x i16> %v
}

// We need wrappers to convert intrisic return structures to vectors
declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.trap(i64, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_v4i16_trap(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.trap(i64 %img, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.trap(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_v4i16_trap(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.trap(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.trap(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_v4i16_trap(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.trap(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.clamp(i64, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_v4i16_clamp(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.clamp(i64 %img, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.clamp(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_v4i16_clamp(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.clamp(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.clamp(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_v4i16_clamp(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.clamp(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.zero(i64, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_1d_v4i16_zero(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.1d.v4i16.zero(i64 %img, i32 %x);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.zero(i64, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_2d_v4i16_zero(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.2d.v4i16.zero(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.zero(i64, i32, i32, i32)
define <4 x i16> @__clc_llvm_nvvm_suld_3d_v4i16_zero(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i16,i16,i16,i16} @llvm.nvvm.suld.3d.v4i16.zero(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i16> @__clc_struct16_to_vector({i16,i16,i16,i16} %0)
  ret <4 x i16> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.trap(i64, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_1d_v4i32_trap(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.trap(i64 %img, i32 %x);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.trap(i64, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_2d_v4i32_trap(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.trap(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.trap(i64, i32, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_3d_v4i32_trap(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.trap(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.clamp(i64, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_1d_v4i32_clamp(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.clamp(i64 %img, i32 %x);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.clamp(i64, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_2d_v4i32_clamp(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.clamp(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.clamp(i64, i32, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_3d_v4i32_clamp(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.clamp(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.zero(i64, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_1d_v4i32_zero(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.1d.v4i32.zero(i64 %img, i32 %x);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.zero(i64, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_2d_v4i32_zero(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.2d.v4i32.zero(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}

declare {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.zero(i64, i32, i32, i32)
define <4 x i32> @__clc_llvm_nvvm_suld_3d_v4i32_zero(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {i32,i32,i32,i32} @llvm.nvvm.suld.3d.v4i32.zero(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x i32> @__clc_struct32_to_vector({i32,i32,i32,i32} %0)
  ret <4 x i32> %1
}


declare {float,float,float,float} @llvm.nvvm.tex.unified.1d.v4f32.s32(i64, i32)
define <4 x float> @__clc_llvm_nvvm_tex_1d_v4f32_s32(i64 %img, i32 %x) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.1d.v4f32.s32(i64 %img, i32 %x);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.2d.v4f32.s32(i64, i32, i32)
define <4 x float> @__clc_llvm_nvvm_tex_2d_v4f32_s32(i64 %img, i32 %x, i32 %y) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.2d.v4f32.s32(i64 %img, i32 %x, i32 %y);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.3d.v4f32.s32(i64, i32, i32, i32)
define <4 x float> @__clc_llvm_nvvm_tex_3d_v4f32_s32(i64 %img, i32 %x, i32 %y, i32 %z) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.3d.v4f32.s32(i64 %img, i32 %x, i32 %y, i32 %z);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.1d.v4f32.f32(i64, float)
define <4 x float> @__clc_llvm_nvvm_tex_1d_v4f32_f32(i64 %img, float %x) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.1d.v4f32.f32(i64 %img, float %x);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.2d.v4f32.f32(i64, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_2d_v4f32_f32(i64 %img, float %x, float %y) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.2d.v4f32.f32(i64 %img, float %x, float %y);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare {float,float,float,float} @llvm.nvvm.tex.unified.3d.v4f32.f32(i64, float, float, float)
define <4 x float> @__clc_llvm_nvvm_tex_3d_v4f32_f32(i64 %img, float %x, float %y, float %z) nounwind alwaysinline {
entry:
  %0 = tail call {float,float,float,float} @llvm.nvvm.tex.unified.3d.v4f32.f32(i64 %img, float %x, float %y, float %z);
  %1 = tail call <4 x float>@__clc_structf32_to_vector({float,float,float,float} %0)
  ret <4 x float> %1
}

declare void @llvm.nvvm.sust.p.1d.v4i32.trap(i64, i32, i32, i32, i32, i32)
define void @__clc_llvm_nvv_sust_p_1d_v4i32_trap(i64 %img, i32 %x, float %r, float %g, float %b, float %a) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  %gi = bitcast float %g to i32
  %bi = bitcast float %b to i32
  %ai = bitcast float %a to i32
  call void @llvm.nvvm.sust.p.1d.v4i32.trap(i64 %img, i32 %x, i32 %ri, i32 %gi, i32 %bi, i32 %ai);
  ret void
}

declare void @llvm.nvvm.sust.p.2d.v4i32.trap(i64, i32, i32, i32, i32, i32, i32)
define void @__clc_llvm_nvv_sust_p_2d_v4i32_trap(i64 %img, i32 %x, i32 %y, float %r, float %g, float %b, float %a) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  %gi = bitcast float %g to i32
  %bi = bitcast float %b to i32
  %ai = bitcast float %a to i32
  call void @llvm.nvvm.sust.p.2d.v4i32.trap(i64 %img, i32 %x, i32 %y, i32 %ri, i32 %gi, i32 %bi, i32 %ai);
  ret void
}

declare void @llvm.nvvm.sust.p.3d.v4i32.trap(i64, i32, i32, i32, i32, i32, i32, i32)
define void @__clc_llvm_nvv_sust_p_3d_v4i32_trap(i64 %img, i32 %x, i32 %y, i32 %z, float %r, float %g, float %b, float %a) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  %gi = bitcast float %g to i32
  %bi = bitcast float %b to i32
  %ai = bitcast float %a to i32
  call void @llvm.nvvm.sust.p.3d.v4i32.trap(i64 %img, i32 %x, i32 %y, i32 %z, i32 %ri, i32 %gi, i32 %bi, i32 %ai);
  ret void
}

declare void @llvm.nvvm.sust.p.1d.i32.trap(i64, i32, i32)
define void @__clc_llvm_nvv_sust_p_1d_i32_trap(i64 %img, i32 %x, float %r) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  call void @llvm.nvvm.sust.p.1d.i32.trap(i64 %img, i32 %x, i32 %ri);
  ret void
}

declare void @llvm.nvvm.sust.p.2d.i32.trap(i64, i32, i32, i32)
define void @__clc_llvm_nvv_sust_p_2d_i32_trap(i64 %img, i32 %x, i32 %y, float %r) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  call void @llvm.nvvm.sust.p.2d.i32.trap(i64 %img, i32 %x, i32 %y, i32 %ri);
  ret void
}

declare void @llvm.nvvm.sust.p.3d.i32.trap(i64, i32, i32, i32, i32)
define void @__clc_llvm_nvv_sust_p_3d_i32_trap(i64 %img, i32 %x, i32 %y, i32 %z, float %r) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  call void @llvm.nvvm.sust.p.3d.i32.trap(i64 %img, i32 %x, i32 %y, i32 %z, i32 %ri);
  ret void
}

declare void @llvm.nvvm.sust.p.1d.v2i32.trap(i64, i32, i32, i32)
define void @__clc_llvm_nvv_sust_p_1d_v2i32_trap(i64 %img, i32 %x, float %r, float %g) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  %gi = bitcast float %g to i32
  call void @llvm.nvvm.sust.p.1d.v2i32.trap(i64 %img, i32 %x, i32 %ri, i32 %gi);
  ret void
}

declare void @llvm.nvvm.sust.p.2d.v2i32.trap(i64, i32, i32, i32, i32)
define void @__clc_llvm_nvv_sust_p_2d_v2i32_trap(i64 %img, i32 %x, i32 %y, float %r, float %g) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  %gi = bitcast float %g to i32
  call void @llvm.nvvm.sust.p.2d.v2i32.trap(i64 %img, i32 %x, i32 %y, i32 %ri, i32 %gi);
  ret void
}

declare void @llvm.nvvm.sust.p.3d.v2i32.trap(i64, i32, i32, i32, i32, i32)
define void @__clc_llvm_nvv_sust_p_3d_v2i32_trap(i64 %img, i32 %x, i32 %y, i32 %z, float %r, float %g) nounwind alwaysinline {
entry:
  %ri = bitcast float %r to i32
  %gi = bitcast float %g to i32
  call void @llvm.nvvm.sust.p.3d.v2i32.trap(i64 %img, i32 %x, i32 %y, i32 %z, i32 %ri, i32 %gi);
  ret void
}
