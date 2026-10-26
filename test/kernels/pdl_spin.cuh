// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// A stand-in, for test_minimax_m31_attn_ll.py's serving chain, for the kernel
// before the M3.1 attention's q|k|v|index GEMM in a served verify step: slot
// hop_mxfp8's attention-input boundary kernel. Launched with PDL, it signals
// its dependent's launch at its entry (`early`, as that kernel does when slot
// attn_ll's low-latency GEMM follows it) or only at its end, waits for its own
// predecessor, runs `spin_ns` nanoseconds (the boundary's all-reduce, ~6 us at
// 8 rows), then writes its dependent's input, `dst` <- `src` (the MXFP8 pair
// the boundary writes), so a dependent that read its input before its own PDL
// wait would read stale bytes.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

namespace sglang {

__device__ __forceinline__ uint64_t pdl_spin_globaltimer() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__global__ void pdl_spin_copy_kernel(const uint4 *src, uint4 *dst,
                                     int64_t num_vecs, uint64_t spin_ns,
                                     bool early) {
  if (early)
    device::PDLTriggerSecondary<true>();
  device::PDLWaitPrimary<true>();
  const uint64_t start = pdl_spin_globaltimer();
  while (pdl_spin_globaltimer() - start < spin_ns) {
  }
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_vecs; i += stride) {
    dst[i] = src[i];
  }
  device::PDLTriggerSecondary<true>();
}

// `src` and `dst`: 1-D uint8 of the same size, a multiple of 16 bytes, 16-byte
// aligned. `blocks` CTAs of 128 threads.
void pdl_spin_copy(tvm::ffi::TensorView src, tvm::ffi::TensorView dst,
                   int64_t spin_ns, bool early, int64_t blocks) {
  using namespace host;
  auto num_bytes = SymbolicSize{"num_bytes"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({num_bytes})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(src)
      .verify(dst);
  const auto bytes = num_bytes.unwrap();
  RuntimeCheck(bytes % 16 == 0, "pdl_spin_copy: ", bytes,
               " bytes is not a whole number of 16-byte vectors");
  RuntimeCheck(reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 == 0,
               "pdl_spin_copy: src and dst must be 16-byte aligned");
  RuntimeCheck(spin_ns >= 0 && blocks > 0,
               "pdl_spin_copy: spin_ns >= 0 and blocks > 0");
  LaunchKernel(dim3(static_cast<uint32_t>(blocks)), dim3(128), device.unwrap())
      .enable_pdl(true)(pdl_spin_copy_kernel,
                        static_cast<const uint4 *>(src.data_ptr()),
                        static_cast<uint4 *>(dst.data_ptr()), bytes / 16,
                        static_cast<uint64_t>(spin_ns), early);
}

} // namespace sglang
