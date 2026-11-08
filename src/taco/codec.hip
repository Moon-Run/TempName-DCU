// TACO H128/E4M3FN arithmetic adapted for HIP on DCU, 2026-11-08.
// Algorithm source: COCCL_TACO ca717e05b6dd6d21fff4d2617d209c95be63f3ef.
// Copyright (c) 2025, Institute of Computing Technology, Chinese Academy of Sciences.
// SPDX-License-Identifier: BSD-3-Clause
// See third_party/taco/LICENSE.txt and NOTICE. This is a new HIP implementation;
// it uses shared-memory collectives and software E4M3FN, not CUDA warp intrinsics.
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
constexpr int kGroup = 128;
constexpr int kAbi = 2;

__device__ __forceinline__ uint32_t bits_of(float x) {
  union { float f; uint32_t u; } value;
  value.f = x;
  return value.u;
}
__device__ __forceinline__ float from_bits(uint32_t x) {
  union { float f; uint32_t u; } value;
  value.u = x;
  return value.f;
}
__device__ __forceinline__ float read_bf16(uint16_t x) {
  return from_bits(uint32_t(x) << 16);
}
__device__ __forceinline__ uint16_t write_bf16(float x) {
  uint32_t bits = bits_of(x);
  if ((bits & 0x7fffffffU) > 0x7f800000U)
    return uint16_t((bits >> 16) | 0x0040U); // Preserve a quiet NaN.
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return uint16_t(bits >> 16);
}
__device__ __forceinline__ float clamp(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}

// Exact RNE conversion for finite FP32 values, with finite saturation at 448.
// E4M3FN uses bias 7; 0x7e is 448, 0x7f is NaN. Signed zero is retained.
__device__ __forceinline__ uint8_t to_e4m3(float x) {
  const uint32_t sign = (bits_of(x) >> 24) & 0x80U;
  float mag = fminf(fabsf(x), 448.0f);
  uint32_t code;
  if (mag < 0.015625f) {
    const float scaled = mag * 512.0f;
    uint32_t lower = uint32_t(scaled);
    const float remainder = scaled - float(lower);
    lower += (remainder > 0.5f || (remainder == 0.5f && (lower & 1U)));
    code = lower;
  } else {
    const uint32_t bits = bits_of(mag);
    code = ((bits + 0x7ffffU + ((bits >> 20) & 1U)) >> 20) - 960U;
  }
  return uint8_t(sign | code);
}
__device__ __forceinline__ float from_e4m3(uint8_t code) {
  const unsigned mag = code & 0x7fU;
  float value;
  if (mag == 0x7fU) value = from_bits(0x7fc00000U);
  else if (mag < 8U) value = float(mag) * 0.001953125f;
  else value = ldexpf(1.0f + float(mag & 7U) * 0.125f, int(mag >> 3) - 7);
  return (code & 0x80U) ? -value : value;
}

__device__ __forceinline__ void hadamard(float* values, unsigned tid) {
  // All 128 threads participate. No warp/wave size assumption.
  for (unsigned stride = 1; stride < kGroup; stride <<= 1) {
    const float own = values[tid];
    const float other = values[tid ^ stride];
    __syncthreads();
    values[tid] = (tid & stride) ? other - own : own + other;
    __syncthreads();
  }
}

__global__ void encode_kernel(const uint16_t* input, uint8_t* packet, int64_t elements) {
  const unsigned tid = threadIdx.x;
  const int64_t group = blockIdx.x;
  const int64_t index = group * kGroup + tid;
  const int64_t groups = elements / kGroup;
  auto* quant_scales = reinterpret_cast<float*>(packet + elements);
  auto* adaptive_scales = quant_scales + groups;
  __shared__ float values[kGroup];
  __shared__ float scratch[kGroup];
  __shared__ float adaptive;
  __shared__ float quant;
  const float x = read_bf16(input[index]);
  scratch[tid] = x * x;
  __syncthreads();
  for (unsigned stride = kGroup / 2; stride; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  if (!tid) {
    const float variance = fmaxf(scratch[0] / float(kGroup) + 1e-6f, 1e-12f);
    adaptive = clamp(448.0f * rsqrtf(variance), 1e-3f, 1e3f);
    adaptive_scales[group] = adaptive;
  }
  __syncthreads();
  values[tid] = x * adaptive;
  __syncthreads();
  hadamard(values, tid);
  values[tid] *= 0.08838834764831844055f; // 1/sqrt(128)
  scratch[tid] = fabsf(values[tid]);
  __syncthreads();
  for (unsigned stride = kGroup / 2; stride; stride >>= 1) {
    if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
    __syncthreads();
  }
  if (!tid) {
    quant = clamp(fmaxf(scratch[0], 1e-12f) / 448.0f, 1e-12f, 1e6f);
    quant_scales[group] = quant;
  }
  __syncthreads();
  const float finite = isfinite(values[tid]) ? values[tid] : 0.0f;
  packet[index] = to_e4m3(clamp(finite / quant, -448.0f, 448.0f));
}

__global__ void decode_kernel(const uint8_t* packet, uint16_t* output, int64_t elements) {
  const unsigned tid = threadIdx.x;
  const int64_t group = blockIdx.x;
  const int64_t index = group * kGroup + tid;
  const int64_t groups = elements / kGroup;
  const auto* quant_scales = reinterpret_cast<const float*>(packet + elements);
  const auto* adaptive_scales = quant_scales + groups;
  __shared__ float values[kGroup];
  values[tid] = from_e4m3(packet[index]) * quant_scales[group];
  __syncthreads();
  hadamard(values, tid);
  const float value = values[tid] * 0.08838834764831844055f;
  output[index] = write_bf16((isfinite(value) ? value : 0.0f) /
                             fmaxf(adaptive_scales[group], 1e-12f));
}

__global__ void reduce_kernel(const uint16_t* local, const uint16_t* remote,
                              uint16_t* output, int64_t elements, int rank, int world) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  uint16_t accum = 0;
  for (int source = 0; source < world; ++source) {
    const int slot = source < rank ? source : source - 1;
    const uint16_t value = source == rank ? local[index] : remote[int64_t(slot) * elements + index];
    accum = write_bf16(read_bf16(accum) + read_bf16(value));
  }
  output[index] = accum;
}

bool valid_elements(int64_t n) {
  return n > 0 && n % kGroup == 0 && n / kGroup <= 0x7fffffffLL;
}

// Diagnostic only: one thread holds one stream for a bounded device-clock
// interval. This kernel is never part of normal forward or timed training.
__global__ void diagnostic_delay_kernel(unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) { }
}
} // namespace

extern "C" int flux_dcu_codec_abi() { return kAbi; }
extern "C" int dcu_codec_abi() { return kAbi; }
extern "C" const char* flux_dcu_codec_error(int error) {
  return hipGetErrorString(static_cast<hipError_t>(error));
}
extern "C" int flux_dcu_diagnostic_delay(uint64_t cycles, void* stream) {
  if (cycles == 0 || cycles > 100000000ULL) return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(diagnostic_delay_kernel, dim3(1), dim3(1), 0,
                    reinterpret_cast<hipStream_t>(stream), static_cast<unsigned long long>(cycles));
  return int(hipGetLastError());
}
extern "C" int flux_dcu_encode(const void* input, void* packet, int64_t elements, void* stream) {
  if (!input || !packet || input == packet || !valid_elements(elements)) return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(encode_kernel, dim3(unsigned(elements / kGroup)), dim3(kGroup), 0,
                    reinterpret_cast<hipStream_t>(stream), static_cast<const uint16_t*>(input),
                    static_cast<uint8_t*>(packet), elements);
  return int(hipGetLastError());
}
extern "C" int flux_dcu_decode(const void* packet, void* output, int64_t elements, void* stream) {
  if (!packet || !output || packet == output || !valid_elements(elements)) return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(decode_kernel, dim3(unsigned(elements / kGroup)), dim3(kGroup), 0,
                    reinterpret_cast<hipStream_t>(stream), static_cast<const uint8_t*>(packet),
                    static_cast<uint16_t*>(output), elements);
  return int(hipGetLastError());
}
extern "C" int flux_dcu_reduce(const void* local, const void* remote, void* output,
                               int64_t elements, int rank, int world, void* stream) {
  if (!local || !remote || !output || elements <= 0 || elements > 0x7fffffffLL * 256 ||
      world < 2 || world > 128 || rank < 0 || rank >= world || output == remote)
    return int(hipErrorInvalidValue);
  hipLaunchKernelGGL(reduce_kernel, dim3(unsigned((elements + 255) / 256)), dim3(256), 0,
                    reinterpret_cast<hipStream_t>(stream), static_cast<const uint16_t*>(local),
                    static_cast<const uint16_t*>(remote), static_cast<uint16_t*>(output),
                    elements, rank, world);
  return int(hipGetLastError());
}
