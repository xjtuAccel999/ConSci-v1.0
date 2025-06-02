// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NCNN_CPU_H
#define NCNN_CPU_H

#include <stddef.h>

#include "platform.h"

namespace ncnn {

class    CpuSet
{
public:
    CpuSet();
    void enable(int cpu);
    void disable(int cpu);
    void disable_all();
    bool is_enabled(int cpu) const;
    int num_enabled() const;
};

// test optional cpu features
// edsp = armv7 edsp
   int cpu_support_arm_edsp();
// neon = armv7 neon or aarch64 asimd
   int cpu_support_arm_neon();
// vfpv4 = armv7 fp16 + fma
   int cpu_support_arm_vfpv4();
// asimdhp = aarch64 asimd half precision
   int cpu_support_arm_asimdhp();
// asimddp = aarch64 asimd dot product
   int cpu_support_arm_asimddp();
// asimdfhm = aarch64 asimd fhm
   int cpu_support_arm_asimdfhm();
// bf16 = aarch64 bf16
   int cpu_support_arm_bf16();
// i8mm = aarch64 i8mm
   int cpu_support_arm_i8mm();
// sve = aarch64 sve
   int cpu_support_arm_sve();
// sve2 = aarch64 sve2
   int cpu_support_arm_sve2();
// svebf16 = aarch64 svebf16
   int cpu_support_arm_svebf16();
// svei8mm = aarch64 svei8mm
   int cpu_support_arm_svei8mm();
// svef32mm = aarch64 svef32mm
   int cpu_support_arm_svef32mm();

// avx = x86 avx
   int cpu_support_x86_avx();
// fma = x86 fma
   int cpu_support_x86_fma();
// xop = x86 xop
   int cpu_support_x86_xop();
// f16c = x86 f16c
   int cpu_support_x86_f16c();
// avx2 = x86 avx2 + fma + f16c
   int cpu_support_x86_avx2();
// avx_vnni = x86 avx vnni
   int cpu_support_x86_avx_vnni();
// avx512 = x86 avx512f + avx512cd + avx512bw + avx512dq + avx512vl
   int cpu_support_x86_avx512();
// avx512_vnni = x86 avx512 vnni
   int cpu_support_x86_avx512_vnni();
// avx512_bf16 = x86 avx512 bf16
   int cpu_support_x86_avx512_bf16();
// avx512_fp16 = x86 avx512 fp16
   int cpu_support_x86_avx512_fp16();

// msa = mips mas
   int cpu_support_mips_msa();
// mmi = loongson mmi
   int cpu_support_loongson_mmi();

// v = riscv vector
   int cpu_support_riscv_v();
// zfh = riscv half-precision float
   int cpu_support_riscv_zfh();
// vlenb = riscv vector length in bytes
   int cpu_riscv_vlenb();

// cpu info
   int get_cpu_count();
   int get_little_cpu_count();
   int get_big_cpu_count();

// bind all threads on little clusters if powersave enabled
// affects HMP arch cpu like ARM big.LITTLE
// only implemented on android at the moment
// switching powersave is expensive and not thread-safe
// 0 = all cores enabled(default)
// 1 = only little clusters enabled
// 2 = only big clusters enabled
// return 0 if success for setter function
   int get_cpu_powersave();
   int set_cpu_powersave(int powersave);

// convenient wrapper
   const CpuSet& get_cpu_thread_affinity_mask(int powersave);

// set explicit thread affinity
   int set_cpu_thread_affinity(const CpuSet& thread_affinity_mask);

// misc function wrapper for openmp routines
   int get_omp_num_threads();
   void set_omp_num_threads(int num_threads);

   int get_omp_dynamic();
   void set_omp_dynamic(int dynamic);

   int get_omp_thread_num();

   int get_kmp_blocktime();
   void set_kmp_blocktime(int time_ms);

// need to flush denormals on Intel Chipset.
// Other architectures such as ARM can be added as needed.
// 0 = DAZ OFF, FTZ OFF
// 1 = DAZ ON , FTZ OFF
// 2 = DAZ OFF, FTZ ON
// 3 = DAZ ON,  FTZ ON
   int get_flush_denormals();
   int set_flush_denormals(int flush_denormals);

} // namespace ncnn

#endif // NCNN_CPU_H
