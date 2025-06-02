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

#include "cpu.h"

#include "platform.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

namespace ncnn {

CpuSet::CpuSet()
{
}

void CpuSet::enable(int /* cpu */)
{
}

void CpuSet::disable(int /* cpu */)
{
}

void CpuSet::disable_all()
{
}

bool CpuSet::is_enabled(int /* cpu */) const
{
    return true;
}

int CpuSet::num_enabled() const
{
    return get_cpu_count();
}


int cpu_support_arm_edsp()
{
    return 0;
}

int cpu_support_arm_neon()
{
    return 0;
}

int cpu_support_arm_vfpv4()
{
    return 0;
}

int cpu_support_arm_asimdhp()
{
    return 0;
}

int cpu_support_arm_asimddp()
{
    return 0;
}

int cpu_support_arm_asimdfhm()
{
    return 0;
}

int cpu_support_arm_bf16()
{
    return 0;
}

int cpu_support_arm_i8mm()
{
    return 0;
}

int cpu_support_arm_sve()
{
    return 0;
}

int cpu_support_arm_sve2()
{
    return 0;
}

int cpu_support_arm_svebf16()
{
    return 0;
}

int cpu_support_arm_svei8mm()
{
    return 0;
}

int cpu_support_arm_svef32mm()
{
    return 0;
}

static const int g_cpu_support_x86_avx = 0;
static const int g_cpu_support_x86_fma = 0;
static const int g_cpu_support_x86_xop = 0;
static const int g_cpu_support_x86_f16c = 0;
static const int g_cpu_support_x86_avx2 = 0;
static const int g_cpu_support_x86_avx_vnni = 0;
static const int g_cpu_support_x86_avx512 = 0;
static const int g_cpu_support_x86_avx512_vnni = 0;
static const int g_cpu_support_x86_avx512_bf16 = 0;
static const int g_cpu_support_x86_avx512_fp16 = 0;

int cpu_support_x86_avx()
{
    return g_cpu_support_x86_avx;
}

int cpu_support_x86_fma()
{
    return g_cpu_support_x86_fma;
}

int cpu_support_x86_xop()
{
    return g_cpu_support_x86_xop;
}

int cpu_support_x86_f16c()
{
    return g_cpu_support_x86_f16c;
}

int cpu_support_x86_avx2()
{
    return g_cpu_support_x86_avx2;
}

int cpu_support_x86_avx_vnni()
{
    return g_cpu_support_x86_avx_vnni;
}

int cpu_support_x86_avx512()
{
    return g_cpu_support_x86_avx512;
}

int cpu_support_x86_avx512_vnni()
{
    return g_cpu_support_x86_avx512_vnni;
}

int cpu_support_x86_avx512_bf16()
{
    return g_cpu_support_x86_avx512_bf16;
}

int cpu_support_x86_avx512_fp16()
{
    return g_cpu_support_x86_avx512_fp16;
}

int cpu_support_mips_msa()
{
    return 0;
}

int cpu_support_loongson_mmi()
{
    return 0;
}

int cpu_support_riscv_v()
{
    return 0;
}

int cpu_support_riscv_zfh()
{
    return 0;
}

int cpu_riscv_vlenb()
{
    return 0;
}

static int get_cpucount()
{
    int count = 1;
    return count;
}

static int g_cpucount = get_cpucount();

int get_cpu_count()
{
    return g_cpucount;
}

int get_little_cpu_count()
{
    return get_cpu_thread_affinity_mask(1).num_enabled();
}

int get_big_cpu_count()
{
    int big_cpu_count = get_cpu_thread_affinity_mask(2).num_enabled();
    return big_cpu_count ? big_cpu_count : g_cpucount;
}


static int g_powersave = 0;

int get_cpu_powersave()
{
    return g_powersave;
}

int set_cpu_powersave(int powersave)
{
    if (powersave < 0 || powersave > 2)
    {
        NCNN_LOGE("powersave %d not supported", powersave);
        return -1;
    }

    const CpuSet& thread_affinity_mask = get_cpu_thread_affinity_mask(powersave);

    int ret = set_cpu_thread_affinity(thread_affinity_mask);
    if (ret != 0)
        return ret;

    g_powersave = powersave;

    return 0;
}

static CpuSet g_thread_affinity_mask_all;
static CpuSet g_thread_affinity_mask_little;
static CpuSet g_thread_affinity_mask_big;

static int setup_thread_affinity_masks()
{
    g_thread_affinity_mask_all.disable_all();

    g_thread_affinity_mask_little.disable_all();
    g_thread_affinity_mask_big = g_thread_affinity_mask_all;

    return 0;
}

const CpuSet& get_cpu_thread_affinity_mask(int powersave)
{
    setup_thread_affinity_masks();

    if (powersave == 0)
        return g_thread_affinity_mask_all;

    if (powersave == 1)
        return g_thread_affinity_mask_little;

    if (powersave == 2)
        return g_thread_affinity_mask_big;

    NCNN_LOGE("powersave %d not supported", powersave);

    // fallback to all cores anyway
    return g_thread_affinity_mask_all;
}

int set_cpu_thread_affinity(const CpuSet& thread_affinity_mask)
{
    // TODO
    (void)thread_affinity_mask;
    return -1;
}

int get_omp_num_threads()
{
    return 1;
}

void set_omp_num_threads(int num_threads)
{
    (void)num_threads;
}

int get_omp_dynamic()
{
    return 0;
}

void set_omp_dynamic(int dynamic)
{
    (void)dynamic;
}

int get_omp_thread_num()
{
    return 0;
}

int get_kmp_blocktime()
{
    return 0;
}

void set_kmp_blocktime(int time_ms)
{
    (void)time_ms;
}

static ncnn::ThreadLocalStorage tls_flush_denormals;

int get_flush_denormals()
{
    return 0;
}

int set_flush_denormals(int flush_denormals)
{
    if (flush_denormals < 0 || flush_denormals > 3)
    {
        NCNN_LOGE("denormals_zero %d not supported", flush_denormals);
        return -1;
    }
    return 0;
}

} // namespace ncnn
