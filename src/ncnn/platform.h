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

#ifndef NCNN_PLATFORM_H
#define NCNN_PLATFORM_H

#define NCNN_STDIO 1
#define NCNN_STRING 1
#define NCNN_SIMPLEOCV 1
#define NCNN_SIMPLEOMP 0
#define NCNN_SIMPLESTL 0
#define NCNN_THREADS 0
#define NCNN_BENCHMARK 0
#define NCNN_C_API 0
#define NCNN_PLATFORM_API 0
#define NCNN_PIXEL 1
#define NCNN_PIXEL_ROTATE 1
#define NCNN_PIXEL_AFFINE 1
#define NCNN_PIXEL_DRAWING 1
#define NCNN_VULKAN 0
#define NCNN_SYSTEM_GLSLANG 0
#define NCNN_RUNTIME_CPU 0
#define NCNN_AVX 0
#define NCNN_XOP 0
#define NCNN_FMA 0
#define NCNN_F16C 0
#define NCNN_AVX2 0
#define NCNN_AVXVNNI 0
#define NCNN_AVX512 0
#define NCNN_AVX512VNNI 0
#define NCNN_AVX512BF16 0
#define NCNN_AVX512FP16 0
#define NCNN_VFPV4 0

#define NCNN_MSA 0
#define NCNN_MMI 0
#define NCNN_RVV 0
#define NCNN_INT8 1
#define NCNN_BF16 0
#define NCNN_FORCE_INLINE 1

#define NCNN_VERSION_STRING "1.0.20220923"

#ifdef __cplusplus


namespace ncnn {

class   Mutex
{
public:
    Mutex() {}
    ~Mutex() {}
    void lock() {}
    void unlock() {}
};

class   ConditionVariable
{
public:
    ConditionVariable() {}
    ~ConditionVariable() {}
    void wait(Mutex& /*mutex*/) {}
    void broadcast() {}
    void signal() {}
};

class   Thread
{
public:
    Thread(void* (*/*start*/)(void*), void* /*args*/ = 0) {}
    ~Thread() {}
    void join() {}
};

class   ThreadLocalStorage
{
public:
    ThreadLocalStorage() { data = 0; }
    ~ThreadLocalStorage() {}
    void set(void* value) { data = value; }
    void* get() { return data; }
private:
    void* data;
};

class   MutexLockGuard
{
public:
    MutexLockGuard(Mutex& _mutex) : mutex(_mutex) { mutex.lock(); }
    ~MutexLockGuard() { mutex.unlock(); }
private:
    Mutex& mutex;
};

} // namespace ncnn

#include <algorithm>
#include <list>
#include <vector>
#include <string>

#endif // __cplusplus

#if NCNN_STDIO
#include <stdio.h>
#define NCNN_LOGE(...) do { \
    fprintf(stderr, ##__VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#else
#define NCNN_LOGE(...)
#endif

#if NCNN_FORCE_INLINE
    #define NCNN_FORCEINLINE inline
#endif

#endif // NCNN_PLATFORM_H
