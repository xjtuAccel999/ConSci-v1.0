// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef NCNN_ALLOCATOR_H
#define NCNN_ALLOCATOR_H

#include "platform.h"

#include <stdlib.h>

namespace ncnn {

// the alignment of all the allocated buffers
// #if NCNN_AVX512
// #define NCNN_MALLOC_ALIGN 64
// #elif NCNN_AVX
// #define NCNN_MALLOC_ALIGN 32
// #else
// #define NCNN_MALLOC_ALIGN 16
// #define NCNN_MALLOC_ALIGN 128
#define NCNN_MALLOC_ALIGN 256
// #endif

// we have some optimized kernels that may overread buffer a bit in loop
// it is common to interleave next-loop data load with arithmetic instructions
// allocating more bytes keeps us safe from SEGV_ACCERR failure
#define NCNN_MALLOC_OVERREAD 64
// #define NCNN_MALLOC_OVERREAD 64

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp>
static NCNN_FORCEINLINE _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static NCNN_FORCEINLINE size_t alignSize(size_t sz, int n)
{
    // int x=64;
    return (sz + n - 1) & -n;
    // return (sz + x - 1) & -x;
}

static NCNN_FORCEINLINE void* fastMalloc(size_t size)
{
    // if(size <512/8) size = 512/8;
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + NCNN_MALLOC_ALIGN + NCNN_MALLOC_OVERREAD);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, NCNN_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

static NCNN_FORCEINLINE void fastFree(void* ptr)
{
    if (ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
    }
}

static NCNN_FORCEINLINE int NCNN_XADD(int* addr, int delta)
{
    int tmp = *addr;
    *addr += delta;
    return tmp;
}


class   Allocator
{
public:
    virtual ~Allocator();
    virtual void* fastMalloc(size_t size) = 0;
    virtual void fastFree(void* ptr) = 0;
};

class PoolAllocatorPrivate;
class   PoolAllocator : public Allocator
{
public:
    PoolAllocator();
    ~PoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    PoolAllocator(const PoolAllocator&);
    PoolAllocator& operator=(const PoolAllocator&);

private:
    PoolAllocatorPrivate* const d;
};

class UnlockedPoolAllocatorPrivate;
class   UnlockedPoolAllocator : public Allocator
{
public:
    UnlockedPoolAllocator();
    ~UnlockedPoolAllocator();

    // ratio range 0 ~ 1
    // default cr = 0.75
    void set_size_compare_ratio(float scr);

    // release all budgets immediately
    void clear();

    virtual void* fastMalloc(size_t size);
    virtual void fastFree(void* ptr);

private:
    UnlockedPoolAllocator(const UnlockedPoolAllocator&);
    UnlockedPoolAllocator& operator=(const UnlockedPoolAllocator&);

private:
    UnlockedPoolAllocatorPrivate* const d;
};

} // namespace ncnn

#endif // NCNN_ALLOCATOR_H
