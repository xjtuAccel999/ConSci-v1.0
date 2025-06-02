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

#include "allocator.h"

namespace ncnn {

Allocator::~Allocator()
{
}

class PoolAllocatorPrivate
{
public:
    Mutex budgets_lock;
    Mutex payouts_lock;
    unsigned int size_compare_ratio; // 0~256
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

PoolAllocator::PoolAllocator()
    : Allocator(), d(new PoolAllocatorPrivate)
{
    d->size_compare_ratio = 192; // 0.75f * 256
}

PoolAllocator::~PoolAllocator()
{
    clear();

    if (!d->payouts.empty())
    {
        NCNN_LOGE("FATAL ERROR! pool allocator destroyed too early");
#if NCNN_STDIO
        std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
        for (; it != d->payouts.end(); ++it)
        {
            void* ptr = it->second;
            NCNN_LOGE("%p still in use", ptr);
        }
#endif
    }

    delete d;
}

PoolAllocator::PoolAllocator(const PoolAllocator&)
    : d(0)
{
}

PoolAllocator& PoolAllocator::operator=(const PoolAllocator&)
{
    return *this;
}

void PoolAllocator::clear()
{
    d->budgets_lock.lock();

    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    d->budgets.clear();

    d->budgets_lock.unlock();
}

void PoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        NCNN_LOGE("invalid size compare ratio %f", scr);
        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void* PoolAllocator::fastMalloc(size_t size)
{
    d->budgets_lock.lock();

    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            d->budgets.erase(it);

            d->budgets_lock.unlock();

            d->payouts_lock.lock();

            d->payouts.push_back(std::make_pair(bs, ptr));

            d->payouts_lock.unlock();

            return ptr;
        }
    }

    d->budgets_lock.unlock();

    // new
    void* ptr = ncnn::fastMalloc(size);

    d->payouts_lock.lock();

    d->payouts.push_back(std::make_pair(size, ptr));

    d->payouts_lock.unlock();

    return ptr;
}

void PoolAllocator::fastFree(void* ptr)
{
    d->payouts_lock.lock();

    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            d->payouts.erase(it);

            d->payouts_lock.unlock();

            d->budgets_lock.lock();

            d->budgets.push_back(std::make_pair(size, ptr));

            d->budgets_lock.unlock();

            return;
        }
    }

    d->payouts_lock.unlock();

    NCNN_LOGE("FATAL ERROR! pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

class UnlockedPoolAllocatorPrivate
{
public:
    unsigned int size_compare_ratio; // 0~256
    std::list<std::pair<size_t, void*> > budgets;
    std::list<std::pair<size_t, void*> > payouts;
};

UnlockedPoolAllocator::UnlockedPoolAllocator()
    : Allocator(), d(new UnlockedPoolAllocatorPrivate)
{
    d->size_compare_ratio = 192; // 0.75f * 256
}

UnlockedPoolAllocator::~UnlockedPoolAllocator()
{
    clear();

    if (!d->payouts.empty())
    {
        NCNN_LOGE("FATAL ERROR! unlocked pool allocator destroyed too early");
#if NCNN_STDIO
        std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
        for (; it != d->payouts.end(); ++it)
        {
            void* ptr = it->second;
            NCNN_LOGE("%p still in use", ptr);
        }
#endif
    }

    delete d;
}

UnlockedPoolAllocator::UnlockedPoolAllocator(const UnlockedPoolAllocator&)
    : d(0)
{
}

UnlockedPoolAllocator& UnlockedPoolAllocator::operator=(const UnlockedPoolAllocator&)
{
    return *this;
}

void UnlockedPoolAllocator::clear()
{
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        void* ptr = it->second;
        ncnn::fastFree(ptr);
    }
    d->budgets.clear();
}

void UnlockedPoolAllocator::set_size_compare_ratio(float scr)
{
    if (scr < 0.f || scr > 1.f)
    {
        NCNN_LOGE("invalid size compare ratio %f", scr);
        return;
    }

    d->size_compare_ratio = (unsigned int)(scr * 256);
}

void* UnlockedPoolAllocator::fastMalloc(size_t size)
{
    // find free budget
    std::list<std::pair<size_t, void*> >::iterator it = d->budgets.begin();
    for (; it != d->budgets.end(); ++it)
    {
        size_t bs = it->first;

        // size_compare_ratio ~ 100%
        if (bs >= size && ((bs * d->size_compare_ratio) >> 8) <= size)
        {
            void* ptr = it->second;

            d->budgets.erase(it);

            d->payouts.push_back(std::make_pair(bs, ptr));

            return ptr;
        }
    }

    // new
    void* ptr = ncnn::fastMalloc(size);

    d->payouts.push_back(std::make_pair(size, ptr));

    return ptr;
}

void UnlockedPoolAllocator::fastFree(void* ptr)
{
    // return to budgets
    std::list<std::pair<size_t, void*> >::iterator it = d->payouts.begin();
    for (; it != d->payouts.end(); ++it)
    {
        if (it->second == ptr)
        {
            size_t size = it->first;

            d->payouts.erase(it);

            d->budgets.push_back(std::make_pair(size, ptr));

            return;
        }
    }

    NCNN_LOGE("FATAL ERROR! unlocked pool allocator get wild %p", ptr);
    ncnn::fastFree(ptr);
}

} // namespace ncnn
