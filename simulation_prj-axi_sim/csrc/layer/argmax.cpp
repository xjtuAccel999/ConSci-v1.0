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

#include "argmax.h"

#include <functional>

namespace ncnn {

ArgMax::ArgMax()
{
    one_blob_only = true;
}

int ArgMax::load_param(const ParamDict& pd)
{
    out_max_val = pd.get(0, 0);
    topk = pd.get(1, 1);

    return 0;
}

//
int ArgMax::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    //bottom_blob是输入特征图
    //top_blob是输出特征图
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward ArgMax on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    int size = bottom_blob.total();

    //如果out_max_val == 1,输出最大值和序号
    //如果out_max_val == 0,输出序号
    if (out_max_val)
        top_blob.create(topk, 2, 4u, opt.blob_allocator);
    else
        top_blob.create(topk, 1, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const float* ptr = bottom_blob;

    // partial sort topk with index
    // optional value
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(ptr[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());
    //less<T> 升序排列 从左到右遍历下标时，数组元素从小到大
    //greater<T> 降序排列 从左到右遍历下标时，数组元素从大到小
    //按照 comp 排序规则，对 [first, last) 范围的数据进行筛选并排序
    //void partial_sort (RandomAccessIterator first,
    //                   RandomAccessIterator middle,
    //                   RandomAccessIterator last,
    //                   Compare comp);
    //partial_sort() 函数会以交换元素存储位置的方式实现部分排序的。
    //具体来说，partial_sort() 会将 [first, last) 范围内最小（或最大）的 middle-first 个元素
    //移动到 [first, middle) 区域中，并对这部分元素做升序（或降序）排序。

    float* outptr = top_blob;
    if (out_max_val)
    {
        float* valptr = outptr + topk;
        //值和序号是分开存放的
        for (int i = 0; i < topk; i++)
        {
            outptr[i] = vec[i].first;
            valptr[i] = vec[i].second;
        }
    }
    else
    {
        for (int i = 0; i < topk; i++)
        {
            outptr[i] = vec[i].second;
        }
    }

    return 0;
}

} // namespace ncnn
