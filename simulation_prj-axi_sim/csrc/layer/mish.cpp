// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "mish.h"

#include <math.h>

namespace ncnn {

Mish::Mish()
{
    one_blob_only = true;
    support_inplace = true;
}

int Mish::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    #ifdef PRINT_LAYER
    printf("[ log ]: Forward Mish on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

     
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            const float MISH_THRESHOLD = 20;
            float x = ptr[i], y;
            if (x > MISH_THRESHOLD)
                y = x;
            else if (x < -MISH_THRESHOLD)
                y = expf(x);
            else
                y = logf(expf(x) + 1);
            ptr[i] = static_cast<float>(x * tanh(y));
        }
    }

    return 0;
}

} // namespace ncnn
