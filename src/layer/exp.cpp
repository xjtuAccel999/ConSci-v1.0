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

#include "exp.h"

#include <math.h>

namespace ncnn {

Exp::Exp()
{
    one_blob_only = true;
    support_inplace = true;
}

int Exp::load_param(const ParamDict& pd)
{
    base = pd.get(0, -1.f);
    scale = pd.get(1, 1.f);
    shift = pd.get(2, 0.f);

    return 0;
}

int Exp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    #ifdef PRINT_LAYER
    printf("[ log ]: Forward Exp on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

    if (base == -1.f)
    {
         
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] = static_cast<float>(exp(shift + ptr[i] * scale));
            }
        }
    }
    else
    {
         
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] = static_cast<float>(pow(base, (shift + ptr[i] * scale)));
            }
        }
    }

    return 0;
}

} // namespace ncnn
