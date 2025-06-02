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

#include "clip.h"
#include <float.h>
#include "assert.h"
#ifndef NCNN_TOOLS
#include "../hw/hw_act.h"
#endif
#include "../hw/accel_params.h"

namespace ncnn {

Clip::Clip()
{
    one_blob_only = true;
    support_inplace = true;
}

int Clip::load_param(const ParamDict& pd)
{
    min = pd.get(0, -FLT_MAX);
    max = pd.get(1, FLT_MAX);

    return 0;
}

int Clip::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    #ifdef FORWARD_ON_CPU_ALU
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward Clip on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    #else 
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward Clip on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    #endif
    #ifdef FORWARD_ON_CPU_ALU
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    int size = w * h;

     
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        for (int i = 0; i < size; i++)
        {
            if (ptr[i] < min)
                ptr[i] = min;
            if (ptr[i] > max)
                ptr[i] = max;
        }
    }
     #endif
    #ifdef FORWARD_ON_NPU_ALU
    accel::hw_act alu_inst;
    alu_inst.act_alpha = min;
    alu_inst.act_beta = max;
    alu_inst.act_op = HW_CLIP;
    // printf("alpha_hw = %f\n",alpha);
    alu_inst.act_forward(bottom_top_blob,bottom_top_blob);

    #endif

    return 0;
}

} // namespace ncnn
