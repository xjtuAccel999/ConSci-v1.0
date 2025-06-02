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

#include "relu.h"
#include "assert.h"
#ifndef NCNN_TOOLS
#include "../hw/hw_act.h"
#endif
#include "../hw/accel_params.h"

namespace ncnn {

ReLU::ReLU()
{
    one_blob_only = true;
    support_inplace = true;
}

int ReLU::load_param(const ParamDict& pd)
{
    slope = pd.get(0, 0.f);

    return 0;
}

int ReLU::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    #ifdef FORWARD_ON_CPU_ALU
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward ReLU on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    #else
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward ReLU on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    #endif

    #ifdef FORWARD_ON_CPU_ALU
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

    if (slope == 0.f)
    {
         
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                if (ptr[i] < 0)
                    ptr[i] = 0;
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
                if (ptr[i] < 0)
                    ptr[i] *= slope;
            }
        }
    }
    #endif

    #ifdef FORWARD_ON_NPU_ALU
    
    accel::hw_act alu_inst;
    alu_inst.act_alpha = slope;
    alu_inst.act_op = HW_LEAKYRELU;

    alu_inst.act_forward(bottom_top_blob,bottom_top_blob);

    #endif

    return 0;
}

} // namespace ncnn
