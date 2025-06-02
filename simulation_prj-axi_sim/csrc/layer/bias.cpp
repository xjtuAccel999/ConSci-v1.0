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

#include "bias.h"
#include "assert.h"
#ifndef NCNN_TOOLS
#include "../hw/hw_math.h"
#endif
#include "../hw/accel_params.h"

namespace ncnn {

Bias::Bias()
{
    one_blob_only = true;
    support_inplace = true;
}

int Bias::load_param(const ParamDict& pd)
{
    bias_data_size = pd.get(0, 0);

    return 0;
}

int Bias::load_model(const ModelBin& mb)
{
    bias_data = mb.load(bias_data_size, 1);
    if (bias_data.empty())
        return -100;

    return 0;
}

int Bias::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    #ifdef FORWARD_ON_CPU_ALU
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward Bias on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    #else
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward Bias on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    #endif


    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int channels = bottom_top_blob.c;
    int size = w * h * d;

     
    for (int q = 0; q < channels; q++)
    {
        float* ptr = bottom_top_blob.channel(q);

        float bias = bias_data[q];

        #ifdef FORWARD_ON_CPU_ALU
            for (int i = 0; i < size; i++)
            {
                ptr[i] += bias;
            }
        #else
            accel::hw_math bias_inst;
            bias_inst.add_en = 1;
            bias_inst.beta = bias;
            bias_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
            bias_inst.add_src1_sel = ADD_SRC1_FROM_BETA;
            bias_inst.math_forward((void*)ptr,(void*)ptr,size);
        #endif
    }

    return 0;
}

} // namespace ncnn
