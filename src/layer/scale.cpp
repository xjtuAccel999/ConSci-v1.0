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

#include "scale.h"
#include "assert.h"
#ifndef NCNN_TOOLS
#include "../hw/hw_math.h"
#endif
#include "../hw/accel_params.h"
#include "xtime_l.h"
namespace ncnn {

Scale::Scale()
{
    one_blob_only = true;
    support_inplace = true;
}

int Scale::load_param(const ParamDict& pd)
{
    scale_data_size = pd.get(0, 0);
    bias_term = pd.get(1, 0);

    if (scale_data_size == -233)
        one_blob_only = false;

    return 0;
}

int Scale::load_model(const ModelBin& mb)
{
    if (scale_data_size == -233)
        return 0;

    scale_data = mb.load(scale_data_size, 1);
    if (scale_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(scale_data_size, 1);
        if (bias_data.empty())
            return -100;
    }

    return 0;
}

int Scale::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{     
    #ifdef SCALE_TIME
		u64 tEnd_sacle, tCur_sacle;
		u32 tUsed_sacle;
        XTime_GetTime(&tCur_sacle);
    #endif
    Mat& bottom_top_blob = bottom_top_blobs[0];
    Mat& scale_blob = bottom_top_blobs[1];

    int dims = bottom_top_blob.dims;

    if (dims == 1)
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        if (bias_term)
        {
            #ifdef FORWARD_ON_CPU_ALU
            for (int i = 0; i < w; i++)
            {
                ptr[i] = ptr[i] * scale_blob[i] + bias_data[i];
            }
            #else
            Mat scale_result(w, 4u, opt.blob_allocator);
            accel::hw_math scale_inst;
            scale_inst.mul_src1_sel = MUL_SRC1_FROM_DMA;
            scale_inst.mul_en = 1;
            scale_inst.math_forward(ptr, scale_blob.data, scale_result, w);

            accel::hw_math bias_inst;
            bias_inst.add_en = 1;
            bias_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
            bias_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
            bias_inst.math_forward(scale_result, bias_data.data, ptr, w);
            #endif
        }
        else
        {
            #ifdef FORWARD_ON_CPU_ALU
            for (int i = 0; i < w; i++)
            {
                ptr[i] *= scale_blob[i];
            }
            #else 
            accel::hw_math scale_inst;
            scale_inst.mul_src1_sel = MUL_SRC1_FROM_DMA;
            scale_inst.mul_en = 1;
            scale_inst.math_forward(ptr, scale_blob.data, ptr, w);
            #endif
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        #ifdef FORWARD_ON_NPU_ALU
        if(w % 4 != 0){
            printf("[ error ]: when dims = 2, mat_w must be aligned to 4\n");
            assert(0);
        }
        #endif
        if (bias_term)
        {
             
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[i];
                float bias = bias_data[i];
                #ifdef FORWARD_ON_CPU_ALU
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = ptr[j] * s + bias;
                }
                #else
                accel::hw_math scale_inst;
                scale_inst.alpha = s;
                scale_inst.beta = bias;
                scale_inst.mul_src1_sel = MUL_SRC1_FROM_ALPHA;
                scale_inst.add_src0_sel = ADD_SRC0_FROM_MUL_O;
                scale_inst.add_src1_sel = ADD_SRC1_FROM_BETA;
                scale_inst.add_en = 1;
                scale_inst.mul_en = 1;
                scale_inst.math_forward(ptr,ptr,w);
                #endif
            }
        }
        else
        {
             
            for (int i = 0; i < h; i++)
            {
                float* ptr = bottom_top_blob.row(i);
                float s = scale_blob[i];
                #ifdef FORWARD_ON_CPU_ALU
                for (int j = 0; j < w; j++)
                {
                    ptr[j] *= s;
                }
                #else
                accel::hw_math scale_inst;
                scale_inst.alpha = s;
                scale_inst.mul_src1_sel = MUL_SRC1_FROM_ALPHA;
                scale_inst.mul_en = 1;
                scale_inst.math_forward(ptr,ptr,w);
                #endif
            }
        }
    }

    if (dims == 3)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        if (bias_term)
        {
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);
                float s = scale_blob[q];
                float bias = bias_data[q];
                #ifdef FORWARD_ON_CPU_ALU
                for (int i = 0; i < size; i++)
                {
                    ptr[i] = ptr[i] * s + bias;
                }
                #else
                accel::hw_math scale_inst;
                scale_inst.alpha = s;
                scale_inst.beta = bias;
                scale_inst.mul_src1_sel = MUL_SRC1_FROM_ALPHA;
                scale_inst.add_src0_sel = ADD_SRC0_FROM_MUL_O;
                scale_inst.add_src1_sel = ADD_SRC1_FROM_BETA;
                scale_inst.add_en = 1;
                scale_inst.mul_en = 1;
                scale_inst.math_forward(ptr,ptr,size);
                #endif
            }
        }
        else
        {
             
            for (int q = 0; q < channels; q++)
            {
                float* ptr = bottom_top_blob.channel(q);

                float s = scale_blob[q];
                
                #ifdef FORWARD_ON_CPU_ALU
                for (int i = 0; i < size; i++)
                {
                    ptr[i] *= s;
                }
                #else
                accel::hw_math scale_inst;
                scale_inst.alpha = s;
                scale_inst.mul_src1_sel = MUL_SRC1_FROM_ALPHA;
                scale_inst.mul_en = 1;
                scale_inst.math_forward(ptr,ptr,size);
                #endif
            }
        }
    }
    #ifdef SCALE_TIME
		XTime_GetTime(&tEnd_sacle);
		tUsed_sacle = ((tEnd_sacle-tCur_sacle)*1000000)/(COUNTS_PER_SECOND);
		printf("time_sacle elapsed is %d us\n",tUsed_sacle);
    #endif
    return 0;
}

int Scale::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    std::vector<Mat> bottom_top_blobs(2);
    bottom_top_blobs[0] = bottom_top_blob;
    bottom_top_blobs[1] = scale_data;
    #ifdef FORWARD_ON_CPU_ALU
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward Scale on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    #else
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward Scale on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    #endif
    
    return forward_inplace(bottom_top_blobs, opt);
    
}

} // namespace ncnn
