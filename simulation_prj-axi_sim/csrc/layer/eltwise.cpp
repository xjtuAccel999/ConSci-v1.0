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

#include "eltwise.h"
#include "assert.h"
#ifndef NCNN_TOOLS
#include "../hw/hw_math.h"
#include "../hw/accel_params.h"
#endif

namespace ncnn {

Eltwise::Eltwise()
{
    one_blob_only = false;
    support_inplace = false; // TODO inplace reduction
}

int Eltwise::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    coeffs = pd.get(1, Mat());

    return 0;
}

int Eltwise::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];

    #ifdef FORWARD_ON_CPU_ALU
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward Eltwise on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    #else
    #ifdef PRINT_LAYER
        if (coeffs.w != 0 && op_type == Operation_SUM)
            printf("[ log ]: Forward Eltwise on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
        else
            printf("[ log ]: Forward Eltwise on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    #endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    Mat& top_blob = top_blobs[0];
    top_blob.create_like(bottom_blob, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    if (op_type == Operation_PROD)
    {
        #ifdef FORWARD_ON_CPU_ALU
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
         
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] * ptr1[i];
            }
        }

        //如果输入blob的个数大于两个，则把剩下的blob的元素按照对应位置，对于的操作处理
        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] *= ptr[i];
                }
            }
        }
        #else 
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blobs[0]);
        Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[1]);

        accel::hw_math alu_inst;
        alu_inst.mul_en = 1;
        alu_inst.mul_src1_sel = MUL_SRC1_FROM_DMA;
        alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);

        //如果输入blob的个数大于两个，则把剩下的blob的元素按照对应位置，对于的操作处理
        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[b]);
            accel::hw_math alu_inst;
            alu_inst.mul_en = 1;
            alu_inst.mul_src1_sel = MUL_SRC1_FROM_DMA;
            alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
        }
        #endif
    }
    else if (op_type == Operation_SUM)
    {
        if (coeffs.w == 0)
        {
            #ifdef FORWARD_ON_CPU_ALU
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] + ptr1[i];
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                 
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i];
                    }
                }
            }
            #else
            Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blobs[0]);
            Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[1]);

            accel::hw_math alu_inst;
            alu_inst.add_en = 1;
            alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
            alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
            alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);

            //如果输入blob的个数大于两个，则把剩下的blob的元素按照对应位置，对于的操作处理
            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[b]);
                accel::hw_math alu_inst;
                alu_inst.add_en = 1;
                alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
                alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
                alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
            }
            #endif
        }
        else
        {
            // first blob
            const Mat& bottom_blob1 = bottom_blobs[1];
            float coeff0 = coeffs[0];
            float coeff1 = coeffs[1];
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);
                const float* ptr1 = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = ptr[i] * coeff0 + ptr1[i] * coeff1;
                }
            }

            for (size_t b = 2; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob1 = bottom_blobs[b];
                float coeff = coeffs[b];
                 
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = bottom_blob1.channel(q);
                    float* outptr = top_blob.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] += ptr[i] * coeff;
                    }
                }
            }
        }
    }
    else if (op_type == Operation_MAX)
    {
        #ifdef FORWARD_ON_CPU_ALU
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
         
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = std::max(ptr[i], ptr1[i]);
            }
        }

        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob1 = bottom_blobs[b];
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob1.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = std::max(outptr[i], ptr[i]);
                }
            }
        }
        #else
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blobs[0]);
        Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[1]);

        accel::hw_math alu_inst;
        alu_inst.add_en = 1;
        alu_inst.sub_en = 1;
        alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
        alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
        alu_inst.max_en = 1;
        alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);

        //如果输入blob的个数大于两个，则把剩下的blob的元素按照对应位置，对于的操作处理
        for (size_t b = 2; b < bottom_blobs.size(); b++)
        {
            Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[b]);
            accel::hw_math alu_inst;
            alu_inst.add_en = 1;
            alu_inst.sub_en = 1;
            alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
            alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
            alu_inst.max_en = 1;
            alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
        }
        #endif
    }

    //the follow is only used for testing! ncnn not support Operation_MIN
    else if (op_type == Operation_MIN)
    {
        #ifdef FORWARD_ON_CPU_ALU
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = std::min(ptr[i], ptr1[i]);
            }
        }
        #else
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blobs[0]);
        Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[1]);

        accel::hw_math alu_inst;
        alu_inst.add_en = 1;
        alu_inst.sub_en = 1;
        alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
        alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
        alu_inst.min_en = 1;
        alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
        #endif
    }

    //the follow is only used for testing! ncnn not support Operation_SUB
    else if (op_type == Operation_SUB)
    {
        #ifdef FORWARD_ON_CPU_ALU
        // first blob
        const Mat& bottom_blob1 = bottom_blobs[1];
        
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_blob.channel(q);
            const float* ptr1 = bottom_blob1.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                outptr[i] = ptr[i] - ptr1[i];
            }
        }
        #else
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blobs[0]);
        Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[1]);

        accel::hw_math alu_inst;
        alu_inst.add_en = 1;
        alu_inst.sub_en = 1;
        alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
        alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
        alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
        #endif
    }


    return 0;
}

} // namespace ncnn
