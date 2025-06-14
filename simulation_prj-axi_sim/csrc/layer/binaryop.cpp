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

#include "binaryop.h"

#include <math.h>
#include "assert.h"

#ifndef NCNN_TOOLS
#include "../hw/hw_math.h"
#include "../hw/accel_params.h"
#endif

namespace ncnn {

BinaryOp::BinaryOp()
{
    one_blob_only = false;
    support_inplace = false;
}

int BinaryOp::load_param(const ParamDict& pd)
{
    op_type = pd.get(0, 0);
    with_scalar = pd.get(1, 0);
    b = pd.get(2, 0.f);

    if (with_scalar != 0)
    {
        one_blob_only = true;
        support_inplace = true;
    }

    return 0;
}

// broadcasting rule
// https://github.com/Tencent/ncnn/wiki/binaryop-broadcasting

template<typename Op>
static int binary_op(const Mat& a, const Mat& b, Mat& c, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;
    size_t elemsize = a.elemsize;

    int w1 = b.w;
    int h1 = b.h;
    int d1 = b.d;
    int channels1 = b.c;
    int size1 = w1 * h1 * d1;

    if (a.dims == 4)
    {
        if (b.dims == 4)
        {
            // type 29
            c.create(w, h, d, channels, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], ptr1[i]);
                }
            }

            return 0;
        }

        c.create(w, h, d, channels, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 3)
        {
            // type 28
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    for (int y = 0; y < h; y++)
                    {
                        const float b0 = ptr1[y];
                        for (int x = 0; x < w; x++)
                        {
                            outptr[x] = op(ptr[x], b0);
                        }

                        ptr += w;
                        outptr += w;
                    }

                    ptr1 += h;
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 27
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d; z++)
                {
                    const float b0 = ptr1[z];
                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            outptr[x] = op(ptr[x], b0);
                        }

                        ptr += w;
                        outptr += w;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1)
            {
                // type 25
                const float b0 = b[0];
                 
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], b0);
                    }
                }

                return 0;
            }

            // type 26
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float b0 = b[q];
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], b0);
                }
            }

            return 0;
        }
    }
    else if (a.dims == 3)
    {
        if (b.dims == 4)
        {
            // type 23
            c.create(w1, h1, d1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

             
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    for (int y = 0; y < h1; y++)
                    {
                        const float a0 = ptr[y];
                        for (int x = 0; x < w1; x++)
                        {
                            outptr[x] = op(a0, ptr1[x]);
                        }

                        ptr1 += w1;
                        outptr += w1;
                    }

                    ptr += h1;
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            if (w1 == 1 && h1 == 1 && channels1 == channels)
            {
                // special type 1
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                 
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* b0 = b.channel(q);
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], b0[0]);
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels1 == 1)
            {
                // special type 2
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                 
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b;
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], ptr1[i]);
                    }
                }

                return 0;
            }

            if (w == 1 && h == 1 && channels1 == channels)
            {
                // special type 3
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                 
                for (int q = 0; q < channels1; q++)
                {
                    const float* a0 = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(a0[0], ptr1[i]);
                    }
                }

                return 0;
            }

            if (w1 == w && h1 == h && channels == 1)
            {
                // special type 4
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                 
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a;
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);
                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(ptr[i], ptr1[i]);
                    }
                }

                return 0;
            }

            if (w != 1 && w1 == 1 && h1 == h && channels1 == channels)
            {
                // special type 5
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                 
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        const float b0 = ptr1[y];
                        for (int x = 0; x < w; x++)
                        {
                            outptr[x] = op(ptr[x], b0);
                        }

                        ptr += w;
                        outptr += w;
                    }
                }

                return 0;
            }

            if (w1 == w && h != 1 && h1 == 1 && channels1 == channels)
            {
                // special type 6
                c.create(w, h, channels, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                 
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h; y++)
                    {
                        for (int x = 0; x < w; x++)
                        {
                            outptr[x] = op(ptr[x], ptr1[x]);
                        }

                        ptr += w;
                        outptr += w;
                    }
                }

                return 0;
            }

            if (w1 != 1 && w == 1 && h1 == h && channels1 == channels)
            {
                // special type 7
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                 
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        const float a0 = ptr[y];
                        for (int x = 0; x < w1; x++)
                        {
                            outptr[x] = op(a0, ptr1[x]);
                        }

                        ptr1 += w1;
                        outptr += w1;
                    }
                }

                return 0;
            }

            if (w1 == w && h1 != 1 && h == 1 && channels1 == channels)
            {
                // special type 8
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                 
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr = a.channel(q);
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            outptr[x] = op(ptr[x], ptr1[x]);
                        }

                        ptr1 += w1;
                        outptr += w1;
                    }
                }

                return 0;
            }

            // type 19
            c.create(w, h, channels, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], ptr1[i]);
                }
            }

            return 0;
        }

        c.create(w, h, channels, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 18
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float* ptr1 = b.row(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h; y++)
                {
                    const float b0 = ptr1[y];
                    for (int x = 0; x < w; x++)
                    {
                        outptr[x] = op(ptr[x], b0);
                    }

                    ptr += w;
                    outptr += w;
                }
            }

            return 0;
        }

        if (b.dims == 1)
        {
            if (b.w == 1)
            {
                // type 16
                const float b0 = b[0];
                 
                for (int q = 0; q < channels; q++)
                {
                    const float* ptr = a.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size; i++)
                    {
                        outptr[i] = op(ptr[i], b0);
                    }
                }

                return 0;
            }

            // type 17
             
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = a.channel(q);
                const float b0 = b[q];
                float* outptr = c.channel(q);

                for (int i = 0; i < size; i++)
                {
                    outptr[i] = op(ptr[i], b0);
                }
            }

            return 0;
        }
    }
    else if (a.dims == 2)
    {
        if (b.dims == 4)
        {
            // type 22
            c.create(w1, h1, d1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

             
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int z = 0; z < d1; z++)
                {
                    const float a0 = ptr[z];
                    for (int y = 0; y < h1; y++)
                    {
                        for (int x = 0; x < w1; x++)
                        {
                            outptr[x] = op(a0, ptr1[x]);
                        }

                        ptr1 += w1;
                        outptr += w1;
                    }
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 14
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

             
            for (int q = 0; q < channels1; q++)
            {
                const float* ptr = a.row(q);
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int y = 0; y < h1; y++)
                {
                    const float a0 = ptr[y];
                    for (int x = 0; x < w1; x++)
                    {
                        outptr[x] = op(a0, ptr1[x]);
                    }

                    ptr1 += w1;
                    outptr += w1;
                }
            }

            return 0;
        }

        c.create(w, h, elemsize, opt.blob_allocator);
        if (c.empty())
            return -100;

        if (b.dims == 2)
        {
            // type 13
            for (int i = 0; i < size; i++)
            {
                c[i] = op(a[i], b[i]);
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, h, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 11
                const float b0 = b[0];
                for (int i = 0; i < size; i++)
                {
                    c[i] = op(a[i], b0);
                }

                return 0;
            }

            // type 12
            const float* ptr = a;
            float* outptr = c;

            for (int y = 0; y < h; y++)
            {
                const float b0 = b[y];
                for (int x = 0; x < w; x++)
                {
                    outptr[x] = op(ptr[x], b0);
                }

                ptr += w;
                outptr += w;
            }

            return 0;
        }
    }
    else if (a.dims == 1)
    {
        if (a.w == 1)
        {
            if (b.dims == 4)
            {
                // type 20
                c.create(w1, h1, d1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = a[0];
                 
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(a0, ptr1[i]);
                    }
                }

                return 0;
            }

            if (b.dims == 3)
            {
                // type 4
                c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = a[0];
                 
                for (int q = 0; q < channels1; q++)
                {
                    const float* ptr1 = b.channel(q);
                    float* outptr = c.channel(q);

                    for (int i = 0; i < size1; i++)
                    {
                        outptr[i] = op(a0, ptr1[i]);
                    }
                }

                return 0;
            }

            if (b.dims == 2)
            {
                // type 3
                c.create(w1, h1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = a[0];
                for (int i = 0; i < size1; i++)
                {
                    c[i] = op(a0, b[i]);
                }

                return 0;
            }

            if (b.dims == 1)
            {
                // type 2
                c.create(w1, elemsize, opt.blob_allocator);
                if (c.empty())
                    return -100;

                const float a0 = a[0];
                for (int i = 0; i < w1; i++)
                {
                    c[i] = op(a0, b[i]);
                }

                return 0;
            }
        }

        if (b.dims == 4)
        {
            // type 21
            c.create(w1, h1, d1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

             
            for (int q = 0; q < channels1; q++)
            {
                const float a0 = a[q];
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }
            }

            return 0;
        }

        if (b.dims == 3)
        {
            // type 9
            c.create(w1, h1, channels1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

             
            for (int q = 0; q < channels1; q++)
            {
                const float a0 = a[q];
                const float* ptr1 = b.channel(q);
                float* outptr = c.channel(q);

                for (int i = 0; i < size1; i++)
                {
                    outptr[i] = op(a0, ptr1[i]);
                }
            }

            return 0;
        }

        if (b.dims == 2)
        {
            // type 8
            c.create(w1, h1, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            const float* ptr1 = b;
            float* outptr = c;

            for (int y = 0; y < h1; y++)
            {
                const float a0 = a[y];
                for (int x = 0; x < w1; x++)
                {
                    outptr[x] = op(a0, ptr1[x]);
                }

                ptr1 += w1;
                outptr += w1;
            }

            return 0;
        }

        if (b.dims == 1)
        {
            c.create(w, elemsize, opt.blob_allocator);
            if (c.empty())
                return -100;

            if (b.w == 1)
            {
                // type 6
                const float b0 = b[0];
                for (int i = 0; i < w; i++)
                {
                    c[i] = op(a[i], b0);
                }

                return 0;
            }

            // type 7
            for (int i = 0; i < w; i++)
            {
                c[i] = op(a[i], b[i]);
            }
        }
    }

    return 0;
}

template<typename Op>
static int binary_op_scalar_inplace(Mat& a, float b, const Option& opt)
{
    Op op;

    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int size = w * h * d;

     
    for (int q = 0; q < channels; q++)
    {
        float* ptr = a.channel(q);

        for (int i = 0; i < size; i++)
        {
            ptr[i] = op(ptr[i], b);
        }
    }

    return 0;
}

struct binary_op_add
{
    float operator()(const float& x, const float& y) const
    {
        return x + y;
    }
};

struct binary_op_sub
{
    float operator()(const float& x, const float& y) const
    {
        return x - y;
    }
};

struct binary_op_mul
{
    float operator()(const float& x, const float& y) const
    {
        return x * y;
    }
};

struct binary_op_div
{
    float operator()(const float& x, const float& y) const
    {
        return x / y;
    }
};

struct binary_op_max
{
    float operator()(const float& x, const float& y) const
    {
        return std::max(x, y);
    }
};

struct binary_op_min
{
    float operator()(const float& x, const float& y) const
    {
        return std::min(x, y);
    }
};

struct binary_op_pow
{
    float operator()(const float& x, const float& y) const
    {
        return (float)pow(x, y);
    }
};

struct binary_op_rsub
{
    float operator()(const float& x, const float& y) const
    {
        return y - x;
    }
};

struct binary_op_rdiv
{
    float operator()(const float& x, const float& y) const
    {
        return y / x;
    }
};

int BinaryOp::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& bottom_blob1 = bottom_blobs[1];

    Mat& top_blob = top_blobs[0];

    #ifdef FORWARD_ON_NPU_ALU
        if(bottom_blob.dims == bottom_blob1.dims && (op_type == Operation_ADD || op_type == Operation_SUB || op_type == Operation_MUL || op_type == Operation_MAX || op_type == Operation_MIN || op_type == Operation_RSUB)){
            #ifdef PRINT_LAYER
                printf("[ log ]: Forward BinaryOp on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
            #endif
            Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blobs[0]);
            Mat& bottom_blob1_cast = const_cast<Mat&>(bottom_blobs[1]);
            top_blob.create_like(bottom_blob_cast, opt.blob_allocator);
            accel::hw_math alu_inst;
            switch (op_type)
            {
            case Operation_ADD:
                alu_inst.add_en = 1;
                alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
                alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
                alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
                break;

            case Operation_SUB:
                alu_inst.add_en = 1;
                alu_inst.sub_en = 1;
                alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
                alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
                alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
                break;
    
            case Operation_MUL:
                alu_inst.mul_en = 1;
                alu_inst.mul_src1_sel = MUL_SRC1_FROM_DMA;
                alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
                break;
    
            case Operation_MAX:
                alu_inst.add_en = 1;
                alu_inst.sub_en = 1;
                alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
                alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
                alu_inst.max_en = 1;
                alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
                break;
    
            case Operation_MIN:
                alu_inst.add_en = 1;
                alu_inst.sub_en = 1;
                alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
                alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
                alu_inst.min_en = 1;
                alu_inst.math_forward(bottom_blob_cast,bottom_blob1_cast,top_blob);
                break;
    
            case Operation_RSUB:
                alu_inst.add_en = 1;
                alu_inst.sub_en = 1;
                alu_inst.add_src0_sel = ADD_SRC0_FROM_DMA;
                alu_inst.add_src1_sel = ADD_SRC1_FROM_DMA;
                alu_inst.math_forward(bottom_blob1_cast,bottom_blob_cast,top_blob);
                break;
    
            default:
                break;
            }

            return 0;
        }   
    #endif

    #ifdef PRINT_LAYER
        printf("[ log ]: Forward BinaryOp on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    printf("[ log ]: binaryop a.dims = %d, b.dims = %d\n",bottom_blob.dims,bottom_blob1.dims);
    if (op_type == Operation_ADD)
        return binary_op<binary_op_add>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_SUB)
        return binary_op<binary_op_sub>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MUL)
        return binary_op<binary_op_mul>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MAX)
        return binary_op<binary_op_max>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_MIN)
        return binary_op<binary_op_min>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_RSUB)
        return binary_op<binary_op_sub>(bottom_blob1, bottom_blob, top_blob, opt);

    if (op_type == Operation_DIV)
        return binary_op<binary_op_div>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_POW)
        return binary_op<binary_op_pow>(bottom_blob, bottom_blob1, top_blob, opt);

    if (op_type == Operation_RDIV)
        return binary_op<binary_op_div>(bottom_blob1, bottom_blob, top_blob, opt);
    
    return 0;
}

int BinaryOp::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    #ifdef FORWARD_ON_NPU_ALU
    if(op_type == Operation_ADD || op_type == Operation_MUL){
        #ifdef PRINT_LAYER
            printf("[ log ]: Forward BinaryOp on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
        #endif
        accel::hw_math alu_inst;
        switch (op_type)
        {
        case Operation_ADD:
            alu_inst.beta = b;
            alu_inst.add_src1_sel = ADD_SRC1_FROM_BETA;
            alu_inst.add_en = 1;
            alu_inst.math_forward(bottom_top_blob.data,bottom_top_blob.data,bottom_top_blob.total());
            break;
        case Operation_MUL:
            alu_inst.alpha = b;
            alu_inst.mul_src1_sel = MUL_SRC1_FROM_ALPHA;
            alu_inst.mul_en = 1;
            alu_inst.math_forward(bottom_top_blob.data,bottom_top_blob.data,bottom_top_blob.total());
            break;
        
        default:
            break;
        }
        return 0;
    }
    #endif

    #ifdef PRINT_LAYER
        printf("[ log ]: Forward BinaryOp on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif

    if (op_type == Operation_ADD)
        return binary_op_scalar_inplace<binary_op_add>(bottom_top_blob, b, opt);

    if (op_type == Operation_SUB)
        return binary_op_scalar_inplace<binary_op_sub>(bottom_top_blob, b, opt);

    if (op_type == Operation_MUL)
        return binary_op_scalar_inplace<binary_op_mul>(bottom_top_blob, b, opt);

    if (op_type == Operation_DIV)
        return binary_op_scalar_inplace<binary_op_div>(bottom_top_blob, b, opt);

    if (op_type == Operation_MAX)
        return binary_op_scalar_inplace<binary_op_max>(bottom_top_blob, b, opt);

    if (op_type == Operation_MIN)
        return binary_op_scalar_inplace<binary_op_min>(bottom_top_blob, b, opt);

    if (op_type == Operation_POW)
        return binary_op_scalar_inplace<binary_op_pow>(bottom_top_blob, b, opt);

    if (op_type == Operation_RSUB)
        return binary_op_scalar_inplace<binary_op_rsub>(bottom_top_blob, b, opt);

    if (op_type == Operation_RDIV)
        return binary_op_scalar_inplace<binary_op_rdiv>(bottom_top_blob, b, opt);

    return 0;
}

} // namespace ncnn
