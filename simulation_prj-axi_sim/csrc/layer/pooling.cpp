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

#include "pooling.h"

#include "../ncnn/layer_type.h"

#include <float.h>

#include "../config.h"
#include "../hw/hw_pool.h"
#include "assert.h"

namespace ncnn {
    // adjust padding
void adjust_padding(accel::hw_pool &inst, int w, int h) {
    // Helper function to adjust padding
    auto adjust = [](int size, int kernel, int stride, int& pad_low, int& pad_high) {
        int tail = (size - kernel + pad_low + pad_high) % stride;
        if (tail == 0) return;

        if (tail <= pad_low + pad_high) {
            // Decrease padding
            while (tail > 0) {
                if (pad_low >= pad_high) {
                    pad_low--;
                } else {
                    pad_high--;
                }
                tail = (size - kernel + pad_low + pad_high) % stride;
            }
        } else {
            // Increase padding
            while (tail > 0) {
                if (pad_low < pad_high) {
                    pad_low++;
                } else {
                    pad_high++;
                }
                tail = (size - kernel + pad_low + pad_high) % stride;
            }
        }
    };

    // Adjust left-right padding
    int original_pad_left = inst.pad_left;
    int original_pad_right = inst.pad_right;
    adjust(w, inst.kernel_w, inst.stride_w, inst.pad_left, inst.pad_right);
    if (inst.pad_left != original_pad_left || inst.pad_right != original_pad_right) {
        printf("\033[93m[ warn ]: Horizontal pad adjusted from [%d, %d] to [%d, %d]\033[0m\n", 
                original_pad_left, original_pad_right, inst.pad_left, inst.pad_right);
    }

    // Adjust top-bottom padding
    int original_pad_top = inst.pad_top;
    int original_pad_bottom = inst.pad_bottom;
    adjust(h, inst.kernel_w, inst.stride_w, inst.pad_top, inst.pad_bottom);
    if (inst.pad_top != original_pad_top || inst.pad_bottom != original_pad_bottom) {
        printf("\033[93m[ warn ]: Vertical pad adjusted from [%d, %d] to [%d, %d]\033[0m\n", 
                original_pad_top, original_pad_bottom, inst.pad_top, inst.pad_bottom);
    }
}

Pooling::Pooling()
{
    one_blob_only = true;
    support_inplace = false;
}

int Pooling::load_param(const ParamDict& pd)
{
    pooling_type = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    stride_w = pd.get(2, 1);
    stride_h = pd.get(12, stride_w);
    pad_left = pd.get(3, 0);
    pad_right = pd.get(14, pad_left);
    pad_top = pd.get(13, pad_left);
    pad_bottom = pd.get(15, pad_top);
    global_pooling = pd.get(4, 0);
    pad_mode = pd.get(5, 0);
    avgpool_count_include_pad = pd.get(6, 0);
    adaptive_pooling = pd.get(7, 0);
    out_w = pd.get(8, 0);
    out_h = pd.get(18, out_w);
    pad_value = pd.get(19,0.f);


    return 0;
}

int Pooling::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    #ifdef FORWARD_ON_CPU_POOL
    #ifdef PRINT_LAYER
            printf("[ log ]: Forward Pooling on CPU, shape input = (%d, %d, %d, %d), dims = %d ,pooling_type = %d,kernel = %d, stride = %d ,padding = (%d, %d, %d, %d) ,pad_mode = %d, pad_value = %f\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims,pooling_type,kernel_w,stride_w,pad_left,pad_right,pad_top,pad_bottom,pad_mode,pad_value);
    #endif
    #else
    #ifdef PRINT_LAYER
        if(global_pooling || adaptive_pooling)
            printf("[ log ]: Forward Pooling on CPU, shape input = (%d, %d, %d, %d), dims = %d ,pooling_type = %d,kernel = %d, stride = %d ,padding = (%d, %d, %d, %d) ,pad_mode = %d, pad_value = %f\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims,pooling_type,kernel_w,stride_w,pad_left,pad_right,pad_top,pad_bottom,pad_mode,pad_value);
        else
            printf("[ log ]: Forward Pooling on NPU, shape input = (%d, %d, %d, %d), dims = %d ,pooling_type = %d,kernel = %d, stride = %d ,padding = (%d, %d, %d, %d) ,pad_mode = %d, pad_value = %f\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims,pooling_type,kernel_w,stride_w,pad_left,pad_right,pad_top,pad_bottom,pad_mode,pad_value);
    #endif
    #endif

    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int size = w * h;

        if (pooling_type == PoolMethod_MAX)
        {
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float max = ptr[0];
                for (int i = 0; i < size; i++)
                {
                    max = std::max(max, ptr[i]);
                }

                top_blob[q] = max;
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
            for (int q = 0; q < channels; q++)
            {
                const float* ptr = bottom_blob.channel(q);

                float sum = 0.f;
                for (int i = 0; i < size; i++)
                {
                    sum += ptr[i];
                }
                top_blob[q] = sum / size;

            }
        }

        return 0;
    }

    if (adaptive_pooling)
    {
        top_blob.create(out_w, out_h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        if (pooling_type == PoolMethod_MAX)
        {
             
            for (int q = 0; q < channels; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < out_h; i++)
                {
                    // floor div
                    const int ih0 = h * i / out_h;
                    // ceil div
                    const int ih1 = (h * (i + 1) + out_h - 1) / out_h;
                    for (int j = 0; j < out_w; j++)
                    {
                        // floor div
                        const int iw0 = w * j / out_w;
                        // ceil div
                        const int iw1 = (w * (j + 1) + out_w - 1) / out_w;

                        float max = inptr[ih0 * w + iw0];
                        for (int ih = ih0; ih < ih1; ih++)
                        {
                            for (int iw = iw0; iw < iw1; iw++)
                            {
                                max = std::max(max, inptr[ih * w + iw]);
                            }
                        }

                        outptr[j] = max;
                    }
                    outptr += out_w;
                }
            }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
             
            for (int q = 0; q < channels; q++)
            {
                const float* inptr = bottom_blob.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < out_h; i++)
                {
                    // floor div
                    const int ih0 = h * i / out_h;
                    // ceil div
                    const int ih1 = (h * (i + 1) + out_h - 1) / out_h;
                    const int hk = ih1 - ih0;
                    for (int j = 0; j < out_w; j++)
                    {
                        // floor div
                        const int iw0 = w * j / out_w;
                        // ceil div
                        const int iw1 = (w * (j + 1) + out_w - 1) / out_w;
                        const int wk = iw1 - iw0;

                        float sum = 0;
                        for (int ih = ih0; ih < ih1; ih++)
                        {
                            for (int iw = iw0; iw < iw1; iw++)
                            {
                                sum += inptr[ih * w + iw];
                            }
                        }

                        outptr[j] = sum / hk / wk;
                    }

                    outptr += out_w;
                }
            }
        }

        return 0;
    }

    #ifdef FORWARD_ON_CPU_POOL
    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    if (pooling_type == PoolMethod_MAX)
    {
         
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        if (avgpool_count_include_pad == 0)
        {
            int wtailpad = 0;
            int htailpad = 0;

            if (pad_mode == 0) // full padding
            {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
            }

             
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        float sum = 0;
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                float val = m.row(sy)[sx];
                                sum += val;
                                area += 1;
                            }
                        }

                        outptr[j] = sum / area;
                    }

                    outptr += outw;
                }
            }
        }
        else // if (avgpool_count_include_pad == 1)
        {
             
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        float sum = 0;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[space_ofs[k]];
                            sum += val;
                        }

                        outptr[j] = sum / maxk;
                    }

                    outptr += outw;
                }
            }
        }
    }
    #endif

    #ifdef FORWARD_ON_NPU_POOL
    if(kernel_w > 3  && pooling_type == POOL_AVG){
        Mat bottom_blob_bordered;
        make_padding(bottom_blob, bottom_blob_bordered, opt);
        if (bottom_blob_bordered.empty())
        return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        int outw = (w - kernel_w) / stride_w + 1;
        int outh = (h - kernel_h) / stride_h + 1;

        top_blob.create(outw, outh, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
        return -100;

        const int maxk = kernel_w * kernel_h;

        // kernel offsets
        std::vector<int> _space_ofs(maxk);
        int* space_ofs = &_space_ofs[0];
        {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
        }

        if (pooling_type == PoolMethod_MAX)
        {
            
        for (int q = 0; q < channels; q++)
        {
            const Mat m = bottom_blob_bordered.channel(q);
            float* outptr = top_blob.channel(q);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    float max = sptr[0];

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        max = std::max(max, val);
                    }

                    outptr[j] = max;
                }

                outptr += outw;
            }
        }
        }
        else if (pooling_type == PoolMethod_AVE)
        {
        if (avgpool_count_include_pad == 0)
        {
            int wtailpad = 0;
            int htailpad = 0;

            if (pad_mode == 0) // full padding
            {
                wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
                htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
            }

                
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    int sy0 = i * stride_h;

                    for (int j = 0; j < outw; j++)
                    {
                        int sx0 = j * stride_w;

                        float sum = 0;
                        int area = 0;

                        for (int ki = 0; ki < kernel_h; ki++)
                        {
                            int sy = sy0 + ki;

                            if (sy < pad_top)
                                continue;

                            if (sy >= h - pad_bottom - htailpad)
                                break;

                            for (int kj = 0; kj < kernel_w; kj++)
                            {
                                int sx = sx0 + kj;

                                if (sx < pad_left)
                                    continue;

                                if (sx >= w - pad_right - wtailpad)
                                    break;

                                float val = m.row(sy)[sx];
                                sum += val;
                                area += 1;
                            }
                        }

                        outptr[j] = sum / area;
                    }

                    outptr += outw;
                }
            }
        }
        else // if (avgpool_count_include_pad == 1)
        {
                
            for (int q = 0; q < channels; q++)
            {
                const Mat m = bottom_blob_bordered.channel(q);
                float* outptr = top_blob.channel(q);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        const float* sptr = m.row(i * stride_h) + j * stride_w;

                        float sum = 0;

                        for (int k = 0; k < maxk; k++)
                        {
                            float val = sptr[space_ofs[k]];
                            sum += val;
                        }

                        outptr[j] = sum / maxk;
                    }

                    outptr += outw;
                }
            }
        }
        }
    }
    // assert(pad_mode == 1);
    // assert(avgpool_count_include_pad == 1);
    // int k_t, int s_t, int type_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_down_t
    //   5          1     POOL_MAX          0                0            0            0  
    //yolov5
    else if(kernel_w == 5 && stride_w == 1 && pooling_type == POOL_MAX) {
        // printf("into kernel = 5\n");
        accel::hw_pool inst;
        inst.pool_type = pooling_type;
        inst.kernel_w = 3;
        inst.kernel_h = 3;
        inst.stride_w = stride_w;
        inst.stride_h = stride_h;
        inst.pad_left = std::max(pad_left-2,0);
        inst.pad_right = std::max(pad_right-2,0);
        inst.pad_top = std::max(pad_top-2,0);
        inst.pad_bottom = std::max(pad_bottom-2,0);
        inst.pad_mode = pad_mode;
        inst.pad_value = pad_value;

        int ofm_w_1 = (w + std::max(pad_left-2,0) + std::max(pad_right-2,0) - 3) + 1;
        int ofm_h_1 = (h + std::max(pad_top-2,0) + std::max(pad_bottom-2,0) - 3) + 1;


        int ofm_w_2 = (ofm_w_1 + pad_left-std::max(pad_left-2,0)+ pad_right-std::max(pad_right-2,0)-3) + 1;
        int ofm_h_2 = (ofm_h_1 + pad_top-std::max(pad_top-2,0)+pad_bottom-std::max(pad_bottom-2,0)-3) + 1;
        
        ncnn::Mat ofm_temp(ofm_w_1, ofm_h_1, channels, 4u, opt.blob_allocator);
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blob);
        top_blob.create(ofm_w_2, ofm_h_2, channels, elemsize, opt.blob_allocator);
        inst.forward(bottom_blob_cast, ofm_temp);

        inst.pad_left = pad_left-inst.pad_left;
        inst.pad_right = pad_right-inst.pad_right;
        inst.pad_top = pad_top-inst.pad_top;
        inst.pad_bottom = pad_bottom-inst.pad_bottom;
        inst.forward(ofm_temp, top_blob);

    }
    // int k_t, int s_t, int type_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_down_t
    //   7          1     POOL_MAX          3                3            3            3   
    else if(kernel_w == 7 && stride_w == 1 && pooling_type == POOL_MAX && pad_left == 3)  {
        printf("into kernel = 7\n");
        accel::hw_pool inst;
        inst.pool_type = pooling_type;
        inst.kernel_w = 3;
        inst.kernel_h = 3;
        inst.stride_w = stride_w;
        inst.stride_h = stride_h;
        inst.pad_left = 0;
        inst.pad_right = 0;
        inst.pad_top = 0;
        inst.pad_bottom = 0;
        inst.pad_mode = pad_mode;
        inst.pad_value = pad_value;
        int ofm_w_1 = w - 2 ;
        int ofm_h_1 = h - 2 ;
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blob);
        ncnn::Mat ofm_temp_1(ofm_w_1, ofm_h_1, channels, 4u, opt.blob_allocator);
        inst.forward(bottom_blob_cast, ofm_temp_1);
        // printf("first pooling is finish\n");
        int ofm_w_2 = w - 2;
        int ofm_h_2 = h - 2;
        ncnn::Mat ofm_temp_2(ofm_w_2, ofm_h_2, channels, 4u, opt.blob_allocator);
        inst.pad_left = 1;
        inst.pad_right = 1;
        inst.pad_top = 1;
        inst.pad_bottom = 1;
        inst.forward(ofm_temp_1, ofm_temp_2);
        // printf("second pooling is finish\n");
        int ofm_w_3 = w ;
        int ofm_h_3 = h ;
        top_blob.create(ofm_w_3, ofm_h_3, channels, elemsize, opt.blob_allocator);
        inst.pad_left = 2;
        inst.pad_right = 2;
        inst.pad_top = 2;
        inst.pad_bottom = 2;
        inst.forward(ofm_temp_2, top_blob);
        // printf("third pooling is finish\n");
    }

    // int k_t, int s_t, int type_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_down_t
    //   9          1     POOL_MAX          4                4            4            4              
    else if(kernel_w == 9 && stride_w == 1 && pooling_type == POOL_MAX && pad_left == 4) {
        printf("into kernel = 9\n");
        accel::hw_pool inst;
        inst.pool_type = pooling_type;
        inst.kernel_w = 3;
        inst.kernel_h = 3;
        inst.stride_w = stride_w;
        inst.stride_h = stride_h;
        inst.pad_left = 0;
        inst.pad_right = 0;
        inst.pad_top = 0;
        inst.pad_bottom = 0;
        inst.pad_mode = pad_mode;
        inst.pad_value = pad_value;

        int ofm_w_1 = w - 2;
        int ofm_h_1 = h - 2;
        ncnn::Mat ofm_temp_1(ofm_w_1, ofm_h_1, channels, 4u, opt.blob_allocator);
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blob);
        inst.forward(bottom_blob_cast, ofm_temp_1);
        // printf("first pooling is finish\n");
        int ofm_w_2 = w - 4;
        int ofm_h_2 = h - 4;
        ncnn::Mat ofm_temp_2(ofm_w_2, ofm_h_2, channels, 4u, opt.blob_allocator);
        inst.forward(ofm_temp_1, ofm_temp_2);
        // printf("second pooling is finish\n");
        inst.pad_left = 2;
        inst.pad_right = 2;
        inst.pad_top = 2;
        inst.pad_bottom = 2;
        int ofm_w_3 = w - 2;
        int ofm_h_3 = h - 2;
        ncnn::Mat ofm_temp_3(ofm_w_3, ofm_h_3, channels, 4u, opt.blob_allocator);
        inst.forward(ofm_temp_2, ofm_temp_3);
        // printf("third pooling is finish\n");
        inst.pad_left = 2;
        inst.pad_right = 2;
        inst.pad_top = 2;
        inst.pad_bottom = 2;
        int ofm_w_4 = w;
        int ofm_h_4 = h;
        top_blob.create(ofm_w_4, ofm_h_4, channels, elemsize, opt.blob_allocator);
        inst.forward(ofm_temp_3, top_blob);
        // printf("fourth pooling is finish\n");
    }
    // int k_t, int s_t, int type_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_down_t
    //   13          1     POOL_MAX          6                6            6            6   
    else if(kernel_w == 13 && stride_w == 1 && pooling_type == POOL_MAX && pad_left == 6 ) {
        printf("into kernel = 13\n");
        accel::hw_pool inst;
        inst.pool_type = pooling_type;
        inst.kernel_w = 3;
        inst.kernel_h = 3;
        inst.stride_w = stride_w;
        inst.stride_h = stride_h;
        inst.pad_left = 0;
        inst.pad_right = 0;
        inst.pad_top = 0;
        inst.pad_bottom = 0;
        inst.pad_mode = pad_mode;
        inst.pad_value = pad_value;
        
        int ofm_w_1 = w - 2;
        int ofm_h_1 = h - 2;
        ncnn::Mat ofm_temp_1(ofm_w_1, ofm_h_1, channels, 4u, opt.blob_allocator);
        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blob);
        inst.forward(bottom_blob_cast, ofm_temp_1);
        printf("first pooling is finish\n");
        int ofm_w_2 = w - 4;
        int ofm_h_2 = h - 4;
        ncnn::Mat ofm_temp_2(ofm_w_2, ofm_h_2, channels, 4u, opt.blob_allocator);
        inst.forward(ofm_temp_1, ofm_temp_2);
        printf("second pooling is finish\n");
        int ofm_w_3 = w - 6;
        int ofm_h_3 = h - 6;
        ncnn::Mat ofm_temp_3(ofm_w_3, ofm_h_3, channels, 4u, opt.blob_allocator);
        inst.forward(ofm_temp_2, ofm_temp_3);
        printf("third pooling is finish\n");
        int ofm_w_4 = w - 4;
        int ofm_h_4 = h - 4;
        inst.pad_left = 2;
        inst.pad_right = 2;
        inst.pad_top = 2;
        inst.pad_bottom = 2;
        ncnn::Mat ofm_temp_4(ofm_w_4, ofm_h_4, channels, 4u, opt.blob_allocator);
        inst.forward(ofm_temp_3, ofm_temp_4);
        printf("fourth pooling is finish\n");
        int ofm_w_5 = w - 2;
        int ofm_h_5 = h - 2;
        ncnn::Mat ofm_temp_5(ofm_w_5, ofm_h_5, channels, 4u, opt.blob_allocator);
        inst.forward(ofm_temp_4, ofm_temp_5);
        printf("fifth pooling is finish\n");
        int ofm_w_6 = w ;
        int ofm_h_6 = h ;
        top_blob.create(ofm_w_6, ofm_h_6, channels, elemsize, opt.blob_allocator);
        inst.forward(ofm_temp_5, top_blob);
        printf("sixth pooling is finish\n");

    }
    
    else {
        accel::hw_pool inst;
        inst.pool_type = pooling_type;
        inst.kernel_w = kernel_w;
        inst.kernel_h = kernel_h;
        inst.stride_w = stride_w;
        inst.stride_h = stride_h;
        inst.pad_left = pad_left;
        inst.pad_right = pad_right;
        inst.pad_top = pad_top;
        inst.pad_bottom = pad_bottom;
        inst.pad_mode = pad_mode;
        inst.pad_value = pad_value;
        // printf("[ log ]: pad_left = %d, pad_right = %d, pad_top = %d, pad_bottom = %d\n",pad_left,pad_right,pad_top,pad_bottom);

        adjust_padding(inst, bottom_blob.w, bottom_blob.h);

        int ofm_w = (w + inst.pad_left + inst.pad_right - inst.kernel_w) / inst.stride_w + 1;
        int ofm_h = (h + inst.pad_top + inst.pad_bottom - inst.kernel_h) / inst.stride_h + 1;
        top_blob.create(ofm_w, ofm_h, channels, elemsize, opt.blob_allocator);

        // inst.pad_right = (ofm_w - 1) * inst.stride_w + inst.kernel_w- w - inst.pad_left;
        // inst.pad_bottom = (ofm_h - 1) * inst.stride_h + inst.kernel_h - h - inst.pad_top;
        

        // printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d\n",w,h,channels);
        // printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n",ofm_w,ofm_h,channels);
        // printf("[ log ]: pool_type = %d, kernel = %d, stride = %d, avgpool_count_include_pad = %d\n",pooling_type,kernel_w,stride_w,avgpool_count_include_pad);
        // printf("[ log ]: pad_left = %d, pad_right = %d, pad_top = %d, pad_bottom = %d\n",inst.pad_left,inst.pad_right,inst.pad_top,inst.pad_bottom);


        Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blob);

        

        inst.forward(bottom_blob_cast,top_blob);
    }
    #endif

    return 0;
}

void Pooling::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX)
    {
        pad_value = bottom_blob.elemsize == 1 ? -128.f : -FLT_MAX;
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 1) // valid padding
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_mode == 3) // onnx padding=SAME_LOWER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

} // namespace ncnn
