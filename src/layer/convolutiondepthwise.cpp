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

#include "convolutiondepthwise.h"
#include "../ncnn/layer_type.h"
#include "fused_activation.h"
#include "../utils/testutil.h"
#include "assert.h"
#ifndef NCNN_TOOLS
#include "../hw/hw_gemm.h"
#include "../testLayer/test_layer.h"
#include "../utils/conv_utils.h"
#endif
#include "../hw/accel_params.h"
#include "../utils/utils.h"
#include "../utils/wgt_reorder.h"
#include "xtime_l.h"

#ifdef LOG_WGT_BIAS
    int dwconv_count = 0;
#endif

namespace ncnn {

ConvolutionDepthWise::ConvolutionDepthWise()
{
    one_blob_only = true;
    support_inplace = false;
}

int ConvolutionDepthWise::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    kernel_w = pd.get(1, 0);
    kernel_h = pd.get(11, kernel_w);
    dilation_w = pd.get(2, 1);
    dilation_h = pd.get(12, dilation_w);
    stride_w = pd.get(3, 1);
    stride_h = pd.get(13, stride_w);
    pad_left = pd.get(4, 0);
    pad_right = pd.get(15, pad_left);
    pad_top = pd.get(14, pad_left);
    pad_bottom = pd.get(16, pad_top);
    pad_value = pd.get(18, 0.f);
    bias_term = pd.get(5, 0);
    weight_data_size = pd.get(6, 0);
    group = pd.get(7, 1);
    int8_scale_term = pd.get(8, 0);
    activation_type = pd.get(9, 0);
    activation_params = pd.get(10, Mat());

    dynamic_weight = pd.get(19, 0);

    layout_en = pd.get(20, 1);
    ifm_w = pd.get(21, 0);
    ifm_h = pd.get(22, 0);
    ifm_c = pd.get(23, 0);

    if (dynamic_weight)
    {
        one_blob_only = false;
    }

    if (num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    if (int8_scale_term)
    {
#if NCNN_INT8
        support_int8_storage = true;
#else
        NCNN_LOGE("please build ncnn with NCNN_INT8 enabled for int8 inference");
        return -1;
#endif
    }

    return 0;
}

int ConvolutionDepthWise::load_model(const ModelBin& mb)
{
    // printf("ConvolutionDepthWise::load_model\n");
    if (dynamic_weight)
        return 0;

    if(int8_scale_term)
        weight_data = mb.load(u_align(num_output, 32) * kernel_w * kernel_h , 0);
    else
        weight_data = mb.load(weight_data_size, 0);

    // printf("int8_scale_term = %d, weight_data_size = %d\n",int8_scale_term,weight_data_size);
    // printf("weight_data_size_align = %d\n", u_align(num_output, 32) * kernel_w * kernel_h);
    // printf("\n");

    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

    #if defined(LOG_WGT_BIAS) && !defined(NCNN_TOOLS) 

    std::string s_weight_data_name = "./log/log_weight_data_dwconv" + std::to_string(dwconv_count) + "_" + std::to_string(ifm_w) + "x" + std::to_string(ifm_h) + "x" + 
                                    std::to_string(ifm_c) + "x" + std::to_string(num_output)+ ".txt";
    std::string s_bias_data_name   = "./log/log_bias_data_dwconv" + std::to_string(dwconv_count) + "_" + std::to_string(ifm_w) + "x" + std::to_string(ifm_h) + "x" + 
                                    std::to_string(ifm_c) + "x" + std::to_string(num_output)+ ".txt";
    
    char weight_data_name[100];
    char bias_data_name[100];
    cstr(s_weight_data_name, weight_data_name);
    cstr(s_bias_data_name,   bias_data_name);

    log_mat_file<float>(bias_data  ,   bias_data_name,-1, 1, 0);
    log_mat_file<float>(weight_data, weight_data_name,-1, 1, 0);

    dwconv_count++;

    #endif

#if NCNN_INT8
    if (int8_scale_term == 1 || int8_scale_term == 101)
    {
        weight_data_int8_scales = mb.load(group, 1);
        bottom_blob_int8_scales = mb.load(1, 1);

        float bottom_blob_int8_scale = bottom_blob_int8_scales[0];
        bottom_blob_int8_scales = Mat(group);
        bottom_blob_int8_scales.fill(bottom_blob_int8_scale);
    }
    else if (int8_scale_term == 2 || int8_scale_term == 102)
    {
        weight_data_int8_scales = mb.load(1, 1);
        bottom_blob_int8_scales = mb.load(1, 1);

        // extend group if only one provided
        float weight_data_int8_scale = weight_data_int8_scales[0];
        weight_data_int8_scales = Mat(group);
        weight_data_int8_scales.fill(weight_data_int8_scale);

        float bottom_blob_int8_scale = bottom_blob_int8_scales[0];
        bottom_blob_int8_scales = Mat(group);
        bottom_blob_int8_scales.fill(bottom_blob_int8_scale);
    }

    if (int8_scale_term > 100)
    {
        top_blob_int8_scales = mb.load(1, 1);

        float top_blob_int8_scale = top_blob_int8_scales[0];
        top_blob_int8_scales = Mat(group);
        top_blob_int8_scales.fill(top_blob_int8_scale);
    }
#endif // NCNN_INT8

    return 0;
}

int ConvolutionDepthWise::create_pipeline(const Option& opt)
{
#if NCNN_INT8
    // runtime quantize the weight data
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)4u && int8_scale_term)
    {
        printf("[ log ]: runtime quantize the weight data\n");
        assert(0);
        Mat int8_weight_data(weight_data_size, (size_t)1u);
        if (int8_weight_data.empty())
            return -100;

        const int weight_data_size_g = weight_data_size / group;

        for (int g = 0; g < group; g++)
        {
            Option opt_q = opt;
            opt_q.blob_allocator = int8_weight_data.allocator;
            opt_q.use_packing_layout = false;

            const Mat weight_data_g = weight_data.range(weight_data_size_g * g, weight_data_size_g);
            Mat int8_weight_data_g = int8_weight_data.range(weight_data_size_g * g, weight_data_size_g);
            const Mat weight_data_int8_scales_g = weight_data_int8_scales.range(g, 1);
            quantize_to_int8(weight_data_g, int8_weight_data_g, weight_data_int8_scales_g, opt_q);
        }

        weight_data = int8_weight_data;
    }
#else
    (void)(opt);
#endif // NCNN_INT8

    return 0;
}

static int convolutiondepthwise(const Mat& bottom_blob, Mat& top_blob, const Mat& weight_data, const Mat& bias_data, int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int group, int activation_type, const Mat& activation_params, const Option& opt)
{
    const int w = bottom_blob.w;
    const int inch = bottom_blob.c;

    const int outw = top_blob.w;
    const int outh = top_blob.h;
    const int outch = top_blob.c;

    const int bias_term = bias_data.empty() ? 0 : 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // depth-wise
    if (inch == group && group == outch)
    {
        for (int g = 0; g < group; g++)
        {
            float* outptr = top_blob.channel(g);
            const float* kptr = (const float*)weight_data + maxk * g;
            const Mat m = bottom_blob.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    float sum = 0.f;

                    if (bias_term)
                        sum = bias_data[g];

                    const float* sptr = m.row(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        float val = sptr[space_ofs[k]];
                        float w = kptr[k];
                        sum += val * w;
                    }

                    outptr[j] = activation_ss(sum, activation_type, activation_params);
                }

                outptr += outw;
            }
        }
    }
    else
    {
        // group convolution
        const int inch_g = inch / group;
        const int outch_g = outch / group;

        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < outch_g; p++)
            {
                float* outptr = top_blob.channel(g * outch_g + p);
                const float* weight_data_ptr = (const float*)weight_data + maxk * inch_g * outch_g * g;

                // shadowed variable for less openmp task args
                const int outw = top_blob.w;
                const int outh = top_blob.h;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                            sum = bias_data[outch_g * g + p];

                        const float* kptr = weight_data_ptr + maxk * inch_g * p;

                        for (int q = 0; q < inch_g; q++)
                        {
                            const Mat m = bottom_blob.channel(inch_g * g + q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        outptr[j] = activation_ss(sum, activation_type, activation_params);
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

int ConvolutionDepthWise::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // convolv with NxN kernel
    // value = value + bias

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {

        #ifdef COMPARE_WITH_NCNN
        int channels = bottom_blob.c;
        assert(!layout_en || channels == group && group == num_output);
            ncnn::Mat top_blob_t;
            forward_int8_cpu(bottom_blob, top_blob, opt);
            forward_int8_npu(bottom_blob, top_blob_t, opt);
            if (CompareMat(top_blob, top_blob_t, 0.001) == 0) {
                printf("\033[;32m[ log ]: TEST LAYER DW CONV BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
                count_success++;
            } else {
                printf("\033[;31m[ log ]: TEST LAYER DW CONV BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
                count_fail ++;
                FILE *file;
                const char *filename = "error_test.txt";
                char buffer[200];
                sprintf(buffer,
                        "dw conv: ifm_w=%d, ifm_h=%d, ifm_c=%d, ofm_c=%d, kernel=%d, stride=%d, pad_left=%d, pad_right=%d, pad_top=%d, pad_bottom=%d\n",
                        ifm_w, ifm_h, ifm_c, num_output, kernel_w, stride_w, pad_left, pad_right, pad_top, pad_bottom);

                file = fopen(filename, "a");
                if (file == NULL) {
                    perror("Error opening file");
                }

                if (fputs(buffer, file) == EOF) {
                    perror("Error writing to file");
                    fclose(file);
                }

                fclose(file);
            }
            return 0;
        #else
            #ifdef FORWARD_ON_CPU_CONV
            return forward_int8_cpu(bottom_blob, top_blob, opt);
            #else
            int channels = bottom_blob.c;
            if (!layout_en || channels == group && group == num_output) {
                return forward_int8_npu(bottom_blob, top_blob, opt);
            } else {
                return forward_int8_cpu(bottom_blob, top_blob, opt);
            }
            #endif
        #endif
    }
#endif

    #ifdef PRINT_LAYER
        printf("[ log ]: Forward ConvolutionDepthWise on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret = convolutiondepthwise(bottom_blob_bordered, top_blob, weight_data, bias_data, kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

int ConvolutionDepthWise::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, _kernel_w, _kernel_h, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    const int w = bottom_blob_bordered.w;
    const int h = bottom_blob_bordered.h;
    const size_t elemsize = bottom_blob_bordered.elemsize;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    const int outw = (w - kernel_extent_w) / stride_w + 1;
    const int outh = (h - kernel_extent_h) / stride_h + 1;

    top_blob.create(outw, outh, _num_output, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    int ret = convolutiondepthwise(bottom_blob_bordered, top_blob, weight_data_flattened, bias_data_flattened, _kernel_w, _kernel_h, stride_w, stride_h, dilation_w, dilation_h, group, activation_type, activation_params, opt);
    if (ret != 0)
        return ret;

    return 0;
}

void ConvolutionDepthWise::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, const Option& opt) const
{
    make_padding(bottom_blob, bottom_blob_bordered, kernel_w, kernel_h, opt);
}

void ConvolutionDepthWise::make_padding(const Mat& bottom_blob, Mat& bottom_blob_bordered, int _kernel_w, int _kernel_h, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_extent_w = dilation_w * (_kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (_kernel_h - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

#if NCNN_INT8
static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

int ConvolutionDepthWise::forward_int8_cpu(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward ConvolutionDepthWise on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    // convolv with NxN kernel
    // value = value + bias

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    //     NCNN_LOGE("ConvolutionDepthWise input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_int8 = bottom_blob;
    if (elemsize != 1)
    {
        const int channels_g = channels / group;

        Mat scales(channels);
        {
            float* ps = scales;
            for (int g = 0; g < group; g++)
            {
                float scale = bottom_blob_int8_scales[g];
                for (int q = 0; q < channels_g; q++)
                {
                    *ps++ = scale;
                }
            }
        }

        Option opt_q = opt;
        opt_q.blob_allocator = opt.workspace_allocator;
        quantize_to_int8(bottom_blob, bottom_blob_int8, scales, opt_q);
    }

    Mat bottom_blob_bordered;
    make_padding(bottom_blob_int8, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // int8
    bool use_int8_requantize = int8_scale_term > 100;
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    // depth-wise
    if (channels == group && group == num_output)
    {
         
        for (int g = 0; g < group; g++)
        {
            signed char* outptr = top_blob.channel(g);
            const signed char* kptr = (const signed char*)weight_data + maxk * g;
            const Mat m = bottom_blob_bordered.channel(g);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    int sum = 0;

                    const signed char* sptr = m.row<signed char>(i * stride_h) + j * stride_w;

                    for (int k = 0; k < maxk; k++)
                    {
                        signed char val = sptr[space_ofs[k]];
                        signed char w = kptr[k];
                        sum += val * w;
                    }

                    float scale_in;
                    if (weight_data_int8_scales[g] == 0)
                        scale_in = 0;
                    else
                        scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                    float sumfp32 = sum * scale_in;

                    if (bias_term)
                        sumfp32 += bias_data[g];

                    sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

                    if (use_int8_requantize)
                    {
                        // requantize
                        float scale_out = top_blob_int8_scales[g];
                        signed char sums8 = float2int8(sumfp32 * scale_out);
                        outptr[0] = sums8;
                        outptr += 1;
                    }
                    else
                    {
                        // dequantize
                        ((float*)outptr)[0] = sumfp32;
                        outptr += 4;
                    }
                }
            }
        }
    }
    else
    {
        // group convolution
        const int channels_g = channels / group;
        const int num_output_g = num_output / group;

        for (int g = 0; g < group; g++)
        {
            for (int p = 0; p < num_output_g; p++)
            {
                signed char* outptr = top_blob.channel(g * num_output_g + p);
                const signed char* weight_data_ptr = (const signed char*)weight_data + maxk * channels_g * num_output_g * g;

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        int sum = 0;

                        const signed char* kptr = weight_data_ptr + maxk * channels_g * p;

                        // channels_g
                        for (int q = 0; q < channels_g; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(channels_g * g + q);
                            const signed char* sptr = m.row<signed char>(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                signed char val = sptr[space_ofs[k]];
                                signed char w = kptr[k];
                                sum += val * w;
                            }

                            kptr += maxk;
                        }

                        float scale_in;
                        if (weight_data_int8_scales[g] == 0)
                            scale_in = 0;
                        else
                            scale_in = 1.f / (bottom_blob_int8_scales[g] * weight_data_int8_scales[g]);

                        float sumfp32 = sum * scale_in;

                        if (bias_term)
                            sumfp32 += bias_data[g * num_output_g + p];

                        sumfp32 = activation_ss(sumfp32, activation_type, activation_params);

                        if (use_int8_requantize)
                        {
                            // requantize
                            float scale_out = top_blob_int8_scales[g];
                            signed char sums8 = float2int8(sumfp32 * scale_out);
                            outptr[0] = sums8;
                            outptr += 1;
                        }
                        else
                        {
                            // dequantize
                            ((float*)outptr)[0] = sumfp32;
                            outptr += 4;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

#ifdef FORWARD_ON_NPU_CONV
int ConvolutionDepthWise::forward_int8_npu(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward ConvolutionDepthWise on NPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    #ifdef CONV_TIME
		u64 tEnd_conv_hw, tCur_conv_hw;
		u32 tUsed_conv_hw;
        XTime_GetTime(&tCur_conv_hw);
    #endif
    accel::hw_gemm inst;
    inst.layout_en = layout_en;
    if(ifm_w == 0 || layout_en == 1){
        // assert(0);
        inst.ifm_w = bottom_blob.w;
        inst.ifm_h = bottom_blob.h;
        inst.ifm_c = bottom_blob.c;
    }
    else{
        inst.ifm_w = ifm_w;
        inst.ifm_h = ifm_h;
        inst.ifm_c = ifm_c;
    }

    inst.kernel = kernel_w;
    inst.stride = stride_w;
    inst.padding_bottom = pad_bottom;
    inst.padding_top = pad_top;
    inst.padding_left = pad_left;
    inst.padding_right = pad_right;
	inst.op = 1;

    inst.ofm_c = num_output;
	conv_get_ofm_pad(inst);

    inst.oscale_en = 1;
    inst.bias_en = bias_term;
    Mat& bias_data_cast = const_cast<Mat&>(bias_data);
    inst.bias_data = (float*)bias_data_cast.data;

    inst.quant_scale = bottom_blob_int8_scales[0];

    inst.act_op = activation_type;
    if(inst.act_op != 0){
        inst.act_alpha = activation_params[0];
        inst.act_beta  = activation_params[1];
    }

    bool use_int8_requantize = int8_scale_term > 100;
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    if(use_int8_requantize){
        inst.requant_en = 1;
        inst.requant_scale = top_blob_int8_scales[0];
        top_blob.create(u_align(inst.ofm_c,32), 1, inst.ofm_w*inst.ofm_h, out_elemsize, opt.blob_allocator);
    }
    else{
        top_blob.create(inst.ofm_w, inst.ofm_h, inst.ofm_c, out_elemsize, opt.blob_allocator);
    }

    Mat& bottom_blob_cast = const_cast<Mat&>(bottom_blob);
    Mat& weight_data_cast = const_cast<Mat&>(weight_data);

    #ifdef PRINT_SHAPE
    printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d\n",inst.ifm_w,inst.ifm_h,inst.ifm_c);
    printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n",inst.ofm_w,inst.ofm_h,inst.ofm_c);
    printf("[ log ]: kernel = %d, stride = %d\n",inst.kernel,inst.stride);
    printf("[ log ]: padding_left = %d, padding_right = %d, padding_top = %d, padding_bottom = %d\n",inst.padding_left,inst.padding_right,inst.padding_top,inst.padding_bottom);
    printf("[ log ]: bias_en = %d, requant_en = %d, act_op = %d, act_alpha = %f\n",inst.bias_en, inst.requant_en, inst.act_op,inst.act_alpha);
    printf("[ log ]: quant_scale = %.5f, dequant_ptr = %p, bias_ptr = %p \n",inst.quant_scale,inst.dequant_scale,inst.bias_data);
    #endif
    
    #ifndef INFERENCE_NET
        Mat dequant_scale(num_output,4u,opt.blob_allocator);
        float* dequant_scale_ptr = dequant_scale;
        for(int i=0; i<num_output; i++)
            dequant_scale_ptr[i] = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[i]);
        inst.dequant_scale = dequant_scale_ptr;

        ncnn::Mat wgt_buffer_res(1, 1, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel, 1u, opt.blob_allocator);
        ncnn::Mat wgt_buffer(32, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel / 32, 1u, opt.blob_allocator);
        wgt_reorder(weight_data_cast, wgt_buffer_res, inst.ifm_c, inst.ofm_c, inst.kernel, 0, 1);
        wgt_gen(wgt_buffer_res, wgt_buffer, inst.ifm_c,inst.ofm_c,inst.kernel, 1, 1);

        #ifdef MAT_LOG
        log_mat_file<unsigned char>(wgt_buffer_res,(char*)"./log/wgt_buffer_res.txt",-1,1,0);
        log_mat_file<unsigned char>(wgt_buffer,(char*)"./log/wgt_buffer.txt",-1,1,0);
        #endif

        inst.gemm_forward(bottom_blob_cast, wgt_buffer, top_blob);
    #else 
        inst.dequant_scale = (float*)weight_data_int8_scales.data;
        inst.gemm_forward(bottom_blob_cast, weight_data_cast, top_blob);
    #endif
    #ifdef CONV_TIME
		XTime_GetTime(&tEnd_conv_hw);
		tUsed_conv_hw = ((tEnd_conv_hw-tCur_conv_hw)*1000000)/(COUNTS_PER_SECOND);
		printf("time_conv_hw elapsed is %d us\n",tUsed_conv_hw);
    #endif



    
    return 0;




}
#endif

#endif // NCNN_INT8

} // namespace ncnn
