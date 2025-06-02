// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include "../ncnn/cpu.h"
#include "../ncnn/layer.h"
#include "../ncnn/mat.h"
#include "../config.h"
#include "prng.h"


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


static struct prng_rand_t g_prng_rand_state;
#define SRAND(seed) prng_srand(seed, &g_prng_rand_state)
#define RAND()      prng_rand(&g_prng_rand_state)

#define TEST_LAYER_DISABLE_AUTO_INPUT_PACKING (1 << 0)
#define TEST_LAYER_DISABLE_AUTO_INPUT_CASTING (1 << 1)
#define TEST_LAYER_DISABLE_GPU_TESTING        (1 << 2)
#define TEST_LAYER_ENABLE_FORCE_INPUT_PACK8   (1 << 3)

static void printf_int32_mat(ncnn::Mat a, char type = 'x'){
    int w = a.w;
    int h = a.h;
    int c = a.c;
    int d = a.d;
    int dims = a.dims;
    printf("[ log ]: shape(w,h,c,d) -> (%d,%d,%d,%d)\n",w,h,c,d);
    printf("[ log ]: dims -> %d, elemsize = %ld\n",dims,a.elemsize);
    if(dims == 4){
        printf("[error]: NO Implement\n");
        assert(0);
    }
    printf("[ log ]: Int32 Mat\n"); 
    if(dims == 1){
        signed int* ptr = a.row<signed int>(0);
        printf("[");
        for(int wi=0; wi<w; wi++)
            if(type == 'x')
                printf("%x ",ptr[wi]);
            else
                printf("%d ",ptr[wi]);
        printf("]\n");
    }
    else if(dims == 2){
        printf("[");
        for(int hi=0; hi<h; hi++){
            signed int* ptr = a.row<signed int>(hi);
            printf("[");
            for(int wi=0; wi<w; wi++)
                if(type == 'x')
                    printf("%x ",ptr[wi]);
                else
                    printf("%d ",ptr[wi]);
            printf("]\n");
        }
        printf("]\n");
    }
    else if(dims == 3){
        printf("[");
        for(int ci=0; ci<c; ci++){
            printf("-------------- ci = %d---------------\n",ci);
            printf("[");
            for(int hi=0; hi<h; hi++){
                signed int* ptr = a.channel(ci).row<signed int>(hi);
                printf("[");
                for(int wi=0; wi<w; wi++)
                    if(type == 'x')
                        printf("%x ",ptr[wi]);
                    else
                        printf("%d ",ptr[wi]);
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
    }
}

static void printf_int8_mat(ncnn::Mat a, char type = 'x'){
    int w = a.w;
    int h = a.h;
    int c = a.c;
    int d = a.d;
    int dims = a.dims;
    printf("[ log ]: shape(w,h,c,d) -> (%d,%d,%d,%d)\n",w,h,c,d);
    printf("[ log ]: dims -> %d, elemsize = %ld\n",dims,a.elemsize);
    if(dims == 4){
        printf("[error]: NO Implement\n");
        assert(0);
    }
    printf("[ log ]: Int8 Mat\n"); 

    if(dims == 1){
        unsigned char* ptr = a.row<unsigned char>(0);
        printf("[");
        for(int wi=0; wi<w; wi++)
            if(type == 'x')
                printf("%x ",ptr[wi]);
            else
                printf("%d ",ptr[wi]);
        printf("]\n");
    }
    else if(dims == 2){
        printf("[");
        for(int hi=0; hi<h; hi++){
            unsigned char* ptr = a.row<unsigned char>(hi);
            printf("[");
            for(int wi=0; wi<w; wi++)
                if(type == 'x')
                    printf("%x ",ptr[wi]);
                else
                    printf("%d ",ptr[wi]);
            printf("]\n");
        }
        printf("]\n");
    }
    else if(dims == 3){
        printf("[");
        for(int ci=0; ci<c; ci++){
            printf("-----------------ci = %d----------------\n",ci);
            unsigned char* ptr = a.channel(ci);
            printf("ci = %d, addr = %p\n",ci,ptr);
            printf("[");
            for(int hi=0; hi<3; hi++){
                printf("ci = %d, addr = %p\n",ci,ptr+hi*a.w);
                printf("[");
                for(int wi=0; wi<w; wi++)
                    if(type == 'x')
                        printf("%x ",ptr[wi+hi*a.w]);
                    else
                        printf("%d ",ptr[wi+hi*a.w]);
                printf("]\n");
            }
            printf("]\n");
        }
        printf("]\n");
    }
}

static void printf_float32_mat(ncnn::Mat a){
    int w = a.w;
    int h = a.h;
    int c = a.c;
    int d = a.d;
    int dims = a.dims;
    printf("[ log ]: shape(w,h,c,d) -> (%d,%d,%d,%d)\n",w,h,c,d);
    printf("[ log ]: dims -> %d\n",dims);
    if(dims == 4){
        printf("[error]: NO Implement\n");
        assert(0);
    }
        printf("[ log ]: Float32 Mat\n");
        if(dims == 1){
            float* ptr = a.row(0);
            printf("[");
            for(int wi=0; wi<w; wi++)
                printf("%f ",ptr[wi]);
            printf("]\n");
        }
        else if(dims == 2){
            printf("[");
            for(int hi=0; hi<h; hi++){
                float* ptr = a.row(hi);
                printf("[");
                for(int wi=0; wi<w; wi++)
                    printf("%f ",ptr[wi]);
                printf("]\n");
            }
            printf("]\n");
        }
        else if(dims == 3){
            printf("[");
            for(int ci=0; ci<c; ci++){
                printf("[");
                for(int hi=0; hi<h; hi++){
                    float* ptr = a.channel(ci).row(hi);
                    printf("[");
                    for(int wi=0; wi<w; wi++)
                        printf("%f ",ptr[wi]);
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
}

static void printf_mat(ncnn::Mat a){
    int w = a.w;
    int h = a.h;
    int c = a.c;
    int d = a.d;
    int dims = a.dims;
    int elesize = a.elemsize;
    printf("[ log ]: shape(w,h,c,d) -> (%d,%d,%d,%d)\n",w,h,c,d);
    printf("[ log ]: dims -> %d  elesize -> %d\n",dims,elesize);

    if(dims == 4){
        printf("[error]: NO Implement\n");
        assert(0);
    }

    if(elesize == 1){
        printf("[ log ]: SInt8 Mat\n"); 
        if(dims == 1){
            signed char* ptr = a.row<signed char>(0);
            printf("[");
            for(int wi=0; wi<w; wi++)
                printf("%d ",ptr[wi]);
            printf("]\n");
        }
        else if(dims == 2){
            printf("[");
            for(int hi=0; hi<h; hi++){
                signed char* ptr = a.row<signed char>(0);
                printf("[");
                for(int wi=0; wi<w; wi++)
                    printf("%d ",ptr[wi]);
                printf("]\n");
            }
            printf("]\n");
        }
        else if(dims == 3){
            printf("[");
            for(int ci=0; ci<c; ci++){
                printf("[");
                for(int hi=0; hi<h; hi++){
                    signed char* ptr = a.channel(ci).row<signed char>(hi);
                    printf("[");
                    for(int wi=0; wi<w; wi++)
                        printf("%d ",ptr[wi]);
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }

    }
    else if(elesize == 4){
        printf("[ log ]: float32 Mat\n");
        if(dims == 1){
            float* ptr = a.row(0);
            printf("[");
            for(int wi=0; wi<w; wi++)
                printf("%f ",ptr[wi]);
            printf("]\n");
        }
        else if(dims == 2){
            printf("[");
            for(int hi=0; hi<h; hi++){
                float* ptr = a.row(hi);
                printf("[");
                for(int wi=0; wi<w; wi++)
                    printf("%f ",ptr[wi]);
                printf("]\n");
            }
            printf("]\n");
        }
        else if(dims == 3){
            printf("[");
            for(int ci=0; ci<c; ci++){
                printf("[");
                for(int hi=0; hi<h; hi++){
                    float* ptr = a.channel(ci).row(hi);
                    printf("[");
                    for(int wi=0; wi<w; wi++)
                        printf("%f ",ptr[wi]);
                    printf("]\n");
                }
                printf("]\n");
            }
            printf("]\n");
        }
    }
}

static float RandomFloat(float a = -1.2f, float b = 1.2f)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    float v = a + r;
    // generate denormal as zero
    if (v < 0.0001 && v > -0.0001)
        v = 0.f;
    return v;
}

static int RandomInt(int a = -10000, int b = 10000)
{
    float random = ((float)RAND()) / (float)uint64_t(-1); //RAND_MAX;
    int diff = b - a;
    float r = random * diff;
    return a + (int)r;
}

static signed char RandomS8()
{
    return (signed char)RandomInt(-127, 127);
}

static void Randomize(ncnn::Mat& m, float a = -1.2f, float b = 1.2f)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        m[i] = RandomFloat(a, b);
    }
}

static void RandomizeInt(ncnn::Mat& m, int a = -10000, int b = 10000)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        ((int*)m)[i] = RandomInt(a, b);
    }
}

static void RandomizeS8(ncnn::Mat& m)
{
    for (size_t i = 0; i < m.total(); i++)
    {
        ((signed char*)m)[i] = RandomS8();
    }
}

static ncnn::Mat RandomMat(int w, float a = -1.2f, float b = 1.2f)
{
    ncnn::Mat m(w);
    Randomize(m, a, b);
    return m;
}

static ncnn::Mat RandomMat(int w, int h, float a = -1.2f, float b = 1.2f)
{
    ncnn::Mat m(w, h);
    Randomize(m, a, b);
    return m;
}

static ncnn::Mat RandomMat(int w, int h, int c, float a = -1.2f, float b = 1.2f)
{
    ncnn::Mat m(w, h, c);
    Randomize(m, a, b);
    return m;
}

static ncnn::Mat RandomMat(int w, int h, int d, int c, float a = -1.2f, float b = 1.2f)
{
    ncnn::Mat m(w, h, d, c);
    Randomize(m, a, b);
    return m;
}

static ncnn::Mat RandomIntMat(int w, float a = -10000.f, float b = 10000.f)
{
    ncnn::Mat m(w);
    RandomizeInt(m, (int)a, (int)b);
    return m;
}

static ncnn::Mat RandomIntMat(int w, int h, float a = -10000.f, float b = 10000.f)
{
    ncnn::Mat m(w, h);
    RandomizeInt(m, (int)a, (int)b);
    return m;
}

static ncnn::Mat RandomIntMat(int w, int h, int c, float a = -10000.f, float b = 10000.f)
{
    ncnn::Mat m(w, h, c);
    RandomizeInt(m, (int)a, (int)b);
    return m;
}

static ncnn::Mat RandomIntMat(int w, int h, int d, int c, float a = -10000.f, float b = 10000.f)
{
    ncnn::Mat m(w, h, d, c);
    RandomizeInt(m, (int)a, (int)b);
    return m;
}

static ncnn::Mat RandomS8Mat(int w)
{
    ncnn::Mat m(w, (size_t)1u);
    RandomizeS8(m);
    return m;
}

static ncnn::Mat RandomS8Mat(int w, int h)
{
    ncnn::Mat m(w, h, (size_t)1u);
    RandomizeS8(m);
    return m;
}

static ncnn::Mat RandomS8Mat(int w, int h, int c)
{
    ncnn::Mat m(w, h, c, (size_t)1u);
    RandomizeS8(m);
    return m;
}

static ncnn::Mat RandomS8Mat(int w, int h, int d, int c)
{
    ncnn::Mat m(w, h, d, c, (size_t)1u);
    RandomizeS8(m);
    return m;
}

static ncnn::Mat scales_mat(const ncnn::Mat& mat, int m, int k, int ldx)
{
    ncnn::Mat weight_scales(m);
    for (int i = 0; i < m; ++i)
    {
        float min = mat[0], _max = mat[0];
        const float* ptr = (const float*)(mat.data) + i * ldx;
        for (int j = 0; j < k; ++j)
        {
            if (min > ptr[j])
            {
                min = ptr[j];
            }
            if (_max < ptr[j])
            {
                _max = ptr[j];
            }
        }
        const float abs_min = abs(min), abs_max = abs(_max);
        weight_scales[i] = 127.f / (abs_min > abs_max ? abs_min : abs_max);
    }
    return weight_scales;
}

static bool NearlyEqual(float a, float b, float epsilon)
{
    if (a == b)
        return true;

    float diff = (float)fabs(a - b);
    if (diff <= epsilon)
        return true;

    // relative error
    return diff < epsilon * std::max(fabs(a), fabs(b));
}

static int Compare(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001)
{
#define CHECK_MEMBER(m)                                                                 \
    if (a.m != b.m)                                                                     \
    {                                                                                   \
        printf("[error]: not match    expect %d but got %d\n", (int)a.m, (int)b.m); \
        return -1;                                                                      \
    }

    CHECK_MEMBER(dims)
    CHECK_MEMBER(w)
    CHECK_MEMBER(h)
    CHECK_MEMBER(d)
    CHECK_MEMBER(c)
    CHECK_MEMBER(elemsize)
    CHECK_MEMBER(elempack)

#undef CHECK_MEMBER

    for (int q = 0; q < a.c; q++)
    {
        const ncnn::Mat ma = a.channel(q);
        const ncnn::Mat mb = b.channel(q);
        for (int z = 0; z < a.d; z++)
        {
            const ncnn::Mat da = ma.depth(z);
            const ncnn::Mat db = mb.depth(z);
            for (int i = 0; i < a.h; i++)
            {
                const float* pa = da.row(i);
                const float* pb = db.row(i);
                for (int j = 0; j < a.w; j++)
                {
                    if (!NearlyEqual(pa[j], pb[j], epsilon))
                    {
                        printf("[error]: value not match  at c:%d d:%d h:%d w:%d    expect %f but got %f\n", q, z, i, j, pa[j], pb[j]);
//                        printf("[ log ]: software data \n");
//                        printf_float32_mat(a.channel(31));
//                        printf_float32_mat(a.channel(63));
//                        printf("[ log ]: hardware data \n");
//                        printf_float32_mat(b.channel(31));
//                        printf_float32_mat(b.channel(63));
                        #ifndef PRINT_COMPARE_ALL
                            return -1;
                        #endif
                    }
                }
            }
        }
    }

    return 0;
}

static int CompareMat(const ncnn::Mat& a, const ncnn::Mat& b, float epsilon = 0.001)
{
    ncnn::Option opt;
    opt.num_threads = 1;

    if (a.elempack != 1)
    {
        ncnn::Mat a1;
        ncnn::convert_packing(a, a1, 1, opt);
        return CompareMat(a1, b, epsilon);
    }

    if (b.elempack != 1)
    {
        ncnn::Mat b1;
        ncnn::convert_packing(b, b1, 1, opt);
        return CompareMat(a, b1, epsilon);
    }

    if (a.elemsize == 2u)
    {
        ncnn::Mat a32;
        cast_float16_to_float32(a, a32, opt);
        return CompareMat(a32, b, epsilon);
    }
    if (a.elemsize == 1u)
    {
        ncnn::Mat a32;
        cast_int8_to_float32(a, a32, opt);
        return CompareMat(a32, b, epsilon);
    }

    if (b.elemsize == 2u)
    {
        ncnn::Mat b32;
        cast_float16_to_float32(b, b32, opt);
        return CompareMat(a, b32, epsilon);
    }
    if (b.elemsize == 1u)
    {
        ncnn::Mat b32;
        cast_int8_to_float32(b, b32, opt);
        return CompareMat(a, b32, epsilon);
    }

    return Compare(a, b, epsilon);
}

static int CompareMat(const std::vector<ncnn::Mat>& a, const std::vector<ncnn::Mat>& b, float epsilon = 0.001)
{
    if (a.size() != b.size())
    {
        printf("[error]: output blob count not match %zu %zu\n", a.size(), b.size());
        return -1;
    }

    for (size_t i = 0; i < a.size(); i++)
    {
        if (CompareMat(a[i], b[i], epsilon))
        {
            printf("[error]: output blob %zu not match\n", i);
            return -1;
        }
    }

    return 0;
}

static int countDifferentElements_mat(const ncnn::Mat& mat1, const ncnn::Mat& mat2) {
    if (mat1.w != mat2.w || mat1.h != mat2.h || mat1.c != mat2.c) {
        // 缁村害涓嶅悓锛屾棤娉曟瘮杈�
        printf("[error]: output blob count not match \n");
        return -1; // 杩斿洖閿欒鐮�
    }

    int differentCount = 0;

    if (mat1.elemsize == 1) {
        // 濡傛灉elemsize涓�1锛岃〃绀烘瘡涓厓绱犲崰涓�涓瓧鑺傦紝鍙兘鏄�8浣嶆暣鏁�
        for (int c = 0; c < mat1.c; ++c) {
            for (int h = 0; h < mat1.h; ++h) {
                for (int w = 0; w < mat1.w; ++w) {
                    int8_t val1 = mat1.channel(c).row(h)[w];
                    int8_t val2 = mat2.channel(c).row(h)[w];

                    // 姣旇緝涓や釜鍏冪礌鏄惁鐩哥瓑
                    if (val1 - val2 > 2) {
                        ++differentCount;
                    }
                }
            }
        }
    } else {
        // elemsize涓嶄负1鏃讹紝鍋囪鍏冪礌涓烘诞鐐规暟
        for (int c = 0; c < mat1.c; ++c) {
            for (int h = 0; h < mat1.h; ++h) {
                for (int w = 0; w < mat1.w; ++w) {
                    float val1 = mat1.channel(c).row(h)[w];
                    float val2 = mat2.channel(c).row(h)[w];

                    // 姣旇緝涓や釜鍏冪礌鏄惁鐩哥瓑
                    if (val1 - val2 >0.001) {
                        ++differentCount;
                    }
                }
            }
        }
    }

    return differentCount;
}


template<typename T>
int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, int top_blob_count, std::vector<ncnn::Mat>& b, void (*func)(T*), int flag)
{
    // printf("test_layer_naive, vector\n");
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func)
    {
        (*func)((T*)op);
    }

    op->load_param(pd);

    if (op->one_blob_only && a.size() != 1)
    {
        printf("[error]: layer with one_blob_only but consume multiple inputs\n");
        delete op;
        return -1;
    }

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    b.resize(top_blob_count);

    if (op->support_inplace)
    {
        for (size_t i = 0; i < a.size(); i++)
        {
            b[i] = a[i].clone();
        }

        ((T*)op)->T::forward_inplace(b, opt);
    }
    else
    {
        ((T*)op)->T::forward(a, b, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

template<typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const std::vector<ncnn::Mat>& a, std::vector<ncnn::Mat>& b, int top_blob_count, const std::vector<ncnn::Mat>& top_shapes = std::vector<ncnn::Mat>(), float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // printf("test 1\n");
    // naive
    // std::vector<ncnn::Mat> b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, top_blob_count, b, func, flag);
        if (ret != 0)
        {
            printf("[error]: test_layer_naive failed\n");
            return -1;
        }
    }

    return 0;
}

template<typename T>
int test_layer_naive(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, ncnn::Mat& b, void (*func)(T*), int flag)
{
    // printf("test_layer_naive, mat\n");
    ncnn::Layer* op = ncnn::create_layer(typeindex);

    if (func)
    {
        (*func)((T*)op);
    }

    op->load_param(pd);

    ncnn::ModelBinFromMatArray mb(weights.data());

    op->load_model(mb);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.lightmode = false;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
    opt.use_bf16_storage = false;
    opt.use_vulkan_compute = false;

    op->create_pipeline(opt);

    if (op->support_inplace)
    {
        b = a.clone();
        ((T*)op)->T::forward_inplace(b, opt);
    }
    else
    {
        ((T*)op)->T::forward(a, b, opt);
    }

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

template<typename T>
int test_layer(int typeindex, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Option& _opt, const ncnn::Mat& a, ncnn::Mat& b, const ncnn::Mat& top_shape = ncnn::Mat(), float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // printf("test 2\n");
    // naive
    // ncnn::Mat b;
    {
        int ret = test_layer_naive(typeindex, pd, weights, a, b, func, flag);
        if (ret != 0)
        {
            printf("[error]: test_layer_naive failed\n");
            return -1;
        }
    }
    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const std::vector<ncnn::Mat>& a, std::vector<ncnn::Mat>& b, int top_blob_count = 1, float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // printf("test 3\n");
    //eltwise
    ncnn::Option opt;

    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
    opt.num_threads = 1;

    std::vector<ncnn::Mat> top_shapes;
    int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights, opt, a, b, top_blob_count, top_shapes, epsilon, func, flag);
    if (ret != 0)
    {
        printf("[error]: test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_fp16_arithmetic=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d use_sgemm_convolution=%d use_winograd_convolution=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_shader_pack8, opt.use_bf16_storage, opt.use_image_storage, opt.use_sgemm_convolution, opt.use_winograd_convolution);
        return ret;
    }

    return 0;
}

template<typename T>
int test_layer(const char* layer_type, const ncnn::ParamDict& pd, const std::vector<ncnn::Mat>& weights, const ncnn::Mat& a, ncnn::Mat& b, float epsilon = 0.001, void (*func)(T*) = 0, int flag = 0)
{
    // printf("test 4\n");
    //conv
    ncnn::Option opt;
    opt.use_packing_layout = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
    opt.use_shader_pack8 = false;
    opt.use_image_storage = false;
    opt.num_threads = 1; 

    ncnn::Mat top_shape;
    int ret = test_layer<T>(ncnn::layer_to_index(layer_type), pd, weights, opt, a, b, top_shape, epsilon, func, flag);
    if (ret != 0)
    {
        printf("[error]: test_layer %s failed use_packing_layout=%d use_fp16_packed=%d use_fp16_storage=%d use_fp16_arithmetic=%d use_shader_pack8=%d use_bf16_storage=%d use_image_storage=%d use_sgemm_convolution=%d use_winograd_convolution=%d\n", layer_type, opt.use_packing_layout, opt.use_fp16_packed, opt.use_fp16_storage, opt.use_fp16_arithmetic, opt.use_shader_pack8, opt.use_bf16_storage, opt.use_image_storage, opt.use_sgemm_convolution, opt.use_winograd_convolution);
        return ret;
    }


    return 0;
}

#endif // TESTUTIL_H
