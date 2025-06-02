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

#include "mat.h"

#include <limits.h>
#include <math.h>
#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL
static int from_rgb(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr0 = rgb[0];
            *ptr1 = rgb[1];
            *ptr2 = rgb[2];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgb += wgap;
    }

    return 0;
}

static void to_rgb(const Mat& m, unsigned char* rgb, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;

        for (; remain > 0; remain--)
        {
            rgb[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgb[2] = SATURATE_CAST_UCHAR(*ptr2);

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgb += wgap;
    }
}

static int from_gray(const unsigned char* gray, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr = *gray;

            gray++;
            ptr++;
        }

        gray += wgap;
    }

    return 0;
}

static void to_gray(const Mat& m, unsigned char* gray, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;

        for (; remain > 0; remain--)
        {
            *gray = SATURATE_CAST_UCHAR(*ptr);

            gray++;
            ptr++;
        }

#undef SATURATE_CAST_UCHAR
        gray += wgap;
    }
}

static int from_rgba(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);
    float* ptr3 = m.channel(3);

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr0 = rgba[0];
            *ptr1 = rgba[1];
            *ptr2 = rgba[2];
            *ptr3 = rgba[3];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

        rgba += wgap;
    }

    return 0;
}

static void to_rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);
    const float* ptr3 = m.channel(3);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;

        for (; remain > 0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[3] = SATURATE_CAST_UCHAR(*ptr3);

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_rgb2bgr(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr0 = rgb[2];
            *ptr1 = rgb[1];
            *ptr2 = rgb[0];

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgb += wgap;
    }

    return 0;
}

static void to_bgr2rgb(const Mat& m, unsigned char* rgb, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;

        for (; remain > 0; remain--)
        {
            rgb[2] = SATURATE_CAST_UCHAR(*ptr0);
            rgb[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgb[0] = SATURATE_CAST_UCHAR(*ptr2);

            rgb += 3;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgb += wgap;
    }
}

static int from_rgb2gray(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((rgb[0] * R2Y + rgb[1] * G2Y + rgb[2] * B2Y) >> Y_shift);

            rgb += 3;
            ptr++;
        }

        rgb += wgap;
    }

    return 0;
}

static int from_rgb2rgba(const unsigned char* rgb, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_rgb(rgb, w, h, stride, rgb_channels, allocator);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_rgb2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;

        for (; remain > 0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[3] = 255;

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_bgr2gray(const unsigned char* bgr, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 3;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((bgr[2] * R2Y + bgr[1] * G2Y + bgr[0] * B2Y) >> Y_shift);

            bgr += 3;
            ptr++;
        }

        bgr += wgap;
    }

    return 0;
}

static int from_bgr2rgba(const unsigned char* bgr, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_rgb2bgr(bgr, w, h, stride, rgb_channels, allocator);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_bgr2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;

        for (; remain > 0; remain--)
        {
            rgba[0] = SATURATE_CAST_UCHAR(*ptr2);
            rgba[1] = SATURATE_CAST_UCHAR(*ptr1);
            rgba[2] = SATURATE_CAST_UCHAR(*ptr0);
            rgba[3] = 255;

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_gray2rgb(const unsigned char* gray, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr0 = *gray;
            *ptr1 = *gray;
            *ptr2 = *gray;

            gray++;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        gray += wgap;
    }

    return 0;
}

static int from_gray2rgba(const unsigned char* gray, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    Mat rgb_channels = m.channel_range(0, 3);
    from_gray2rgb(gray, w, h, stride, rgb_channels, allocator);

    Mat alpha_channel = m.channel(3);
    alpha_channel.fill(255.f);

    return 0;
}

static void to_gray2rgba(const Mat& m, unsigned char* rgba, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr = m;

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;

        for (; remain > 0; remain--)
        {
            unsigned char gray = SATURATE_CAST_UCHAR(*ptr);
            rgba[0] = gray;
            rgba[1] = gray;
            rgba[2] = gray;
            rgba[3] = 255;

            rgba += 4;
            ptr++;
        }

#undef SATURATE_CAST_UCHAR
        rgba += wgap;
    }
}

static int from_rgba2rgb(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr0 = rgba[0];
            *ptr1 = rgba[1];
            *ptr2 = rgba[2];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2bgr(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 3, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr0 = rgba[2];
            *ptr1 = rgba[1];
            *ptr2 = rgba[0];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2gray(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((rgba[0] * R2Y + rgba[1] * G2Y + rgba[2] * B2Y) >> Y_shift);

            rgba += 4;
            ptr++;
        }

        rgba += wgap;
    }

    return 0;
}

static int from_rgba2bgra(const unsigned char* rgba, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    m.create(w, h, 4, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr0 = m.channel(0);
    float* ptr1 = m.channel(1);
    float* ptr2 = m.channel(2);
    float* ptr3 = m.channel(3);

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr0 = rgba[2];
            *ptr1 = rgba[1];
            *ptr2 = rgba[0];
            *ptr3 = rgba[3];

            rgba += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

        rgba += wgap;
    }

    return 0;
}

static void to_rgba2bgra(const Mat& m, unsigned char* bgra, int stride)
{
    int w = m.w;
    int h = m.h;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    const float* ptr0 = m.channel(0);
    const float* ptr1 = m.channel(1);
    const float* ptr2 = m.channel(2);
    const float* ptr3 = m.channel(3);

    for (int y = 0; y < h; y++)
    {
#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);

        int remain = w;

        for (; remain > 0; remain--)
        {
            bgra[0] = SATURATE_CAST_UCHAR(*ptr2);
            bgra[1] = SATURATE_CAST_UCHAR(*ptr1);
            bgra[2] = SATURATE_CAST_UCHAR(*ptr0);
            bgra[3] = SATURATE_CAST_UCHAR(*ptr3);

            bgra += 4;
            ptr0++;
            ptr1++;
            ptr2++;
            ptr3++;
        }

#undef SATURATE_CAST_UCHAR
        bgra += wgap;
    }
}

static int from_bgra2gray(const unsigned char* bgra, int w, int h, int stride, Mat& m, Allocator* allocator)
{
    // coeffs for r g b = 0.299f, 0.587f, 0.114f
    const unsigned char Y_shift = 8; //14
    const unsigned char R2Y = 77;
    const unsigned char G2Y = 150;
    const unsigned char B2Y = 29;

    m.create(w, h, 1, 4u, allocator);
    if (m.empty())
        return -100;

    const int wgap = stride - w * 4;
    if (wgap == 0)
    {
        w = w * h;
        h = 1;
    }

    float* ptr = m;

    for (int y = 0; y < h; y++)
    {
        int remain = w;

        for (; remain > 0; remain--)
        {
            *ptr = static_cast<float>((bgra[2] * R2Y + bgra[1] * G2Y + bgra[0] * B2Y) >> Y_shift);

            bgra += 4;
            ptr++;
        }

        bgra += wgap;
    }

    return 0;
}

void yuv420sp2rgb(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb)
{
    const unsigned char* yptr = yuv420sp;
    const unsigned char* vuptr = yuv420sp + w * h;

    for (int y = 0; y < h; y += 2)
    {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = rgb;
        unsigned char* rgb1 = rgb + w * 3;

        int remain = w;

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);
        for (; remain > 0; remain -= 2)
        {
            // R = 1.164 * yy + 1.596 * vv
            // G = 1.164 * yy - 0.813 * vv - 0.391 * uu
            // B = 1.164 * yy              + 2.018 * uu

            // R = Y + (1.370705 * (V-128))
            // G = Y - (0.698001 * (V-128)) - (0.337633 * (U-128))
            // B = Y + (1.732446 * (U-128))

            // R = ((Y << 6) + 87.72512 * (V-128)) >> 6
            // G = ((Y << 6) - 44.672064 * (V-128) - 21.608512 * (U-128)) >> 6
            // B = ((Y << 6) + 110.876544 * (U-128)) >> 6

            // R = ((Y << 6) + 90 * (V-128)) >> 6
            // G = ((Y << 6) - 46 * (V-128) - 22 * (U-128)) >> 6
            // B = ((Y << 6) + 113 * (U-128)) >> 6

            // R = (yy + 90 * vv) >> 6
            // G = (yy - 46 * vv - 22 * uu) >> 6
            // B = (yy + 113 * uu) >> 6

            int v = vuptr[0] - 128;
            int u = vuptr[1] - 128;

            int ruv = 90 * v;
            int guv = -46 * v + -22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            rgb0[0] = SATURATE_CAST_UCHAR((y00 + ruv) >> 6);
            rgb0[1] = SATURATE_CAST_UCHAR((y00 + guv) >> 6);
            rgb0[2] = SATURATE_CAST_UCHAR((y00 + buv) >> 6);

            int y01 = yptr0[1] << 6;
            rgb0[3] = SATURATE_CAST_UCHAR((y01 + ruv) >> 6);
            rgb0[4] = SATURATE_CAST_UCHAR((y01 + guv) >> 6);
            rgb0[5] = SATURATE_CAST_UCHAR((y01 + buv) >> 6);

            int y10 = yptr1[0] << 6;
            rgb1[0] = SATURATE_CAST_UCHAR((y10 + ruv) >> 6);
            rgb1[1] = SATURATE_CAST_UCHAR((y10 + guv) >> 6);
            rgb1[2] = SATURATE_CAST_UCHAR((y10 + buv) >> 6);

            int y11 = yptr1[1] << 6;
            rgb1[3] = SATURATE_CAST_UCHAR((y11 + ruv) >> 6);
            rgb1[4] = SATURATE_CAST_UCHAR((y11 + guv) >> 6);
            rgb1[5] = SATURATE_CAST_UCHAR((y11 + buv) >> 6);

            yptr0 += 2;
            yptr1 += 2;
            vuptr += 2;
            rgb0 += 6;
            rgb1 += 6;
        }
#undef SATURATE_CAST_UCHAR

        yptr += 2 * w;
        rgb += 2 * 3 * w;
    }
}

void yuv420sp2rgb_nv12(const unsigned char* yuv420sp, int w, int h, unsigned char* rgb)
{
    const unsigned char* yptr = yuv420sp;
    const unsigned char* uvptr = yuv420sp + w * h;

    for (int y = 0; y < h; y += 2)
    {
        const unsigned char* yptr0 = yptr;
        const unsigned char* yptr1 = yptr + w;
        unsigned char* rgb0 = rgb;
        unsigned char* rgb1 = rgb + w * 3;

        int remain = w;

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);
        for (; remain > 0; remain -= 2)
        {
            // R = 1.164 * yy + 1.596 * vv
            // G = 1.164 * yy - 0.813 * vv - 0.391 * uu
            // B = 1.164 * yy              + 2.018 * uu

            // R = Y + (1.370705 * (V-128))
            // G = Y - (0.698001 * (V-128)) - (0.337633 * (U-128))
            // B = Y + (1.732446 * (U-128))

            // R = ((Y << 6) + 87.72512 * (V-128)) >> 6
            // G = ((Y << 6) - 44.672064 * (V-128) - 21.608512 * (U-128)) >> 6
            // B = ((Y << 6) + 110.876544 * (U-128)) >> 6

            // R = ((Y << 6) + 90 * (V-128)) >> 6
            // G = ((Y << 6) - 46 * (V-128) - 22 * (U-128)) >> 6
            // B = ((Y << 6) + 113 * (U-128)) >> 6

            // R = (yy + 90 * vv) >> 6
            // G = (yy - 46 * vv - 22 * uu) >> 6
            // B = (yy + 113 * uu) >> 6

            int u = uvptr[0] - 128;
            int v = uvptr[1] - 128;

            int ruv = 90 * v;
            int guv = -46 * v + -22 * u;
            int buv = 113 * u;

            int y00 = yptr0[0] << 6;
            rgb0[0] = SATURATE_CAST_UCHAR((y00 + ruv) >> 6);
            rgb0[1] = SATURATE_CAST_UCHAR((y00 + guv) >> 6);
            rgb0[2] = SATURATE_CAST_UCHAR((y00 + buv) >> 6);

            int y01 = yptr0[1] << 6;
            rgb0[3] = SATURATE_CAST_UCHAR((y01 + ruv) >> 6);
            rgb0[4] = SATURATE_CAST_UCHAR((y01 + guv) >> 6);
            rgb0[5] = SATURATE_CAST_UCHAR((y01 + buv) >> 6);

            int y10 = yptr1[0] << 6;
            rgb1[0] = SATURATE_CAST_UCHAR((y10 + ruv) >> 6);
            rgb1[1] = SATURATE_CAST_UCHAR((y10 + guv) >> 6);
            rgb1[2] = SATURATE_CAST_UCHAR((y10 + buv) >> 6);

            int y11 = yptr1[1] << 6;
            rgb1[3] = SATURATE_CAST_UCHAR((y11 + ruv) >> 6);
            rgb1[4] = SATURATE_CAST_UCHAR((y11 + guv) >> 6);
            rgb1[5] = SATURATE_CAST_UCHAR((y11 + buv) >> 6);

            yptr0 += 2;
            yptr1 += 2;
            uvptr += 2;
            rgb0 += 6;
            rgb1 += 6;
        }
#undef SATURATE_CAST_UCHAR

        yptr += 2 * w;
        rgb += 2 * 3 * w;
    }
}

void yuv420sp2rgb_half(const unsigned char* yuv, int w, int h, unsigned char* rgb)
{
    const unsigned char* puv = yuv + w * h;
    const unsigned char *py0 = yuv, *py1 = yuv + w;
    const int hstep = h / 2;

    const int tailstep = w / 2;

    for (int i = 0; i < hstep; ++i)
    {
        for (int idx = 0; idx < tailstep; ++idx)
        {
            int y = (static_cast<int>(py0[0]) + py0[1] + py1[2] + py1[1]) << 4;
            int v = static_cast<int>(puv[0]) - 128;
            int u = static_cast<int>(puv[1]) - 128;

            int ruv = 90 * v;
            int guv = -46 * v + -22 * u;
            int buv = 113 * u;

#define SATURATE_CAST_UCHAR(X) (unsigned char)::std::min(::std::max((int)(X), 0), 255);
            rgb[0] = SATURATE_CAST_UCHAR((y + ruv) >> 6);
            rgb[1] = SATURATE_CAST_UCHAR((y + guv) >> 6);
            rgb[2] = SATURATE_CAST_UCHAR((y + buv) >> 6);
#undef SATURATE_CAST_UCHAR

            rgb += 3;
            py0 += 2;
            py1 += 2;
            puv += 2;
        }
        // next two row
        py0 = py1;
        py1 = py0 + w;
    }
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, Allocator* allocator)
{
    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 3, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 1, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return Mat::from_pixels(pixels, type, w, h, w * 4, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels(const unsigned char* pixels, int type, int w, int h, int stride, Allocator* allocator)
{
    Mat m;

    if (type & PIXEL_CONVERT_MASK)
    {
        switch (type)
        {
        case PIXEL_RGB2BGR:
        case PIXEL_BGR2RGB:
            from_rgb2bgr(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGB2GRAY:
            from_rgb2gray(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGB2RGBA:
        case PIXEL_BGR2BGRA:
            from_rgb2rgba(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_BGR2GRAY:
            from_bgr2gray(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_BGR2RGBA:
        case PIXEL_RGB2BGRA:
            from_bgr2rgba(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_GRAY2RGB:
        case PIXEL_GRAY2BGR:
            from_gray2rgb(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_GRAY2RGBA:
        case PIXEL_GRAY2BGRA:
            from_gray2rgba(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGBA2RGB:
        case PIXEL_BGRA2BGR:
            from_rgba2rgb(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGBA2BGR:
        case PIXEL_BGRA2RGB:
            from_rgba2bgr(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGBA2GRAY:
            from_rgba2gray(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_RGBA2BGRA:
        case PIXEL_BGRA2RGBA:
            from_rgba2bgra(pixels, w, h, stride, m, allocator);
            break;
        case PIXEL_BGRA2GRAY:
            from_bgra2gray(pixels, w, h, stride, m, allocator);
            break;
        default:
            // unimplemented convert type
            NCNN_LOGE("unimplemented convert type %d", type);
            break;
        }
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            from_rgb(pixels, w, h, stride, m, allocator);

        if (type == PIXEL_GRAY)
            from_gray(pixels, w, h, stride, m, allocator);

        if (type == PIXEL_RGBA || type == PIXEL_BGRA)
            from_rgba(pixels, w, h, stride, m, allocator);
    }

    return m;
}

Mat Mat::from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int target_width, int target_height, Allocator* allocator)
{
    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 3, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 1, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return Mat::from_pixels_resize(pixels, type, w, h, w * 4, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_resize(const unsigned char* pixels, int type, int w, int h, int stride, int target_width, int target_height, Allocator* allocator)
{
    if (w == target_width && h == target_height)
        return Mat::from_pixels(pixels, type, w, h, stride, allocator);

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        Mat dst(target_width, target_height, (size_t)3u, 3);
        resize_bilinear_c3(pixels, w, h, stride, dst, target_width, target_height, target_width * 3);

        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        Mat dst(target_width, target_height, (size_t)1u, 1);
        resize_bilinear_c1(pixels, w, h, stride, dst, target_width, target_height, target_width * 1);

        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        Mat dst(target_width, target_height, (size_t)4u, 4);
        resize_bilinear_c4(pixels, w, h, stride, dst, target_width, target_height, target_width * 4);

        return Mat::from_pixels(dst, type, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels(pixels + (roiy * w + roix) * 3, type, roiw, roih, w * 3, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels(pixels + (roiy * w + roix) * 1, type, roiw, roih, w * 1, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels(pixels + (roiy * w + roix) * 4, type, roiw, roih, w * 4, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels(pixels + roiy * stride + roix * 3, type, roiw, roih, stride, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels(pixels + roiy * stride + roix * 1, type, roiw, roih, stride, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels(pixels + roiy * stride + roix * 4, type, roiw, roih, stride, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 3, type, roiw, roih, w * 3, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 1, type, roiw, roih, w * 1, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels_resize(pixels + (roiy * w + roix) * 4, type, roiw, roih, w * 4, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

Mat Mat::from_pixels_roi_resize(const unsigned char* pixels, int type, int w, int h, int stride, int roix, int roiy, int roiw, int roih, int target_width, int target_height, Allocator* allocator)
{
    if (roix < 0 || roiy < 0 || roiw <= 0 || roih <= 0 || roix + roiw > w || roiy + roih > h)
    {
        NCNN_LOGE("roi %d %d %d %d out of image %d %d", roix, roiy, roiw, roih, w, h);
        return Mat();
    }

    int type_from = type & PIXEL_FORMAT_MASK;

    if (type_from == PIXEL_RGB || type_from == PIXEL_BGR)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 3, type, roiw, roih, stride, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_GRAY)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 1, type, roiw, roih, stride, target_width, target_height, allocator);
    }
    else if (type_from == PIXEL_RGBA || type_from == PIXEL_BGRA)
    {
        return from_pixels_resize(pixels + roiy * stride + roix * 4, type, roiw, roih, stride, target_width, target_height, allocator);
    }

    // unknown convert type
    NCNN_LOGE("unknown convert type %d", type);
    return Mat();
}

void Mat::to_pixels(unsigned char* pixels, int type) const
{
    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        to_pixels(pixels, type, w * 3);
    }
    else if (type_to == PIXEL_GRAY)
    {
        to_pixels(pixels, type, w * 1);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        to_pixels(pixels, type, w * 4);
    }
}

void Mat::to_pixels(unsigned char* pixels, int type, int stride) const
{
    if (type & PIXEL_CONVERT_MASK)
    {
        switch (type)
        {
        case PIXEL_RGB2BGR:
        case PIXEL_BGR2RGB:
            to_bgr2rgb(*this, pixels, stride);
            break;
        case PIXEL_RGB2RGBA:
        case PIXEL_BGR2BGRA:
            to_rgb2rgba(*this, pixels, stride);
            break;
        case PIXEL_BGR2RGBA:
        case PIXEL_RGB2BGRA:
            to_bgr2rgba(*this, pixels, stride);
            break;
        case PIXEL_GRAY2RGBA:
        case PIXEL_GRAY2BGRA:
            to_gray2rgba(*this, pixels, stride);
            break;
        case PIXEL_RGBA2BGRA:
        case PIXEL_BGRA2RGBA:
            to_rgba2bgra(*this, pixels, stride);
            break;
        default:
            // unimplemented convert type
            NCNN_LOGE("unimplemented convert type %d", type);
            break;
        }
    }
    else
    {
        if (type == PIXEL_RGB || type == PIXEL_BGR)
            to_rgb(*this, pixels, stride);

        if (type == PIXEL_GRAY)
            to_gray(*this, pixels, stride);

        if (type == PIXEL_RGBA || type == PIXEL_BGRA)
            to_rgba(*this, pixels, stride);
    }
}

void Mat::to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height) const
{
    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 3);
    }
    else if (type_to == PIXEL_GRAY)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 1);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        to_pixels_resize(pixels, type, target_width, target_height, target_width * 4);
    }
}

void Mat::to_pixels_resize(unsigned char* pixels, int type, int target_width, int target_height, int target_stride) const
{
    if (w == target_width && h == target_height)
        return to_pixels(pixels, type);

    int type_to = (type & PIXEL_CONVERT_MASK) ? (type >> PIXEL_CONVERT_SHIFT) : (type & PIXEL_FORMAT_MASK);

    if (type_to == PIXEL_RGB || type_to == PIXEL_BGR)
    {
        Mat src(w, h, (size_t)3u, 3);

        to_pixels(src, type);

        resize_bilinear_c3(src, w, h, w * 3, pixels, target_width, target_height, target_stride);
    }
    else if (type_to == PIXEL_GRAY)
    {
        Mat src(w, h, (size_t)1u, 1);

        to_pixels(src, type);

        resize_bilinear_c1(src, w, h, w * 1, pixels, target_width, target_height, target_stride);
    }
    else if (type_to == PIXEL_RGBA || type_to == PIXEL_BGRA)
    {
        Mat src(w, h, (size_t)4u, 4);

        to_pixels(src, type);

        resize_bilinear_c4(src, w, h, w * 4, pixels, target_width, target_height, target_stride);
    }
}
#endif // NCNN_PIXEL

} // namespace ncnn
