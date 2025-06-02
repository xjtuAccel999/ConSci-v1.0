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

#include "mat.h"
#include "platform.h"

namespace ncnn {

#if NCNN_PIXEL_ROTATE
// should be a kanna ascii art here in my local branch
// but we shall ask the original art author for permission first ...
// https://www.reddit.com/r/anime/comments/5uxjn4/i_recreated_the_kanna_ascii_art_from_kobayashisan/

static void kanna_rotate_1_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride - w;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }

    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_1_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 2;
    const int wgap = stride - w * 2;

    int size = srcw * 2;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }

    for (; y < srch; y++)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_1_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 3;
    const int wgap = stride - w * 3;

    int size = srcw * 3;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }

    for (; y < srch; y++)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_1_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 4;
    const int wgap = stride - w * 4;

    int size = srcw * 4;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dst;
    unsigned char* dst1 = dst + stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 += wgap + stride;
        dst1 += wgap + stride;
    }

    for (; y < srch; y++)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_2_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride + w;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w - 1;

    int y = 0;
    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= 1;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_2_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 2;
    const int wgap = stride + w * 2;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w * 2 - 2;

    int y = 0;
    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= 2;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_2_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 3;
    const int wgap = stride + w * 3;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w * 3 - 3;

    int y = 0;
    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= 3;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_2_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 4;
    const int wgap = stride + w * 4;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dst + w * 4 - 4;

    int y = 0;
    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= 4;
        }

        src0 += srcwgap;
        dst0 += wgap;
    }
}

static void kanna_rotate_3_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride - w;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 1;

    int y = 0;
    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= 1;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_3_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 2;
    const int wgap = stride - w * 2;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 2;

    int y = 0;
    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= 2;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_3_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 3;
    const int wgap = stride - w * 3;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 3;

    int y = 0;
    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= 3;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_3_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 4;
    const int wgap = stride - w * 4;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * h - wgap;

    const unsigned char* src0 = src;
    unsigned char* dst0 = dstend - 4;

    int y = 0;
    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= 4;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_4_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw;
    const int wgap = stride + w;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
        int remain = srcw;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_4_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 2;
    const int wgap = stride + w * 2;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    int size = srcw * 2;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_4_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 3;
    const int wgap = stride + w * 3;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    int size = srcw * 3;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_4_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 4;
    const int wgap = stride + w * 4;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    int size = srcw * 4;

    const unsigned char* src0 = src;
    const unsigned char* src1 = src + srcstride;
    unsigned char* dst0 = dstend;
    unsigned char* dst1 = dstend - stride;

    int y = 0;
    for (; y + 1 < srch; y += 2)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
            *dst1++ = *src1++;
        }

        src0 += srcwgap + srcstride;
        src1 += srcwgap + srcstride;
        dst0 -= wgap + stride;
        dst1 -= wgap + stride;
    }

    for (; y < srch; y++)
    {
        int remain = size;

        for (; remain > 0; remain--)
        {
            *dst0++ = *src0++;
        }

        src0 += srcwgap;
        dst0 -= wgap;
    }
}

static void kanna_rotate_5_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_5_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 2;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y * 2;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_5_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 3;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y * 3;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_5_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 4;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dst + y * 4;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_6_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y - 1;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_6_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 2;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w * 2;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 2 - 2;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_6_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 3;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w * 3;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 3 - 3;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_6_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int /*h*/, int stride)
{
    const int srcwgap = srcstride - srcw * 4;

    // point to the last dst pixel in row
    unsigned char* dstend = dst + w * 4;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 4 - 4;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 += stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_7_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * (h - 1) + w;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y - 1;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_7_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 2;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * (h - 1) + w * 2;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 2 - 2;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_7_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 3;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * (h - 1) + w * 3;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 3 - 3;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_7_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 4;

    // point to the last dst pixel
    unsigned char* dstend = dst + stride * (h - 1) + w * 4;

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend - y * 4 - 4;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_8_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int h, int stride)
{
    const int srcwgap = srcstride - srcw;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend + y;

        int x = 0;
        for (; x < srcw; x++)
        {
            *dst0 = *src0;

            src0 += 1;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_8_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 2;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend + y * 2;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];

            src0 += 2;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_8_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 3;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend + y * 3;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];

            src0 += 3;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

static void kanna_rotate_8_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int /*w*/, int h, int stride)
{
    const int srcwgap = srcstride - srcw * 4;

    // point to the last dst pixel row
    unsigned char* dstend = dst + stride * (h - 1);

    const unsigned char* src0 = src;

    int y = 0;

    for (; y < srch; y++)
    {
        unsigned char* dst0 = dstend + y * 4;

        int x = 0;
        for (; x < srcw; x++)
        {
            dst0[0] = src0[0];
            dst0[1] = src0[1];
            dst0[2] = src0[2];
            dst0[3] = src0[3];

            src0 += 4;
            dst0 -= stride;
        }

        src0 += srcwgap;
    }
}

void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    return kanna_rotate_c1(src, srcw, srch, srcw, dst, w, h, w, type);
}

void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    return kanna_rotate_c2(src, srcw, srch, srcw * 2, dst, w, h, w * 2, type);
}

void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    return kanna_rotate_c3(src, srcw, srch, srcw * 3, dst, w, h, w * 3, type);
}

void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    return kanna_rotate_c4(src, srcw, srch, srcw * 4, dst, w, h, w * 4, type);
}

void kanna_rotate_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c1(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}

void kanna_rotate_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c2(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}

void kanna_rotate_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c3(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}

void kanna_rotate_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride, int type)
{
    // assert srcw == w && srch == h for type 1234
    // assert srcw == h && srch == w for type 5678

    switch (type)
    {
    case 1:
        kanna_rotate_1_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 2:
        kanna_rotate_2_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 3:
        kanna_rotate_3_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 4:
        kanna_rotate_4_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 5:
        kanna_rotate_5_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 6:
        kanna_rotate_6_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 7:
        kanna_rotate_7_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    case 8:
        kanna_rotate_8_c4(src, srcw, srch, srcstride, dst, w, h, stride);
        break;
    default:
        // unsupported rotate type
        break;
    }
}

void kanna_rotate_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h, int type)
{
    // assert srcw % 2 == 0
    // assert srch % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const unsigned char* srcY = src;
    unsigned char* dstY = dst;
    kanna_rotate_c1(srcY, srcw, srch, dstY, w, h, type);

    const unsigned char* srcUV = src + srcw * srch;
    unsigned char* dstUV = dst + w * h;
    kanna_rotate_c2(srcUV, srcw / 2, srch / 2, dstUV, w / 2, h / 2, type);
}
#endif // NCNN_PIXEL_ROTATE

} // namespace ncnn
