// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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
void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c1(src, srcw, srch, srcw, dst, w, h, w);
}

void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c2(src, srcw, srch, srcw * 2, dst, w, h, w * 2);
}

void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c3(src, srcw, srch, srcw * 3, dst, w, h, w * 3);
}

void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    return resize_bilinear_c4(src, srcw, srch, srcw * 4, dst, w, h, w * 4);
}

void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

    // double scale_x = (double)srcw / w;
    // double scale_y = (double)srch / h;
    double scale_x_temp = (double)srcw / w * 8192;
    double scale_y_temp = (double)srch / h * 8192;
    int scale_x = static_cast<int>(scale_x_temp);
    int scale_y = static_cast<int>(scale_y_temp);

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    // float fx;
    // float fy;
    int fx;
    int fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        // fx = (float)((dx + 0.5) * scale_x - 0.5);
        // sx = static_cast<int>(floor(fx));
        // fx -= sx;
        fx = (2 * dx + 1) * scale_x - 8192;
        sx = fx >> 14;
        fx = (fx & 0x3FF8) >> 3;

        if (sx < 0)
        {
            sx = 0;
            // fx = 0.f;
            fx = 0;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            // fx = 1.f;
            fx = 2048;
        }

        xofs[dx] = sx;

        // float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        // float a1 = fx * INTER_RESIZE_COEF_SCALE;
        int a0 = 2048 - fx;
        int a1 = fx;

        // ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        // ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
        ialpha[dx * 2] = static_cast<short>(a0);
        ialpha[dx * 2 + 1] = static_cast<short>(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        // fy = (float)((dy + 0.5) * scale_y - 0.5);
        // sy = static_cast<int>(floor(fy));
        // fy -= sy;
        fy = (2 * dy + 1) * scale_y - 8192;
        sy = fy >> 14;
        fy = (fy & 0x3FF8) >> 3;

        if (sy < 0)
        {
            sy = 0;
            // fy = 0.f;
            fy = 0;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            // fy = 1.f;
            fy = 2048;
        }

        yofs[dy] = sy;

        // float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        // float b1 = fy * INTER_RESIZE_COEF_SCALE;
        int b0 = 2048 - fy;
        int b1 = fy;

        // ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        // ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
        ibeta[dy * 2] = static_cast<short>(b0);
        ibeta[dy * 2 + 1] = static_cast<short>(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w, (size_t)2u);
    Mat rowsbuf1(w, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;
                rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;
                rows0p[dx] = (S0p[0] * a0 + S0p[1] * a1) >> 4;
                rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

                ialphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

        int nn = 0;
        int remain = w - (nn << 3);

        for (; remain; --remain)
        {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

    double scale_x = (double)srcw / w;
    double scale_y = (double)srch / h;

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    float fx;
    float fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        fx = (float)((dx + 0.5) * scale_x - 0.5);
        sx = static_cast<int>(floor(fx));
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            fx = 1.f;
        }

        xofs[dx] = sx * 2;

        float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        float a1 = fx * INTER_RESIZE_COEF_SCALE;

        ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        fy = (float)((dy + 0.5) * scale_y - 0.5);
        sy = static_cast<int>(floor(fy));
        fy -= sy;

        if (sy < 0)
        {
            sy = 0;
            fy = 0.f;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            fy = 1.f;
        }

        yofs[dy] = sy;

        float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        float b1 = fy * INTER_RESIZE_COEF_SCALE;

        ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 2 + 2, (size_t)2u);
    Mat rowsbuf1(w * 2 + 2, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];

                const unsigned char* S1p = S1 + sx;

                short a0 = ialphap[0];
                short a1 = ialphap[1];

                rows1p[0] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;

                ialphap += 2;
                rows1p += 2;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;

                rows0p[0] = (S0p[0] * a0 + S0p[2] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[3] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[2] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[3] * a1) >> 4;

                ialphap += 2;
                rows0p += 2;
                rows1p += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

        int nn = 0;

        int remain = (w * 2) - (nn << 3);

        for (; remain; --remain)
        {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

    // double scale_x = (double)srcw / w;
    // double scale_y = (double)srch / h;
    double scale_x_temp = (double)srcw / w * 8192;
    double scale_y_temp = (double)srch / h * 8192;
    int scale_x = static_cast<int>(scale_x_temp);
    int scale_y = static_cast<int>(scale_y_temp);

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    // float fx;
    // float fy;
    int fx;
    int fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        // fx = (float)((dx + 0.5) * scale_x - 0.5);
        // sx = static_cast<int>(floor(fx));
        // fx -= sx;
        fx = (2 * dx + 1) * scale_x - 8192;
        sx = fx >> 14;
        fx = (fx & 0x3FF8) >> 3;

        if (sx < 0)
        {
            sx = 0;
            // fx = 0.f;
            fx = 0;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            // fx = 1.f;
            fx = 2048;
        }

        xofs[dx] = sx * 3;

        // float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        // float a1 = fx * INTER_RESIZE_COEF_SCALE;
        int a0 = 2048 - fx;
        int a1 = fx;

        // ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        // ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
        ialpha[dx * 2] = static_cast<short>(a0);
        ialpha[dx * 2 + 1] = static_cast<short>(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        // fy = (float)((dy + 0.5) * scale_y - 0.5);
        // sy = static_cast<int>(floor(fy));
        // fy -= sy;
        fy = (2 * dy + 1) * scale_y - 8192;
        sy = fy >> 14;
        fy = (fy & 0x3FF8) >> 3;

        if (sy < 0)
        {
            sy = 0;
            // fy = 0.f;
            fy = 0;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            // fy = 1.f;
            fy = 2048;
        }

        yofs[dy] = sy;

        // float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        // float b1 = fy * INTER_RESIZE_COEF_SCALE;
        int b0 = 2048 - fy;
        int b1 = fy;

        // ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        // ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
        ibeta[dy * 2] = static_cast<short>(b0);
        ibeta[dy * 2 + 1] = static_cast<short>(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 3 + 1, (size_t)2u);
    Mat rowsbuf1(w * 3 + 1, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;

                rows1p[0] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;

                ialphap += 2;
                rows1p += 3;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;

                rows0p[0] = (S0p[0] * a0 + S0p[3] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[4] * a1) >> 4;
                rows0p[2] = (S0p[2] * a0 + S0p[5] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[3] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[4] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[5] * a1) >> 4;

                ialphap += 2;
                rows0p += 3;
                rows1p += 3;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

        int nn = 0;

        int remain = (w * 3) - (nn << 3);

        for (; remain; --remain)
        {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride, unsigned char* dst, int w, int h, int stride)
{
    const int INTER_RESIZE_COEF_BITS = 11;
    const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
    //     const int ONE=INTER_RESIZE_COEF_SCALE;

    // double scale_x = (double)srcw / w;
    // double scale_y = (double)srch / h;
    double scale_x_temp = (double)srcw / w * 8192;
    double scale_y_temp = (double)srch / h * 8192;
    int scale_x = static_cast<int>(scale_x_temp);
    int scale_y = static_cast<int>(scale_y_temp);

    int* buf = new int[w + h + w + h];

    int* xofs = buf;     //new int[w];
    int* yofs = buf + w; //new int[h];

    short* ialpha = (short*)(buf + w + h);    //new short[w * 2];
    short* ibeta = (short*)(buf + w + h + w); //new short[h * 2];

    // float fx;
    // float fy;
    int fx;
    int fy;
    int sx;
    int sy;

#define SATURATE_CAST_SHORT(X) (short)::std::min(::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

    for (int dx = 0; dx < w; dx++)
    {
        // fx = (float)((dx + 0.5) * scale_x - 0.5);
        // sx = static_cast<int>(floor(fx));
        // fx -= sx;
        fx = (2 * dx + 1) * scale_x - 8192;
        sx = fx >> 14;
        fx = (fx & 0x3FF8) >> 3;

        if (sx < 0)
        {
            sx = 0;
            // fx = 0.f;
            fx = 0;
        }
        if (sx >= srcw - 1)
        {
            sx = srcw - 2;
            // fx = 1.f;
            fx = 2048;
        }

        xofs[dx] = sx * 4;

        // float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
        // float a1 = fx * INTER_RESIZE_COEF_SCALE;
        int a0 = 2048 - fx;
        int a1 = fx;

        // ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
        // ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
        ialpha[dx * 2] = static_cast<short>(a0);
        ialpha[dx * 2 + 1] = static_cast<short>(a1);
    }

    for (int dy = 0; dy < h; dy++)
    {
        // fy = (float)((dy + 0.5) * scale_y - 0.5);
        // sy = static_cast<int>(floor(fy));
        // fy -= sy;
        fy = (2 * dy + 1) * scale_y - 8192;
        sy = fy >> 14;
        fy = (fy & 0x3FF8) >> 3;

        if (sy < 0)
        {
            sy = 0;
            // fy = 0.f;
            fy = 0;
        }
        if (sy >= srch - 1)
        {
            sy = srch - 2;
            // fy = 1.f;
            fy = 2048;
        }

        yofs[dy] = sy;

        // float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
        // float b1 = fy * INTER_RESIZE_COEF_SCALE;
        int b0 = 2048 - fy;
        int b1 = fy;

        // ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
        // ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
        ibeta[dy * 2] = static_cast<short>(b0);
        ibeta[dy * 2 + 1] = static_cast<short>(b1);
    }

#undef SATURATE_CAST_SHORT

    // loop body
    Mat rowsbuf0(w * 4, (size_t)2u);
    Mat rowsbuf1(w * 4, (size_t)2u);
    short* rows0 = (short*)rowsbuf0.data;
    short* rows1 = (short*)rowsbuf1.data;

    int prev_sy1 = -2;

    for (int dy = 0; dy < h; dy++)
    {
        sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            short* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S1p = S1 + sx;

                rows1p[0] = (S1p[0] * a0 + S1p[4] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;
                rows1p[3] = (S1p[3] * a0 + S1p[7] * a1) >> 4;

                ialphap += 2;
                rows1p += 4;
            }
        }
        else
        {
            // hresize two rows
            const unsigned char* S0 = src + srcstride * (sy);
            const unsigned char* S1 = src + srcstride * (sy + 1);

            const short* ialphap = ialpha;
            short* rows0p = rows0;
            short* rows1p = rows1;
            for (int dx = 0; dx < w; dx++)
            {
                sx = xofs[dx];
                short a0 = ialphap[0];
                short a1 = ialphap[1];

                const unsigned char* S0p = S0 + sx;
                const unsigned char* S1p = S1 + sx;

                rows0p[0] = (S0p[0] * a0 + S0p[4] * a1) >> 4;
                rows0p[1] = (S0p[1] * a0 + S0p[5] * a1) >> 4;
                rows0p[2] = (S0p[2] * a0 + S0p[6] * a1) >> 4;
                rows0p[3] = (S0p[3] * a0 + S0p[7] * a1) >> 4;
                rows1p[0] = (S1p[0] * a0 + S1p[4] * a1) >> 4;
                rows1p[1] = (S1p[1] * a0 + S1p[5] * a1) >> 4;
                rows1p[2] = (S1p[2] * a0 + S1p[6] * a1) >> 4;
                rows1p[3] = (S1p[3] * a0 + S1p[7] * a1) >> 4;

                ialphap += 2;
                rows0p += 4;
                rows1p += 4;
            }
        }

        prev_sy1 = sy;

        // vresize
        short b0 = ibeta[0];
        short b1 = ibeta[1];

        short* rows0p = rows0;
        short* rows1p = rows1;
        unsigned char* Dp = dst + stride * (dy);

        int nn = 0;

        int remain = (w * 4) - (nn << 3);

        for (; remain; --remain)
        {
            //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
            *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) + (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >> 2);
        }

        ibeta += 2;
    }

    delete[] buf;
}

void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h)
{
    // assert srcw % 2 == 0
    // assert srch % 2 == 0
    // assert w % 2 == 0
    // assert h % 2 == 0

    const unsigned char* srcY = src;
    unsigned char* dstY = dst;
    resize_bilinear_c1(srcY, srcw, srch, dstY, w, h);

    const unsigned char* srcUV = src + srcw * srch;
    unsigned char* dstUV = dst + w * h;
    resize_bilinear_c2(srcUV, srcw / 2, srch / 2, dstUV, w / 2, h / 2);
}
#endif // NCNN_PIXEL

} // namespace ncnn
