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

#include "concat.h"
#include "xtime_l.h"
namespace ncnn {

Concat::Concat()
{
    one_blob_only = false;
    support_inplace = false;
}

int Concat::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    return 0;
}

int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward Concat on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blobs[0].w,bottom_blobs[0].h,bottom_blobs[0].c,bottom_blobs[0].d,bottom_blobs[0].dims);
    #endif
    #ifdef CONCAT_TIME
		u64 tEnd, tCur;
		u32 tUsed;
        XTime_GetTime(&tCur);
    #endif
    int dims = bottom_blobs[0].dims;
    size_t elemsize = bottom_blobs[0].elemsize;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int w = bottom_blob.w;

            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, w * elemsize);

            outptr += w * elemsize;
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        unsigned char* outptr = top_blob;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int size = w * bottom_blob.h;

            const unsigned char* ptr = bottom_blob;
            memcpy(outptr, ptr, size * elemsize);

            outptr += size * elemsize;
        }

        return 0;
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

         
        for (int i = 0; i < h; i++)
        {
            unsigned char* outptr = top_blob.row<unsigned char>(i);
            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                const unsigned char* ptr = bottom_blob.row<const unsigned char>(i);
                memcpy(outptr, ptr, bottom_blob.w * elemsize);

                outptr += bottom_blob.w * elemsize;
            }
        }

        return 0;
    }

    if (dims == 3 && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        // total channels
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.c;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        int q = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];

            int channels = bottom_blob.c;
            size_t size = bottom_blob.cstep * channels;

            const unsigned char* ptr = bottom_blob;
            unsigned char* outptr = top_blob.channel(q);
            memcpy(outptr, ptr, size * elemsize);

            q += channels;
        }

        return 0;
    }

    if (dims == 3 && positive_axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

         
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (size_t b = 0; b < bottom_blobs.size(); b++)
            {
                const Mat& bottom_blob = bottom_blobs[b];

                int size = bottom_blob.w * bottom_blob.h;

                const unsigned char* ptr = bottom_blob.channel(q);
                memcpy(outptr, ptr, size * elemsize);

                outptr += size * elemsize;
            }
        }

        return 0;
    }

    if (dims == 3 && positive_axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const Mat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        Mat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

         
        for (int q = 0; q < channels; q++)
        {
            unsigned char* outptr = top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                for (size_t b = 0; b < bottom_blobs.size(); b++)
                {
                    const Mat& bottom_blob = bottom_blobs[b];

                    const unsigned char* ptr = bottom_blob.channel(q).row<const unsigned char>(i);
                    memcpy(outptr, ptr, bottom_blob.w * elemsize);

                    outptr += bottom_blob.w * elemsize;
                }
            }
        }

        return 0;
    }
    #ifdef CONCAT_TIME
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		printf("time_concat elapsed is %d us\n",tUsed);
    #endif
    return 0;
}

} // namespace ncnn
