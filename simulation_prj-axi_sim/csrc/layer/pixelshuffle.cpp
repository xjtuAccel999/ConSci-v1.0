// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "pixelshuffle.h"

namespace ncnn {

PixelShuffle::PixelShuffle()
{
    one_blob_only = true;
    support_inplace = false;
}

int PixelShuffle::load_param(const ParamDict& pd)
{
    upscale_factor = pd.get(0, 1);
    mode = pd.get(1, 0);

    return 0;
}

int PixelShuffle::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    #ifdef PRINT_LAYER
        printf("[ log ]: Forward PixelShuffle on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    int outw = w * upscale_factor;
    int outh = h * upscale_factor;
    int outc = channels / (upscale_factor * upscale_factor);

    top_blob.create(outw, outh, outc, elemsize, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

     
    for (int p = 0; p < outc; p++)
    {
        Mat m = top_blob.channel(p);

        for (int sh = 0; sh < upscale_factor; sh++)
        {
            for (int sw = 0; sw < upscale_factor; sw++)
            {
                int q;
                if (mode == 0)  //通道连着的插到wh中
                    q = p * upscale_factor * upscale_factor + sh * upscale_factor + sw;
                else // if (mode == 1)   //通道分散开的
                    q = (sh * upscale_factor + sw) * outc + p;

                const float* sptr = bottom_blob.channel(q);

                for (int i = 0; i < h; i++)
                {
                    float* outptr = m.row(i * upscale_factor + sh) + sw;
                    for (int j = 0; j < w; j++)
                    {
                        outptr[0] = sptr[0];

                        sptr++;
                        outptr += upscale_factor;
                    }
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
