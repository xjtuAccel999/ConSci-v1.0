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

#include "softmax.h"

#include <float.h>
#include <math.h>

namespace ncnn {

Softmax::Softmax()
{
    one_blob_only = true;
    support_inplace = true;
}

int Softmax::load_param(const ParamDict& pd)
{
    axis = pd.get(0, 0);

    // the original softmax handle axis on 3-dim blob incorrectly
    // ask user to regenerate param instead of producing wrong result
    int fixbug0 = pd.get(1, 0);
    if (fixbug0 == 0 && axis != 0)
    {
        NCNN_LOGE("param is too old, please regenerate!");
        return -1;
    }

    return 0;
}

int Softmax::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
    #ifdef PRINT_LAYER
    printf("[ log ]: Forward Softmax on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_top_blob.w,bottom_top_blob.h,bottom_top_blob.c,bottom_top_blob.d,bottom_top_blob.dims);
    #endif
    // value = exp( value - global max value )
    // sum all value
    // value = value / sum

    int dims = bottom_top_blob.dims;
    size_t elemsize = bottom_top_blob.elemsize;
    int positive_axis = axis < 0 ? dims + axis : axis;

    if (dims == 1) // positive_axis == 0
    {
        int w = bottom_top_blob.w;

        float* ptr = bottom_top_blob;

        float max = -FLT_MAX;
        for (int i = 0; i < w; i++)
        {
            max = std::max(max, ptr[i]);
        }

        float sum = 0.f;
        for (int i = 0; i < w; i++)
        {
            ptr[i] = static_cast<float>(exp(ptr[i] - max));
            sum += ptr[i];
        }

        for (int i = 0; i < w; i++)
        {
            ptr[i] /= sum;
        }
    }

    if (dims == 2 && positive_axis == 0)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        Mat max;
        max.create(w, elemsize, opt.workspace_allocator);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);

        for (int i = 0; i < h; i++)
        {
            const float* ptr = bottom_top_blob.row(i);
            for (int j = 0; j < w; j++)
            {
                max[j] = std::max(max[j], ptr[j]);
            }
        }

        Mat sum;
        sum.create(w, elemsize, opt.workspace_allocator);
        if (sum.empty())
            return -100;
        sum.fill(0.f);

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            for (int j = 0; j < w; j++)
            {
                ptr[j] = static_cast<float>(exp(ptr[j] - max[j]));
                sum[j] += ptr[j];
            }
        }

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            for (int j = 0; j < w; j++)
            {
                ptr[j] /= sum[j];
            }
        }
    }

    if (dims == 2 && positive_axis == 1)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;

        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float m = -FLT_MAX;
            for (int j = 0; j < w; j++)
            {
                m = std::max(m, ptr[j]);
            }

            float s = 0.f;
            for (int j = 0; j < w; j++)
            {
                ptr[j] = static_cast<float>(exp(ptr[j] - m));
                s += ptr[j];
            }

            for (int j = 0; j < w; j++)
            {
                ptr[j] /= s;
            }
        }
    }

    if (dims == 3 && positive_axis == 0)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;
        int size = w * h;

        Mat max;
        max.create(w, h, elemsize, opt.workspace_allocator);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                max[i] = std::max(max[i], ptr[i]);
            }
        }

        Mat sum;
        sum.create(w, h, elemsize, opt.workspace_allocator);
        if (sum.empty())
            return -100;
        sum.fill(0.f);
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] = static_cast<float>(exp(ptr[i] - max[i]));
                sum[i] += ptr[i];
            }
        }

         
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < size; i++)
            {
                ptr[i] /= sum[i];
            }
        }
    }

    if (dims == 3 && positive_axis == 1)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;

        Mat max;
        max.create(w, channels, elemsize, opt.workspace_allocator);
        if (max.empty())
            return -100;
        max.fill(-FLT_MAX);
         
        for (int q = 0; q < channels; q++)
        {
            const float* ptr = bottom_top_blob.channel(q);
            float* maxptr = max.row(q);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    maxptr[j] = std::max(maxptr[j], ptr[j]);
                }

                ptr += w;
            }
        }

        Mat sum;
        sum.create(w, channels, elemsize, opt.workspace_allocator);
        if (sum.empty())
            return -100;
        sum.fill(0.f);
         
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* maxptr = max.row(q);
            float* sumptr = sum.row(q);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = static_cast<float>(exp(ptr[j] - maxptr[j]));
                    sumptr[j] += ptr[j];
                }

                ptr += w;
            }
        }

         
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float* sumptr = sum.row(q);

            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    ptr[j] /= sumptr[j];
                }

                ptr += w;
            }
        }
    }

    if (dims == 3 && positive_axis == 2)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int channels = bottom_top_blob.c;

         
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            for (int i = 0; i < h; i++)
            {
                float max = -FLT_MAX;
                for (int j = 0; j < w; j++)
                {
                    max = std::max(max, ptr[j]);
                }

                float sum = 0.f;
                for (int j = 0; j < w; j++)
                {
                    ptr[j] = static_cast<float>(exp(ptr[j] - max));
                    sum += ptr[j];
                }

                for (int j = 0; j < w; j++)
                {
                    ptr[j] /= sum;
                }

                ptr += w;
            }
        }
    }

    return 0;
}

} // namespace ncnn
