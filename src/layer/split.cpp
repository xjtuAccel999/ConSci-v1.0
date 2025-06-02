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

#include "split.h"
#include "../ncnn/cpu.h"
#include "xtime_l.h"
namespace ncnn {

Split::Split()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;
    support_packing = false;
    // support_packing = true;
    support_fp16_storage = cpu_support_arm_asimdhp() || cpu_support_riscv_zfh();
    support_bf16_storage = true;
    support_image_storage = true;
}

int Split::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& /*opt*/) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    #ifdef PRINT_LAYER
    printf("[ log ]: Forward Split on CPU, shape = (%d, %d, %d, %d), dims = %d\n",bottom_blob.w,bottom_blob.h,bottom_blob.c,bottom_blob.d,bottom_blob.dims);
    #endif
    #ifdef SPLIT_TIME
		u64 tEnd_split, tCur_split;
		u32 tUsed_split;
        XTime_GetTime(&tCur_split);
    #endif
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }
    #ifdef SPLIT_TIME
		XTime_GetTime(&tEnd_split);
		tUsed_split = ((tEnd_split-tCur_split)*1000000)/(COUNTS_PER_SECOND);
		printf("time_split elapsed is %d us\n",tUsed_split);
    #endif
    return 0;
}

} // namespace ncnn
