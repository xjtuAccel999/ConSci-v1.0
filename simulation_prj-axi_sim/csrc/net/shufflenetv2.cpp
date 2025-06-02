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

#include "cnn_demo.h"
#include "../ncnn/net.h"
#include "../ncnn/simpleocv.h"

#include <algorithm>
#include <stdio.h>
#include <vector>
#include "../config.h"
#include "../veri.h"
namespace demo{
static int detect_shufflenetv2(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net shufflenetv2;

    shufflenetv2.opt.use_vulkan_compute = true;

    // https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe
    // models can be downloaded from https://github.com/miaow1988/ShuffleNet_V2_pytorch_caffe/releases
    // if (shufflenetv2.load_param("/home/zmx/ncnnAccel_hw/ncnn/file/shufflenet_v2_x0.5.param"))
    //     exit(-1);
    // if (shufflenetv2.load_model("/home/zmx/ncnnAccel_hw/ncnn/file/shufflenet_v2_x0.5.bin"))
    //     exit(-1);
    // if (shufflenetv2.load_param_mem(shufflenetv2.load_file2mem("models/shufflenetv2/shufflenetv2_opt.param")))
    //     exit(-1);
    // if (shufflenetv2.load_model_mem(shufflenetv2.load_file2mem("models/shufflenetv2/shufflenetv2_opt.bin")))
    //     exit(-1);

    if (shufflenetv2.load_param_mem(shufflenetv2.load_file2mem("models/shufflenetv2/shufflenetv2_int8.param")))
        exit(-1);
    if (shufflenetv2.load_model_mem(shufflenetv2.load_file2mem("models/shufflenetv2/shufflenetv2_int8.bin")))
        exit(-1);

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = shufflenetv2.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("fc", out);

    // manually call softmax on the fc output
    // convert result into probability
    // skip if your model already has softmax operation
    {
        ncnn::Layer* softmax = ncnn::create_layer("Softmax");

        ncnn::ParamDict pd;
        softmax->load_param(pd);

        softmax->forward_inplace(out, shufflenetv2.opt);

        delete softmax;
    }

    out = out.reshape(out.w * out.h * out.c);

    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++)
    {
        cls_scores[j] = out[j];
    }

    return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // print topk and score
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}

int shufflenetv2_inference()
{
    cv::Mat m = cv::imread(IMAGE_PATH, 1);
    if (m.empty())
    {
        printf("[error]: cv::imread is failed\n");
        return -1;
    }

    std::vector<float> cls_scores;
    detect_shufflenetv2(m, cls_scores);

    print_topk(cls_scores, 3);
    
    double time_ms = (main_time/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    double time_pool_ms = (time_pool/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    double time_gemm_ms = (time_gemm/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    printf("[ log ]: FREQ = %dMHz, CYCLES = %ld, TIME = %fms\n",HW_FREQ_MHZ,main_time/2,time_ms);
    printf("[ log ]: POOL_TIME = %f, %.5f%%\n", time_pool_ms, time_pool_ms/time_ms*100.f);
    printf("[ log ]: GEMM_TIME = %f, %.5f%%\n", time_gemm_ms, time_gemm_ms/time_ms*100.f);

    

    return 0;
}
}
