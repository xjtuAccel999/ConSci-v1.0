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

#include "cnn_demo.h"
#include "../ncnn/net.h"
#include "../ncnn/simpleocv.h"
#include <stdio.h>
#include <vector>
#include "../config.h"
#include "xtime_l.h"
#include "../hw/dri_sd.h"
//#include "../veri.h"
namespace demo {
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static int detect_mobilenet(const cv::Mat& bgr, std::vector<Object>& objects)
{
	
    ncnn::Net mobilenet;

    mobilenet.opt.use_int8_inference = true;

    if (mobilenet.load_param_mem(load_file_align("mobilenet_ssd_int8.param")))
        exit(-1);
    if (mobilenet.load_model_mem(load_file_align("mobilenet_ssd_int8.bin")))
        exit(-1);

    // if (mobilenet.load_param_mem(mobilenet.load_file2mem("models/mobilenet_ssd/mobilenet_ssd_opt.param")))
    //     exit(-1);
    // if (mobilenet.load_model_mem(mobilenet.load_file2mem("models/mobilenet_ssd/mobilenet_ssd_opt.bin")))
    //     exit(-1);


    const int target_size = 300;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = mobilenet.create_extractor();

    ex.input("data", in);

    ncnn::Mat out;
    #ifdef FORWRD_TIME
		u64 tEnd, tCur;
		u32 tUsed;
        XTime_GetTime(&tCur);
    #endif
    ex.extract("detection_out", out);

	#ifdef FORWRD_TIME
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd-tCur)*1000000)/(COUNTS_PER_SECOND);
		double fps = 1000000.f / tUsed;
		printf("time_mobilenetssd elapsed is %d us, fps = %.3f\r\n",tUsed,fps);
    #endif

    //     printf("%d %d %d\n", out.w, out.h, out.c);
    objects.clear();
    for (int i = 0; i < out.h; i++)
    {
        const float* values = out.row(i);

        Object object;
        object.label = values[0];
        object.prob = values[1];
        object.rect.x = values[2] * img_w;
        object.rect.y = values[3] * img_h;
        object.rect.width = values[4] * img_w - object.rect.x;
        object.rect.height = values[5] * img_h - object.rect.y;

        objects.push_back(object);
    }

    return 0;
}


static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {"background",
                                        "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair",
                                        "cow", "diningtable", "dog", "horse",
                                        "motorbike", "person", "pottedplant",
                                        "sheep", "sofa", "train", "tvmonitor"
                                       };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
		#ifndef VIDEO_MODE
        	fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
		#endif
        cv::rectangle(image, obj.rect, cv::Scalar(0, 0,255),2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

	#ifdef VIDEO_MODE
        bgr2abgr(image.data,GFrame,SCREEN_W,SCREEN_H);
        Xil_DCacheFlushRange((INTPTR)GFrame, IMG_SIZE);
        // video_buffer_update((u8*)GFrame,&RunCfg);
        graphic_buffer_update((u8*)GFrame,&RunCfg);
    #else
        cv::imshow("image", image);
        cv::waitKey(0);
    #endif
}

int mobilenet_ssd_inference()
{
    cv::Mat m = cv::imread(IMAGE_PATH, 1);
    if (m.empty())
    {
        printf("[error]: cv::imread is failed\n");
        return -1;
    }

    std::vector<Object> objects;
    detect_mobilenet(m, objects);

    draw_objects(m, objects);

    // double time_ms = (main_time/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    // double time_pool_ms = (time_pool/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    // double time_gemm_ms = (time_gemm/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    // printf("[ log ]: FREQ = %dMHz, CYCLES = %ld, TIME = %.3fms, FPS = %.3f\n",HW_FREQ_MHZ,main_time/2,time_ms,1000.f/time_ms);
    // printf("[ log ]: POOL_TIME = %f, %.5f%%\n", time_pool_ms, time_pool_ms/time_ms*100.f);
    // printf("[ log ]: GEMM_TIME = %f, %.5f%%\n", time_gemm_ms, time_gemm_ms/time_ms*100.f);

    return 0;
}

}
