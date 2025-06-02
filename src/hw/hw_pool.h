#ifndef __HW_POOL_H_
#define __HW_POOL_H_

#include "../ncnn/mat.h"

#define POOL_MAX 0
#define POOL_AVG 1

#define SINGLE_CHANNEL0 1
#define SINGLE_CHANNEL1 2
#define DOUBLE_CHANNELS 3

namespace accel {

class hw_pool {
  public:
    int pool_type;
    int channels;
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int pad_mode;
    int pad_left;
    int pad_right;
    int pad_top;
    int pad_bottom;
    float pad_value;

  public:
    hw_pool();
    void forward(ncnn::Mat &ifm, ncnn::Mat &ofm);
};

} // namespace accel

#endif