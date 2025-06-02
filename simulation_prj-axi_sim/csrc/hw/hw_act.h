#ifndef __HW_ACT_H_
#define __HW_ACT_H_

#include "../ncnn/mat.h"

#define HW_NONE        0
#define HW_RELU        1
#define HW_LEAKYRELU   2
#define HW_CLIP        3 
#define HW_SIGMOID     4
#define HW_MISH        5  //no support
#define HW_HARDSWISH   6
#define HW_TANH        7  
#define HW_SWISH       8
#define HW_PRELU       9
#define HW_ELU         10
#define HW_SELU        11
#define HW_HARDSIGMOID 12


// typedef struct HW_ACT_COFFE
// {
//   float range[4];
//   float coffe_a[5];
//   float coffe_b[5];
//   float coffe_c[5];
// }hw_act_coffe;


#define ACT_FROM_DMA 1
#define ACT_FROM_OSCALE_BIAS 2

#define ACT_TO_DMA 1
#define ACT_TO_OPFUSION 2

namespace accel {

class hw_act {
  public:
    int act_op;
    float act_alpha;
    float act_beta;
    int act_prop;
    float range[4];
    float coffe_a[5];
    float coffe_b[5];
    float coffe_c[5];

  public:
    hw_act();
    void cfg_param();
    void send_coffe();
    void act_forward(ncnn::Mat &src0, ncnn::Mat &dst);
    void act_forward(float* src_dst_ptr, int len);
};

} // namespace accel

#endif