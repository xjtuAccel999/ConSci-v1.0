#ifndef __HW_INNERPROD_H_
#define __HW_INNERPROD_H_

#include "../ncnn/mat.h"
#include "hw_math.h"
#include "hw_act.h"

namespace accel {

class hw_innerprod : public hw_act{
  public:
    float math_alpha;
    float math_beta;

  public:
    hw_innerprod();
    float innerprod_get_data();
    void  innerprod_forward(ncnn::Mat &src0, void* wgt_addr);
};

} // namespace accel

#endif