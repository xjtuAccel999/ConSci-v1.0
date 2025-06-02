#ifndef __HW_MATH_H_
#define __HW_MATH_H_

#include "../ncnn/mat.h"

#define MUL_SRC1_FROM_DMA 1
#define MUL_SRC1_FROM_ALPHA 2

#define ADD_SRC0_FROM_DMA 1
#define ADD_SRC0_FROM_MUL_O 2

#define ADD_SRC1_FROM_DMA 1
#define ADD_SRC1_FROM_BETA 2
#define ADD_SRC1_FROM_ADD_O 3

#define MATH_OP_ABS 1
#define MATH_OP_THRESHOLD 2
#define MATH_OP_EQUAL 3

namespace accel {

class hw_math {
  public:
    int mul_src1_sel;
    int add_src0_sel;
    int add_src1_sel;
    int sub_en;
    int add_en;
    int mul_en;
    int max_en;
    int min_en;
    int op;
    float alpha;
    float beta;

  public:
    hw_math();

    void math_forward(ncnn::Mat &src0, ncnn::Mat &src1, ncnn::Mat &dst);
    void math_forward(ncnn::Mat &src0, ncnn::Mat &dst);
    void math_forward(void* src0_addr0, void* src1_addr0, void* dst_addr0, int len);
    void math_forward(void* src0_addr0, void* dst_addr0, int len);
    void math_forward(void* src0_addr0, int len);
};

} // namespace accel

#endif