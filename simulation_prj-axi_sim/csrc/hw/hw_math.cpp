#include "hw_math.h"
#include "../veri.h"
#include "accel_params.h"

namespace accel {

hw_math::hw_math() {
    this->mul_src1_sel = 0;
    this->add_src0_sel = 0;
    this->add_src1_sel = 0;
    this->sub_en = 0;
    this->add_en = 0;
    this->mul_en = 0;
    this->max_en = 0;
    this->min_en = 0;
    this->op = 0;
    this->alpha = 0.f;
    this->beta = 0.f;
}

void hw_math::math_forward(ncnn::Mat &src0, ncnn::Mat &src1, ncnn::Mat &dst) {
    ALU_RESET;
    int *src0_addr0 = src0;
    int *src1_addr0 = src1;
    int *dst_addr0 = dst;
    int len = src0.total();

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = src0_addr0 + len_ch0;
    int *src1_addr1 = src1_addr0 + len_ch0;
    int *dst_addr1 = dst_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint64_t)src0_addr0);
    ALU_SRC1_ADDR_CH0_SET((uint64_t)src1_addr0);
    ALU_DST_ADDR_CH0_SET((uint64_t)dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint64_t)src0_addr1);
    ALU_SRC1_ADDR_CH1_SET((uint64_t)src1_addr1);
    ALU_DST_ADDR_CH1_SET((uint64_t)dst_addr1);

    ALU_MATHFUNC_CTRL_SET(1,2,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,op);

    dma_wait();
    ALU_RESET;
}

void hw_math::math_forward(ncnn::Mat &src0, ncnn::Mat &dst) {
    ALU_RESET;
    int *src0_addr0 = src0;
    int *dst_addr0 = dst;
    int len = src0.total();

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = src0_addr0 + len_ch0;
    int *dst_addr1 = dst_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint64_t)src0_addr0);
    ALU_DST_ADDR_CH0_SET((uint64_t)dst_addr0);

    printf("src0_addr0 = %p\n",src0_addr0);
    printf("dst_addr0 = %p\n",dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint64_t)src0_addr1);
    ALU_DST_ADDR_CH1_SET((uint64_t)dst_addr1);
    printf("src0_addr1 = %p\n",src0_addr1);
    printf("dst_addr1 = %p\n",dst_addr1);

    ALU_MATHFUNC_CTRL_SET(1,1,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,op);

    dma_wait();
    ALU_RESET;
}

void hw_math::math_forward(void* src0_addr0, void* src1_addr0, void* dst_addr0, int len) {
    ALU_RESET;

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = (int*)src0_addr0 + len_ch0;
    int *src1_addr1 = (int*)src1_addr0 + len_ch0;
    int *dst_addr1 = (int*)dst_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint64_t)src0_addr0);
    ALU_SRC1_ADDR_CH0_SET((uint64_t)src1_addr0);
    ALU_DST_ADDR_CH0_SET((uint64_t)dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint64_t)src0_addr1);
    ALU_SRC1_ADDR_CH1_SET((uint64_t)src1_addr1);
    ALU_DST_ADDR_CH1_SET((uint64_t)dst_addr1);

    ALU_MATHFUNC_CTRL_SET(1,2,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,op);

    dma_wait();
    ALU_RESET;
}

void hw_math::math_forward(void* src0_addr0, void* dst_addr0, int len) {
    ALU_RESET;

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = (int*)src0_addr0 + len_ch0;
    int *dst_addr1 = (int*)dst_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint64_t)src0_addr0);
    ALU_DST_ADDR_CH0_SET((uint64_t)dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint64_t)src0_addr1);
    ALU_DST_ADDR_CH1_SET((uint64_t)dst_addr1);

    ALU_MATHFUNC_CTRL_SET(1,1,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,op);

    dma_wait();
    ALU_RESET;
}

void hw_math::math_forward(void* src0_addr0, int len) {
    ALU_RESET;

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = (int*)src0_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint64_t)src0_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint64_t)src0_addr1);

    ALU_MATHFUNC_CTRL_SET(1,1,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,op);

    dma_wait();
    ALU_RESET;
}

} // namespace accel