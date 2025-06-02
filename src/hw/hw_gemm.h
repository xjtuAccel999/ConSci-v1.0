#ifndef __HW_GEMM_H_
#define __HW_GEMM_H_

#include "../ncnn/mat.h"
#include "hw_act.h"

#define HW_CONV 0
#define HW_DEPTHCONV 1
#define HW_MATMUL 2

#define PD_MODE_ZERO 0
#define PD_MODE_EDGE 1

namespace accel {

class hw_gemm : public hw_act{
  public:
    // gemm
    int op;
    int kernel;
    int stride;
    int padding_mode;
    int padding_left;
    int padding_right;
    int padding_top;
    int padding_bottom;
    int bias_en;
    int oscale_en;
    int requant_en;
    int layout_en; // requant not need layout
    float quant_scale;
    float requant_scale;
    float *dequant_scale;
    float *bias_data;
    int ifm_w;
    int ifm_h;
    int ifm_c;
    int ifm_cstep;
    int ofm_w;
    int ofm_h;
    int ofm_c;
    int ofm_cstep;
    int ofm_total;

    // activation
    int act_dst_sel;

    //block
    int block_ic_flag;
    int block_ic_limit;
    int block_ic_base;
    int block_ic_offset;
    int block_use_bias;
    int block_oc_flag;
    int block_oc_limit;
    int block_oc_base;
    int block_oc_offset;
    int div_ifm_c_en;

  public:
    hw_gemm();
    void block_channel_check();
    void gemm_forward_s(ncnn::Mat &ifm, ncnn::Mat &wgt, ncnn::Mat &ofm);
    void gemm_forward_block(float* ifm_baseaddr, unsigned char* wgt_baseaddr, float* ofm_baseaddr);
    void gemm_forward_block_oc(float* ifm_baseaddr, unsigned char* wgt_baseaddr, float* ofm_baseaddr);
    void gemm_forward(ncnn::Mat &ifm, ncnn::Mat &wgt, ncnn::Mat &ofm);
};

} // namespace accel

#endif