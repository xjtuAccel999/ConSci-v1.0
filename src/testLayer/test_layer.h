#ifndef __TEST_LAYER_H_
#define __TEST_LAYER_H_

#include "../hw/hw_act.h"
#include "../hw/hw_gemm.h"
#include "../hw/hw_math.h"
#include "../hw/hw_pool.h"
#include "../hw/hw_innerprod.h"
#include "../layer/absval.h"
#include "../layer/convolution.h"
#include "../layer/convolutiondepthwise.h"
#include "../layer/eltwise.h"
#include "../layer/pooling.h"
#include "../layer/relu.h"
#include "../layer/tanh.h"
#include "../layer/sigmoid.h"
#include "../layer/swish.h"
#include "../layer/elu.h"
#include "../layer/selu.h"
#include "../layer/clip.h"
#include "../layer/hardsigmoid.h"
#include "../layer/hardswish.h"
#include "../layer/bias.h"
#include "../layer/dropout.h"
#include "../layer/threshold.h"
#include "../layer/scale.h"
#include "../layer/innerproduct.h"
#include "../layer/binaryop.h"
#include "../layer/fused_activation.h"

#include "../config.h"
#include "../ncnn/mat.h"
#include "../ncnn/option.h"
#include "../utils/testutil.h"
#include "../utils/utils.h"
//#include "../veri.h"
#include "../hw/accel_params.h"
#include <assert.h>
#include <math.h>

//#include <omp.h>

extern int count_total;
extern int count_success;
extern int count_fail;

namespace test {
// TEST_LAYER_CONV
void forward_ncnn_conv(accel::hw_gemm &inst, std::vector<ncnn::Mat> &weights, ncnn::Mat &ifm_data, ncnn::Mat &activation_params, ncnn::Mat &ofm_data);
void mat_quant_fp32Toint8(ncnn::Mat &i_data, ncnn::Mat &o_data, float scale);
int ifm_layout(ncnn::Mat &i_data, ncnn::Mat &o_data, accel::hw_gemm &inst);
void fetch_wgt(ncnn::Mat &wgt_buffer, ncnn::Mat &wgt_32x32, int h_index, int v_index);
void fetch_ifm(ncnn::Mat &ifm_buffer, ncnn::Mat &ifm_32x32, accel::hw_gemm &inst, int ow_cnt, int oh_cnt, int kw_cnt, int kh_cnt, int ic_cnt);
void gemm_32x32(ncnn::Mat &a, ncnn::Mat &b, ncnn::Mat &c);
void acc_mem(ncnn::Mat &ofm_data, ncnn::Mat &psum_data);
template <typename T> void opfusion_32x32(ncnn::Mat &i_data, ncnn::Mat &o_data, accel::hw_gemm &inst, int c_index);
void add_block2ofm_nchw(ncnn::Mat &block, ncnn::Mat &ofm, accel::hw_gemm inst, int w_t, int h_t, int c_t);
void gemm_forward_sim(ncnn::Mat &ifm_buffer, ncnn::Mat &wgt_buffer, ncnn::Mat &ofm_buffer, accel::hw_gemm &inst);
void forward_sim_conv(accel::hw_gemm inst, std::vector<ncnn::Mat> &weights, ncnn::Mat &ifm_data, ncnn::Mat &ofm_data);
void test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_bottom_t, int op, int layout_en, int requant_en);
void test_layer_conv_batch();
void del_duplicate_passed_error(const char* error_txt, const char* right_txt);
void compute_pad(int iw_t, int ih_t, int k_t, int s_t, int &pd_left_t, int &pd_right_t, int &pd_top_t, int &pd_bottom_t);

// TEST_LAYER_POOL
void forward_ncnn_pool(ncnn::Mat &ifm, ncnn::Mat &ofm, accel::hw_pool &inst);
void forward_sim_pool(ncnn::Mat &ifm, ncnn::Mat &ofm, accel::hw_pool &inst);
void test_layer_pool(int iw_t, int ih_t, int ic_t, int k_t, int s_t, int type_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_down_t, int pad_mode, float pad_value);
void test_layer_pool_batch();
int rand_int(int min, int max);




// TEST_LAYER_ELTWISE
void forward_sim_eltwise(std::vector<ncnn::Mat> &in, ncnn::Mat &out, int op_type);
void forward_ncnn_eltwise(std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out, ncnn::ParamDict &pd);
void test_layer_eltwise(int w, int h, int c, int op_type);
void test_layer_eltwise_batch();

// TEST_LAYER_BINARYOP
void forward_sim_binaryop(std::vector<ncnn::Mat> &in, ncnn::Mat &out, int op_type);
void forward_ncnn_binaryop(std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out, ncnn::ParamDict &pd);
void test_layer_binaryop(int w, int h, int c, int op_type);
void test_layer_binaryop_batch();

// TEST_LAYER_ABSVAL
void forward_ncnn_absval(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd);
void forward_sim_absval(ncnn::Mat &in, ncnn::Mat &out);
void test_layer_bias(const ncnn::Mat& data_in);
void test_layer_absval_batch();

//TEST_LAYER_BIAS
void forward_ncnn_bias(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, ncnn::ParamDict &pd);
void forward_sim_bias(ncnn::Mat &in, ncnn::Mat& data_bias, ncnn::Mat &out);
void test_layer_bias(const ncnn::Mat& data_in);
void test_layer_bias_batch();

//TEST_LAYER_DROPOUT
void forward_ncnn_dropout(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd);
void forward_sim_dropout(ncnn::Mat &in, float scale, ncnn::Mat &out);
void test_layer_dropout(const ncnn::Mat& data_in, float scale);
void test_layer_dropout_batch();

//TEST_LAYER_THRESHOLD
void forward_ncnn_threshold(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd);
void forward_sim_threshold(ncnn::Mat &in, float threshold, ncnn::Mat &out);
void test_layer_threshold();

//TEST_LAYER_SCALE
void forward_ncnn_scale(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, ncnn::ParamDict &pd);
void forward_sim_scale(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, int bias_en);
void test_layer_scale(const ncnn::Mat& data_in, int bias_en);
void test_layer_scale_batch();

//TEST_LAYER_RELU
void forward_ncnn_relu(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd);
void forward_sim_relu(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd);
void test_layer_relu();

//TEST_LAYER_CLIP
void forward_ncnn_clip(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd);
void forward_sim_clip(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd);
void test_layer_clip();

//TEST_TANH
void forward_ncnn_tanh(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void forward_sim_tanh(ncnn::Mat &in, ncnn::Mat &out);
void test_layer_tanh();

//TEST_SIGMOID
void forward_ncnn_sigmoid(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void forward_sim_sigmoid(ncnn::Mat &in, ncnn::Mat &out);
void test_layer_sigmoid();

//TEST_SWISH
void forward_ncnn_swish(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void forward_sim_swish(ncnn::Mat &in, ncnn::Mat &out);
void test_layer_swish();

//TEST_ELU
void forward_ncnn_elu(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void forward_sim_elu(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void test_layer_elu();

//TEST_SELU
void forward_ncnn_selu(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void forward_sim_selu(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void test_layer_selu();

//TEST_HARDSIGMOID
void forward_ncnn_hardsigmoid(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void forward_sim_hardsigmoid(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void test_layer_hardsigmoid();

//TEST_HARDSWISH
void forward_ncnn_hardswish(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void forward_sim_hardswish(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd);
void test_layer_hardswish();

//TEST_LAYER_INNERPROD
void forward_ncnn_innerprod(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, ncnn::ParamDict &pd);
void forward_sim_innerprod(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, ncnn::ParamDict &pd);
void test_layer_innerprod(const ncnn::Mat& data_in, int num_output, int bias_en, int act_sel);
void test_layer_innerprod_batch();
} // namespace test

#endif
