#ifndef __TEST_OP_H_
#define __TEST_OP_H_

#include "../veri.h"
#include "../hw/accel_params.h"
#include "../utils/testutil.h"
#include "../ncnn/mat.h"
#include "../ncnn/option.h"
#include <assert.h>
#include <math.h>
#include "../config.h"


namespace test{
    #ifdef TEST_FP32_ADD
    void test_fp32_add();
    #endif
    #ifdef TEST_FP32_MUL
    void test_fp32_mul();
    #endif
    #ifdef TEST_PE
    void test_pe_single();
    #endif
    #ifdef TEST_FP32_TO_INT8
    void test_fp32ToInt8();
    #endif
    #ifdef TEST_INT32_TO_FP32
    void test_Int32Tofp32();
    #endif

    //TEST_IFMBUF
    int read_ifm(ncnn::Mat& gemm_ifm_buffer, int line_len);
    void test_ifmbuf_single(int data_format, ncnn::Mat& data_in);
    void test_ifmbuf_batch();

    //TEST_WGTBUF
    void test_wgtbuf(const ncnn::Mat& in, int ic, int ow, int oh, int oc, int kernel, int op);
    void test_wgtbuf_single(int ow, int oh, int oc, int ic, int kernel,int op );
    void test_wgtbuf_batch();

}

#endif