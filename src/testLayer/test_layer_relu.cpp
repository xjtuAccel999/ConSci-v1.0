#include "../utils/utils.h"
#include "test_layer.h"
#include "xtime_l.h"
namespace test {

void forward_ncnn_relu(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd) {
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::ReLU>("ReLU", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_relu\n");
        assert(0);
    }
}

void forward_sim_relu(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd) {
    float slope = pd.get(0, 0.f);
    if (slope == 0.f) {
        for (int c = 0; c < in.c; c++) {
            float *i_ptr = in.channel(c);
            float *o_ptr = out.channel(c);

            for (int i = 0; i < in.w * in.h; i++) {
                if (i_ptr[i] < 0)
                    o_ptr[i] = 0.f;
                else
                    o_ptr[i] = i_ptr[i];
            }
        }
    } else {
        for (int c = 0; c < in.c; c++) {
            float *i_ptr = in.channel(c);
            float *o_ptr = out.channel(c);

            for (int i = 0; i < in.w * in.h; i++) {
                if (i_ptr[i] < 0)
                    o_ptr[i] = i_ptr[i] * slope;
                else
                    o_ptr[i] = i_ptr[i];
            }
        }
    }
}

void test_layer_relu() {
    SRAND(7767517);
    // int op_type = RELU;

    ncnn::ParamDict pd;
    float slope = 2.0f;
    pd.set(0, slope);

    ncnn::Option opt;
    // ncnn::Mat data_in = RandomMat(20, 30, 300);
    ncnn::Mat data_in = RandomMat(10, 50, 100);
    // ncnn::Mat data_in = RandomMat(256);

    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);
    #ifdef ALU_TIME
		u64 tEnd_relu_sw, tCur_relu_sw;
		u32 tUsed_relu_sw;
        XTime_GetTime(&tCur_relu_sw);
    #endif
    forward_sim_relu(data_in, c_sim, pd);
    #ifdef ALU_TIME
		XTime_GetTime(&tEnd_relu_sw);
		tUsed_relu_sw = ((tEnd_relu_sw-tCur_relu_sw)*1000000)/(COUNTS_PER_SECOND);
		printf("time_relu_sw elapsed is %d us\n",tUsed_relu_sw);
    #endif
    forward_ncnn_relu(data_in, c_ncnn, pd);
    // printf("ofm_sw\n");
    // printf_float32_mat(c_sim);
    // printf("ofm_hw\n");
    // printf_float32_mat(c_ncnn);

    if (CompareMat(c_ncnn, c_sim, 0.001) == 0){
        if(slope==0.f)
            printf("\033[;32m[ log ]: TEST LAYER RELU BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
        else 
            printf("\033[;32m[ log ]: TEST LAYER RELU BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    }
        
    else{
        if(slope==0.f)
            printf("\033[;31m[ log ]: TEST RELU BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
        
        else 
            printf("\033[;31m[ log ]: TEST LAYER RELU BETWEEN NCNN AND SIMULATION FAILED\n\033[0m"); 
    }
        
        
}

} // namespace test
