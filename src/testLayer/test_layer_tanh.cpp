#include "../utils/utils.h"
#include "test_layer.h"
#include "xtime_l.h"
namespace test {

void forward_ncnn_tanh(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd) {
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::TanH>("TanH", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_tanh\n");
        assert(0);
    }
}

void forward_sim_tanh(ncnn::Mat &in, ncnn::Mat &out) {
    for (int q = 0; q < in.c; q++)
    {
        float* i_ptr = in.channel(q);
        float* o_ptr = out.channel(q);

        for (int i = 0; i < in.w*in.h; i++)
        {
            o_ptr[i] = static_cast<float>(tanh(i_ptr[i]));
        }
    }
}

void test_layer_tanh() {
    SRAND(7767517);

    ncnn::ParamDict pd;
    pd.set(0, 0);

    ncnn::Option opt;
    ncnn::Mat data_in = RandomMat(10, 100, 100);
    // ncnn::Mat data_in = RandomMat(256);

    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);
    #ifdef ALU_TIME
		u64 tEnd_tanh_sw, tCur_tanh_sw;
		u32 tUsed_tanh_sw;
        XTime_GetTime(&tCur_tanh_sw);
    #endif
    forward_sim_tanh(data_in, c_sim);
    #ifdef ALU_TIME
		XTime_GetTime(&tEnd_tanh_sw);
		tUsed_tanh_sw = ((tEnd_tanh_sw-tCur_tanh_sw)*1000000)/(COUNTS_PER_SECOND);
		printf("time_tanh_sw elapsed is %d us\n",tUsed_tanh_sw);
    #endif
    forward_ncnn_tanh(data_in, c_ncnn, pd);

    if (CompareMat(c_ncnn, c_sim, 0.01) == 0)
        printf("\033[;32m[ log ]: TEST TANH BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST TANH BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
}

} // namespace test
