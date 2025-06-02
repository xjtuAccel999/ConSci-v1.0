#include "../utils/utils.h"
#include "test_layer.h"
#include "xtime_l.h"
namespace test {

void forward_ncnn_elu(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd) {
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::ELU>("ELU", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_elu\n");
        assert(0);
    }
}

void forward_sim_elu(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd) {
    float alpha = pd.get(0, 0.f);
    // printf("alpha_sim = %f\n",alpha);
    for (int q = 0; q < in.c; q++)
    {
        float* i_ptr = in.channel(q);
        float* o_ptr = out.channel(q);

        for (int i = 0; i < in.w * in.h; i++)
        {
            if (i_ptr[i] < 0.f)
                o_ptr[i] = static_cast<float>(alpha * (exp(i_ptr[i]) - 1.f));
            else 
                o_ptr[i] = i_ptr[i];
        }
    }
}

void test_layer_elu() {
    SRAND(7767517);

    
    ncnn::ParamDict pd;
    pd.set(0, 0.122f);

    ncnn::Option opt;
    // ncnn::Mat data_in = RandomMat(10);
    ncnn::Mat data_in = RandomMat(256,10,100);

    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);

    #ifdef ALU_TIME
		u64 tEnd_elu_sw, tCur_elu_sw;
		u32 tUsed_elu_sw;
        XTime_GetTime(&tCur_elu_sw);
    #endif
    forward_sim_elu(data_in, c_sim, pd);
    #ifdef ALU_TIME
		XTime_GetTime(&tEnd_elu_sw);
		tUsed_elu_sw = ((tEnd_elu_sw-tCur_elu_sw)*1000000)/(COUNTS_PER_SECOND);
		printf("time_elu_sw elapsed is %d us\n",tUsed_elu_sw);
    #endif
    forward_ncnn_elu(data_in, c_ncnn, pd);
    // printf("data_in\n");
    // printf_float32_mat(data_in);
    // printf("ofm_sw\n");
    // printf_float32_mat(c_sim);
    // printf("ofm_hw\n");
    // printf_float32_mat(c_ncnn);

    if (CompareMat(c_ncnn, c_sim, 0.002) == 0)
        printf("\033[;32m[ log ]: TEST ELU BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST ELU BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
}

} // namespace test
