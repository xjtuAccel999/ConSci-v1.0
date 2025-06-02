#include "../utils/utils.h"
#include "test_layer.h"

namespace test {

void forward_ncnn_hardsigmoid(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd) {
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::HardSigmoid>("HardSigmoid", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_elu\n");
        assert(0);
    }
}

void forward_sim_hardsigmoid(ncnn::Mat &in, ncnn::Mat &out,ncnn::ParamDict &pd) {
    float alpha = pd.get(0, 0.f);
    float beta = pd.get(1, 0.f);
    float lower = -beta / alpha;
    float upper = (1.f / alpha) + lower;
    // printf(" alpha = %f, beta = %f, lower = %f, upper = %f\n",alpha ,beta,lower,upper);
    for (int q = 0; q < in.c; q++)
    {
        float* i_ptr = in.channel(q);
        float* o_ptr = out.channel(q);

        for (int i = 0; i < in.w*in.h; i++)
        {
            if (i_ptr[i] < lower)
                o_ptr[i] = 0.f;
            else if (i_ptr[i] > upper)
                o_ptr[i] = 1.f;
            else
                o_ptr[i] = i_ptr[i] * alpha + beta;
        }
    }
}

void test_layer_hardsigmoid() {
    SRAND(7767517);

    ncnn::ParamDict pd;
    pd.set(0, 0.2f);
    pd.set(1, 0.5f);

    ncnn::Option opt;
    ncnn::Mat data_in = RandomMat(10,10,10);
    // ncnn::Mat data_in = RandomMat(256);

    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);

    forward_sim_hardsigmoid(data_in, c_sim, pd);
    forward_ncnn_hardsigmoid(data_in, c_ncnn, pd);
    // printf("data_in\n");
    // printf_float32_mat(data_in);
    // printf("ofm_sw\n");
    // printf_float32_mat(c_sim);
    // printf("ofm_hw\n");
    // printf_float32_mat(c_ncnn);

    if (CompareMat(c_ncnn, c_sim, 0.001) == 0)
        printf("\033[;32m[ log ]: TEST HARDSIGMOID BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST HARDSIGMOID BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
}

} // namespace test
