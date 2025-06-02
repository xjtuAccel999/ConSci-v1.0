#include "../utils/utils.h"
#include "test_layer.h"

namespace test {

void forward_ncnn_threshold(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd) {
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::Threshold>("Threshold", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_threshold\n");
        assert(0);
    }
}

void forward_sim_threshold(ncnn::Mat &in, float threshold, ncnn::Mat &out) {
    for (int c = 0; c < in.c; c++) {
        float *i_ptr = in.channel(c);
        float *o_ptr = out.channel(c);

        for (int i = 0; i < in.w * in.h; i++) {
            o_ptr[i] = i_ptr[i] > threshold ? 1.f : 0.f; 
        }
    }
}

void test_layer_threshold() {
    SRAND(7767517);

    ncnn::ParamDict pd;
    ncnn::Option opt;

    ncnn::Mat data_in = RandomMat(20, 30, 300);
    // ncnn::Mat data_in = RandomMat(256);
    float threshold = RandomFloat(-0.5f,0.5f);
    pd.set(0, threshold);

    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);

    forward_sim_threshold(data_in, threshold, c_sim);
    forward_ncnn_threshold(data_in, c_ncnn, pd);

    if (CompareMat(c_ncnn, c_sim, 0.001) == 0)
        printf("\033[;32m[ log ]: TEST LAYER THRESHOLD BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST LAYER THRESHOLD BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
}

} // namespace test
