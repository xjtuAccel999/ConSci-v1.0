#include "../utils/utils.h"
#include "test_layer.h"

namespace test {

void forward_ncnn_bias(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, ncnn::ParamDict &pd) {
    int ret = test_layer<ncnn::Bias>("Bias", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_bias\n");
        assert(0);
    }
}

void forward_sim_bias(ncnn::Mat &in, ncnn::Mat& data_bias, ncnn::Mat &out) {
    for (int c = 0; c < in.c; c++) {
        float *i_ptr = in.channel(c);
        float *o_ptr = out.channel(c);
        float bias = data_bias[c];

        for (int i = 0; i < in.w * in.h; i++) {
            o_ptr[i] = i_ptr[i] + bias;
        }
    }
}

void test_layer_bias(const ncnn::Mat& data_in) {

    if (data_in.dims == 1) printf("[ log ]: data.dims = %d, data.shape = (%d)\n",1,data_in.w);
    if (data_in.dims == 2) printf("[ log ]: data.dims = %d, data.shape = (%d,%d)\n",2,data_in.w,data_in.h);
    if (data_in.dims == 3) printf("[ log ]: data.dims = %d, data.shape = (%d,%d,%d)\n",3,data_in.w,data_in.h,data_in.c);

    ncnn::ParamDict pd;
    ncnn::Option opt;

    ncnn::Mat& data_in_cast = const_cast<ncnn::Mat&>(data_in);
    
    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(data_in.c);
    pd.set(0, data_in.c);

    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);

    forward_sim_bias(data_in_cast, weights[0], c_sim);
    forward_ncnn_bias(data_in_cast, weights, c_ncnn, pd);

    if (CompareMat(c_ncnn, c_sim, 0.001) == 0)
        printf("\033[;32m[ log ]: TEST LAYER BIAS BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST LAYER BIAS BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
    printf("\n");
}

void test_layer_bias_batch(){
    SRAND(7767517);

    test_layer_bias(RandomMat(20, 300, 20));
    test_layer_bias(RandomMat(5, 7, 24));
    test_layer_bias(RandomMat(7, 9, 12));
    test_layer_bias(RandomMat(3, 5, 13));
}



} // namespace test
