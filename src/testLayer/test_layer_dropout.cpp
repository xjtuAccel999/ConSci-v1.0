#include "../utils/utils.h"
#include "test_layer.h"

namespace test {

void forward_ncnn_dropout(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd) {
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::Dropout>("Dropout", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_dropout\n");
        assert(0);
    }
}

void forward_sim_dropout(ncnn::Mat &in, float scale, ncnn::Mat &out) {
    for (int c = 0; c < in.c; c++) {
        float *i_ptr = in.channel(c);
        float *o_ptr = out.channel(c);

        for (int i = 0; i < in.w * in.h; i++) {
            o_ptr[i] = i_ptr[i] * scale;
        }
    }
}

void test_layer_dropout(const ncnn::Mat& data_in, float scale) {
    
    ncnn::Option opt;
    ncnn::ParamDict pd;
    pd.set(0, scale);

    if (data_in.dims == 1) printf("[ log ]: data.dims = %d, data.shape = (%d)\n",1,data_in.w);
    if (data_in.dims == 2) printf("[ log ]: data.dims = %d, data.shape = (%d,%d)\n",2,data_in.w,data_in.h);
    if (data_in.dims == 3) printf("[ log ]: data.dims = %d, data.shape = (%d,%d,%d)\n",3,data_in.w,data_in.h,data_in.c);
    printf("[ log ]: scale = %.3f\n",scale);

    ncnn::Mat& data_in_cast = const_cast<ncnn::Mat&>(data_in);
    
    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);

    forward_sim_dropout(data_in_cast, scale, c_sim);
    forward_ncnn_dropout(data_in_cast, c_ncnn, pd);

    if (CompareMat(c_ncnn, c_sim, 0.001) == 0)
        printf("\033[;32m[ log ]: TEST LAYER DROPOUT BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST LAYER DROPOUT BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
    printf("\n");
}

void test_layer_dropout_batch(){

    SRAND(7767517);

    test_layer_dropout(RandomMat(5, 7, 24), 1.f );
    test_layer_dropout(RandomMat(5, 7, 24), 0.2f);
    test_layer_dropout(RandomMat(7, 9, 12), 1.f );
    test_layer_dropout(RandomMat(7, 9, 12), 0.3f);
    test_layer_dropout(RandomMat(3, 5, 13), 1.f );
    test_layer_dropout(RandomMat(3, 5, 13), 0.5f);

    test_layer_dropout(RandomMat(15, 24), 1.f );
    test_layer_dropout(RandomMat(15, 24), 0.6f);
    test_layer_dropout(RandomMat(19, 12), 1.f );
    test_layer_dropout(RandomMat(19, 12), 0.4f);
    test_layer_dropout(RandomMat(17, 15), 1.f );
    test_layer_dropout(RandomMat(17, 15), 0.7f);

    test_layer_dropout(RandomMat(128), 1.f );
    test_layer_dropout(RandomMat(128), 0.4f);
    test_layer_dropout(RandomMat(124), 1.f );
    test_layer_dropout(RandomMat(124), 0.1f);
    test_layer_dropout(RandomMat(127), 1.f );
    test_layer_dropout(RandomMat(127), 0.5f);
}

} // namespace test
