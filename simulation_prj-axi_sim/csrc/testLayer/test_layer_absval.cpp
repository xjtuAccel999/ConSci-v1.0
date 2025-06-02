#include "../utils/utils.h"
#include "test_layer.h"

namespace test {

void forward_ncnn_absval(ncnn::Mat &in, ncnn::Mat &out, ncnn::ParamDict &pd) {
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::AbsVal>("AbsVal", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_absval\n");
        assert(0);
    }
}

void forward_sim_absval(ncnn::Mat &in, ncnn::Mat &out) {
    for (int c = 0; c < in.c; c++) {
        float *i_ptr = in.channel(c);
        float *o_ptr = out.channel(c);

        for (int i = 0; i < in.w * in.h; i++) {
            if (i_ptr[i] < 0)
                o_ptr[i] = -i_ptr[i];
            else
                o_ptr[i] = i_ptr[i];
        }
    }
}

void test_layer_absval(const ncnn::Mat& data_in) {

    if (data_in.dims == 1) printf("[ log ]: data.dims = %d, data.shape = (%d)\n",1,data_in.w);
    if (data_in.dims == 2) printf("[ log ]: data.dims = %d, data.shape = (%d,%d)\n",2,data_in.w,data_in.h);
    if (data_in.dims == 3) printf("[ log ]: data.dims = %d, data.shape = (%d,%d,%d)\n",3,data_in.w,data_in.h,data_in.c);

    ncnn::ParamDict pd;
    ncnn::Option opt;

    ncnn::Mat& data_in_cast = const_cast<ncnn::Mat&>(data_in);

    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);

    forward_sim_absval(data_in_cast, c_sim);
    forward_ncnn_absval(data_in_cast, c_ncnn, pd);

    if (CompareMat(c_sim, c_ncnn, 0.001) == 0)
        printf("\033[;32m[ log ]: TEST LAYER ABSVAL BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST LAYER ABSVAL BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
    printf("\n");
}

void test_layer_absval_batch(){
    SRAND(7767517);

    test_layer_absval(RandomMat(5, 7, 24));
    // test_layer_absval(RandomMat(7, 9, 12));
    // test_layer_absval(RandomMat(3, 5, 13));
    // test_layer_absval(RandomMat(15, 24));
    // test_layer_absval(RandomMat(19, 12));
    // test_layer_absval(RandomMat(17, 15));
    // test_layer_absval(RandomMat(128));
    // test_layer_absval(RandomMat(124));
    // test_layer_absval(RandomMat(127));
    // test_layer_absval(RandomMat(255));
    // test_layer_absval(RandomMat(20, 30, 300));
}

} // namespace test
