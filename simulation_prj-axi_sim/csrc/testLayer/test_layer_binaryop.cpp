#include "../utils/utils.h"
#include "test_layer.h"

namespace test {

enum OperationType
{
    Operation_ADD = 0,
    Operation_SUB = 1,
    Operation_MUL = 2,
    Operation_DIV = 3,
    Operation_MAX = 4,
    Operation_MIN = 5,
    Operation_POW = 6,
    Operation_RSUB = 7,
    Operation_RDIV = 8
};

void forward_sim_binaryop(std::vector<ncnn::Mat> &in, ncnn::Mat &out, int op_type) {
    for (int c = 0; c < out.c; c++) {
        float *out_ptr = out.channel(c);
        float *in_0_ptr = in[0].channel(c);
        float *in_1_ptr = in[1].channel(c);
        for (int wh = 0; wh < out.w * out.h; wh++) {
            switch (op_type)
            {
            case Operation_MUL:
                *out_ptr++ = *in_0_ptr++ * *in_1_ptr++;
                break;
            case Operation_ADD:
                *out_ptr++ = *in_0_ptr++ + *in_1_ptr++;
                break;
            case Operation_MAX:
                *out_ptr++ = std::max(*in_0_ptr++, *in_1_ptr++);
                break;
            case Operation_MIN:
                *out_ptr++ = std::min(*in_0_ptr++, *in_1_ptr++);
                break;
            case Operation_SUB:
                *out_ptr++ = *in_0_ptr++ - *in_1_ptr++;
                break;
            default:
                break;
            }                
        }
    }
}

void forward_ncnn_binaryop(std::vector<ncnn::Mat> &in, std::vector<ncnn::Mat> &out, ncnn::ParamDict &pd) {
    std::vector<ncnn::Mat> weights(0);
    int ret = test_layer<ncnn::BinaryOp>("BinaryOp", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_binaryop\n");
        assert(0);
    }
}

void test_layer_binaryop(int w, int h, int c, int op_type) {

    printf("[ log ]: data.dims = %d, data.shape = (%d,%d,%d)\n",3,w,h,c);
    printf("[ log ]: op_type = %d\n",op_type);

    std::vector<ncnn::Mat> data_in(2);

    data_in[0] = RandomMat(w, h, c);
    data_in[1] = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, op_type);

    ncnn::Option opt;
    ncnn::Mat c_sim;
    c_sim.create_like(data_in[0], opt.blob_allocator);
    std::vector<ncnn::Mat> c_ncnn;

    forward_sim_binaryop(data_in, c_sim, op_type);
    forward_ncnn_binaryop(data_in, c_ncnn, pd);

    if (CompareMat(c_ncnn[0], c_sim, 0.001) == 0)
        printf("\033[;32m[ log ]: TEST LAYER binaryop BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST LAYER binaryop BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
    printf("\n");
}

void test_layer_binaryop_batch(){
    SRAND(7767517);

    test_layer_binaryop(13, 34, 35, Operation_ADD);
    test_layer_binaryop(13, 34, 35, Operation_SUB);
    test_layer_binaryop(13, 34, 35, Operation_MUL);
    test_layer_binaryop(13, 34, 35, Operation_MAX);
    test_layer_binaryop(13, 34, 35, Operation_MIN);
    test_layer_binaryop(3, 4, 5, Operation_ADD);
    test_layer_binaryop(3, 4, 5, Operation_SUB);
    test_layer_binaryop(3, 4, 5, Operation_MUL);
    test_layer_binaryop(3, 4, 5, Operation_MAX);
    test_layer_binaryop(3, 4, 5, Operation_MIN);
}

} // namespace test
