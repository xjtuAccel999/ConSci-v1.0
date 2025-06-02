#include "../utils/utils.h"
#include "test_layer.h"

namespace test {

void forward_ncnn_scale(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, ncnn::ParamDict &pd) {
    int ret = test_layer<ncnn::Scale>("Scale", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_scale\n");
        assert(0);
    }
}

void forward_sim_scale(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, int bias_en){
    int dims = in.dims;
    float* scale_ptr = weights[0];

    if (dims == 1)
    {
        int w = in.w;
        float* i_ptr = in;
        float* o_ptr = out;
        if (bias_en)
        {
            float* bias_ptr = weights[1];
            for (int i = 0; i < w; i++)
            {
                o_ptr[i] = i_ptr[i] * scale_ptr[i] + bias_ptr[i];
            }
        }
        else
        {
            for (int i = 0; i < w; i++)
            {
                o_ptr[i] = i_ptr[i] * scale_ptr[i];
            }
        }
    }

    if (dims == 2)
    {
        int w = in.w;
        int h = in.h;
        if (bias_en)
        {
            float* bias_ptr = weights[1];
            for (int i = 0; i < h; i++)
            {
                float* i_ptr = in.row(i);
                float* o_ptr = out.row(i);
                float s = scale_ptr[i];
                float bias = bias_ptr[i];
                for (int j = 0; j < w; j++)
                {
                    o_ptr[j] = i_ptr[j] * s + bias;
                }
            }
        }
        else
        {
            for (int i = 0; i < h; i++)
            {
                float* i_ptr = in.row(i);
                float* o_ptr = out.row(i);
                float s = scale_ptr[i];
                for (int j = 0; j < w; j++)
                {
                    o_ptr[j] = i_ptr[j] * s;
                }
            }
        }
    }

    if (dims == 3)
    {
        int w = in.w;
        int h = in.h;
        int channels = in.c;
        int size = w * h;
        if (bias_en)
        {
            float* bias_ptr = weights[1];
            for (int q = 0; q < channels; q++)
            {
                float* i_ptr = in.channel(q);
                float* o_ptr = out.channel(q);
                float s = scale_ptr[q];
                float bias = bias_ptr[q];
                for (int i = 0; i < size; i++)
                {
                    o_ptr[i] = i_ptr[i] * s + bias;
                }
            }
        }
        else
        {
            for (int q = 0; q < channels; q++)
            {
                float* i_ptr = in.channel(q);
                float* o_ptr = out.channel(q);
                float s = scale_ptr[q];
                for (int i = 0; i < size; i++)
                {
                    o_ptr[i] = i_ptr[i] * s;
                }
            }
        }
    }
}

void test_layer_scale(const ncnn::Mat& data_in, int bias_en) {
    ncnn::Mat& data_in_cast = const_cast<ncnn::Mat&>(data_in);
    
    ncnn::Option opt;
    int scale_data_size;
    if (data_in.dims == 1) scale_data_size = data_in.w;
    if (data_in.dims == 2) scale_data_size = data_in.h;
    if (data_in.dims == 3) scale_data_size = data_in.c;

    if (data_in.dims == 1) printf("[ log ]: data.dims = %d, data.shape = (%d)\n",1,data_in.w);
    if (data_in.dims == 2) printf("[ log ]: data.dims = %d, data.shape = (%d,%d)\n",2,data_in.w,data_in.h);
    if (data_in.dims == 3) printf("[ log ]: data.dims = %d, data.shape = (%d,%d,%d)\n",3,data_in.w,data_in.h,data_in.c);
    printf("[ log ]: scale_data_size = %d, bias_en = %d\n",scale_data_size,bias_en);

    ncnn::ParamDict pd;
    pd.set(0, scale_data_size);
    pd.set(1, bias_en);

    std::vector<ncnn::Mat> weights(bias_en ? 2 : 1);
    weights[0] = RandomMat(scale_data_size);
    if (bias_en)
        weights[1] = RandomMat(scale_data_size);

    // printf_int32_mat(data_in);
    // printf_int32_mat(weights[0]);
    // if(bias_en)
    //     printf_int32_mat(weights[1]);

    ncnn::Mat c_sim;
    c_sim.create_like(data_in, opt.blob_allocator);
    ncnn::Mat c_ncnn;
    c_ncnn.create_like(data_in, opt.blob_allocator);

    forward_sim_scale(data_in_cast, weights, c_sim, bias_en);
    forward_ncnn_scale(data_in_cast, weights, c_ncnn, pd);

#ifdef MAT_LOG
    log_mat_file<float>(c_sim, (char *)"./log/log_scale_sim.txt", -1, 1, 0);
    log_mat_file<float>(c_ncnn, (char *)"./log/log_scale_ncnn.txt", -1, 1, 0);
#endif

    if (CompareMat(c_ncnn, c_sim, 0.001) == 0)
        printf("\033[;32m[ log ]: TEST LAYER SCALE BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST LAYER SCALE BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
    printf("\n");
}

void test_layer_scale_batch(){
    SRAND(7767517);

    test_layer_scale(RandomMat(5, 3, 48), 0);
    test_layer_scale(RandomMat(5, 3, 48), 1);
    test_layer_scale(RandomMat(5, 7, 24), 0);
    test_layer_scale(RandomMat(5, 7, 24), 1);
    test_layer_scale(RandomMat(7, 9, 12), 0);
    test_layer_scale(RandomMat(7, 9, 12), 1);
    test_layer_scale(RandomMat(3, 5, 13), 0);
    test_layer_scale(RandomMat(3, 5, 13), 1);

    //when dims = 2,  w must be aligned to 4!!!
    test_layer_scale(RandomMat(12, 48), 0);
    test_layer_scale(RandomMat(12, 48), 1);
    test_layer_scale(RandomMat(16, 24), 0);
    test_layer_scale(RandomMat(16, 24), 1);
    test_layer_scale(RandomMat(20, 12), 0);
    test_layer_scale(RandomMat(20, 12), 1);
    test_layer_scale(RandomMat(24, 15), 0);
    test_layer_scale(RandomMat(24, 15), 1);

    test_layer_scale(RandomMat(128), 0);
    test_layer_scale(RandomMat(128), 1);
    test_layer_scale(RandomMat(124), 0);
    test_layer_scale(RandomMat(124), 1);
    test_layer_scale(RandomMat(127), 0);
    test_layer_scale(RandomMat(127), 1);
}

} // namespace test







