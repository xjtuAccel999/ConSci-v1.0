#include "../utils/utils.h"
#include "test_layer.h"
#include "xtime_l.h"
namespace test {

void forward_ncnn_innerprod(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, ncnn::ParamDict &pd) {
    int ret = test_layer<ncnn::InnerProduct>("InnerProduct", pd, weights, in, out);
    if (ret != 0) {
        printf("[error]: forward_ncnn_innerprod\n");
        assert(0);
    }
}

void forward_sim_innerprod(ncnn::Mat &in, std::vector<ncnn::Mat>& weights, ncnn::Mat &out, ncnn::ParamDict &pd) {
    int num_output = pd.get(0,0);
    int bias_en = pd.get(1, 0);
    int activation_type = pd.get(9, 0);
    ncnn::Mat activation_params = pd.get(10, ncnn::Mat());

    int channels = in.c;
    int size = in.w * in.h;

    ncnn::Mat weight_data_int8_scales;
    ncnn::Mat bottom_blob_int8_scales;
    ncnn::Mat bias_data;

    if(bias_en){
        bias_data = weights[1];
        weight_data_int8_scales = weights[2];
        bottom_blob_int8_scales = weights[3];
    }
    else{
        weight_data_int8_scales = weights[1];
        bottom_blob_int8_scales = weights[2];
    }

    for (int p = 0; p < num_output; p++)
    {
        float* outptr = out;
        int sum = 0;

        // int offset = size * channels * p;
        int offset = in.total() * p;
        // channels
        for (int q = 0; q < channels; q++)
        {
            // signed char* w = (signed char*)weights[0] + offset + size * q;
            signed char* w = (signed char*)weights[0] + offset + in.cstep * q;
            signed char* m = in.channel(q);

            for (int i = 0; i < size; i++)
            {
                sum += m[i] * w[i];
            }
        }

        // dequantize and relu
        float scale_in;
        if (weight_data_int8_scales[p] == 0)
            scale_in = 0;
        else
            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

        float sumfp32 = sum * scale_in;

        if(bias_en)
            sumfp32 += bias_data[p];

        outptr[p] = activation_ss(sumfp32, activation_type, activation_params);
    }
}

void test_layer_innerprod(const ncnn::Mat& data_in, int num_output, int bias_en, int act_sel) {

    ncnn::ParamDict pd;
    ncnn::Option opt;

    ncnn::Mat& data_in_cast = const_cast<ncnn::Mat&>(data_in);

    pd.set(0, num_output); // num_output
    pd.set(1, bias_en);  // bias_term
    pd.set(2, num_output * data_in.w * data_in.h * data_in.c);
    pd.set(8, 1); // int8_scale_term

    int activation_type = act_sel; 
    ncnn::Mat activation_params(2);
    activation_params[0] = (activation_type == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                               // beta
    pd.set(9, activation_type);
    pd.set(10, activation_params);
    int weight_data_size_align = u_align(data_in.total(),16)*num_output;
    pd.set(11, weight_data_size_align);

    std::vector<ncnn::Mat> weights(bias_en ? 4 : 3);
    int size = data_in.w * data_in.h;
    int k = size * data_in.c;
    ncnn::Mat weight_scales = RandomMat(num_output);
    ncnn::Mat input_scales = scales_mat(data_in_cast, 1, k, k);
    ncnn::Mat data_in_int8(data_in.w, data_in.h, data_in.c, 1u, opt.blob_allocator);
    mat_quant_fp32Toint8(data_in_cast, data_in_int8, input_scales[0]);

    ncnn::Mat weight_int8 = RandomS8Mat(num_output * k);
    ncnn::Mat weight_reorder(u_align(data_in_int8.total(),16)*num_output, 1u, opt.blob_allocator); 
    memset(weight_reorder.data, 0, weight_reorder.total());

    for(int i=0 ;i<num_output; i++){
        char* weight_int8_ptr = (char*)weight_int8.data + i*k;
        char* weight_reorder_ptr = (char*)weight_reorder.data + i*data_in_int8.total();
        for(int j=0; j<data_in.c; j++){
            memcpy(weight_reorder_ptr, weight_int8_ptr, size);
            weight_int8_ptr += size;
            weight_reorder_ptr += data_in_int8.cstep;
        }
    }

    // printf("k = %d, data_in_int.total = %ld, size = %d, data_in_int8.cstp = %d\n",k,data_in_int8.total(),size,data_in_int8.cstep);
    // printf("weight_int8\n");
    // printf_int8_mat(weight_int8);
    // printf("weight_reorder\n");
    // printf_int8_mat(weight_reorder);

    weights[0] = weight_reorder;

    // weights[0] = RandomS8Mat(num_output * k);
    // ncnn::Mat weight_scales = scales_mat(weights[0], num_output, k, k);

    if(bias_en){
        weights[1] = RandomMat(num_output, -10.f, 10.f);
        weights[2] = weight_scales;
        weights[3] = input_scales;
    }
    else{
        weights[1] = weight_scales;
        weights[2] = input_scales;
    }

    printf("[ log ]: data_in.shape = (%d, %d, %d), data_in_int8.total() = %ld\n",data_in.w, data_in.h, data_in.c, data_in_int8.total());
    printf("[ log ]: num_output = %d, bias_en = %d, act_sel = %d\n",num_output, bias_en, act_sel);
    
    ncnn::Mat c_sim(num_output, 4u, opt.blob_allocator);
    ncnn::Mat c_ncnn(num_output, 4u, opt.blob_allocator);

#ifdef MAT_LOG
    log_mat_file<unsigned char>(data_in_int8, (char *)"./log/log_innerprod_a.txt", -1, 1, 0);
    log_mat_file<unsigned char>(weights[0], (char *)"./log/log_innerprod_b.txt", -1, 1, 0);
#endif


	#ifdef INNERPROD_TIME
		u64 tEnd_innerprod_sw, tCur_innerprod_sw;
		u32 tUsed_innerprod_sw;
        XTime_GetTime(&tCur_innerprod_sw);
    #endif
    forward_sim_innerprod(data_in_int8, weights, c_sim, pd);
	#ifdef INNERPROD_TIME
		XTime_GetTime(&tEnd_innerprod_sw);
		tUsed_innerprod_sw = ((tEnd_innerprod_sw-tCur_innerprod_sw)*1000000)/(COUNTS_PER_SECOND);
		printf("time_innerprod_sw elapsed is %d us\n",tUsed_innerprod_sw);
    #endif
    forward_ncnn_innerprod(data_in_cast, weights, c_ncnn, pd);

#ifdef MAT_LOG
    log_mat_file<float>(c_sim, (char *)"./log/log_innerprod_data_sim.txt", -1, 1, 0);
    log_mat_file<float>(c_ncnn, (char *)"./log/log_innerprod_data_ncnn.txt", -1, 1, 0);
#endif

    if (CompareMat(c_sim, c_ncnn, 0.001) == 0)
        printf("\033[;32m[ log ]: TEST LAYER INNERPROD BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
    else
        printf("\033[;31m[ log ]: TEST LAYER INNERPROD BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
    printf("\n");
}

void test_layer_innerprod_batch(){
    SRAND(7767517);
    // void test_layer_innerprod(const ncnn::Mat& data_in, int num_output, int bias_en, int act_sel)
//    test_layer_innerprod(RandomMat(4,  4,  2),  4, 1, 2);
//    test_layer_innerprod(RandomMat(1,  3, 16),  1, 1, 1);
//    test_layer_innerprod(RandomMat(1,  3,  2),  1, 1, 1);
//    test_layer_innerprod(RandomMat(1,  3,  8),  1, 1, 1);
//    test_layer_innerprod(RandomMat(4,  3,  2),  1, 1, 1);
//    test_layer_innerprod(RandomMat(4,  4,  2),  4, 1, 2);
//    test_layer_innerprod(RandomMat(5,  3, 32),  3, 1, 2);
//    test_layer_innerprod(RandomMat(7,  2, 32), 12, 1, 2);
//
//    test_layer_innerprod(RandomMat(13, 81, 91), 101, 1, 2);
//    test_layer_innerprod(RandomMat( 4,  2, 52),  13, 1, 0);
//    test_layer_innerprod(RandomMat( 7,  5, 91),  55, 0, 1);
    test_layer_innerprod(RandomMat( 512,  0, 0),  1000, 0, 1);
    test_layer_innerprod(RandomMat( 512,  0, 0),  1000, 0, 1);
    test_layer_innerprod(RandomMat( 512,  0, 0),  1000, 0, 1);
    test_layer_innerprod(RandomMat( 512,  0, 0),  1000, 0, 1);
}

} // namespace test
