//#include "../testOp/test_op.h"
#include "test_layer.h"
#include "time.h"
#include <iostream>
#include <string>
#include <fstream>
#include "../utils/wgt_reorder.h"
#include "../utils/conv_utils.h"
#include "../hw/dri_sd.h"
#include "xtime_l.h"
unsigned char log_id = 0;
std::string ifmbuf_block_path = "./log/ifmbufctl/log_ifmbuf_block_path_";
std::string wgtbuf_block_path = "./log/wgtbuf/log_wgtbuf_block_path_";
std::string accmem_block_path = "./log/accmem/log_accmem_block_path_";
std::string ofmbuf_block_path = "./log/ofmbuf/log_ofmbuf_block_path_";
std::string opfusion_block_path = "./log/opfusion/log_opfusion_block_path_";
std::string postfix = ".txt";
char log_path[100];
int count_total = 0;
int count_success = 0;
int count_fail = 0;

namespace test {

//----------------------------------------- forward_ncnn_conv ---------------------------------------------//
void forward_ncnn_conv(accel::hw_gemm &inst, std::vector<ncnn::Mat> &weights, ncnn::Mat &ifm_data, ncnn::Mat &activation_params, ncnn::Mat &ofm_data) {
    // set pd parameters
    ncnn::ParamDict pd;
    pd.set(0, inst.ofm_c);
    pd.set(1, inst.kernel);
    pd.set(2, 1); // dilation
    pd.set(3, inst.stride);
    pd.set(4, inst.padding_left); // pad
    pd.set(5, inst.bias_en);
    pd.set(6, inst.ofm_c * inst.ifm_c * inst.kernel * inst.kernel);
	pd.set(7, inst.op ? inst.ofm_c : 1); // group
    pd.set(8, inst.requant_en ? 101 : 1); // int8_scale_term
    pd.set(9, inst.act_op);
    pd.set(10, activation_params);

    pd.set(14, inst.padding_top);    // pad
    pd.set(15, inst.padding_right);  // pad
    pd.set(16, inst.padding_bottom); // pad

    pd.set(20, inst.layout_en);
    pd.set(21, inst.ifm_w);
    pd.set(22, inst.ifm_h);
    pd.set(23, inst.ifm_c);

    ncnn::Option opt;

    int flag = TEST_LAYER_DISABLE_GPU_TESTING;
    int ret;
    if (inst.op == 0) {
        ret = test_layer<ncnn::Convolution>("Convolution", pd, weights, ifm_data, ofm_data, inst.requant_en ? 1.0f : 0.001f, 0, flag);
    } else {
        ret = test_layer<ncnn::ConvolutionDepthWise>("ConvolutionDepthWise", pd, weights, ifm_data, ofm_data, inst.requant_en ? 1.0f : 0.001f, 0, flag);
    }
    if (ret != 0) {
        printf("[error]: test_convolution_int8 failed w=%d h=%d c=%d outch=%d kernel=%d stride=%d bias=%d requant=%d\n", inst.ifm_w, inst.ifm_h, inst.ifm_c, inst.ofm_c, inst.kernel, inst.stride,
               inst.bias_en, inst.requant_en);
        assert(0);
    }
}

//----------------------------------------- forward_sim_conv ---------------------------------------------//
static inline signed char float2int8(float v) {
    int int32 = static_cast<int>(round(v));
    if (int32 > 127)
        return 127;
    if (int32 < -127)
        return -127;
    return (signed char)int32;
}

void mat_quant_fp32Toint8(ncnn::Mat &i_data, ncnn::Mat &o_data, float scale) {
    for (int ic = 0; ic < i_data.c; ic++) {
        float *ptr_i_data = i_data.channel(ic);
        char *ptr_o_data = o_data.channel(ic);
        for (int i = 0; i < i_data.w * i_data.h; i++) {
            ptr_o_data[i] = float2int8(ptr_i_data[i] * scale);
        }
    }
#ifdef MAT_LOG
    log_mat_file<unsigned char>(o_data, (char *)"./log/log_int8_ifm.txt");
#endif
}

int ifm_layout(ncnn::Mat &i_data, ncnn::Mat &o_data, accel::hw_gemm &inst) {
    // NCWH -> N(WH)C

    char *o_data_ptr = o_data;
	// printf("cstep:%d\n",o_data.cstep);
    memset(o_data_ptr, 0, o_data.cstep * u_align(inst.ifm_c, 32) / 32 * inst.ifm_w * inst.ifm_h);

    if (inst.layout_en == 0) { // NHWC INT8
        // assert(inst.ifm_c % 32 == 0);
        for (int ih = 0; ih < i_data.c; ih++) {
            for (int iw = 0; iw < i_data.h; iw++) {
                char *i_data_ptr = i_data.channel(ih).row<char>(iw);
                memcpy(o_data_ptr, i_data_ptr, i_data.w);
                // o_data_ptr += i_data.w;
                o_data_ptr += u_align(i_data.w,32);
            }
        }
#ifdef MAT_LOG
        log_mat_file<unsigned char>(o_data, (char *)"./log/log_layout_ifm.txt", i_data.c * i_data.h);
#endif
        return i_data.c * i_data.h * i_data.w / 32;
    } else {
        // after quant, fp32 -> int8
        for (int ih = 0; ih < i_data.h; ih++) {
            for (int iw = 0; iw < i_data.w; iw++) {
                for (int ic = 0; ic < u_align(i_data.c, 32); ic++) {
                    if (ic >= inst.ifm_c)
                        *o_data_ptr++ = 0;
                    else
                        *o_data_ptr++ = i_data.channel(ic).row<char>(ih)[iw];
                }
            }
        }
#ifdef MAT_LOG
        log_mat_file<unsigned char>(o_data, (char *)"./log/log_layout_ifm.txt", i_data.w * i_data.h);
#endif
        return i_data.w * i_data.h * u_align(i_data.c, 32) / 32;
    }
    return 0;
}

void fetch_wgt(ncnn::Mat &wgt_buffer, ncnn::Mat &wgt_32x32, int h_index, int v_index) {
    // get 32x32 block from wgt_buffer(oc,1,maxk*ic)
    for (int i = 0; i < 32; i++) {
        char *i_ptr = wgt_buffer.channel(v_index + i);
        i_ptr += h_index;
        char *o_ptr = wgt_32x32.channel(i);
        memcpy(o_ptr, i_ptr, 32);
    }
}

void fetch_wgt_dw(ncnn::Mat &wgt_buffer, ncnn::Mat &wgt_32x32, int block_index) {
    // get 32x32 block from wgt_buffer(oc,1,maxk*ic)
    char *i_ptr = wgt_buffer.channel(0).row<char>(block_index);
    for (int i = 0; i < 32; i++) {
        char *o_ptr = wgt_32x32.channel(i);
        memcpy(o_ptr, i_ptr, 32);
    }
}

void fetch_ifm(ncnn::Mat &ifm_buffer, ncnn::Mat &ifm_32x32, accel::hw_gemm &inst, int ow_cnt, int oh_cnt, int kw_cnt, int kh_cnt, int ic_cnt) {
    // get 32x32 block from ifm_buffer std::vector<ncnn::Mat> (2) (32,1,ifm_buffer_depth)
    //  printf("enter fetch_ifm\n");
    int low_w = inst.padding_left;
    int high_w = inst.padding_left + inst.ifm_w - 1;
    int low_h = inst.padding_top;
    int high_h = inst.padding_top + inst.ifm_h - 1;
    int ifm_pd_w = inst.ifm_w + inst.padding_left + inst.padding_right;
    int ifm_pd_h = inst.ifm_h + inst.padding_top + inst.padding_bottom;

    int ifm_pd_w_base = ow_cnt * inst.stride + kw_cnt;
    int ifm_pd_h_base = oh_cnt * inst.stride + kh_cnt;
    int ifm_buffer_addr_offset = ic_cnt / 32;
    int ic_align_div32 = u_align(inst.ifm_c, 32) / 32;

    char *i_data_ptr = ifm_buffer;
    char *o_data_ptr = ifm_32x32;

    int remain_row = (inst.ofm_h - oh_cnt - 1) * inst.ofm_w + inst.ofm_w - ow_cnt;

    for (int i = 0; i < 32; i++) {
        if (i < remain_row) {
            if (ifm_pd_w_base < low_w || ifm_pd_w_base > high_w || ifm_pd_h_base < low_h || ifm_pd_h_base > high_h) {
                memset(o_data_ptr, 0, 32);
            } else {
                int ifm_index_w = ifm_pd_w_base - low_w;
                int ifm_index_h = ifm_pd_h_base - low_h;
                int ifm_buffer_addr_base = (ifm_index_h * inst.ifm_w + ifm_index_w) * ic_align_div32;
                int ifm_buffer_addr = ifm_buffer_addr_base + ifm_buffer_addr_offset;
                i_data_ptr = ifm_buffer.channel(ifm_buffer_addr);
                memcpy(o_data_ptr, i_data_ptr, 32);
            }
            o_data_ptr += 32;

            if (ifm_pd_w_base + inst.kernel - 1 - kw_cnt >= ifm_pd_w - 1) {
                ifm_pd_w_base = kw_cnt;
                ifm_pd_h_base += inst.stride;
            } else {
                ifm_pd_w_base += inst.stride;
            }
        } else {
            memset(o_data_ptr, 0, 32);
            o_data_ptr += 32;
        }
    }
}

void gemm_32x32(ncnn::Mat &a, ncnn::Mat &b, ncnn::Mat &c) {
    // c = a * b
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            int sum = 0;
            for (int k = 0; k < 32; k++) {
                int8_t *a_ptr = a.channel(i);
                int8_t *b_ptr = b.channel(k);
                sum += a_ptr[k] * b_ptr[j];
            }
            int *c_ptr = c.channel(i);
            c_ptr[j] = sum;
        }
    }
}

void dot_mul_32x32(ncnn::Mat &a, ncnn::Mat &b, ncnn::Mat &c) {
    // c = a * b
    for (int i = 0; i < 32; i++) {
        int8_t *a_ptr = a.channel(i);
        int8_t *b_ptr = b.channel(i);
        int *c_ptr = c.channel(i);
        for (int j = 0; j < 32; j++) {
            c_ptr[j] = a_ptr[j] * b_ptr[j];
        }
    }
}

void acc_mem(ncnn::Mat &ofm_data, ncnn::Mat &psum_data) {
    for (int c = 0; c < 32; c++) {
        int *i_ptr = psum_data.channel(c);
        int *o_ptr = ofm_data.channel(c);
        for (int w = 0; w < 32; w++) {
            *o_ptr++ = *o_ptr + *i_ptr++;
        }
    }
}

template <typename T> void opfusion_32x32(ncnn::Mat &i_data, ncnn::Mat &o_data, accel::hw_gemm &inst, int c_index) {
    // c_index: 0,1,2,3  oc = c_index*32
    float tmp;
    for (int owh = 0; owh < 32; owh++) {
        int *i_ptr = i_data.channel(owh);
        T *o_ptr = o_data.channel(owh);
        for (int oc = 0; oc < 32; oc++) {
            tmp = (float)(i_ptr[oc]);
            tmp *= inst.dequant_scale[c_index * 32 + oc];

            if (inst.bias_en)
                tmp += inst.bias_data[c_index * 32 + oc];
            if (inst.act_op == HW_RELU && tmp < 0)
                tmp = 0.f;
            else if (inst.act_op == HW_LEAKYRELU && tmp < 0)
                tmp *= inst.act_alpha;

            o_ptr[oc] = (T)tmp;
        }
    }
}

void add_block2ofm_nchw(ncnn::Mat &block, ncnn::Mat &ofm, accel::hw_gemm inst, int w_t, int h_t, int c_t) {
    // block (oc,1,wh) oc=32 wh=32
    int w = w_t, h = h_t;
    for (int wh = 0; wh < 32; wh++) {
        float *block_ptr = block.channel(wh);
        if (w < inst.ofm_w && h < inst.ofm_h) {
            for (int oc = 0; oc < 32; oc++) {
                if (c_t + oc < inst.ofm_c) {
                    ofm.channel(c_t + oc).row<float>(h)[w] = block_ptr[oc];
                    // printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n",w,h,oc+c_t);
                }
            }
        }
        w = (w == ofm.w - 1) ? 0 : w + 1;
        h = (w == 0) ? h + 1 : h;
        if (h > ofm.h - 1)
            break;
    }
}

void gemm_forward_sim(ncnn::Mat &ifm_buffer, ncnn::Mat &wgt_buffer, ncnn::Mat &ofm_buffer, accel::hw_gemm &inst) {
    int ifm_c_align_div32 = u_align(inst.ifm_c, 32) / 32;
	int ofm_c_align_div32 = u_align(inst.ofm_c, 32) / 32;
    int ofm_wh_align_div32 = u_align(inst.ofm_w * inst.ofm_h, 32) / 32;

    ncnn::Option opt;
    ncnn::Mat ifm_32x32(32, 1, 32, 1u, opt.blob_allocator);
    ncnn::Mat wgt_32x32(32, 1, 32, 1u, opt.blob_allocator);
    ncnn::Mat psum_32x32(32, 1, 32, 4u, opt.blob_allocator);
    ncnn::Mat ofm_32x32(32, 1, 32, 4u, opt.blob_allocator);
    ncnn::Mat opfusion_fp32_32x32(32, 1, 32, 4u, opt.blob_allocator);

    int *ofm_ptr = ofm_32x32;

    if (inst.op == 0) {

        #ifdef USE_OPENMP
        #pragma omp parallel num_threads(OPENMP_THREADS)
        #endif
        for (int oc = 0; oc < ofm_c_align_div32; oc++) {
            int ow = 0, oh = 0;
            for (int owh = 0; owh < ofm_wh_align_div32; owh++) {
                memset(ofm_ptr, 0, ofm_32x32.total() * 4);
                for (int kernel_h = 0; kernel_h < inst.kernel; kernel_h++) {
                    for (int kernel_w = 0; kernel_w < inst.kernel; kernel_w++) {
                        int kernel_index = kernel_w + kernel_h * inst.kernel;
                        for (int ic = 0; ic < ifm_c_align_div32; ic++) {
                            fetch_ifm(ifm_buffer, ifm_32x32, inst, ow, oh, kernel_w, kernel_h, ic * 32);
                            fetch_wgt(wgt_buffer, wgt_32x32, oc * 32, kernel_index * u_align(inst.ifm_c, 32) + ic * 32);
#ifdef MAT_LOG
                            if (oc == 0)
                                log_block32x32_file<unsigned char>(ifm_32x32, (char *)"./log/log_block_ifm.txt", ow, oh, oc, owh, kernel_h, kernel_w,
                                                                   ic);
                            if (owh == 0)
                                log_block32x32_file<unsigned char>(wgt_32x32, (char *)"./log/log_block_wgt.txt", ow, oh, oc, owh, kernel_h, kernel_w,
                                                                   ic);
#endif
                            gemm_32x32(ifm_32x32, wgt_32x32, psum_32x32);
#ifdef MAT_LOG
                            log_block32x32_file<int>(psum_32x32, (char *)"./log/log_block_psum.txt", ow, oh, oc, owh, kernel_h, kernel_w, ic, 1, 0);
#endif
                            acc_mem(ofm_32x32, psum_32x32);
                        }
                    }
                }
                if (inst.oscale_en) {
#ifdef MAT_LOG
                    log_block32x32_file<int>(ofm_32x32, (char *)"./log/log_block_ofm.txt", ow, oh, oc, owh, 0, 0, 0, 1, 0);
#endif
                    opfusion_32x32<float>(ofm_32x32, opfusion_fp32_32x32, inst, oc);
#ifdef MAT_LOG
                    log_block32x32_file<float>(opfusion_fp32_32x32, (char *)"./log/log_block_opfusion.txt", ow, oh, oc, owh, 0, 0, 0, 0, 1);
#endif
                    add_block2ofm_nchw(opfusion_fp32_32x32, ofm_buffer, inst, ow, oh, oc * 32);
                } else {
                    add_block2ofm_nchw(ofm_32x32, ofm_buffer, inst, ow, oh, oc * 32);
                }
                oh += (ow + 32) / inst.ofm_w;
                ow = (ow + 32) % inst.ofm_w;
            }
        }
    } else {
        #ifdef USE_OPENMP
        #pragma omp parallel num_threads(OPENMP_THREADS)
        #endif
        for (int oc = 0; oc < ofm_c_align_div32; oc++) { // ofm block x
            int ow = 0, oh = 0;
            for (int owh = 0; owh < ofm_wh_align_div32; owh++) { // ofm block y
                memset(ofm_ptr, 0, ofm_32x32.total() * 4);
                for (int kernel_h = 0; kernel_h < inst.kernel; kernel_h++) {
                    for (int kernel_w = 0; kernel_w < inst.kernel; kernel_w++) {
                        int kernel_index = kernel_w + kernel_h * inst.kernel;
                        fetch_ifm(ifm_buffer, ifm_32x32, inst, ow, oh, kernel_w, kernel_h, oc * 32);
#ifdef MAT_LOG
                        std::string ifmblock_filename = ifmbuf_block_path + uchar2string(log_id) + postfix;
                        cstr(ifmblock_filename, log_path);
                        log_mat_file<unsigned char>(ifm_32x32, log_path, -1, 1, 0);
#endif
                        fetch_wgt_dw(wgt_buffer, wgt_32x32, oc + ofm_c_align_div32 * kernel_index);
#ifdef MAT_LOG
                        std::string wgtblock_filename = wgtbuf_block_path + uchar2string(log_id) + postfix;
                        cstr(wgtblock_filename, log_path);
                        log_mat_file<unsigned char>(wgt_32x32, log_path, -1, 1, 0);
#endif
                        dot_mul_32x32(ifm_32x32, wgt_32x32, psum_32x32);
                        acc_mem(ofm_32x32, psum_32x32);
#ifdef MAT_LOG
                        std::string accmemblock_filename = accmem_block_path + uchar2string(log_id) + postfix;
                        log_id++;
                        cstr(accmemblock_filename, log_path);
                        log_mat_file<unsigned char>(ofm_32x32, log_path, -1, 1, 0);
#endif
                    }
                }

                if (inst.oscale_en) {
#ifdef MAT_LOG
                    log_block32x32_file<int>(ofm_32x32, (char *)"./log/log_block_ofm.txt", ow, oh, oc, owh, 0, 0, 0, 1, 0);
#endif
                    opfusion_32x32<float>(ofm_32x32, opfusion_fp32_32x32, inst, oc);
#ifdef MAT_LOG
                    log_block32x32_file<float>(opfusion_fp32_32x32, (char *)"./log/log_block_opfusion.txt", ow, oh, oc, owh, 0, 0, 0, 0, 1);
#endif
                    add_block2ofm_nchw(opfusion_fp32_32x32, ofm_buffer, inst, ow, oh, oc * 32);
                } else {
                    add_block2ofm_nchw(ofm_32x32, ofm_buffer, inst, ow, oh, oc * 32);
                }

                oh += (ow + 32) / inst.ofm_w;
                ow = (ow + 32) % inst.ofm_w;
            }
        }
    }
}

void forward_sim_conv(accel::hw_gemm inst, std::vector<ncnn::Mat> &weights, ncnn::Mat &ifm_data, ncnn::Mat &ofm_data) {
#ifdef PRINT_SHAPE
    printf("\n");
    printf("[ log ]: start forward sim conv ...\n");
    printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d\n", inst.ifm_w, inst.ifm_h, inst.ifm_c);
    printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n", inst.ofm_w, inst.ofm_h, inst.ofm_c);
    printf("[ log ]: kernel = %d, stride = %d\n", inst.kernel, inst.stride);
    printf("[ log ]: padding_left = %d, padding_right = %d, padding_top = %d, padding_bottom = %d\n", inst.padding_left, inst.padding_right, inst.padding_top, inst.padding_bottom);
    printf("[ log ]: bias_en = %d, requant_en = %d, act_op = %d, act_param = %f\n", inst.bias_en, inst.requant_en, inst.act_op, inst.act_alpha);
    printf("[ log ]: quant_scale = %.5f, dequant_ptr = %p, bias_ptr = %p \n", inst.quant_scale, inst.dequant_scale, inst.bias_data);

    int calculation_quantity_M = (long long)(inst.ofm_w * inst.ofm_h) * (long long)(inst.ifm_c * inst.ofm_c) * inst.kernel * inst.kernel / 1000000;
    printf("[ log ]: sim: calculation_quantity = " ANSI_FMT("%dM", ANSI_FG_YELLOW) " muls\n", calculation_quantity_M);
    clock_t start_time = clock();
#endif
    #ifdef CONV_TIME
        u64 tEnd_conv_sw, tCur_conv_sw;
        u32 tUsed_conv_sw;
        XTime_GetTime(&tCur_conv_sw);
    #endif
    ncnn::Option opt;
    
	int oc_align = (inst.op == 0) ? u_align(inst.ofm_c, 32) : 1;
    ncnn::Mat gemm_wgt_buffer(oc_align, 1, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel, 1u, opt.blob_allocator);
    wgt_reorder(weights[0], gemm_wgt_buffer, inst.ifm_c, inst.ofm_c, inst.kernel, 0, inst.op);
    ncnn::Mat hw_wgt_buffer_res(oc_align, 1, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel, 1u, opt.blob_allocator);
    ncnn::Mat hw_quant_wgt(32, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel * oc_align / 32, 1u, opt.blob_allocator);
    wgt_reorder(weights[0], hw_wgt_buffer_res, inst.ifm_c, inst.ofm_c, inst.kernel, !inst.op, inst.op);
    wgt_gen(hw_wgt_buffer_res, hw_quant_wgt, inst.ifm_c,inst.ofm_c,inst.kernel, inst.op);
    // log_mat_file<int8_t>(hw_quant_wgt, (char *)"./log/hw_quant_wgt.txt", -1, 0, 0);


	ncnn::Mat gemm_ifm_buffer(32, 1, u_align(inst.ifm_c, 32) / 32 * inst.ifm_w * inst.ifm_h, 1u, opt.blob_allocator);
    if(inst.layout_en == 1){
        ncnn::Mat quant_ifm(ifm_data.w, ifm_data.h, ifm_data.c, 1u, opt.blob_allocator);
        mat_quant_fp32Toint8(ifm_data, quant_ifm, inst.quant_scale);
        ifm_layout(quant_ifm, gemm_ifm_buffer, inst);
    }
    else{
        ifm_layout(ifm_data, gemm_ifm_buffer, inst);
    }

    if(inst.requant_en == 1){
        ncnn::Mat ofm_nchw(inst.ofm_w, inst.ofm_h, inst.ofm_c, 4u, opt.blob_allocator);
        gemm_forward_sim(gemm_ifm_buffer, (inst.op?hw_quant_wgt:gemm_wgt_buffer), ofm_nchw, inst);
        ncnn::Mat ofm_nhwc(u_align(inst.ofm_c,32), 1, inst.ofm_w*inst.ofm_h, 4u, opt.blob_allocator);
        mat_nchw2nhwc(ofm_nchw, ofm_nhwc);
        mat_quant_fp32Toint8(ofm_nhwc, ofm_data, inst.requant_scale);
    }
    else{
        gemm_forward_sim(gemm_ifm_buffer, (inst.op?hw_quant_wgt:gemm_wgt_buffer), ofm_data, inst);
    }
    #ifdef CONV_TIME
		XTime_GetTime(&tEnd_conv_sw);
		tUsed_conv_sw = ((tEnd_conv_sw-tCur_conv_sw)*1000000)/(COUNTS_PER_SECOND);
		printf("time_conv_sw elapsed is %d us\n",tUsed_conv_sw);
    #endif
//    clock_t  end_time = clock();
//	double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
//    printf("[ log ]: test_layer_conv consume time = %f s\n", total_time);
//    double M_per_second = calculation_quantity_M / total_time ;
//    printf("[ log ]: M_per_second = %f M/s\n\n", M_per_second);

}

//----------------------------------------- forward_hw_conv ---------------------------------------------//
void forward_hw_conv(accel::hw_gemm inst, std::vector<ncnn::Mat> &weights, ncnn::Mat &ifm_data, ncnn::Mat &ofm_data, ncnn::Mat &wgt_buffer) {
    ncnn::Option opt;
    int oc_align = (inst.op == 0) ? u_align(inst.ofm_c, 32) : 1;
    ncnn::Mat wgt_buffer_res(oc_align, 1, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel, 1u, opt.blob_allocator);
    wgt_buffer.create(32, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel * oc_align / 32, 1u, opt.blob_allocator);
    // ncnn::Mat wgt_buffer(32,u_align(inst.ifm_c*inst.kernel*inst.kernel,32)*u_align(inst.ofm_c,32)/32, 1u, opt.blob_allocator);
    wgt_reorder(weights[0], wgt_buffer_res, inst.ifm_c, inst.ofm_c, inst.kernel, !inst.op, inst.op);
    wgt_gen(wgt_buffer_res, wgt_buffer, inst.ifm_c, inst.ofm_c, inst.kernel, inst.op);
    log_mat_file<int8_t>(wgt_buffer, (char *)"./log/hw_quant_wgt.txt", -1, 0, 0);
    log_mat_file<int8_t>(wgt_buffer, (char *)"./log/hw_quant_wgt_hex.txt", -1, 1, 0);

    inst.gemm_forward_s(ifm_data, wgt_buffer, ofm_data);
}

//----------------------------------------- test_ifmbufctl_addr_ctrl ---------------------------------------------//
template <typename T> void block32x32_gather(ncnn::Mat &block, ncnn::Mat &map, int x_index, int y_index, int x_max, int place, accel::hw_gemm &inst) {
    // x_index block 闂傚倷绶氬濠氭嚈瑜版帒绾х憸鐗堝笚閻撴盯鏌涚仦鍓х煁闁搞倧鎷烽梻浣筋嚃閸犳鍒掗幘璇茬畾闁告劦鍠栫粈瀣亜閹捐泛浠滃ù婊冪埣閹鈻撻崹顔界彲闂佺懓鍤栭幏锟�
    // y_index block 闂傚倷绶氬濠氭嚈瑜版帒绾ф繛宸簼閻撴盯鏌涚仦鍓х煁闁搞倧鎷烽梻浣筋嚃閸犳鍒掗幘璇茬畾闁告劦鍠栫粈瀣亜閹捐泛浠滃ù婊冪埣閹鈻撻崹顔界彲闂佺懓鍤栭幏锟�
    // place = 1 -> 婵犲痉鏉匡拷鏇㈠磹閸︻厽绠掔紓鍌氬�哥粙鍕箯閿燂拷; place = 0  -> 婵犵數鍋犻幓顏嗗緤閻ｅ矁濮抽柤娴嬫櫇缁�濠囨煥閻曞倹瀚�
    int ofm_wh = inst.ofm_w * inst.ofm_h;
    uint32_t *map_ptr = map.channel(y_index * x_max + x_index);
    if (!place)
        memset(map_ptr, 0, 1024 * 4);
    for (int c = 0; c < 32; c++) {
        if ((y_index * 2 + place) * 32 + c < ofm_wh) {
            T *block_ptr = block.channel(c);
            for (int w = 0; w < 32; w++) {
                if (place)
                    map_ptr[c * 32 + w] |= block_ptr[w] << 16;
                else
                    map_ptr[c * 32 + w] |= block_ptr[w];
            }
        }
    }
}

#ifdef TEST_IFMBUFCTL
void test_ifmbufctl_addr_ctrl(accel::hw_gemm inst, std::vector<ncnn::Mat> &weights, ncnn::Mat &ifm_data, ncnn::Mat &ofm_data) {
    ncnn::Option opt;
    ncnn::Mat quant_ifm(ifm_data.w, ifm_data.h, ifm_data.c, 1u, opt.blob_allocator);
    ncnn::Mat gemm_ifm_buffer(32, 1, u_align(inst.ifm_c, 32) / 32 * inst.ifm_w * inst.ifm_h, 1u, opt.blob_allocator);
    mat_quant_fp32Toint8(ifm_data, quant_ifm, inst.quant_scale);
    if (!inst.layout_en) {
        inst.layout_en = 1;
        ifm_layout(quant_ifm, gemm_ifm_buffer, inst);
        inst.layout_en = 0;
    } else {
        ifm_layout(quant_ifm, gemm_ifm_buffer, inst);
    }

    int oc_align = (inst.op == 0) ? u_align(inst.ofm_c, 32) : 1;
    ncnn::Mat wgt_buffer_res(oc_align, 1, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel, 1u, opt.blob_allocator);
    ncnn::Mat wgt_buffer(32, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel * oc_align / 32, 1u, opt.blob_allocator);

    if (inst.op == 1) {
        wgt_reorder(weights[0], wgt_buffer_res, inst.ifm_c, inst.ofm_c, inst.kernel, 0, 1);
    } else {
        wgt_reorder(weights[0], wgt_buffer_res, inst.ifm_c, inst.ofm_c, inst.kernel, 1);
    }
    wgt_gen(wgt_buffer_res, wgt_buffer, inst.ifm_c, inst.ofm_c, inst.kernel, inst.op);

#ifdef MAT_LOG
    log_mat_file<uint8_t>(wgt_buffer_res, (char *)"./log/ifmbufctl/wgt_buffer_res.txt", -1, 1, 0);
    log_mat_file<uint8_t>(wgt_buffer, (char *)"./log/ifmbufctl/wgt_buffer.txt", -1, 1, 0);
#endif

    printf("[ log ]: start sim ifmbufctl on npu ...\n");
    printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d\n", inst.ifm_w, inst.ifm_h, inst.ifm_c);
    printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n", inst.ofm_w, inst.ofm_h, inst.ofm_c);
    printf("[ log ]: kernel = %d, stride = %d\n", inst.kernel, inst.stride);
    printf("[ log ]: padding_left = %d, padding_right = %d, padding_top = %d, padding_bottom = %d\n", inst.padding_left, inst.padding_right, inst.padding_top,
           inst.padding_bottom);
    printf("-----------------------------------------------------------\n");

    int ifm_c_align_div32 = u_align(inst.ifm_c, 32) / 32;
    int ofm_wh_align_div32 = u_align(inst.ofm_w * inst.ofm_h, 32) / 32;
    int ofm_wh_align_div64 = u_align(inst.ofm_w * inst.ofm_h, 64) / 64;

    ncnn::Mat ifm_32x32(32, 1, 32, 1u, opt.blob_allocator);
    ncnn::Mat gemm_input_ifm(32 * 32, 1, inst.kernel * inst.kernel * ifm_c_align_div32 * ofm_wh_align_div64, 4u, opt.blob_allocator);
    
    bool flip = true;
    int ow = 0;
    int oh = 0;
    int x_index_max = inst.kernel * inst.kernel * ifm_c_align_div32;
    for (int owh = 0; owh < ofm_wh_align_div32; owh++) {
        flip = !flip;
        for (int kernel_h = 0; kernel_h < inst.kernel; kernel_h++) {
            for (int kernel_w = 0; kernel_w < inst.kernel; kernel_w++) {
                int kernel_index = kernel_w + kernel_h * inst.kernel;
                for (int ic = 0; ic < ifm_c_align_div32; ic++) {
                    fetch_ifm(gemm_ifm_buffer, ifm_32x32, inst, ow, oh, kernel_w, kernel_h, ic * 32);
                    // printf("x_index = %d, y_index = %d, x_index_max = %d, flip = %d\n",kernel_index*ifm_c_align_div32+ic,owh/2,x_index_max,flip);
                    block32x32_gather<unsigned char>(ifm_32x32, gemm_input_ifm, kernel_index * ifm_c_align_div32 + ic, owh / 2, x_index_max, flip, inst);
#ifdef MAT_LOG
                    std::string ifmblock_filename = ifmbuf_block_path + uchar2string(log_id) + postfix;
                    log_id++;
                    cstr(ifmblock_filename, log_path);
                    log_mat_file<unsigned char>(ifm_32x32, log_path, -1, 1, 0);
// log_block32x32_file<unsigned char>(ifm_32x32,(char*)"./log/log_block_ifm.txt",ow,oh,oc,owh,kernel_h,kernel_w,ic);
#endif
                }
            }
        }
        oh += (ow + 32) / inst.ofm_w;
        ow = (ow + 32) % inst.ofm_w;
    }
#ifdef MAT_LOG
    log_mat_file<uint32_t>(gemm_input_ifm, (char *)"./log/ifmbufctl/gemm_input_ifm.txt", -1, 1, 0);
#endif

    if (inst.layout_en == 0) { // sometime bug
        printf("[ log ]: no use layout, ifm format is NHWC\n");
        ncnn::Mat ifm_nhwc(u_align(inst.ifm_c, 32), 1, inst.ifm_w * inst.ifm_h, 4u, opt.blob_allocator);
        mat_nchw2nhwc(ifm_data, ifm_nhwc);
        ncnn::Mat ifm_nhwc_quant(u_align(inst.ifm_c, 32), 1, inst.ifm_w * inst.ifm_h, 1u, opt.blob_allocator);
        mat_quant_fp32Toint8(ifm_nhwc, ifm_nhwc_quant, inst.quant_scale);
        inst.gemm_forward_s(ifm_nhwc_quant, wgt_buffer, ofm_data);
    } else {
        inst.gemm_forward_s(ifm_data, wgt_buffer, ofm_data);
    }

    top->io_ifmctl_odata_ready = 0;
    printf("------\nhw sim begin\n");
    dma_wait(gemm_input_ifm, inst);
}
#endif

#ifdef TEST_ACCMEM
template <typename T> void accblock32x32_gather(ncnn::Mat &block, ncnn::Mat &map, int x_index, int y_index, int y_max, int place, accel::hw_gemm &inst) {
    uint32_t *map_ptr = map.channel(x_index * y_max + y_index);
    uint32_t *block_ptr = block;
    memcpy(map_ptr, block_ptr, 1024 * 4);
}
//----------------------------------------- test_acc_mem ---------------------------------------------//
void test_acc_mem(accel::hw_gemm inst, std::vector<ncnn::Mat> &weights, ncnn::Mat &ifm_data, ncnn::Mat &ofm_data) {
    printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d\n", inst.ifm_w, inst.ifm_h, inst.ifm_c);
    printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n", inst.ofm_w, inst.ofm_h, inst.ofm_c);
    printf("[ log ]: kernel = %d, stride = %d\n", inst.kernel, inst.stride);
    printf("[ log ]: padding_left = %d, padding_right = %d, padding_top = %d, padding_bottom = %d\n", inst.padding_left, inst.padding_right, inst.padding_top, inst.padding_bottom);
    printf("[ log ]: bias_en = %d, requant_en = %d\n", inst.bias_en, inst.requant_en);
    printf("[ log ]: quant_scale = %.5f, dequant_ptr = %p, bias_ptr = %p \n", inst.quant_scale, inst.dequant_scale, inst.bias_data);

	if(inst.op == 0){
		assert(u_align(inst.ifm_c, 32)/32 * inst.ifm_w * inst.ifm_h <= IFM_BUFFER_DEPTH);
		assert(u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel <= WGT_BUFFER_DEPTH);
	}else{
		assert(u_align(inst.ifm_c, 32)/32 * inst.ifm_w * inst.ifm_h <= IFM_BUFFER_DEPTH);
		assert(u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel / 32 <= WGT_BUFFER_DEPTH);
	}
	
    ncnn::Option opt;
    // IFM quant
    ncnn::Mat quant_ifm(ifm_data.w, ifm_data.h, ifm_data.c, 1u, opt.blob_allocator);
    mat_quant_fp32Toint8(ifm_data, quant_ifm, inst.quant_scale);
    // C*H*W -> HWIC/32 * 32
    ncnn::Mat gemm_ifm_buffer(IFM_BUFFER_WIDTH / 8, 1, IFM_BUFFER_DEPTH, 1u, opt.blob_allocator);
    ifm_layout(quant_ifm, gemm_ifm_buffer, inst);

    // W 1*NCKK -> KKIC*OC ; gold use
    int oc_align = (inst.op == 0) ? u_align(inst.ofm_c, 32) : 1;
    ncnn::Mat wgt_buffer_res(oc_align, 1, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel, 1u, opt.blob_allocator);
    wgt_reorder(weights[0], wgt_buffer_res, inst.ifm_c, inst.ofm_c, inst.kernel, 0, inst.op);

    // W 1*NCKK -> KKIC*OC and flip each 32line ; hardware use
    ncnn::Mat hw_wgt_buffer_res(oc_align, 1, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel, 1u, opt.blob_allocator);
    ncnn::Mat hw_quant_wgt(32, u_align(inst.ifm_c, 32) * inst.kernel * inst.kernel * oc_align / 32, 1u, opt.blob_allocator);
    // ncnn::Mat quant_wgt(32,u_align(inst.ifm_c*inst.kernel*inst.kernel,32)*u_align(inst.ofm_c,32)/32, 1u, opt.blob_allocator);
    wgt_reorder(weights[0], hw_wgt_buffer_res, inst.ifm_c, inst.ofm_c, inst.kernel, !inst.op, inst.op);
    wgt_gen(hw_wgt_buffer_res, hw_quant_wgt, inst.ifm_c, inst.ofm_c, inst.kernel, inst.op);

    // gen gold result
    int ifm_c_align_div32 = u_align(inst.ifm_c, 32) / 32;
    int ofm_c_align_div32 = u_align(inst.ofm_c, 32) / 32;
    int ofm_wh_align_div32 = u_align(inst.ofm_w * inst.ofm_h, 32) / 32;

    ncnn::Mat ifm_32x32(32, 1, 32, 1u, opt.blob_allocator);
    ncnn::Mat wgt_32x32(32, 1, 32, 1u, opt.blob_allocator);
    ncnn::Mat psum_32x32(32, 1, 32, 4u, opt.blob_allocator);
    ncnn::Mat ofm_32x32(32, 1, 32, 4u, opt.blob_allocator);
    ncnn::Mat gemm_output_accmem(32 * 32, 1, u_align(inst.ofm_w * inst.ofm_h, 64) / 32 * ofm_c_align_div32, 4u, opt.blob_allocator);

    ncnn::Mat zero_32x32(32, 1, 32, 4u, opt.blob_allocator);
    zero_32x32.fill(0);

    int *ofm_ptr = ofm_32x32;
    if (inst.op == 0) {
        for (int oc = 0; oc < ofm_c_align_div32; oc++) { // ofm block x
            int ow = 0, oh = 0;
            for (int owh = 0; owh < ofm_wh_align_div32; owh++) { // ofm block y
                memset(ofm_ptr, 0, ofm_32x32.total() * 4);

                for (int kernel_h = 0; kernel_h < inst.kernel; kernel_h++) {
                    for (int kernel_w = 0; kernel_w < inst.kernel; kernel_w++) {
                        int kernel_index = kernel_w + kernel_h * inst.kernel;

                        for (int ic = 0; ic < ifm_c_align_div32; ic++) { // ifm one block
                            fetch_ifm(gemm_ifm_buffer, ifm_32x32, inst, ow, oh, kernel_w, kernel_h, ic * 32);
                            fetch_wgt(wgt_buffer_res, wgt_32x32, oc * 32, kernel_index * u_align(inst.ifm_c, 32) + ic * 32);
                            if (oc == 0 && owh == 5 && kernel_index == 0 && ic == 0) {
                                std::string ifm_32x32_filename = accmem_block_path + "ifm_32x32";
                                std::string wgt_32x32_filename = accmem_block_path + "wgt_32x32";
                                cstr(ifm_32x32_filename, log_path);
                                log_mat_file<unsigned char>(ifm_32x32, log_path, -1, 0, 0);
                                cstr(wgt_32x32_filename, log_path);
                                log_mat_file<unsigned char>(wgt_32x32, log_path, -1, 0, 0);
                            }
                            gemm_32x32(ifm_32x32, wgt_32x32, psum_32x32);
                            log_mat_file<uint32_t>(psum_32x32, (char *)"./log/psum_32x32.txt", -1, 1, 0);

                            acc_mem(ofm_32x32, psum_32x32);
                            log_mat_file<uint32_t>(ofm_32x32, (char *)"./log/ofm_32x32.txt", -1, 1, 0);
                        }
                    }
                }

#ifdef MAT_LOG
                std::string accmemblock_filename = accmem_block_path + uchar2string(log_id) + postfix;
                log_id++;
                cstr(accmemblock_filename, log_path);
                log_mat_file<uint32_t>(ofm_32x32, log_path, -1, 1, 0);
#endif
                // part 32x32 acc result gather
                accblock32x32_gather<unsigned int>(ofm_32x32, gemm_output_accmem, oc, owh, u_align(inst.ofm_w * inst.ofm_h, 64) / 32, 0, inst);
                if (owh == ofm_wh_align_div32 - 1 && owh % 2 == 0) {
                    accblock32x32_gather<unsigned int>(zero_32x32, gemm_output_accmem, oc, owh + 1, u_align(inst.ofm_w * inst.ofm_h, 64) / 32, 0, inst);
                }

                oh += (ow + 32) / inst.ofm_w;
                ow = (ow + 32) % inst.ofm_w;
            }
        }
    } else {
        for (int oc = 0; oc < ofm_c_align_div32; oc++) { // ofm block x
            int ow = 0, oh = 0;
            for (int owh = 0; owh < ofm_wh_align_div32; owh++) { // ofm block y
                memset(ofm_ptr, 0, ofm_32x32.total() * 4);

                for (int kernel_h = 0; kernel_h < inst.kernel; kernel_h++) {
                    for (int kernel_w = 0; kernel_w < inst.kernel; kernel_w++) {
                        int kernel_index = kernel_w + kernel_h * inst.kernel;

                        fetch_ifm(gemm_ifm_buffer, ifm_32x32, inst, ow, oh, kernel_w, kernel_h, oc * 32);
                        fetch_wgt_dw(hw_quant_wgt, wgt_32x32, oc + ofm_c_align_div32 * kernel_index);

                        dot_mul_32x32(ifm_32x32, wgt_32x32, psum_32x32);
                        log_mat_file<uint32_t>(psum_32x32, (char *)"./log/psum_32x32.txt", -1, 1, 0);

                        acc_mem(ofm_32x32, psum_32x32);
                        log_mat_file<uint32_t>(ofm_32x32, (char *)"./log/ofm_32x32.txt", -1, 1, 0);
						
                    }
                }

#ifdef MAT_LOG
                std::string accmemblock_filename = accmem_block_path + uchar2string(log_id) + postfix;
                log_id++;
                cstr(accmemblock_filename, log_path);
                log_mat_file<uint32_t>(ofm_32x32, log_path, -1, 1, 0);
#endif
                // part 32x32 acc result gather
                accblock32x32_gather<unsigned int>(ofm_32x32, gemm_output_accmem, oc, owh, u_align(inst.ofm_w * inst.ofm_h, 64) / 32, 0, inst);
                if (owh == ofm_wh_align_div32 - 1 && owh % 2 == 0) {
                    accblock32x32_gather<unsigned int>(zero_32x32, gemm_output_accmem, oc, owh + 1, u_align(inst.ofm_w * inst.ofm_h, 64) / 32, 0,
                                                       inst);
                }

                oh += (ow + 32) / inst.ofm_w;
                ow = (ow + 32) % inst.ofm_w;
            }
        }
    }

#ifdef MAT_LOG
    log_mat_file<uint32_t>(gemm_output_accmem, (char *)"./log/accmem/gemm_output_accmem.txt", -1, 1, 0);
#endif
    inst.gemm_forward_s(ifm_data, hw_quant_wgt, ofm_data);
    dma_wait(gemm_output_accmem, inst);
}
#endif

#ifdef TEST_OFM
void test_ofm_flatten(accel::hw_gemm inst, ncnn::Mat &ofm, ncnn::Mat &flatten_data) {
    // generate align ofm
    ncnn::Option opt;
    ncnn::Mat ofm_align_data(u_align(ofm.w * ofm.h, 64), 1, u_align(ofm.c, 32), 4u, opt.blob_allocator);
    memset(ofm_align_data.data, 0, ofm_align_data.total() * 4);
    for (int oc = 0; oc < ofm.c; oc++) {
        int *oc_ofm_ptr = ofm.channel(oc);
        int *oc_ofm_align_ptr = ofm_align_data.channel(oc);
        memcpy(oc_ofm_align_ptr, oc_ofm_ptr, ofm.w * ofm.h * 4);
    }
    log_mat_file<float>(ofm_align_data, (char *)"./log/log_ofm_align_data.txt", -1, 1, 0);

    // flatten ofm
    memset(flatten_data.data, 0, flatten_data.total() * 4);
    int *flatten_ptr0 = flatten_data.channel(0);
    int *flatten_ptr1 = flatten_data.channel(1);

    for (int c_block = 0; c_block < ofm_align_data.c / 32; c_block++) {
        for (int wh_block = 0; wh_block < u_align(ofm.w * ofm.h, 64) / 64; wh_block++) {
            for (int c_index = 0; c_index < 32; c_index++) {
                int *c_ptr = ofm_align_data.channel(c_block * 32 + c_index);
                c_ptr += wh_block * 64;
                memcpy(flatten_ptr0, c_ptr, 32 * 4);
                memcpy(flatten_ptr1, c_ptr + 32, 32 * 4);
                flatten_ptr0 += 32;
                flatten_ptr1 += 32;
            }
        }
    }
}

#endif

void test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_bottom_t, int op = 0, int layout_en = 1, int requant_en = 0) {
    
    #ifdef COMPARE_WITH_NCNN
        assert(requant_en == 0);
        assert(layout_en == 1);
    #endif

    #ifndef COMPARE_WITH_NCNN
        if(k_t < s_t){
            printf("\033[;31m[error]: kernel is smaller than stride, please open COMPARE_WITH_NCNN\n\033[0m");
            assert(0);
        }
    #endif

    count_total ++ ;

    ncnn::Option opt;

    // hw_gemm parameters inst
    accel::hw_gemm inst;
    inst.padding_mode = PD_MODE_ZERO;
    inst.padding_left = pd_left_t;
    inst.padding_right = pd_right_t;
    inst.padding_top = pd_top_t;
    inst.padding_bottom = pd_bottom_t;

    inst.kernel = k_t;
    inst.stride = s_t;
    inst.ifm_w = iw_t;
    inst.ifm_h = ih_t;
    inst.ifm_c = ic_t;
    inst.ofm_c = oc_t;
    inst.op = op;
    if (op == 1) {
        assert(oc_t == ic_t);
        printf("[ log ]: this conv is a depthwise conv!\n");
    }
	conv_get_ofm_pad(inst);

    inst.oscale_en = 1;  //when sim SIM_OCALE_BIAS, oscale_en must be set 1
    inst.bias_en = 1;
    //requant_en = 1 -> output nhwc Mat; requant_en = 0 -> output nchw Mat
    inst.requant_en = requant_en;
    //layout_en = 1 -> input nchw Mat; layout_en = 0 -> input nhwc Mat
    inst.layout_en = layout_en; 
    // inst.act_op = HW_NONE;
    inst.act_op = HW_RELU;
    // inst.act_op = HW_LEAKYRELU;

#if defined(TEST_LAYER_CONV) && !defined(TEST_IFMBUFCTL) && !defined(TEST_ACCMEM) && !defined(TEST_OFM) 
	assert(inst.oscale_en == 1);
#endif

    // generate ifm
    ncnn::Mat ifm_data = RandomMat(inst.ifm_w, inst.ifm_h, inst.ifm_c);
#ifdef MAT_LOG
    log_mat_file<float>(ifm_data, (char *)"./log/log_ifm_data.txt",-1, 1, 0);
#endif
    float ifm_mem_size = ((float)(u_align(inst.ifm_c, 32) * inst.ifm_w * inst.ifm_h)) / 1024.f / 1024.f;

    // generate activation params
    ncnn::Mat activation_params(2);
    activation_params[0] = (inst.act_op == 6) ? RandomFloat(0, 1) : RandomFloat(-1, 0); // alpha
    activation_params[1] = RandomFloat(0, 1);                                           // beta
    inst.act_alpha = activation_params[0];
    inst.act_beta = activation_params[1];
    #ifdef PRINT_SHAPE
    printf("[ log ]: the ifm size is %.7f MB\n", ifm_mem_size);
    printf("[ log ]: inst.act_alpha = %f => %x\n", inst.act_alpha, *((uint32_t *)&inst.act_alpha));
    printf("[ log ]: inst.act_beta = %f => %x\n", inst.act_beta, *((uint32_t *)&inst.act_beta));
    printf("[ log ]: inst.ofm_w=%d, inst.ofm_h=%d, inst.ofm_c=%d\n", inst.ofm_w, inst.ofm_h, inst.ofm_c);
    printf("[ log ]: inst.ifm_w=%d, inst.ifm_h=%d, inst.ifm_c=%d\n", inst.ifm_w, inst.ifm_h, inst.ifm_c);
    printf("[ log ]: inst.kernel=%d, inst.stride=%d\n", inst.kernel, inst.stride);
    #endif

    // generate wgt scale bias
    std::vector<ncnn::Mat> weights(inst.bias_en ? 5 : 4);
    weights[0] = RandomS8Mat(inst.ofm_c * inst.ifm_c * inst.kernel * inst.kernel);
    ncnn::Mat weight_scales = RandomMat(inst.ofm_c, 90.f, 100.f);
    ncnn::Mat input_scales = scales_mat(ifm_data, 1, inst.ifm_w * inst.ifm_h * inst.ifm_c, ifm_data.cstep);
    ncnn::Mat top_scales = inst.requant_en ? RandomMat(1, 20.f, 25.f) : ncnn::Mat();
    ncnn::Mat bias_data = RandomMat(inst.ofm_c);

    if (inst.bias_en) {
        weights[1] = bias_data;
        weights[2] = weight_scales;
        weights[3] = input_scales;
        weights[4] = top_scales;
    } else {
        weights[1] = weight_scales;
        weights[2] = input_scales;
        weights[3] = top_scales;
    }

    inst.quant_scale = input_scales[0];
    inst.requant_scale = inst.requant_en ? top_scales[0] : 0.f;
    ncnn::Mat dequant_scale(inst.ofm_c, 4u, opt.blob_allocator);
    for (int i = 0; i < inst.ofm_c; i++)
        dequant_scale[i] = 1.f / (input_scales[0] * weight_scales[i]);
    inst.dequant_scale = (float*)dequant_scale.data;
    inst.bias_data = (float*)bias_data.data;

// simulation
#if defined(TEST_IFMBUFCTL)
    ncnn::Mat ofm_data_sim(inst.ofm_w, inst.ofm_h, inst.ofm_c, 4u, opt.blob_allocator);
    test_ifmbufctl_addr_ctrl(inst, weights, ifm_data, ofm_data_sim);
#elif defined(TEST_ACCMEM)
    ncnn::Mat ofm_data_sim(inst.ofm_w, inst.ofm_h, inst.ofm_c, 4u, opt.blob_allocator);
    test_acc_mem(inst, weights, ifm_data, ofm_data_sim);
#elif defined(TEST_OFM)
    ncnn::Mat ofm_data_sim(inst.ofm_w, inst.ofm_h, inst.ofm_c, 4u, opt.blob_allocator);
    ncnn::Mat ofm_data_hw(inst.ofm_w, inst.ofm_h, inst.ofm_c, 4u, opt.blob_allocator);
	ncnn::Mat wgt_buffer;

    if (inst.layout_en == 0) {// error
        printf("[ log ]: no use layout, ifm format is NHWC\n");
        ncnn::Mat ifm_nhwc(u_align(inst.ifm_c, 32), 1, inst.ifm_w * inst.ifm_h, 4u, opt.blob_allocator);
        mat_nchw2nhwc(ifm_data, ifm_nhwc);
        ncnn::Mat ifm_nhwc_quant(u_align(inst.ifm_c, 32), 1, inst.ifm_w * inst.ifm_h, 1u, opt.blob_allocator);
        mat_quant_fp32Toint8(ifm_nhwc, ifm_nhwc_quant, inst.quant_scale);
        forward_sim_conv(inst, weights, ifm_nhwc_quant, ofm_data_sim);
        forward_hw_conv(inst, weights, ifm_nhwc_quant, ofm_data_hw, wgt_buffer);
    } else {
        forward_sim_conv(inst, weights, ifm_data, ofm_data_sim);
        forward_hw_conv(inst, weights, ifm_data, ofm_data_hw, wgt_buffer);
    }

    int flatten_len = u_align(ofm_data_sim.w * ofm_data_sim.h, 64) * u_align(ofm_data_sim.c, 32) / 8;
    ncnn::Mat ofm_flatten_data(4, flatten_len, 2, 4u, opt.blob_allocator);
    test_ofm_flatten(inst, ofm_data_sim, ofm_flatten_data);

    log_mat_file<float>(ofm_data_sim, (char *)"./log/log_ofm_data_sim.txt", -1, 1, 0);
    log_mat_file<float>(ofm_flatten_data, (char *)"./log/log_ofm_flatten_data.txt", -1, 1, 0);

    dma_wait(ofm_flatten_data, inst);
#else

#ifdef FORWARD_ON_CPU_CONV
    assert(inst.layout_en == 1);
#endif
    
    ncnn::Mat ofm_data_sim;
    ncnn::Mat ofm_data_ncnn;
    if(inst.requant_en){
        printf("[ log ]: use requant, ofm format is NHWC\n");
        // ofm_data_sim.create(inst.ofm_c, 1, inst.ofm_w*inst.ofm_h, 1u, opt.blob_allocator);
        // ofm_data_ncnn.create(inst.ofm_c, 1, inst.ofm_w*inst.ofm_h, 1u, opt.blob_allocator);
        ofm_data_sim.create(u_align(inst.ofm_c,32), 1, inst.ofm_w*inst.ofm_h, 1u, opt.blob_allocator);
        ofm_data_ncnn.create(u_align(inst.ofm_c,32), 1, inst.ofm_w*inst.ofm_h, 1u, opt.blob_allocator);
        memset(ofm_data_ncnn.data, 0xff, ofm_data_ncnn.total());
    }
    else{
        ofm_data_sim.create(inst.ofm_w, inst.ofm_h, inst.ofm_c, 4u, opt.blob_allocator);
        ofm_data_ncnn.create(inst.ofm_w, inst.ofm_h, inst.ofm_c, 4u, opt.blob_allocator);
    }

    if(inst.layout_en == 0){
        printf("[ log ]: no use layout, ifm format is NHWC\n");
        ncnn::Mat ifm_nhwc(u_align(inst.ifm_c,32), 1, inst.ifm_w*inst.ifm_h, 4u, opt.blob_allocator);
        mat_nchw2nhwc(ifm_data, ifm_nhwc);
        ncnn::Mat ifm_nhwc_quant(u_align(inst.ifm_c,32), 1, inst.ifm_w*inst.ifm_h, 1u, opt.blob_allocator);
        mat_quant_fp32Toint8(ifm_nhwc, ifm_nhwc_quant, inst.quant_scale);

//        log_mat_file<unsigned char>(ifm_nhwc_quant, (char *)"./log/log_ifm_quant_data.txt", -1, 1, 0);
//        log_mat_file<float>(ifm_nhwc, (char *)"./log/log_ifm_data.txt", -1, 1, 0);


        #ifndef COMPARE_WITH_NCNN
        forward_sim_conv(inst, weights, ifm_nhwc_quant, ofm_data_sim);
        #endif
        forward_ncnn_conv(inst, weights, ifm_nhwc_quant, activation_params, ofm_data_ncnn);
    }
    else{
        #ifndef COMPARE_WITH_NCNN
        forward_sim_conv(inst, weights, ifm_data, ofm_data_sim);
        #endif
        forward_ncnn_conv(inst, weights, ifm_data, activation_params, ofm_data_ncnn);
    }

#ifdef MAT_LOG
    if(inst.requant_en){
        log_mat_file<unsigned char>(ofm_data_sim, (char *)"./log/log_ofm_data_sim.txt", -1, 1, 0);
        log_mat_file<unsigned char>(ofm_data_ncnn, (char *)"./log/log_ofm_data_ncnn.txt", -1, 1, 0);
    }
    else{
        log_mat_file<float>(ofm_data_sim, (char *)"./log/log_ofm_data_sim.txt", -1, 1, 0);
        log_mat_file<float>(ofm_data_ncnn, (char *)"./log/log_ofm_data_ncnn.txt", -1, 1, 0);
    }
#endif

    #ifndef COMPARE_WITH_NCNN
    int compare_status = -1;
    if(ofm_data_sim.elemsize == 4u)
        compare_status = CompareMat(ofm_data_sim, ofm_data_ncnn, 0.001);
    else
        compare_status = CompareMat(ofm_data_sim, ofm_data_ncnn, 1.f);

    if (compare_status == 0) {
        printf("\033[;32m[ log ]: TEST LAYER CONV BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
        count_success++;
        // printf("differentCount = %d\n",countDifferentElements_mat(ofm_data_sim, ofm_data_ncnn));
        // FILE *file;
        // const char *filename = "right_test.txt";
        // char buffer[200];
        // sprintf(buffer,
        //         "iw_t=%d, ih_t=%d, ic_t=%d, oc_t=%d, k_t=%d, s_t=%d, pd_left_t=%d, pd_right_t=%d, pd_top_t=%d, pd_bottom_t=%d, "
        //         "op=%d, layout_en=%d, requant_en=%d\n",
        //         iw_t, ih_t, ic_t, oc_t, k_t, s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, op, layout_en, requant_en);

        // file = fopen(filename, "w");
        // if (file == NULL) {
        //     perror("Error opening file");
        // }

        // if (fputs(buffer, file) == EOF) {
        //     perror("Error writing to file");
        //     fclose(file);
        // }

        // fclose(file);
        // sd_write("test.txt");
        // sd_write("ofm_data_sim.txt",(int8_t*)ofm_data_sim.data,ofm_data_sim.total()*ofm_data_sim.elemsize);
        // sd_write("ofm_data_sim.txt",ofm_data_sim);
        // sd_write("ofm_data_ncnn.txt",ofm_data_ncnn);
    } else {
        printf("\033[;31m[ log ]: TEST LAYER CONV BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
        count_fail ++;
        printf("differentCount = %d\n",countDifferentElements_mat(ofm_data_sim, ofm_data_ncnn));
        // FILE *file;
        // const char *filename = "error_test.txt";
        // char buffer[200];
        // sprintf(buffer,
        //         "iw_t=%d, ih_t=%d, ic_t=%d, oc_t=%d, k_t=%d, s_t=%d, pd_left_t=%d, pd_right_t=%d, pd_top_t=%d, pd_bottom_t=%d, "
        //         "op=%d, layout_en=%d, requant_en=%d\n",
        //         iw_t, ih_t, ic_t, oc_t, k_t, s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, op, layout_en, requant_en);

        // file = fopen(filename, "a");
        // if (file == NULL) {
        //     perror("Error opening file");
        // }

        // if (fputs(buffer, file) == EOF) {
        //     perror("Error writing to file");
        //     fclose(file);
        // }

        // fclose(file);
    }
    #endif
    printf("[ log ]: success => test_summary [%d / %d]\n",count_success,BATCH_TEST_CYCYLES);
    printf("[ log ]: fail => test_summary [%d / %d]\n",count_fail,BATCH_TEST_CYCYLES);
#endif

    printf("\n");
}

void compute_pad(int iw_t, int ih_t, int k_t, int s_t, int &pd_left_t, int &pd_right_t, int &pd_top_t, int &pd_bottom_t) {
    if ((iw_t - k_t) % s_t) {
        int p = ((iw_t - k_t) / s_t + 1) * s_t + k_t - iw_t;
        pd_right_t = p / 2;
        pd_left_t = p - pd_right_t;
    } else {
        pd_left_t = 0;
        pd_right_t = 0;
    }
    if ((ih_t - k_t) % s_t) {
        int p = ((ih_t - k_t) / s_t + 1) * s_t + k_t - ih_t;
        pd_bottom_t = p / 2;
        pd_top_t = p - pd_bottom_t;
    } else {
        pd_top_t = 0;
        pd_bottom_t = 0;
    }
}

//void del_duplicate_passed_error(const char* error_txt, const char* right_txt){
//    const std::string GREEN = "\033[32m";
//    const std::string RESET = "\033[0m";
//
//    std::string fileA(error_txt);
//    std::string fileB(right_txt);
//
//    std::set<std::string> linesInB;
//    std::string line;
//
//    std::ifstream fileBStream(fileB);
//    if (!fileBStream.is_open()) {
//        std::cerr << "Error opening " << fileB << std::endl;
//    }
//
//    while (getline(fileBStream, line)) {
//        linesInB.insert(line);
//    }
//    fileBStream.close();
//
//    std::ifstream fileAStream(fileA);
//    std::set<std::string> uniqueLinesInA;
//    std::vector<std::string> linesToRemove;
//
//    if (!fileAStream.is_open()) {
//        std::cerr << "Error opening " << fileA << std::endl;
//    }
//
//    while (getline(fileAStream, line)) {
//        if (linesInB.find(line) != linesInB.end()) {
//            std::cout << GREEN << "Line removed from ERROR.txt because it's present in RIGHT.txt: " << line << RESET << std::endl;
//            linesToRemove.push_back(line);
//        } else if (!uniqueLinesInA.insert(line).second) {
//            std::cout << GREEN << "Duplicate line removed from ERROR.txt: " << line << RESET << std::endl;
//            linesToRemove.push_back(line);
//        }
//    }
//    fileAStream.close();
//
//    std::ofstream fileAOutStream(fileA, std::ofstream::out | std::ofstream::trunc);
//    if (!fileAOutStream.is_open()) {
//        std::cerr << "Error opening " << fileA << " for writing" << std::endl;
//    }
//
//    for (const auto &uniqueLine : uniqueLinesInA) {
//        if (std::find(linesToRemove.begin(), linesToRemove.end(), uniqueLine) == linesToRemove.end()) {
//            fileAOutStream << uniqueLine << "\n";
//        }
//    }
//
//    fileAOutStream.close();
//}


#ifdef BATCH_ERROR_TEST_CONV
void test_layer_conv_batch() {
    int ifm_w, ifm_h, ifm_c, ofm_c, kernel, stride, pad_left, pad_right, pad_top, pad_bottom;
    const char *filename = "error_test.txt";
    const char *right_filename = "right_test.txt";
    del_duplicate_passed_error(filename, right_filename);

    FILE *file;
    file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        assert(0);
    }
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line,
                   "dw conv: ifm_w=%d, ifm_h=%d, ifm_c=%d, ofm_c=%d, kernel=%d, stride=%d, pad_left=%d, pad_right=%d, pad_top=%d, pad_bottom=%d",
                   &ifm_w, &ifm_h, &ifm_c, &ofm_c, &kernel, &stride, &pad_left, &pad_right, &pad_top, &pad_bottom) == 10) {
            // not support
            for (int i = 0; ifm_w * u_align(ifm_c, 32) / 32 * 8 > IFM_BUFFER_DEPTH; i++) {
                if (i == 0) {
                    printf("[ log ]: ic * iw too large not support, scale the ic or iw\n");
                    FILE *file;
                    const char *right_filename = "right_test.txt";
                    const char *error_filename = "error_test.txt";

                    file = fopen(right_filename, "a");
                    if (file == NULL) {
                        perror("Error opening file");
                    }

                    if (fputs(line, file) == EOF) {
                        perror("Error writing to file");
                        fclose(file);
                    }

                    fclose(file);
					del_duplicate_passed_error(error_filename, right_filename);
                }
                int idx = random(2);
                switch (idx) {
                case 0:
                    ifm_w /= 2;
                    break;
                case 1:
                    ifm_c /= 2;
                default:
                    break;
                }
            }
            printf("dw conv => Matched data: ifm_w=%d, ifm_h=%d, ifm_c=%d, ofm_c=%d, kernel=%d, stride=%d, pad_left=%d, pad_right=%d, "
                   "pad_top=%d, pad_bottom=%d\n",
                   ifm_w, ifm_h, ifm_c, ofm_c, kernel, stride, pad_left, pad_right, pad_top, pad_bottom);

            test_layer_conv(ifm_w, ifm_h, ifm_c, ofm_c, kernel, stride, pad_left, pad_right, pad_top, pad_bottom, 1);
        } else if (sscanf(line,
                          "conv   : ifm_w=%d, ifm_h=%d, ifm_c=%d, ofm_c=%d, kernel=%d, stride=%d, pad_left=%d, pad_right=%d, pad_top=%d, "
                          "pad_bottom=%d",
                          &ifm_w, &ifm_h, &ifm_c, &ofm_c, &kernel, &stride, &pad_left, &pad_right, &pad_top, &pad_bottom) == 10) {
            // not support
            for (int i = 0; ifm_w * u_align(ifm_c, 32) / 32 * 8 > IFM_BUFFER_DEPTH; i++) {
                if (i == 0) {
                    printf("[ log ]: ic * iw too large not support, scale the ic or iw\n");
                    FILE *file;
                    const char *filename = "right_test.txt";

                    file = fopen(filename, "a");
                    if (file == NULL) {
                        perror("Error opening file");
                    }

                    if (fputs(line, file) == EOF) {
                        perror("Error writing to file");
                        fclose(file);
                    }

                    fclose(file);
                }
                int idx = random(2);
                switch (idx) {
                case 0:
                    ifm_w /= 2;
                    break;
                case 1:
                    ifm_c /= 2;
                default:
                    break;
                }
            }
            printf("conv => Matched data: ifm_w=%d, ifm_h=%d, ifm_c=%d, ofm_c=%d, kernel=%d, stride=%d, pad_left=%d, pad_right=%d, pad_top=%d, "
                   "pad_bottom=%d\n",
                   ifm_w, ifm_h, ifm_c, ofm_c, kernel, stride, pad_left, pad_right, pad_top, pad_bottom);
            test_layer_conv(ifm_w, ifm_h, ifm_c, ofm_c, kernel, stride, pad_left, pad_right, pad_top, pad_bottom, 0);
        }
    }
    fclose(file);
}

#elif defined(BATCH_TEST_CONV)
  void test_layer_conv_batch(){
        SRAND(7767517);

        int kernel_list[] = {1, 3, 5};
        int stride_list[] = {1, 2};
        int cal_list[] = {1, 2, 4, 8, 10};

        int ic_t, oc_t;

        while(count_total < BATCH_TEST_CYCYLES){

            #ifdef COMPARE_WITH_NCNN
            int layout_en = 1;
            int requant_en = 0;
            #else
            int layout_en = random(2);
            int requant_en = random(2);
            #endif

            int k_t = kernel_list[random(3)];
            int iw_t = random(512) + 2 * k_t;
            int ih_t = random(256) + 2 * k_t;
            if(iw_t > 256 || ih_t > 256){
                ic_t = random(512) + 1;
                oc_t = random(512) + 1;
            } else{
                ic_t = random(1024) + 1;
                oc_t = random(1024) + 1;
            }

            int s_t;
            if (k_t == 1)
                s_t = 1;
            else
                s_t = stride_list[random(2)];
            int pd_left_t;
            int pd_right_t;
            int pd_top_t;
            int pd_bottom_t;
            compute_pad(iw_t, ih_t, k_t, s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t);
            int op = random(2);
            if (op) {
                oc_t = ic_t;
            }

            int calculation_quantity_limit = cal_list[random(5)] * 1024; // 10B
            int ow_t = (iw_t - k_t + pd_left_t + pd_right_t) / s_t + 1;
            int oh_t = (ih_t - k_t + pd_top_t + pd_bottom_t) / s_t + 1;

            int calculation_quantity_M = (long long)(ow_t * oh_t) * (long long)(ic_t * oc_t) * k_t * k_t / 1000000;
            printf("[ log ]: origin: calculation_quantity_limit = %d M mul\n", calculation_quantity_limit);
            printf("[ log ]: origin: calculation_quantity_M = %d M mul\n", calculation_quantity_M);
            printf("[ log ]: origin: iw_t=%d, ih_t=%d, ic_t=%d, oc_t=%d, k_t=%d, s_t=%d, pd_left_t=%d, pd_right_t=%d, pd_top_t=%d, pd_bottom_t=%d, "
                "op=%d, layout_en=%d, requant_en=%d\n",
                iw_t, ih_t, ic_t, oc_t, k_t, s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, op, layout_en, requant_en);

            for (; calculation_quantity_M > calculation_quantity_limit;) {
                int idx = random(4);
                switch (idx) {
                case 0:
                    iw_t /= 2;
                    break;
                case 1:
                    ih_t /= 2;
                    break;
                case 2:
                    ic_t /= 2;
                    break;
                case 3:
                    oc_t /= 2;
                    break;
                }
                if (op) {
                    oc_t = ic_t;
                }
                compute_pad(iw_t, ih_t, k_t, s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t);
                ow_t = (iw_t - k_t + pd_left_t + pd_right_t) / s_t + 1;
                oh_t = (ih_t - k_t + pd_top_t + pd_bottom_t) / s_t + 1;
                calculation_quantity_M = (long long)(ow_t * oh_t) * (long long)(ic_t * oc_t) * k_t * k_t / 1000000;
            }

			// not support
            for (; iw_t * u_align(ic_t, 32) / 32 * 12 > IFM_BUFFER_DEPTH;) {
                int idx = random(2);
                switch (idx) {
                case 0:
                    iw_t /= 2;
                    break;
                case 1:
                    ic_t /= 2;
                default:
                    break;
                }
            }

            if (k_t * k_t * u_align(ic_t, 32) > WGT_BUFFER_DEPTH || oc_t > OSCALE_BUFFER_DEPTH) {
                requant_en = 0;
            }


            if (op) {
                oc_t = ic_t;
            }

            compute_pad(iw_t, ih_t, k_t, s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t);

            if(ih_t < k_t || iw_t < k_t){
                continue;
            }
            
            printf("[ log ]: change: calculation_quantity_M = %d M mul\n", calculation_quantity_M);
            printf("[ log ]: change: iw_t=%d, ih_t=%d, ic_t=%d, oc_t=%d, k_t=%d, s_t=%d, pd_left_t=%d, pd_right_t=%d, pd_top_t=%d, pd_bottom_t=%d, "
                "op=%d, layout_en=%d, requant_en=%d\n",
                iw_t, ih_t, ic_t, oc_t, k_t, s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, op, layout_en, requant_en);

            test_layer_conv(iw_t, ih_t, ic_t, oc_t, k_t, s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, op, layout_en, requant_en);

            printf("[ log ]: test_summary [%d / %d]\n",count_success,count_total);
        }
    }
#else
    void test_layer_conv_batch() {
        SRAND(7767517);
        // // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);
//		test_layer_conv(        387,       34,      56,       56,       3,       1,         0,          0,        0,           0,      1,             0,              0);
		//  test_layer_conv(          251,        26,        126,        126,       5,       1,         0,          0,        0,           0,      1,             1  , 0);
        // test_layer_conv(          117,        219,        189,        859,       1,       1,         0,          0,        0,           0,      0,             1  , 0);
        // test_layer_conv(          146,        128,        204,        341,       3,       2,         1,          0,        1,           0,      0,             1  , 0);
        // test_layer_conv(          106,        264,        291,        33,       5,       2,         1,          0,        1,           0,      0,             1  , 0);
        // test_layer_conv(          410,        160,        78,        78,       3,       2,         1,          0,        1,           0,      1 ,           1  , 0);
        
        // test_layer_conv(        179,       49,      184,       17,       5,       2,         0,          0,        0,           0,      0,             0,              0);
		// test_layer_conv(        180,       49,       64,       32,       5,       2,         0,          0,        0,           0,      0,             1,              0);
		// test_layer_conv(        179,       46,      64,       32,       5,       2,         0,          0,        0,           0,      0,             1,              0);  //error
		// test_layer_conv(        178,       49,       64,       32,       5,       2,         0,          0,        0,           0,      0,             1,              0);
		// test_layer_conv(        179,       49,      184,       17,       5,       2,         0,          0,        0,           0,      0,             1,              0); //error
        // test_layer_conv(         55,       82,      266,      104,       5,       2,         0,          0,        1,           0,      0,             1,              0);
        // test_layer_conv(        191,       28,      223,      223,       5,       2,         0,          0,        1,           0,      1);
        // test_layer_conv(        150,      150,       64,       64,       3,       2,         1,          0,        1,           0,      1);     //error
        // test_layer_conv(         35,       35,      512,       32,       3,       1,         1,          1,        1,           1);

//		 test_layer_conv(         32,       10,      180,       16,       5,       2,         1,          2,        1,           0,      0,             0,              0);
//		 test_layer_conv(        169,       20,      180,       16,       5,       2,         0,          0,        1,           0,      0,             0,              0);
//		 test_layer_conv(        179,       49,      184,       17,       5,       2,         0,          0,        0,           0,      0,             0,              0);
//		 test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0,      0,             0,              1);
//         test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0,      0,             1,              0);
//		 test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0,      0,             0,              1);
		//  test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0,      0,             0,              0);
//		 test_layer_conv(        371,        4,       85,       82,       1,       1,         0,          0,        0,           0);
//         test_layer_conv(         99,       60,      347,      347,       1,       1,         0,          0,        0,           0,      1);
//         test_layer_conv(        460,      111,       58,       82,       3,       2,         1,          0,        0,           0,      0);
//         test_layer_conv(        120,      111,       58,       82,       3,       2,         1,          0,        0,           0,      0);
//         test_layer_conv(        105,      135,      251,       67,       5,       2,         1,          0,        0,           0);  //!
//         test_layer_conv(        640,      512,        3,       32,       6,       2,         2,          0,        2,           0);
//         test_layer_conv(        107,      214,      375,       72,       1,       1,         0,          0,        0,           0,      0,              1,             1);
//         test_layer_conv(         13,       13,      512,     1024,       3,       1,         1,          1,        1,           1,      0,              0,             0);
//         test_layer_conv(        107,      214,      375,       72,       1,       1,         0,          0,        0,           0,      0,              1,             1);
//        test_layer_conv(         300,       300,      3,     32,       3,       2,         1,          0,        1,           0,      1,              0,             1);
//        test_layer_conv(         13,       13,      512,     64,       3,       1,         1,          1,        1,           1,      0,              0,             0);

//        test_layer_conv(         13,       13,      511,     32,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         13,       13,      512,     32,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         13,       13,      511,     64,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         13,       13,      512,     64,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         13,       13,      256,     128,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         13,       13,      511,     128,       3,       1,         1,          1,        1,           1,      0,              1,             0);  //

//        test_layer_conv(         5,       5,      512,     128,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         6,       6,      512,     128,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         8,       8,      256,     64,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         8,       8,      512,     64,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         189,       34,       113,     113,       3,       2,         0,          0,        1,           0,      1,             1, 0);
//        test_layer_conv(         8,       8,      256,     128,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         8,       8,      512,     128,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         10,       10,      512,     128,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         13,       13,      512,     128,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         13,       13,      512,     256,       3,       1,         1,          1,        1,           1,      0,              1,             0);
//        test_layer_conv(         13,       13,      512,     1024,       3,       1,         1,          1,        1,           1,      0,              1,             0);
		//  test_layer_conv(          121,        79,        181,        449,       1,       1,         0,          0,        0,           0,      0,             1  , 0);
		//  test_layer_conv(          50,        230,        556,        556,       3,       2,         1,          0,        1,           0,      1,             1  , 0);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);
//         test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0);
//         test_layer_conv(         16,        8,       32,       32,       1,       1,         0,          0,        0,           0,      0,             0,              0);
//         test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0,      0,             1,              1);
         test_layer_conv(        19,      19,       64,       32,       3,       1,         1,          1,        1,           1);
         test_layer_conv(        32,      32,       32,       16,       3,       1,         1,          1,        1,           1);
         test_layer_conv(        58,      58,       16,       8,       3,       1,         1,          1,        1,           1);
         test_layer_conv(        110,      110,       8,       4,       3,       1,         1,          1,        1,           1);
         test_layer_conv(        214,      214,       4,       2,       3,       1,         1,          1,        1,           1);
//         test_layer_conv(        160,      160,       8,       16,       3,       1,         1,          1,        1,           1);


		// // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0); // layout_en = 0
		// test_layer_conv(          5,        5,        3,        4,       1,       1,         0,          0,        0,           0,      0,             0);
        // test_layer_conv(          8,        8,       32,       32,       1,       1,         0,          0,        0,           0,      0,             0);
        // test_layer_conv(          5,        5,        3,        3,       1,       1,         0,          0,        0,           0,      1,             0);
        // test_layer_conv(          8,        8,       32,       32,       1,       1,         0,          0,        0,           0,      1,             0);
		// test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0,      0,             0,              1);
        // test_layer_conv(         99,       60,      347,      347,       1,       1,         0,          0,        0,           0,      1,             0);
        // test_layer_conv(        120,      111,       58,       82,       3,       2,         1,          0,        0,           0,      0,             0);
        // test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0,      0,             0,              1);
        // test_layer_conv(         16,        8,       32,       64,       1,       1,         0,          0,        0,           0,      0,             0,              0);
		// test_layer_conv(        416,      416,         3,      16,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(        208,      208,        16,      32,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(        104,      104,        32,      64,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(         52,       52,        64,     128,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(         26,       26,       128,     256,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(         13,       13,       256,     512,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(         13,       13,       512,    1024,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(         13,       13,      1024,     256,       1,       1,         0,          0,        0,           0,      0,             0);
        // test_layer_conv(         13,       13,       256,     128,       1,       1,         0,          0,        0,           0,      0,             0);
        // test_layer_conv(         13,       13,       256,     512,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(         26,       26,       384,     256,       3,       1,         1,          1,        1,           1,      0,             0);
        // test_layer_conv(         13,       13,       512,     255,       1,       1,         0,          0,        0,           0,      0,             0);
        // test_layer_conv(         26,       26,       256,     255,       1,       1,         0,          0,        0,           0,      0,             0);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);  // yolov3_tiny
//         test_layer_conv(        416,      416,         3,      16,       3,       1,         1,          1,        1,           1);
//         test_layer_conv(        208,      208,        16,      32,       3,       1,         1,          1,        1,           1);
//         test_layer_conv(        104,      104,        32,      64,       3,       1,         1,          1,        1,           1);
//         test_layer_conv(         52,       52,        64,     128,       3,       1,         1,          1,        1,           1);
//         test_layer_conv(         26,       26,       128,     256,       3,       1,         1,          1,        1,           1);
//         test_layer_conv(         13,       13,       256,     512,       3,       1,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,       512,    1024,       3,       1,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,      1024,     256,       1,       1,         0,          0,        0,           0);
        // test_layer_conv(         13,       13,       256,     128,       1,       1,         0,          0,        0,           0);
        // test_layer_conv(         13,       13,       256,     512,       3,       1,         1,          1,        1,           1);
        // test_layer_conv(         26,       26,       384,     256,       3,       1,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,       512,     255,       1,       1,         0,          0,        0,           0);
        // test_layer_conv(         26,       26,       256,     255,       1,       1,         0,          0,        0,           0);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);  // dw conv
        // test_layer_conv(         16,        8,       32,       32,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(          8,        8,       64,       64,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         16,        8,       32,       32,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         16,        9,       32,       32,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         13,       13,       32,       32,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         13,       13,       16,       16,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         13,       13,       48,       48,       3,       1,         1,          1,        1,           1,      1,             0,              0);
        // test_layer_conv(         13,       13,       48,       48,       3,       1,         1,          1,        1,           1,      1,             0,              1);
        // test_layer_conv(         13,       13,       48,       48,       3,       1,         1,          1,        1,           1,      1,             1,              0);
        // test_layer_conv(         13,       13,       48,       48,       3,       1,         1,          1,        1,           1,      1,             1,              1);
        // test_layer_conv(         13,       13,       48,       48,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(         48,       48,       20,       20,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         26,       26,      128,      128,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         13,       13,      256,      256,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         13,       13,      512,      512,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         13,       13,      256,      256,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         13,       13,      256,      256,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         13,       13,     1024,     1024,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         26,       26,      384,      384,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         13,       13,      512,      512,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         52,       52,       32,       32,       1,       1,         0,          0,        0,           0,      0);
        // test_layer_conv(         26,       26,      256,      256,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         52,       52,       64,       64,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(        104,      104,       64,       64,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(        208,      157,       64,       64,       1,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(        208,      208,       64,       64,       3,       1,         1,          1,        1,           1,      1);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);  // mobilenet v1 dw
        // test_layer_conv(        112,      112,       32,       32,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(        112,      112,       64,       64,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(         56,       56,      128,      128,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         56,       56,      128,      128,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(         28,       28,      256,      256,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         28,       28,      256,      256,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(         14,       14,      512,      512,       3,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(         14,       14,      512,      512,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(          7,        7,     1024,     1024,       3,       2,         1,          1,        1,           1,      1);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);  // small size
        // test_layer_conv(          5,        5,        3,        4,       1,       1,         0,          0,        0,           0);
        // test_layer_conv(          8,        8,       32,       32,       1,       1,         0,          0,        0,           0);
        // test_layer_conv(          5,        5,        3,        3,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(          8,        8,       32,       32,       1,       1,         0,          0,        0,           0,      1 , 0,1);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);  // oc block
        // test_layer_conv(          7,        7,       32,     1100,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,       64,     1100,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,       64,     1100,       1,       1,         1,          1,        1,           1);
        // test_layer_conv(          7,        7,     1024,     1024,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(          8,        8,     1024,     1024,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(          7,        7,     1500,     1500,       1,       1,         1,          1,        1,           1,      1);
        // test_layer_conv(          7,        7,     1500,     1500,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(          8,        8,     1500,     1500,       3,       2,         1,          1,        1,           1,      1);
        // test_layer_conv(          8,        8,     2048,     2048,       3,       2,         1,          1,        1,           1,      1);


        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);  // ic_block only common conv +> error
        // test_layer_conv(          8,        8,     1024,      600,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,     2048,      700,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(          7,        7,     1500,     1024,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,     2800,       32,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(          8,        8,     1024,      600,       1,       1,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,     2048,      700,       1,       1,         1,          1,        1,           1);
        // test_layer_conv(          7,        7,     1500,     1024,       1,       1,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,     2800,       32,       1,       1,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,     2800,       32,       1,       1,         1,          1,        1,           1);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);  // block ic and oc
        // test_layer_conv(         16,       16,     1024,     1024,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(         16,       16,     1024, 1024+256,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(         13,       13,     2048,      700,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(          7,        7,     1500,     1024,       3,       2,         1,          1,        1,           1);
        // test_layer_conv(         72,       80,      128,       32,       5,       2,         1,          0,        1,           0,      0,             0,              0);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);
        // test_layer_conv(         11,       35,      821,      821,       3,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         11,       35,      821,      821,       3,       1,         0,          0,        0,           0,      0);
        // test_layer_conv(          5,       13,      263,      365,       1,       1,         0,          0,        0,           0,      0);
        // test_layer_conv(          5,       13,      263,      263,       1,       1,         0,          0,        0,           0,      1);
        // test_layer_conv(         27,       19,      967,      939,       1,       1,         0,          0,        0,           0,      0);
        // test_layer_conv(         38,       46,      316,      316,       1,       1,         0,          0,        0,           0,      1,             0,              0);

        // // test_layer_conv(int iw_t, int ih_t, int ic_t, int oc_t, int k_t, int s_t, pd_left_t, pd_right_t, pd_top_t, pd_bottom_t, int op, layout_en = 1, requant_en = 0);  // k < s
        // test_layer_conv(         56,       56,       64,      128,       1,       2,         0,          0,        0,           0);

//        test_layer_conv(         208,       208,       32,      32,       3,       1,         0,          0,        0,           0,       1);
        // test_layer_conv(         208,       208,       64,      64,       3,       2,         0,          0,        0,           0,       1);
        // test_layer_conv(         104,       104,       128,    128,       3,       1,         0,          0,        0,           0,       1);
        // test_layer_conv(         104,       104,       128,    128,       3,       2,         0,          0,        0,           0,       1);
        // test_layer_conv(         52 ,       52 ,       256,    256,       3,       1,         0,          0,        0,           0,       1);
        // test_layer_conv(         52 ,       52 ,       256,    256,       3,       2,         0,          0,        0,           0,       1);


    }
#endif



} // namespace test
