#include "hw_gemm.h"
#include "../config.h"
#include "../testLayer/test_layer.h"
#include "../utils/utils.h"
#include "../veri.h"
#include "accel_params.h"
#include "time.h"
namespace accel {

#ifdef PRINT_CONV_TIME
int conv_layer_count = 0;
uint64_t conv_time = 0;
#endif
vluint64_t start_time = 0;

hw_gemm::hw_gemm() {
    // gemm
    this->op = HW_CONV;
    this->kernel = 3;
    this->stride = 1;
    this->padding_mode = PD_MODE_ZERO;
    this->padding_left = 1;
    this->padding_right = 1;
    this->padding_top = 1;
    this->padding_bottom = 1;
    this->bias_en = 1;
    this->oscale_en = 1;
    this->requant_en = 0;
    this->layout_en = 1;
    this->quant_scale = 0.f;
    this->dequant_scale = NULL;
    this->requant_scale = 0.f;
    this->bias_data = NULL;
    this->ifm_w = 0;
    this->ifm_h = 0;
    this->ifm_c = 0;
    this->ofm_w = 0;
    this->ofm_h = 0;
    this->ofm_c = 0;

    // activation
    this->act_dst_sel = ACT_TO_OPFUSION;

    //block check
    this->block_ic_flag = 0;
    this->block_ic_limit = 0;
    this->block_ic_base = 0;
    this->block_ic_offset = 0;
    this->block_use_bias = 1;
    this->block_oc_flag = 0;
    this->block_oc_limit = OSCALE_BUFFER_DEPTH;
    this->block_oc_base = 0;
    this->block_oc_offset = 0;
    this->div_ifm_c_en = 0;
}

void hw_gemm::block_channel_check(){
    if(op == HW_CONV){
        int weight_size = kernel * kernel * u_align(ifm_c,32);
        block_ic_flag = weight_size > WGT_BUFFER_DEPTH;
        block_ic_limit = (WGT_BUFFER_DEPTH/kernel/kernel) / 32 *32;
    }

    block_oc_flag = ofm_c > block_oc_limit;

    if(block_ic_flag){
        assert(requant_en == 0);
    }
    else{
        block_ic_base = 0;
        block_ic_offset = ifm_c;
    }

    if(block_oc_flag){
        assert(requant_en == 0);
    }
    else{
        block_oc_base = 0;
        block_oc_offset = ofm_c;
    }
}

// no blocking
void hw_gemm::gemm_forward_s(ncnn::Mat& ifm, ncnn::Mat& wgt, ncnn::Mat& ofm){
    if(layout_en)
        assert(u_align(ifm_c,32)/32 * ifm_w * ifm_h <= IFM_BUFFER_DEPTH);
    else
        assert(ifm_c/32*ifm_w*ifm_h <= IFM_BUFFER_DEPTH);
    assert(u_align(ifm_c,32) * kernel * kernel <= 1024 * 16);

    GEMM_RESET;
    ALU_ACT_RESET;

    if(act_op != 0){
        cfg_param();
        send_coffe();
    }

    int* ifm_addr = ifm;
    int* wgt_addr = wgt;
    int* ofm_addr = ofm;
    GEMM_IFM_ADDR_SET((uint64_t)ifm_addr);
    GEMM_WGT_ADDR_SET((uint64_t)wgt_addr);
    GEMM_OFM_ADDR_SET((uint64_t)ofm_addr);

    GEMM_IFM_WH_SET(ifm_w,ifm_h);
    GEMM_IFM_C_SET(ifm_c);
    GEMM_IFM_CSTEP_SET(ifm.cstep);

    GEMM_QUANT_DATA_SET(*((uint32_t*)&quant_scale));
    GEMM_DEQUANT_ADDR_SET((uint64_t)dequant_scale);

    if(requant_en)
        GEMM_REQUANT_DATA_SET(*((uint32_t*)&requant_scale));
    if(bias_en)
        GEMM_BIAS_ADDR_SET((uint64_t)bias_data);

    GEMM_OFM_WH_SET(ofm_w,ofm_h);
    GEMM_OFM_C_SET(ofm_c);
    GEMM_OFM_CSTEP_SET(ofm.cstep);
    ALU_ACTFUNC_CTRL_SET(1,act_prop,ACT_FROM_OSCALE_BIAS,act_dst_sel,act_op);
    GEMM_CTRL_SET(1,op,kernel,stride,padding_mode,padding_left,padding_right,padding_top,padding_bottom,bias_en,requant_en,layout_en,oscale_en,0);
}

void hw_gemm::gemm_forward_block(float* ifm_baseaddr_t, unsigned char* wgt_baseaddr_t, float* ofm_baseaddr_t) {

    int align_down_value = 0;
    if(ofm_w % 4 == 0)
        align_down_value = 1;
    else if(ofm_w % 2 == 0)
        align_down_value = 2;
    else    
        align_down_value = 4;

    int u_padding_bottom = 0;
    int u_padding_top = 0;
    int u_ih = 0;
    int u_oh = 0;
    int u_oh_aligndown = 0; 
    int *ifm_base_addr = (int*)ifm_baseaddr_t;
    int *wgt_base_addr = (int*)wgt_baseaddr_t;
    int *ofm_base_addr = (int*)ofm_baseaddr_t;
    int is_finish = 0;

    int ic_align32 = u_align(block_ic_offset, 32);
    int oc_align32 = u_align(block_oc_offset, 32);
    int ih_step = IFM_BUFFER_DEPTH * 32 / u_align(block_ic_offset, 32) / ifm_w;

    int ih_cnt_cur = 0;
    int oh_cnt_cur = 0;

    int count = 0;

    int *ifm_addr = NULL;
    int *ofm_addr = NULL;

    if(block_oc_flag){
        if(op == HW_CONV){
            div_ifm_c_en = block_ic_flag;
        }
        else if(op == HW_DEPTHCONV){
            div_ifm_c_en = 1;
        }
    }
    if(block_ic_flag && layout_en == 0){
        div_ifm_c_en = 1;
    }

    while (!is_finish) {
        // cal parameters
        int ih_cnt_next = ih_cnt_cur + ih_step;
        if (ih_cnt_next >= ifm_h) {
            is_finish = 1;
            u_padding_bottom = padding_bottom;
            if (ih_cnt_cur == 0) {
                u_padding_top = padding_top;
                u_ih = ifm_h;
            } else {
                u_padding_top = 0;
                u_ih = ifm_h - ih_cnt_cur;
            }
        } else {
            u_padding_bottom = 0;
            u_ih = ih_step;
            if (ih_cnt_cur == 0)
                u_padding_top = padding_top;
            else
                u_padding_top = 0;
        }

        u_oh = (u_ih - kernel + u_padding_bottom + u_padding_top) / stride + 1;
        u_oh_aligndown = u_oh / align_down_value * align_down_value;

        if(layout_en == 1){
            ifm_addr = ifm_base_addr + ih_cnt_cur * ifm_w;
        }
        else{
            if(div_ifm_c_en)
                ifm_addr = ifm_base_addr + ih_cnt_cur * ifm_w * u_align(ifm_c,32) / 4;
            else
                ifm_addr = ifm_base_addr + ih_cnt_cur * ifm_w * ic_align32 / 4;
        }

        if(requant_en == 1){
            ofm_addr = ofm_base_addr + oh_cnt_cur * ofm_w * oc_align32 / 4;
        }
        else{
            ofm_addr = ofm_base_addr + oh_cnt_cur * ofm_w;
        }

		if(op == HW_DEPTHCONV){
			assert(block_ic_offset == block_oc_offset);
		}
#ifdef PRINT_SHAPE_PLUS
        printf("[ log ]: --------------------- task %d ---------------------\n", count);
        printf("[ log ]: u_padding_top = %d, u_padding_bottom = %d\n", u_padding_top, u_padding_bottom);
        printf("[ log ]: u_ih = %d, u_oh = %d\n", u_ih, u_oh);
        printf("[ log ]: ih_cnt_cur = %d, ih_cnt_next = %d, oh_cnt_cur = %d\n", ih_cnt_cur, ih_cnt_next, oh_cnt_cur);
        printf("[ log ]: ifm_addr = %p, ofm_addr = %p\n", ifm_addr, ofm_addr);
		printf("[ log ]: ifm_w = %d, ifm_h = %d, block_ic_offset = %d\n", ifm_w, u_ih, block_ic_offset);
    	printf("[ log ]: ofm_w = %d, ofm_h = %d, block_oc_offset = %d\n", ofm_w, u_oh, block_oc_offset);
    	printf("[ log ]: oscale_addr = %p, bias_addr = %p\n", (dequant_scale+block_oc_base), (bias_data+block_oc_base));
    	printf("[ log ]: block_use_bias & bias_en = %d, block_use_bias = %d, block_oc_base = %d\n",bias_en & block_use_bias, block_use_bias,block_oc_base);
#endif

        // oh_cnt_cur += u_oh;
        oh_cnt_cur += u_oh_aligndown;
        ih_cnt_cur = oh_cnt_cur * stride - u_padding_bottom - padding_top;
        count++;

        // send instruction
        GEMM_RESET;
        ALU_ACT_RESET;
        
        cfg_param();
        send_coffe();

        GEMM_IFM_ADDR_SET((uint64_t)ifm_addr);
        GEMM_WGT_ADDR_SET((uint64_t)wgt_base_addr);
        GEMM_OFM_ADDR_SET((uint64_t)ofm_addr);
        GEMM_IFM_WH_SET(ifm_w, u_ih);
        GEMM_IFM_C_SET(block_ic_offset);
        GEMM_IFM_CSTEP_SET(ifm_cstep);

        if(div_ifm_c_en)
            GEMM_DIV_IFM_C_SET(u_align(ifm_c,32));
            // GEMM_DIV_IFM_C_SET(u_align(block_ic_offset,32));

        GEMM_QUANT_DATA_SET(*((uint32_t *)&quant_scale));
        GEMM_DEQUANT_ADDR_SET((uint64_t)(dequant_scale+block_oc_base));

        if (requant_en)
            GEMM_REQUANT_DATA_SET(*((uint32_t *)&requant_scale));
        if (bias_en & block_use_bias)
            GEMM_BIAS_ADDR_SET((uint64_t)(bias_data+block_oc_base));

        GEMM_OFM_WH_SET(ofm_w, u_oh);
        GEMM_OFM_C_SET(block_oc_offset);
        GEMM_OFM_CSTEP_SET(ofm_cstep);

        if(block_ic_flag){
            ALU_ACTFUNC_CTRL_SET(1, act_prop, ACT_FROM_OSCALE_BIAS, act_dst_sel, 0);
        }
        else{
            ALU_ACTFUNC_CTRL_SET(1, act_prop, ACT_FROM_OSCALE_BIAS, act_dst_sel, act_op);
        } 

        GEMM_CTRL_SET(1, op, kernel, stride, padding_mode, padding_left, padding_right, u_padding_top, u_padding_bottom, (bias_en & block_use_bias), requant_en, layout_en, oscale_en, div_ifm_c_en);


        dma_wait();
        GEMM_RESET;
        ALU_ACT_RESET;
    }
}

void hw_gemm::gemm_forward_block_oc(float* ifm_baseaddr_t, unsigned char* wgt_baseaddr_t, float* ofm_baseaddr_t) {
    if(block_oc_flag){
        block_oc_base = 0;
        block_oc_offset = 0;
        int is_finish = 0;
        int count = 0;
        unsigned char* block_wgt_baseaddr = wgt_baseaddr_t;
        float* block_ofm_baseaddr = ofm_baseaddr_t;
        float* block_ifm_baseaddr = ifm_baseaddr_t;
        while(!is_finish){
            if(block_oc_base + block_oc_limit > ofm_c){
                block_oc_offset = ofm_c - block_oc_base;
            }
            else{
                block_oc_offset = block_oc_limit;
            }

            #ifdef PRINT_SHAPE_PLUS
                printf("[ log ]: --------------------- block oc task %d ---------------------\n", count);
                printf("[ log ]: block_oc_limit = %d, ofm_c = %d, is_finish = %d\n", block_oc_limit, ofm_c, is_finish);
                printf("[ log ]: block_oc_base = %d, block_oc_offset = %d\n", block_oc_base, block_oc_offset);
                printf("[ log ]: block_wgt_baseaddr = %p, block_ofm_baseaddr = %p\n", block_wgt_baseaddr, block_ofm_baseaddr);
            #endif
			if(op==HW_DEPTHCONV){
                block_ic_offset = block_oc_offset;
			}

            gemm_forward_block(block_ifm_baseaddr, block_wgt_baseaddr, block_ofm_baseaddr);

            is_finish = block_oc_base + block_oc_offset >= ofm_c;
             
            if(!is_finish){
                block_oc_base += block_oc_limit;
                block_wgt_baseaddr += kernel * kernel * (op ? 1 : u_align(block_ic_offset, 32)) * block_oc_limit;
                block_ofm_baseaddr = ofm_baseaddr_t + block_oc_base * ofm_cstep;
                if (op == HW_DEPTHCONV) {
                    if(layout_en)
                        block_ifm_baseaddr = ifm_baseaddr_t + block_oc_base * ifm_cstep;
                    else
                        block_ifm_baseaddr = ifm_baseaddr_t + block_oc_base / 4;
                }
            }
            count ++ ;
            #ifdef PRINT_SHAPE_PLUS
            printf("\n");
            #endif
        }
    }
    else{
        gemm_forward_block(ifm_baseaddr_t, wgt_baseaddr_t, ofm_baseaddr_t);
    }
}

void hw_gemm::gemm_forward(ncnn::Mat &ifm, ncnn::Mat &wgt, ncnn::Mat &ofm){

#ifdef PRINT_CONV_TIME
    time_dma_ch0_r = 0;
    time_dma_ch1_r = 0;
    time_dma_ch0_w = 0;
    time_dma_ch1_w = 0;
    time_ifmbuf_load = 0;
    time_ofmbuf_congest = 0;
    time_axisendbuf_congest = 0;
#endif
    start_time = main_time;

    ifm_cstep = ifm.cstep;
    ofm_cstep = ofm.cstep;
    ofm_total = ofm.total();

    float* ifm_baseaddr_t = (float*)ifm.data;
    float* ofm_baseaddr_t = (float*)ofm.data;
    unsigned char* wgt_baseaddr_t = (unsigned char*)wgt.data;

#ifdef PRINT_SHAPE_PLUS
    printf("[ log ]: --------------------- origin ---------------------\n");
    printf("[ log ]: padding_top = %d, padding_bottom = %d\n", padding_top, padding_bottom);
    printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d\n", ifm_w, ifm_h, ifm_c);
    printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n", ofm_w, ofm_h, ofm_c);
    printf("[ log ]: ifm_base_addr = %p, ofm_base_addr = %p\n", ifm_baseaddr_t, ofm_baseaddr_t);
    printf("\n");
#endif

    block_channel_check();

    if(block_ic_flag){
                
        int is_finish = 0;
        int count = 0;
        unsigned char* wgt_baseaddr = wgt_baseaddr_t;
        float* ifm_baseaddr = ifm_baseaddr_t;

        ncnn::Mat ofm_t;
        ncnn::Option opt;
        ofm_t.create_like(ofm, opt.blob_allocator);

        while(!is_finish){
            ncnn::Mat& ofm_o = (count == 0) ? ofm : ofm_t;
            block_use_bias = count == 0 ;
            if(block_ic_base + block_ic_limit > ifm_c){
                block_ic_offset = ifm_c - block_ic_base;
            }
            else{
                block_ic_offset = block_ic_limit;
            }

            #ifdef PRINT_SHAPE_PLUS
                printf("[ log ]: -----------------------------------------------------------------------------\n");
                printf("[ log ]: --------------------- block ic task %d ---------------------\n", count);
                printf("[ log ]: block_ic_limit = %d, ifm_c = %d, is_finish = %d\n", block_ic_limit, ifm_c, is_finish);
                printf("[ log ]: block_ic_base = %d, block_ic_offset = %d\n", block_ic_base, block_ic_offset);
                printf("[ log ]: wgt_baseaddr = %p, ifm_baseaddr = %p\n", wgt_baseaddr, ifm_baseaddr);
                printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n",ofm_o.w, ofm_o.h, ofm_o.c);
            #endif

            gemm_forward_block_oc(ifm_baseaddr, wgt_baseaddr, ofm_o);

            is_finish = block_ic_base + block_ic_offset >= ifm_c;

            if(!is_finish){
                block_ic_base += block_ic_limit;
                wgt_baseaddr += kernel * kernel * block_ic_limit * u_align(ofm_c,32);
                if(layout_en)
                    ifm_baseaddr = ifm_baseaddr_t + block_ic_base * ifm_cstep;
                else
                    ifm_baseaddr = ifm_baseaddr_t + block_ic_base / 4;
            }

            if(count != 0){
                hw_math mat_add;
                mat_add.add_en = 1;
                mat_add.add_src0_sel = ADD_SRC0_FROM_DMA;
                mat_add.add_src1_sel = ADD_SRC1_FROM_DMA;
                mat_add.math_forward(ofm,ofm_t,ofm);
            }
            count ++ ;
            #ifdef PRINT_SHAPE_PLUS
            printf("\n");
            #endif
        }
        if(act_op != 0){
            act_forward(ofm, ofm);
        }
    }
    else{
        gemm_forward_block_oc(ifm_baseaddr_t, wgt_baseaddr_t, ofm_baseaddr_t);
    }

    #ifdef PRINT_CONV_TIME
        conv_time = (main_time - start_time)/2;
        printf("[ log ]: current conv layer id is %d\n",conv_layer_count);
        printf("[ log ]: (iw, ih, ic, oc) = (%d, %d, %d, %d), op = %d\n", ifm_w, ifm_h, ifm_c, ofm_c, op);
        printf("[ log ]: conv_time = %ld\n", conv_time);
        printf("[ log ]: time_dma_ch0_r = %ld, => %.3f%%\n", time_dma_ch0_r, (double)time_dma_ch0_r/(double)conv_time*100.f);
        printf("[ log ]: time_dma_ch1_r = %ld, => %.3f%%\n", time_dma_ch1_r, (double)time_dma_ch1_r/(double)conv_time*100.f);
        printf("[ log ]: time_dma_ch0_w = %ld, => %.3f%%\n", time_dma_ch0_w, (double)time_dma_ch0_w/(double)conv_time*100.f);
        printf("[ log ]: time_dma_ch1_w = %ld, => %.3f%%\n", time_dma_ch1_w, (double)time_dma_ch1_w/(double)conv_time*100.f);
        printf("[ log ]: time_ifmbuf_load = %ld, => %.3f%%\n", time_ifmbuf_load, (double)time_ifmbuf_load/(double)conv_time*100.f);
        printf("[ log ]: time_ofmbuf_congest = %ld, => %.3f%%\n", time_ofmbuf_congest, (double)time_ofmbuf_congest/(double)conv_time*100.f);
        printf("[ log ]: time_axisendbuf_congest = %ld, => %.3f%%\n", time_axisendbuf_congest, (double)time_axisendbuf_congest/(double)conv_time*100.f);
        printf("\n\n");
        conv_layer_count ++;
    #endif
    time_gemm += main_time - start_time;
}

} // namespace accel