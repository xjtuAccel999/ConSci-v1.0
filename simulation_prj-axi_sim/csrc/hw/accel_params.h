#ifndef __ACCEL_PARAMS_H_
#define __ACCEL_PARAMS_H_


#define u_align(x,n)  ((x+n-1) & -n)  

#ifndef NCNN_TOOLS
#include "../veri.h"
/*****************************ALU*******************************/
#define ALU_RESET \
        top->io_sim_alu_reg_mathfunc_ctrl_reg = 0; \
        top->io_sim_alu_reg_actfunc_ctrl_reg = 0; \
        top->io_sim_alu_reg_innerprod_ctrl_reg = 0; update()
#define ALU_MATH_RESET \
        top->io_sim_alu_reg_mathfunc_ctrl_reg = 0; update()
#define ALU_ACT_RESET \
        top->io_sim_alu_reg_actfunc_ctrl_reg = 0; update()
#define ALU_INNERPROD_RESET \
        top->io_sim_alu_reg_innerprod_ctrl_reg = 0; update()
#define ALU_MATHFUNC_CTRL_SET(en,src_num,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,op) \
        top->io_sim_alu_reg_mathfunc_ctrl_reg=(en | src_num<<1 | mul_src1_sel<<3 | add_src0_sel<<6 | add_src1_sel<<9 | sub_en<<12 | add_en<<13 | mul_en<<14 | max_en<<15 | min_en<<16 | op<<23); update()
#define ALU_ACTFUNC_CTRL_SET(en,prop,src_sel,dst_sel,op) \
        top->io_sim_alu_reg_actfunc_ctrl_reg=(en | prop<<1 | src_sel<<3 | dst_sel<<5 | op<<16); update()
#define ALU_INNERPROD_CTRL_SET(en) \
        top->io_sim_alu_reg_innerprod_ctrl_reg=en; update()

#define ALU_VECLEN_CH0_SET(x) \
        top->io_sim_alu_reg_alu_veclen_ch0_reg=x; update()
#define ALU_SRC0_ADDR_CH0_SET(x) \
        top->io_sim_alu_reg_src0_addr_ch0_reg=x; update()
#define ALU_SRC1_ADDR_CH0_SET(x) \
        top->io_sim_alu_reg_src1_addr_ch0_reg=x; update()
#define ALU_DST_ADDR_CH0_SET(x) \
        top->io_sim_alu_reg_dst_addr_ch0_reg=x; update()
#define ALU_VECLEN_CH1_SET(x) \
        top->io_sim_alu_reg_alu_veclen_ch1_reg=x; update()
#define ALU_SRC0_ADDR_CH1_SET(x) \
        top->io_sim_alu_reg_src0_addr_ch1_reg=x; update()
#define ALU_SRC1_ADDR_CH1_SET(x) \
        top->io_sim_alu_reg_src1_addr_ch1_reg=x; update()
#define ALU_DST_ADDR_CH1_SET(x) \
        top->io_sim_alu_reg_dst_addr_ch1_reg=x; update()

#define ALU_MATH_ALPHA_SET(x) \
        top->io_sim_alu_reg_math_alpha_reg=x; update()
#define ALU_MATH_BETA_SET(x) \
        top->io_sim_alu_reg_math_beta_reg=x; update()

#define ALU_ACT_COEFF_A_0_SET(x) \
        top->io_sim_alu_reg_act_coefficient_a_reg_0=x; update()
#define ALU_ACT_COEFF_A_1_SET(x) \
        top->io_sim_alu_reg_act_coefficient_a_reg_1=x; update()
#define ALU_ACT_COEFF_A_2_SET(x) \
        top->io_sim_alu_reg_act_coefficient_a_reg_2=x; update()
#define ALU_ACT_COEFF_A_3_SET(x) \
        top->io_sim_alu_reg_act_coefficient_a_reg_3=x; update()
#define ALU_ACT_COEFF_A_4_SET(x) \
        top->io_sim_alu_reg_act_coefficient_a_reg_4=x; update()
#define ALU_ACT_COEFF_B_0_SET(x) \
        top->io_sim_alu_reg_act_coefficient_b_reg_0=x; update()
#define ALU_ACT_COEFF_B_1_SET(x) \
        top->io_sim_alu_reg_act_coefficient_b_reg_1=x; update()
#define ALU_ACT_COEFF_B_2_SET(x) \
        top->io_sim_alu_reg_act_coefficient_b_reg_2=x; update()
#define ALU_ACT_COEFF_B_3_SET(x) \
        top->io_sim_alu_reg_act_coefficient_b_reg_3=x; update()
#define ALU_ACT_COEFF_B_4_SET(x) \
        top->io_sim_alu_reg_act_coefficient_b_reg_4=x; update()
#define ALU_ACT_COEFF_C_0_SET(x) \
        top->io_sim_alu_reg_act_coefficient_c_reg_0=x; update()
#define ALU_ACT_COEFF_C_1_SET(x) \
        top->io_sim_alu_reg_act_coefficient_c_reg_1=x; update()
#define ALU_ACT_COEFF_C_2_SET(x) \
        top->io_sim_alu_reg_act_coefficient_c_reg_2=x; update()
#define ALU_ACT_COEFF_C_3_SET(x) \
        top->io_sim_alu_reg_act_coefficient_c_reg_3=x; update()
#define ALU_ACT_COEFF_C_4_SET(x) \
        top->io_sim_alu_reg_act_coefficient_c_reg_4=x; update()

#define ALU_ACT_RANGE_0_SET(x) \
        top->io_sim_alu_reg_act_range_reg_0=x; update()
#define ALU_ACT_RANGE_1_SET(x) \
        top->io_sim_alu_reg_act_range_reg_1=x; update()
#define ALU_ACT_RANGE_2_SET(x) \
        top->io_sim_alu_reg_act_range_reg_2=x; update()
#define ALU_ACT_RANGE_3_SET(x) \
        top->io_sim_alu_reg_act_range_reg_3=x; update()





/*****************************POOL*******************************/
#define POOL_RESET \
        top->io_sim_pool_reg_pool_ctrl_reg=0;update()   
#define POOL_CTRL_SET(en,op,kernel_w,kernel_h,stride_w,stride_h,src_sel,pad_bottom,pad_top,pad_right,pad_left,pad_mode) \
        top->io_sim_pool_reg_pool_ctrl_reg=(en | op<<1 | kernel_w<<3 | kernel_h<<5 | stride_w<<7 | stride_h<<9 | src_sel<<11 | pad_bottom<<16 | pad_top<<18 | pad_right<<20 | pad_left<<22 | pad_mode<<24); update()
#define POOL_IFM_C_SET(c) \
        top->io_sim_pool_reg_pool_shape_ic_reg=c; update()
#define POOL_IFM_WH_SET(w,h) \
        top->io_sim_pool_reg_pool_shape_iwh_reg=(w<<16 | h); update()
#define POOL_IFM_CSTEP_SET(x) \
        top->io_sim_pool_reg_pool_shape_icstep_reg=x; update()
#define POOL_OFM_C_SET(c) \
        top->io_sim_pool_reg_pool_shape_oc_reg=c; update()
#define POOL_OFM_WH_SET(w,h) \
        top->io_sim_pool_reg_pool_shape_owh_reg=(w<<16 | h); update()
#define POOL_OFM_CSTEP_SET(x) \
        top->io_sim_pool_reg_pool_shape_ocstep_reg=x; update()
#define POOL_IFM_ADDR_SET(x) \
        top->io_sim_pool_reg_pool_ifm_addr_reg=x; update()
#define POOL_OFM_ADDR_SET(x) \
        top->io_sim_pool_reg_pool_ofm_addr_reg=x; update()
#define POOL_PAD_VALUE_SET(x) \
        top->io_sim_pool_reg_pool_pad_value_reg=x; update()


/*****************************GEMM*******************************/
#define GEMM_RESET \
        top->io_sim_gemm_reg_gemm_ctrl_reg = 0; update()
#define GEMM_CTRL_SET(en,op,kernel,stride,pad_mode,pad_left,pad_right,pad_top,pad_bottom,bias_en,requant_en,layout_en,oscale_en,div_ifm_c_en) \
        top->io_sim_gemm_reg_gemm_ctrl_reg=(en | op<<1 | kernel<<3 | stride<<6 | pad_mode<<9 | pad_left<<11 | pad_right<<13 | pad_top<<15 | pad_bottom<<17 | bias_en<<19 | requant_en<<20 | layout_en<<21 | oscale_en<<22 | div_ifm_c_en<<23); update()
#define GEMM_QUANT_DATA_SET(x) \
        top->io_sim_gemm_reg_quant_data_reg=x; update()
#define GEMM_REQUANT_DATA_SET(x) \
        top->io_sim_gemm_reg_requant_data_reg=x; update()
#define GEMM_DEQUANT_ADDR_SET(x) \
        top->io_sim_gemm_reg_dequant_addr_reg=x; update()
#define GEMM_BIAS_ADDR_SET(x) \
        top->io_sim_gemm_reg_bias_addr_reg=x; update()
#define GEMM_IFM_C_SET(c) \
        top->io_sim_gemm_reg_ifm_shape_c_reg=c; update()
#define GEMM_IFM_WH_SET(w,h) \
        top->io_sim_gemm_reg_ifm_shape_wh_reg=(w<<16 | h); update()
#define GEMM_IFM_CSTEP_SET(x) \
        top->io_sim_gemm_reg_ifm_shape_cstep_reg=x; update()
#define GEMM_OFM_C_SET(c) \
        top->io_sim_gemm_reg_ofm_shape_c_reg=c; update()
#define GEMM_OFM_WH_SET(w,h) \
        top->io_sim_gemm_reg_ofm_shape_wh_reg=(w<<16 | h); update()
#define GEMM_OFM_CSTEP_SET(x) \
        top->io_sim_gemm_reg_ofm_shape_cstep_reg=x; update()
#define GEMM_WGT_LEN_SET(x) \
        top->io_sim_gemm_reg_wgt_len_reg=x; update()
#define GEMM_IFM_ADDR_SET(x) \
        top->io_sim_gemm_reg_ifm_baseaddr_reg=x; update()
#define GEMM_WGT_ADDR_SET(x) \
        top->io_sim_gemm_reg_wgt_baseaddr_reg=x; update()
#define GEMM_OFM_ADDR_SET(x) \
        top->io_sim_gemm_reg_ofm_baseaddr_reg=x; update()
#define GEMM_DIV_IFM_C_SET(x) \
        top->io_sim_gemm_reg_div_ifm_c_reg=x; update()

/*****************************SOFT RESET*******************************/
#define SOFT_RESET \
        top->io_sim_reset_reg_reset_reg = 1; update()

#define SOFT_RESET_CTRL_SET \
        top->io_sim_reset_reg_reset_reg = 0; update()
#endif





#endif