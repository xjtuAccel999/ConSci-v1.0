#ifndef __ACCEL_PARAMS_H_
#define __ACCEL_PARAMS_H_

#include "xil_io.h"

#define ACCEL_REG_BASEADDR 0x80020000

#define u_align(x,n)  ((x+n-1) & -n)  
//ALU
#define ACCEL_REG_MATHFUNC_CTRL     ACCEL_REG_BASEADDR + 0*4
#define ACCEL_REG_ACTFUNC_CTRL      ACCEL_REG_BASEADDR + 1*4
#define ACCEL_REG_ALU_VECLEN0       ACCEL_REG_BASEADDR + 2*4
#define ACCEL_REG_SRC0_ADDR0        ACCEL_REG_BASEADDR + 3*4
#define ACCEL_REG_SRC1_ADDR0        ACCEL_REG_BASEADDR + 4*4
#define ACCEL_REG_DST_ADDR0         ACCEL_REG_BASEADDR + 5*4
#define ACCEL_REG_ALU_VECLEN1       ACCEL_REG_BASEADDR + 6*4
#define ACCEL_REG_SRC0_ADDR1        ACCEL_REG_BASEADDR + 7*4
#define ACCEL_REG_SRC1_ADDR1        ACCEL_REG_BASEADDR + 8*4
#define ACCEL_REG_DST_ADDR1         ACCEL_REG_BASEADDR + 9*4
#define ACCEL_REG_MATH_ALPHA        ACCEL_REG_BASEADDR + 10*4
#define ACCEL_REG_MATH_BETA         ACCEL_REG_BASEADDR + 11*4
#define ACCEL_REG_ACT_RANG_0        ACCEL_REG_BASEADDR + 12*4
#define ACCEL_REG_ACT_RANG_1        ACCEL_REG_BASEADDR + 13*4
#define ACCEL_REG_ACT_RANG_2        ACCEL_REG_BASEADDR + 14*4
#define ACCEL_REG_ACT_RANG_3        ACCEL_REG_BASEADDR + 15*4
#define ACCEL_REG_ACT_COEFF_A0      ACCEL_REG_BASEADDR + 16*4
#define ACCEL_REG_ACT_COEFF_A1      ACCEL_REG_BASEADDR + 17*4
#define ACCEL_REG_ACT_COEFF_A2      ACCEL_REG_BASEADDR + 18*4
#define ACCEL_REG_ACT_COEFF_A3      ACCEL_REG_BASEADDR + 19*4
#define ACCEL_REG_ACT_COEFF_A4      ACCEL_REG_BASEADDR + 20*4
#define ACCEL_REG_ACT_COEFF_B0      ACCEL_REG_BASEADDR + 21*4
#define ACCEL_REG_ACT_COEFF_B1      ACCEL_REG_BASEADDR + 22*4
#define ACCEL_REG_ACT_COEFF_B2      ACCEL_REG_BASEADDR + 23*4
#define ACCEL_REG_ACT_COEFF_B3      ACCEL_REG_BASEADDR + 24*4
#define ACCEL_REG_ACT_COEFF_B4      ACCEL_REG_BASEADDR + 25*4
#define ACCEL_REG_ACT_COEFF_C0      ACCEL_REG_BASEADDR + 26*4
#define ACCEL_REG_ACT_COEFF_C1      ACCEL_REG_BASEADDR + 27*4
#define ACCEL_REG_ACT_COEFF_C2      ACCEL_REG_BASEADDR + 28*4
#define ACCEL_REG_ACT_COEFF_C3      ACCEL_REG_BASEADDR + 29*4
#define ACCEL_REG_ACT_COEFF_C4      ACCEL_REG_BASEADDR + 30*4
#define ACCEL_REG_INNER_CTRL        ACCEL_REG_BASEADDR + 31*4
#define ACCEL_REG_ALU_ODATA         ACCEL_REG_BASEADDR + 34*4


//POOL
#define ACCEL_REG_POOL_CTRL         ACCEL_REG_BASEADDR + 40*4
#define ACCEL_REG_POOL_SHAPE_IC     ACCEL_REG_BASEADDR + 41*4
#define ACCEL_REG_POOL_SHAPE_IWH    ACCEL_REG_BASEADDR + 42*4
#define ACCEL_REG_POOL_SHAPE_ICSTEP ACCEL_REG_BASEADDR + 43*4
#define ACCEL_REG_POOL_SHAPE_OC     ACCEL_REG_BASEADDR + 44*4
#define ACCEL_REG_POOL_SHAPE_OWH    ACCEL_REG_BASEADDR + 45*4
#define ACCEL_REG_POOL_SHAPE_OCSTEP ACCEL_REG_BASEADDR + 46*4
#define ACCEL_REG_POOL_IFM_ADDR     ACCEL_REG_BASEADDR + 47*4
#define ACCEL_REG_POOL_OFM_ADDR     ACCEL_REG_BASEADDR + 48*4
#define ACCEL_REG_POOL_PAD_VALUE    ACCEL_REG_BASEADDR + 49*4


//GEMM
#define ACCEL_REG_GEMM_CTRL         ACCEL_REG_BASEADDR + 60*4
#define ACCEL_REG_QUANT_DATA        ACCEL_REG_BASEADDR + 61*4
#define ACCEL_REG_REQUANT_DATA      ACCEL_REG_BASEADDR + 62*4
#define ACCEL_REG_DEQUANT_ADDR      ACCEL_REG_BASEADDR + 63*4
#define ACCEL_REG_BIAS_ADDR         ACCEL_REG_BASEADDR + 64*4
#define ACCEL_REG_SHAPE_IFM_C       ACCEL_REG_BASEADDR + 65*4
#define ACCEL_REG_SHAPE_IFM_WH      ACCEL_REG_BASEADDR + 66*4
#define ACCEL_REG_SHAPE_IFM_CSTEP   ACCEL_REG_BASEADDR + 67*4
#define ACCEL_REG_SHAPE_OFM_C       ACCEL_REG_BASEADDR + 68*4
#define ACCEL_REG_SHAPE_OFM_WH      ACCEL_REG_BASEADDR + 69*4
#define ACCEL_REG_SHAPE_OFM_CSTEP   ACCEL_REG_BASEADDR + 70*4
#define ACCEL_REG_WGT_LEN           ACCEL_REG_BASEADDR + 71*4
#define ACCEL_REG_IFM_BASEADDR      ACCEL_REG_BASEADDR + 72*4
#define ACCEL_REG_WGT_BASEADDR      ACCEL_REG_BASEADDR + 73*4
#define ACCEL_REG_OFM_BASEADDR      ACCEL_REG_BASEADDR + 74*4
#define ACCEL_REG_DIV_IFM_C         ACCEL_REG_BASEADDR + 75*4

//SOFT RESET
#define ACCEL_REG_SOFT_RESET        ACCEL_REG_BASEADDR + 76*4

/*****************************ALU*******************************/
#define ALU_RESET     \
                                  Xil_Out32(ACCEL_REG_MATHFUNC_CTRL, 0); \
                                  Xil_Out32(ACCEL_REG_ACTFUNC_CTRL, 0); \
                                  Xil_Out32(ACCEL_REG_INNER_CTRL, 0);
#define ALU_MATH_RESET            Xil_Out32(ACCEL_REG_MATHFUNC_CTRL, 0)
#define ALU_ACT_RESET             Xil_Out32(ACCEL_REG_ACTFUNC_CTRL, 0)
#define ALU_INNERPROD_RESET       Xil_Out32(ACCEL_REG_INNER_CTRL, 0)
#define ALU_MATHFUNC_CTRL_SET(en,src_num,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,op) \
                                  Xil_Out32(ACCEL_REG_MATHFUNC_CTRL,en | src_num<<1 | mul_src1_sel<<3 | add_src0_sel<<6 | add_src1_sel<<9 | sub_en<<12 | add_en<<13 | mul_en<<14 | max_en<<15 | min_en<<16 | op<<23)
#define ALU_ACTFUNC_CTRL_SET(en,prop,src_sel,dst_sel,op) \
                                  Xil_Out32(ACCEL_REG_ACTFUNC_CTRL, en | prop<<1 | src_sel<<3 | dst_sel<<5 | op<<16)
#define ALU_INNERPROD_CTRL_SET(en) \
                                  Xil_Out32(ACCEL_REG_INNER_CTRL, en)
#define ALU_VECLEN_CH0_SET(x)     Xil_Out32(ACCEL_REG_ALU_VECLEN0, x)
#define ALU_SRC0_ADDR_CH0_SET(x)  Xil_Out32(ACCEL_REG_SRC0_ADDR0, x)
#define ALU_SRC1_ADDR_CH0_SET(x)  Xil_Out32(ACCEL_REG_SRC1_ADDR0, x)
#define ALU_DST_ADDR_CH0_SET(x)   Xil_Out32(ACCEL_REG_DST_ADDR0, x)
#define ALU_VECLEN_CH1_SET(x)     Xil_Out32(ACCEL_REG_ALU_VECLEN1, x)
#define ALU_SRC0_ADDR_CH1_SET(x)  Xil_Out32(ACCEL_REG_SRC0_ADDR1, x)
#define ALU_SRC1_ADDR_CH1_SET(x)  Xil_Out32(ACCEL_REG_SRC1_ADDR1, x)
#define ALU_DST_ADDR_CH1_SET(x)   Xil_Out32(ACCEL_REG_DST_ADDR1, x)
#define ALU_MATH_ALPHA_SET(x)     Xil_Out32(ACCEL_REG_MATH_ALPHA, x)
#define ALU_MATH_BETA_SET(x)      Xil_Out32(ACCEL_REG_MATH_BETA, x)

#define ALU_ACT_COEFF_A_0_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_A0, x)
#define ALU_ACT_COEFF_A_1_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_A1, x)
#define ALU_ACT_COEFF_A_2_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_A2, x)
#define ALU_ACT_COEFF_A_3_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_A3, x)
#define ALU_ACT_COEFF_A_4_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_A4, x)
#define ALU_ACT_COEFF_B_0_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_B0, x)
#define ALU_ACT_COEFF_B_1_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_B1, x)
#define ALU_ACT_COEFF_B_2_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_B2, x)
#define ALU_ACT_COEFF_B_3_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_B3, x)
#define ALU_ACT_COEFF_B_4_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_B4, x)
#define ALU_ACT_COEFF_C_0_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_C0, x)
#define ALU_ACT_COEFF_C_1_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_C1, x)
#define ALU_ACT_COEFF_C_2_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_C2, x)
#define ALU_ACT_COEFF_C_3_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_C3, x)
#define ALU_ACT_COEFF_C_4_SET(x)  Xil_Out32(ACCEL_REG_ACT_COEFF_C4, x)

#define ALU_ACT_RANGE_0_SET(x)    Xil_Out32(ACCEL_REG_ACT_RANG_0, x)
#define ALU_ACT_RANGE_1_SET(x)    Xil_Out32(ACCEL_REG_ACT_RANG_1, x)
#define ALU_ACT_RANGE_2_SET(x)    Xil_Out32(ACCEL_REG_ACT_RANG_2, x)
#define ALU_ACT_RANGE_3_SET(x)    Xil_Out32(ACCEL_REG_ACT_RANG_3, x)

/*****************************POOL*******************************/
#define POOL_RESET                Xil_Out32(ACCEL_REG_POOL_CTRL, 0)
#define POOL_CTRL_SET(en,op,kernel_w,kernel_h,stride_w,stride_h,src_sel,pad_bottom,pad_top,pad_right,pad_left,pad_mode) \
                                  Xil_Out32(ACCEL_REG_POOL_CTRL, en | op<<1 | kernel_w<<3 | kernel_h<<5 | stride_w<<7 | stride_h<<9 | src_sel<<11 | pad_bottom<<16 | pad_top<<18 | pad_right<<20 | pad_left<<22 | pad_mode<<24)
#define POOL_IFM_C_SET(c)         Xil_Out32(ACCEL_REG_POOL_SHAPE_IC, c)
#define POOL_IFM_WH_SET(w,h)      Xil_Out32(ACCEL_REG_POOL_SHAPE_IWH, w<<16 | h)
#define POOL_IFM_CSTEP_SET(x)     Xil_Out32(ACCEL_REG_POOL_SHAPE_ICSTEP, x)
#define POOL_OFM_C_SET(c)         Xil_Out32(ACCEL_REG_POOL_SHAPE_OC, c)
#define POOL_OFM_WH_SET(w,h)      Xil_Out32(ACCEL_REG_POOL_SHAPE_OWH, w<<16 | h)
#define POOL_OFM_CSTEP_SET(x)     Xil_Out32(ACCEL_REG_POOL_SHAPE_OCSTEP, x)
#define POOL_IFM_ADDR_SET(x)      Xil_Out32(ACCEL_REG_POOL_IFM_ADDR, x)
#define POOL_OFM_ADDR_SET(x)      Xil_Out32(ACCEL_REG_POOL_OFM_ADDR, x)
#define POOL_PAD_VALUE_SET(x)     Xil_Out32(ACCEL_REG_POOL_PAD_VALUE, x)
/*****************************GEMM*******************************/
#define GEMM_RESET                Xil_Out32(ACCEL_REG_GEMM_CTRL, 0)
#define GEMM_CTRL_SET(en,op,kernel,stride,pad_mode,pad_left,pad_right,pad_top,pad_bottom,bias_en,requant_en,layout_en,oscale_en,div_ifm_c_en) \
                                  Xil_Out32(ACCEL_REG_GEMM_CTRL, en | op<<1 | kernel<<3 | stride<<6 | pad_mode<<9 | pad_left<<11 | pad_right<<13 | pad_top<<15 | pad_bottom<<17 | bias_en<<19 | requant_en<<20 | layout_en<<21 | oscale_en<<22 | div_ifm_c_en<<23)
#define GEMM_QUANT_DATA_SET(x)    Xil_Out32(ACCEL_REG_QUANT_DATA, x)
#define GEMM_REQUANT_DATA_SET(x)  Xil_Out32(ACCEL_REG_REQUANT_DATA, x)
#define GEMM_DEQUANT_ADDR_SET(x)  Xil_Out32(ACCEL_REG_DEQUANT_ADDR, x)
#define GEMM_BIAS_ADDR_SET(x)     Xil_Out32(ACCEL_REG_BIAS_ADDR, x)
#define GEMM_IFM_C_SET(c)         Xil_Out32(ACCEL_REG_SHAPE_IFM_C, c)
#define GEMM_IFM_WH_SET(w,h)      Xil_Out32(ACCEL_REG_SHAPE_IFM_WH, w<<16 | h)
#define GEMM_IFM_CSTEP_SET(x)     Xil_Out32(ACCEL_REG_SHAPE_IFM_CSTEP, x)
#define GEMM_OFM_C_SET(c)         Xil_Out32(ACCEL_REG_SHAPE_OFM_C, c)
#define GEMM_OFM_WH_SET(w,h)      Xil_Out32(ACCEL_REG_SHAPE_OFM_WH, w<<16 | h)
#define GEMM_OFM_CSTEP_SET(x)     Xil_Out32(ACCEL_REG_SHAPE_OFM_CSTEP, x)
#define GEMM_WGT_LEN_SET(x)       Xil_Out32(ACCEL_REG_WGT_LEN, x)
#define GEMM_IFM_ADDR_SET(x)      Xil_Out32(ACCEL_REG_IFM_BASEADDR, x)
#define GEMM_WGT_ADDR_SET(x)      Xil_Out32(ACCEL_REG_WGT_BASEADDR, x)
#define GEMM_OFM_ADDR_SET(x)      Xil_Out32(ACCEL_REG_OFM_BASEADDR, x)
#define GEMM_DIV_IFM_C_SET(x)     Xil_Out32(ACCEL_REG_DIV_IFM_C, x)


/*****************************SOFT RESET*******************************/
#define SOFT_RESET                Xil_Out32(ACCEL_REG_SOFT_RESET, 1)
#define SOFT_RESET_CTRL_SET       Xil_Out32(ACCEL_REG_SOFT_RESET, 0)

#endif
