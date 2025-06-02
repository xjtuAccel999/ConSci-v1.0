#include "hw_act.h"
// #include "../veri.h"
#include "accel_params.h"
#include "accel.h"
#include "xtime_l.h"
namespace accel {
hw_act::hw_act() {
    this->act_op =0;
    this->act_alpha = 0.0f;
    this->act_beta = 0.f;
}

void hw_act::cfg_param(){
    switch (act_op)
    {
    case HW_TANH:  //Tanh
        act_prop = 1;
        range[0] = 1.0f;
        range[1] = 2.0f;
        range[2] = 4.0f;
        range[3] = 8.0f;
        coffe_a[0] = -0.33031377;
        coffe_a[1] = -0.16859024;
        coffe_a[2] = -0.0128121;
        coffe_a[3] = -6.87e-05;
        coffe_a[4] = 0.0f;
        coffe_b[0] = 1.10203762;
        coffe_b[1] = 0.69955814;
        coffe_b[2] = 0.09117782;
        coffe_b[3] = 9.18e-04;
        coffe_b[4] = 0.0f;
        coffe_c[0] = -0.00713339;
        coffe_c[1] = 0.23526194;
        coffe_c[2] = 0.83713885;
        coffe_c[3] = 9.97e-01;
        coffe_c[4] = 0.0f;
        break;
    case HW_SIGMOID: //Sigmoid
        act_prop = 2;
        range[0] = 1.0f;
        range[1] = 2.0f;
        range[2] = 5.0f;
        range[3] = 8.0f;
        coffe_a[0] = -0.02786673;
        coffe_a[1] = -0.04669766;
        coffe_a[2] = -0.01474262;
        coffe_a[3] = -0.000872138178;
        coffe_a[4] = 0.0f;
        coffe_b[0] = 0.26049149;
        coffe_b[1] = 0.28960504;
        coffe_b[2] = 0.13705577;
        coffe_b[3] = 0.0131977678;
        coffe_b[4] = 0.0f;
        
        coffe_c[0] = 0.49915767;
        coffe_c[1] = 0.48821998;
        coffe_c[2] = 0.67188801;
        coffe_c[3] = 0.949589803;
        coffe_c[4] = 1.0f;
        break;
    case HW_SWISH: //Swish
        act_prop = 0;
        range[0] = -8.0;
        range[1] = -2.0;
        range[2] = 2.0;
        range[3] = 8.0;
        coffe_a[0] = 0.0f;
        coffe_a[1] =-0.00999307;
        coffe_a[2] = 0.19631972;
        coffe_a[3] =-0.00999307;
        coffe_a[4] = 0.0f;
        coffe_b[0] = 0.0f;
        coffe_b[1] =-0.13541545;
        coffe_b[2] = 0.5;
        coffe_b[3] = 1.13541545;
        coffe_b[4] = 1.0f;
        coffe_c[0] = 0.0f;
        coffe_c[1] =-0.46091293;
        coffe_c[2] = 0.01944111;
        coffe_c[3] =-0.46091293;
        coffe_c[4] = 0.0f;
        break;
    case HW_RELU: //relu
        act_prop = 0;
        range[0] = 0.0f;
        range[1] = 1.0f;
        range[2] = 2.0f;
        range[3] = 3.0f;
        coffe_a[0] = 0.0f;
        coffe_a[1] = 0.0f;
        coffe_a[2] = 0.0f;
        coffe_a[3] = 0.0f;
        coffe_a[4] = 0.0f;
        coffe_b[0] = 0.0f;
        coffe_b[1] = 1.0f;
        coffe_b[2] = 1.0f;
        coffe_b[3] = 1.0f;
        coffe_b[4] = 1.0f;
        coffe_c[0] = 0.0f;
        coffe_c[1] = 0.0f;
        coffe_c[2] = 0.0f;
        coffe_c[3] = 0.0f;
        coffe_c[4] = 0.0f;
        break;
    case HW_LEAKYRELU: //Leaky relu
        act_prop = 0;
        range[0] = 0.0f;
        range[1] = 1.0f;
        range[2] = 2.0f;
        range[3] = 3.0f;
        coffe_a[0] = 0.0f;
        coffe_a[1] = 0.0f;
        coffe_a[2] = 0.0f;
        coffe_a[3] = 0.0f;
        coffe_a[4] = 0.0f;
        coffe_b[0] = act_alpha;
        coffe_b[1] = 1.0f;
        coffe_b[2] = 1.0f;
        coffe_b[3] = 1.0f;
        coffe_b[4] = 1.0f;
        coffe_c[0] = 0.0f;
        coffe_c[1] = 0.0f;
        coffe_c[2] = 0.0f;
        coffe_c[3] = 0.0f;
        coffe_c[4] = 0.0f;
        break;
    case HW_PRELU: //Prelu
        act_prop = 0;
        range[0] = 0.0f;
        range[1] = 1.0f;
        range[2] = 2.0f;
        range[3] = 3.0f;
        coffe_a[0] = 0.0f;
        coffe_a[1] = 0.0f;
        coffe_a[2] = 0.0f;
        coffe_a[3] = 0.0f;
        coffe_a[4] = 0.0f;
        coffe_b[0] = act_alpha;
        coffe_b[1] = 1.0f;
        coffe_b[2] = 1.0f;
        coffe_b[3] = 1.0f;
        coffe_b[4] = 1.0f;
        coffe_c[0] = 0.0f;
        coffe_c[1] = 0.0f;
        coffe_c[2] = 0.0f;
        coffe_c[3] = 0.0f;
        coffe_c[4] = 0.0f;
        break;
    case HW_ELU: //ELU
        act_prop = 0;
        range[0] =-8.0f;
        range[1] =-3.0f;
        range[2] =-1.0f;
        range[3] = 0.0f;
        coffe_a[0] = 0.0f * act_alpha;
        coffe_a[1] = 0.00312992f * act_alpha;
        coffe_a[2] = 0.07263736f * act_alpha;
        coffe_a[3] = 0.30871853f * act_alpha;
        coffe_a[4] = 0.0f ;
        coffe_b[0] = 0.0f * act_alpha;
        coffe_b[1] = 0.04171119f * act_alpha;
        coffe_b[2] = 0.43991065f * act_alpha;
        coffe_b[3] = 0.93054848f * act_alpha;
        coffe_b[4] = 1.0f ;
        coffe_c[0] =-1.0f * act_alpha;
        coffe_c[1] =-0.86189893f * act_alpha;
        coffe_c[2] =-0.27589442f * act_alpha;
        coffe_c[3] =-0.00551138f * act_alpha;
        coffe_c[4] = 0.0f ;
        break;
    case HW_SELU : //SELU
        act_prop = 0;
        range[0] =-8.0f;
        range[1] =-3.0f;
        range[2] =-1.0f;
        range[3] = 0.0f;
        coffe_a[0] = 0.0f * act_alpha * act_beta;
        coffe_a[1] = 0.00312992f * act_alpha * act_beta;
        coffe_a[2] = 0.07263736f * act_alpha * act_beta; 
        coffe_a[3] = 0.30871853f * act_alpha * act_beta;
        coffe_a[4] = 0.0f *act_beta;
        coffe_b[0] = 0.0f * act_alpha * act_beta;
        coffe_b[1] = 0.04171119f * act_alpha * act_beta;
        coffe_b[2] = 0.43991065f * act_alpha * act_beta;
        coffe_b[3] = 0.93054848f * act_alpha * act_beta;
        coffe_b[4] = 1.0f * act_beta;
        coffe_c[0] =-1.0f * act_alpha * act_beta;
        coffe_c[1] =-0.86189893f * act_alpha * act_beta; 
        coffe_c[2] =-0.27589442f * act_alpha * act_beta;
        coffe_c[3] =-0.00551138f * act_alpha * act_beta;
        coffe_c[4] = 0.0f * act_beta;
        break;
    case HW_CLIP: //CLIP
        act_prop = 0;
        range[0] = act_alpha;
        range[1] = act_alpha;
        range[2] = act_alpha;
        range[3] = act_beta ;
        coffe_a[0] = 0.0f;
        coffe_a[1] = 0.0f;
        coffe_a[2] = 0.0f;
        coffe_a[3] = 0.0f;
        coffe_a[4] = 0.0f;
        coffe_b[0] = 0.0f;
        coffe_b[1] = 1.0f;
        coffe_b[2] = 1.0f;
        coffe_b[3] = 1.0f;
        coffe_b[4] = 0.0f;
        coffe_c[0] = act_alpha;
        coffe_c[1] = 0.0f;
        coffe_c[2] = 0.0f;
        coffe_c[3] = 0.0f;
        coffe_c[4] = act_beta;
        break;
    case HW_HARDSIGMOID: //HardSigmoid
        act_prop = 0;
        range[0] = static_cast<float>(-(act_beta/act_alpha));
        range[1] = 0.0f;
        range[2] = static_cast<float>((1-act_beta)/act_alpha);
        range[3] = 10.0f;
        coffe_a[0] = 0.0f;
        coffe_a[1] = 0.0f;
        coffe_a[2] = 0.0f;
        coffe_a[3] = 0.0f;
        coffe_a[4] = 0.0f;
        coffe_b[0] = 0.0f;
        coffe_b[1] = act_alpha;
        coffe_b[2] = act_alpha;
        coffe_b[3] = 0.0f;
        coffe_b[4] = 0.0f;
        coffe_c[0] = 0.0f;
        coffe_c[1] = act_beta;
        coffe_c[2] = act_beta;
        coffe_c[3] = 1.0f;
        coffe_c[4] = 1.0f;
        break;
    case HW_HARDSWISH: //HardSwish
        act_prop = 0;
        range[0] = static_cast<float>(-(act_beta/act_alpha));
        range[1] = 0.0f;
        range[2] = static_cast<float>((1.f-act_beta)/act_alpha);
        range[3] = 10.0f;
        coffe_a[0] = 0.0f;
        coffe_a[1] = act_alpha;
        coffe_a[2] = act_alpha;
        coffe_a[3] = 0.0f;
        coffe_a[4] = 0.0f;
        coffe_b[0] = 0.0f;
        coffe_b[1] = act_beta;
        coffe_b[2] = act_beta;
        coffe_b[3] = 1.0f;
        coffe_b[4] = 1.0f;
        coffe_c[0] = 0.0f;
        coffe_c[1] = 0.0f;
        coffe_c[2] = 0.0f;
        coffe_c[3] = 0.0f;
        coffe_c[4] = 0.0f;
        break;
    
    default:
        act_prop = 0;
        range[0] = 0.0f;
        range[1] = 0.0f;
        range[2] = 0.0f;
        range[3] = 0.0f;
        coffe_a[0] = 0.0f;
        coffe_a[1] = 0.0f;
        coffe_a[2] = 0.0f;
        coffe_a[3] = 0.0f;
        coffe_a[4] = 0.0f;
        coffe_b[0] = 0.0f;
        coffe_b[1] = 0.0f;
        coffe_b[2] = 0.0f;
        coffe_b[3] = 0.0f;
        coffe_b[4] = 0.0f;
        coffe_c[0] = 0.0f;
        coffe_c[1] = 0.0f;
        coffe_c[2] = 0.0f;
        coffe_c[3] = 0.0f;
        coffe_c[4] = 0.0f;
        break;
    }
}

void hw_act::send_coffe(){
    ALU_ACT_COEFF_A_0_SET(((uint32_t*)coffe_a)[0]);
    ALU_ACT_COEFF_A_1_SET(((uint32_t*)coffe_a)[1]);
    ALU_ACT_COEFF_A_2_SET(((uint32_t*)coffe_a)[2]); 
    ALU_ACT_COEFF_A_3_SET(((uint32_t*)coffe_a)[3]); 
    ALU_ACT_COEFF_A_4_SET(((uint32_t*)coffe_a)[4]); 
        
    ALU_ACT_COEFF_B_0_SET(((uint32_t*)coffe_b)[0]);
    ALU_ACT_COEFF_B_1_SET(((uint32_t*)coffe_b)[1]);
    ALU_ACT_COEFF_B_2_SET(((uint32_t*)coffe_b)[2]); 
    ALU_ACT_COEFF_B_3_SET(((uint32_t*)coffe_b)[3]); 
    ALU_ACT_COEFF_B_4_SET(((uint32_t*)coffe_b)[4]); 

    ALU_ACT_COEFF_C_0_SET(((uint32_t*)coffe_c)[0]);
    ALU_ACT_COEFF_C_1_SET(((uint32_t*)coffe_c)[1]);
    ALU_ACT_COEFF_C_2_SET(((uint32_t*)coffe_c)[2]); 
    ALU_ACT_COEFF_C_3_SET(((uint32_t*)coffe_c)[3]); 
    ALU_ACT_COEFF_C_4_SET(((uint32_t*)coffe_c)[4]);

    ALU_ACT_RANGE_0_SET(((uint32_t*)range)[0]);
    ALU_ACT_RANGE_1_SET(((uint32_t*)range)[1]);
    ALU_ACT_RANGE_2_SET(((uint32_t*)range)[2]);
    ALU_ACT_RANGE_3_SET(((uint32_t*)range)[3]);
}

void hw_act::act_forward(ncnn::Mat &src0, ncnn::Mat &dst) {


    #ifdef ALU_TIME
		u64 tEnd_alu_hw, tCur_alu_hw;
		u32 tUsed_alu_hw;
        XTime_GetTime(&tCur_alu_hw);
    #endif
    cfg_param();

    ALU_ACT_RESET;
    Xil_DCacheFlush();
    int *src0_addr0 = src0;
    int *dst_addr0 = dst;
    int len = src0.total();

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = src0_addr0 + len_ch0;
    int *dst_addr1 = dst_addr0 + len_ch0;

    send_coffe();

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint64_t)src0_addr0);
    ALU_DST_ADDR_CH0_SET((uint64_t)dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint64_t)src0_addr1);
    ALU_DST_ADDR_CH1_SET((uint64_t)dst_addr1);

    ALU_ACTFUNC_CTRL_SET(1, act_prop, ACT_FROM_DMA, ACT_TO_DMA, act_op);

    wait_alu_mat_done();
    Xil_DCacheInvalidate();
    ALU_ACT_RESET;
    #ifdef ALU_TIME
		XTime_GetTime(&tEnd_alu_hw);
		tUsed_alu_hw = ((tEnd_alu_hw-tCur_alu_hw)*1000000)/(COUNTS_PER_SECOND);
		printf("time_alu_hw elapsed is %d us\n",tUsed_alu_hw);
    #endif
}

void hw_act::act_forward(float* src_dst_ptr, int len) {

    cfg_param();

    ALU_ACT_RESET;
    Xil_DCacheFlush();
    int *src0_dst_addr0 = (int*)src_dst_ptr;

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = len - len_ch0;
    int *src0_dst_addr1 = src0_dst_addr0 + len_ch0;

    send_coffe();

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint64_t)src0_dst_addr0);
    ALU_DST_ADDR_CH0_SET((uint64_t)src0_dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint64_t)src0_dst_addr1);
    ALU_DST_ADDR_CH1_SET((uint64_t)src0_dst_addr1);

    ALU_ACTFUNC_CTRL_SET(1, act_prop, ACT_FROM_DMA, ACT_TO_DMA, act_op);

    wait_alu_mat_done();
	Xil_DCacheInvalidate();
    ALU_ACT_RESET;
}

} // namespace accel