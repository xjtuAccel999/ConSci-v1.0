#include "hw_innerprod.h"
#include "../veri.h"
#include "accel_params.h"

namespace accel {

hw_innerprod::hw_innerprod() {
    this->math_alpha = 0.f;
    this->math_beta = 0.f;
}

float hw_innerprod::innerprod_get_data(){
    float odata =  *((float*)(&top->io_sim_alu_o_reg));
    ALU_RESET;
    return odata;

}

void hw_innerprod::innerprod_forward(ncnn::Mat &src0, void* wgt_addr) {

    cfg_param();
    send_coffe();

    ALU_RESET;
    int *src0_addr0 = src0;
    int *src1_addr0 = (int*)wgt_addr;
    int len = u_align(src0.total(),16)/4;

    int len_ch0 = u_align(len / 2, 4);
    int len_ch1 = len - len_ch0;
    int *src0_addr1 = src0_addr0 + len_ch0;
    int *src1_addr1 = src1_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&math_alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&math_beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint64_t)src0_addr0);
    ALU_SRC1_ADDR_CH0_SET((uint64_t)src1_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint64_t)src0_addr1);
    ALU_SRC1_ADDR_CH1_SET((uint64_t)src1_addr1);

    ALU_MATHFUNC_CTRL_SET(1,2,MUL_SRC1_FROM_ALPHA,ADD_SRC0_FROM_MUL_O,ADD_SRC1_FROM_BETA,0,1,1,0,0,0);
    
    ALU_ACTFUNC_CTRL_SET(1, act_prop, 0, 0, act_op);
    ALU_INNERPROD_CTRL_SET(1);

    dma_wait();
    ALU_RESET;
}

} // namespace accel