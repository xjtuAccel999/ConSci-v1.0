#include "hw_pool.h"
#include "../veri.h"
#include "accel_params.h"

namespace accel {

hw_pool::hw_pool() {
    this->pool_type = POOL_MAX;
    this->channels = DOUBLE_CHANNELS;
    this->kernel_w = 2;
    this->kernel_h = 2;
    this->stride_w = 2;
    this->stride_h = 2;
    this->pad_mode = 0;
    this->pad_left = 0;
    this->pad_right = 0;
    this->pad_top = 0;
    this->pad_bottom = 0;
    this->pad_value = 0.f;
}

void hw_pool::forward(ncnn::Mat &ifm, ncnn::Mat &ofm) {
    uint64_t start_time = main_time;
    POOL_RESET;
    int* src0_addr0 = ifm;
    int* dst_addr0 = ofm;

    POOL_IFM_ADDR_SET((uint64_t)src0_addr0);
    POOL_OFM_ADDR_SET((uint64_t)dst_addr0);
    POOL_IFM_WH_SET(ifm.w,ifm.h);
    POOL_IFM_C_SET(ifm.c);
    POOL_IFM_CSTEP_SET(ifm.cstep);
    POOL_OFM_WH_SET(ofm.w,ofm.h);
    POOL_OFM_C_SET(ofm.c);
    POOL_OFM_CSTEP_SET(ofm.cstep);
    POOL_PAD_VALUE_SET(*((uint32_t *)&pad_value));
    POOL_CTRL_SET(1, pool_type,kernel_w,kernel_h,stride_w,stride_h,1,pad_bottom,pad_top,pad_right,pad_left,pad_mode);

    dma_wait();

    POOL_RESET;
    time_pool += main_time - start_time;
}

} // namespace accel