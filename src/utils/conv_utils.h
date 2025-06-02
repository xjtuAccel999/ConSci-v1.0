#ifndef __CONV_UTILS_H_
#define __CONV_UTILS_H_

#include "../testLayer/test_layer.h"

static void conv_get_ofm_pad(accel::hw_gemm &inst) {
    inst.ofm_w = (inst.ifm_w - inst.kernel + inst.padding_left + inst.padding_right) / inst.stride + 1;
    inst.ofm_h = (inst.ifm_h - inst.kernel + inst.padding_top + inst.padding_bottom) / inst.stride + 1;
    if (inst.stride != 1 && inst.kernel >= inst.stride) {
        inst.padding_right = (inst.ofm_w - 1) * inst.stride + inst.kernel - inst.ifm_w - inst.padding_left;
        inst.padding_bottom = (inst.ofm_h - 1) * inst.stride + inst.kernel - inst.ifm_h - inst.padding_top;
        if (inst.padding_right < 0) {
            inst.ofm_w++;
            inst.padding_right = (inst.ofm_w - 1) * inst.stride + inst.kernel - inst.ifm_w - inst.padding_left;
        }
        if (inst.padding_bottom < 0) {
            inst.ofm_h++;
            inst.padding_bottom = (inst.ofm_h - 1) * inst.stride + inst.kernel - inst.ifm_h - inst.padding_top;
        }
    }
}

#endif