#ifndef __WEIGHT_ORDER_H_
#define __WEIGHT_ORDER_H_

// ncnn public header
#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

#define u_align(x, n) ((x + n - 1) & -n)

namespace accel {

class weightorder
{
public:
    int kernel;
    int ifm_c;
    int ofm_c;

public:
    weightorder(int ifm_c, int ofm_c, int kernel);
    void wgt_reorder(ncnn::Mat &i_data, ncnn::Mat &o_data, int ifm_c, int ofm_c, int kernel, int flip_en, int op);
    void wgt_gen(ncnn::Mat &wgt_buffer_res, ncnn::Mat &wgt_buffer, int ifm_c, int ofm_c, int kernel, int op, int oc_segmentation);
};

} // namespace accel

#endif