#include "u_weightorder.h"

namespace accel {

weightorder::weightorder(int ifm_c, int ofm_c, int kernel)
{
    this->ifm_c = ifm_c;
    this->ofm_c = ofm_c;
    this->kernel = kernel;
}

void weightorder::wgt_reorder(ncnn::Mat &i_data, ncnn::Mat &o_data, int ifm_c, int ofm_c, int kernel, int flip_en = 0, int op = 0) {
#ifdef MAT_LOG
    log_origin_wgt(i_data, inst);
#endif
    int ifm_c_align32 = ncnn::alignSize(ifm_c, 32);
    int maxk = kernel * kernel;
    int oc_step = maxk * ifm_c;
    ncnn::Option opt;
    if (op == 0) {
        // first step: maxk*ic*oc -> (oc,1,maxk*ic)
        ncnn::Mat wgt_buffer(ncnn::alignSize(ofm_c, 32), 1, ifm_c_align32 * maxk, 1u, opt.blob_allocator);
        char *wgt_i_ptr = i_data;
        char *wgt_o_ptr = flip_en ? wgt_buffer : o_data;
        memset(wgt_o_ptr, 0, sizeof(char) * o_data.cstep * ifm_c_align32 * maxk);

        for (int k = 0; k < maxk; k++) {
            for (int ic = 0; ic < ifm_c; ic++) {
                wgt_o_ptr = flip_en ? wgt_buffer.channel(ifm_c_align32 * k + ic) : o_data.channel(ifm_c_align32 * k + ic);
                for (int oc = 0; oc < ofm_c; oc++)
                    *wgt_o_ptr++ = wgt_i_ptr[oc * oc_step + ic * maxk + k];
            }
        }
#ifdef MAT_LOG
        log_mat_file<unsigned char>(wgt_buffer, (char *)"./log/log_before_reorder_wgt.txt");
#endif
        // second step: small block flip32
        if (flip_en) {
            for (int i = 0; i < ncnn::alignSize(ifm_c, 32) * maxk / 32; i++) {
                for (int k = 0; k < 32; k++) {
                    char *wgt_i_ptr = wgt_buffer.channel(i * 32 + k);
                    char *wgt_o_ptr = o_data.channel(i * 32 + 31 - k);
                    memcpy(wgt_o_ptr, wgt_i_ptr, sizeof(char) * ncnn::alignSize(ofm_c, 32));
                }
            }
        }
    } else if (op == 1) {
        // first step: maxk*ic -> (1,1,maxk*ic)
        char *wgt_i_ptr = i_data;
        char *wgt_o_ptr = o_data;
        memset(wgt_o_ptr, 0, sizeof(char) * o_data.cstep * ifm_c_align32 * maxk);

        for (int k = 0; k < maxk; k++) {
            for (int ic = 0; ic < ifm_c; ic++) {
                wgt_o_ptr = o_data.channel(ifm_c_align32 * k + ic);
                *wgt_o_ptr = wgt_i_ptr[ic * maxk + k];
            }
        }
    }

#ifdef MAT_LOG
    log_mat_file<unsigned char>(o_data, (char *)"./log/log_reorder_wgt.txt");
#endif
}

void weightorder::wgt_gen(ncnn::Mat &wgt_buffer_res, ncnn::Mat &wgt_buffer, int ifm_c, int ofm_c, int kernel, int op=0, int oc_segmentation=0) {
    int oc = ofm_c;
    int oc_align = ncnn::alignSize(oc, 32);
    int k2ic_align = kernel * kernel * ncnn::alignSize(ifm_c, 32);
    if (op == 0) {
        unsigned char *o_data_ptr = wgt_buffer;
        memset(wgt_buffer, 0, oc_align * k2ic_align);
        if (oc_segmentation == 1 && k2ic_align > WGT_BUFFER_DEPTH) {
            int ic_block = WGT_BUFFER_DEPTH / (kernel * kernel) / 32 * 32;
            int ic_segment = ifm_c / ic_block;
            int ic_remainder = ifm_c % ic_block;

            for (int i = 0; i < ic_segment; i++) {
                for (int j = 0; j < oc_align / 32; j++) {
                    for (int m = 0; m < kernel * kernel; m++) {
                        for (int n = 0; n < ic_block; n++) {
                            unsigned char *i_data_ptr = wgt_buffer_res.channel(n + m * ncnn::alignSize(ifm_c, 32) + i * ic_block);
                            memcpy(o_data_ptr, i_data_ptr + j * 32, 32);
                            o_data_ptr += 32;
                        }
                    }
                }
            }
            for (int i = 0; i < oc_align / 32; i++) {
                for (int j = 0; j < kernel * kernel; j++) {
                    for (int m = 0; m < ncnn::alignSize(ic_remainder, 32); m++) {
                        unsigned char *i_data_ptr = wgt_buffer_res.channel(m + j * ncnn::alignSize(ifm_c, 32) + ic_segment * ic_block);
                        memcpy(o_data_ptr, i_data_ptr + i * 32, 32);
                        o_data_ptr += 32;
                    }
                }
            }

        } else {
            for (int i = 0; i < oc_align / 32; i++) {
                for (int j = 0; j < k2ic_align; j++) {
                    unsigned char *i_data_ptr = wgt_buffer_res.channel(j);
                    memcpy(o_data_ptr, i_data_ptr + i * 32, 32);
                    o_data_ptr += 32;
                }
            }
        }

    } else if (op == 1) {
        unsigned char *o_data_ptr = wgt_buffer;
        unsigned char *i_data_ptr = wgt_buffer_res;
        memset(wgt_buffer, 0, k2ic_align);
        if (oc_segmentation == 1 && oc_align > OSCALE_BUFFER_DEPTH) {
            int oc_segment = oc_align / OSCALE_BUFFER_DEPTH;
            int oc_remainder = oc_align % OSCALE_BUFFER_DEPTH;
            for (int k = 0; k < oc_segment; k++) {
                for (int j = 0; j < kernel * kernel; j++) {
                    i_data_ptr = (unsigned char *)wgt_buffer_res.data + (oc_align * j + k * OSCALE_BUFFER_DEPTH) * wgt_buffer_res.cstep;
                    for (int i = 0; i < OSCALE_BUFFER_DEPTH; i++) {
                        memcpy(o_data_ptr++, i_data_ptr, 1);
                        i_data_ptr += wgt_buffer_res.cstep;
                    }
                }
            }
            for (int j = 0; j < kernel * kernel; j++) {
                i_data_ptr = (unsigned char *)wgt_buffer_res.data + (oc_align * j + oc_segment * OSCALE_BUFFER_DEPTH) * wgt_buffer_res.cstep;
                for (int i = 0; i < oc_remainder; i++) {
                    memcpy(o_data_ptr++, i_data_ptr, 1);
                    i_data_ptr += wgt_buffer_res.cstep;
                }
            }
        } else {
            for (int i = 0; i < k2ic_align; i++) {
                memcpy(o_data_ptr++, i_data_ptr, 1);
                i_data_ptr += wgt_buffer_res.cstep;
            }
        }
    }
}

} // namespace accel