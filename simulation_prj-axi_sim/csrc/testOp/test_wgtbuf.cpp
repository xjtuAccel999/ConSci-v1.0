#include "../hw/hw_gemm.h"
#include "test_op.h"
#include "../testLayer/test_layer.h"
#include "../utils/utils.h"
#include "../utils/wgt_reorder.h"



namespace test {

void test_wgtbuf(const ncnn::Mat& in, int ic, int ow, int oh, int oc, int kernel, int op=0){
    accel::hw_gemm inst;
    inst.ifm_c = ic;
    inst.ofm_w = ow;
    inst.ofm_h = oh;
    inst.ofm_c = oc;
    inst.kernel = kernel;
    inst.op = op;
    int ic_align32 = u_align(ic, 32);
    ncnn::Option opt;
    ncnn::Mat &wgt = const_cast<ncnn::Mat &>(in);

    int oc_align = (op == 0) ? u_align(oc, 32) : 1;

    ncnn::Mat wgt_buffer_res(oc_align, 1, ic_align32 * kernel * kernel, 1u, opt.blob_allocator);
    ncnn::Mat wgt_buffer(32, ic_align32 * kernel * kernel * oc_align / 32, 1u, opt.blob_allocator);

    wgt_reorder(wgt, wgt_buffer_res, inst.ifm_c, inst.ofm_c, inst.kernel, 0, op);
    wgt_gen(wgt_buffer_res, wgt_buffer, inst.ifm_c, inst.ofm_c, inst.kernel, op);

#ifdef MAT_LOG
    log_mat_file<unsigned char>(wgt_buffer_res, (char *)"./log/wgtbuf/log_wgt_buffer_res.txt", -1, 1, 0);
    log_mat_file<unsigned char>(wgt_buffer, (char *)"./log/wgtbuf/log_wgt_buffer.txt", -1, 1, 0);
#endif
    ncnn::Mat ifm(ow, oh, ic, 1u, opt.blob_allocator);
    ncnn::Mat ofm(ow, oh, oc, 1u, opt.blob_allocator);
    inst.gemm_forward_s(ifm, wgt_buffer, ofm);
#ifdef TEST_WGTBUF
    top->io_wgt_odata_ready = 1;
#endif
    if(op==0){
    dma_wait(wgt_buffer_res, inst);
    }else if(op==1){
    dma_wait(wgt_buffer, inst);
    }
    GEMM_RESET;
}

void test_wgtbuf_single(int ow, int oh, int oc, int ic, int kernel, int op = 0) {
    if (op == 0) {
        assert(u_align(ic, 32) * kernel * kernel <= WGT_BUFFER_DEPTH);
        printf("[ log ]: ow = %d, oh = %d, oc = %d, ic = %d, kernel = %d\n", ow, oh, oc, ic, kernel);
    } else if (op == 1) {
        assert(oc == ic);
        assert(u_align(ic, 32) / 32 * kernel * kernel <= WGT_BUFFER_DEPTH);
        printf("[ log ]:(dw comv) ow = %d, oh = %d, oc = %d, ic = %d, kernel = %d\n", ow, oh, oc, ic, kernel);
    }
    test_wgtbuf(RandomS8Mat(oc * ic * kernel * kernel), ic, ow, oh, oc, kernel, op);
}

void test_wgtbuf_batch(){
    SRAND(7767517);
    // test_wgtbuf_single(int ow, int oh, int oc, int ic, int kernel, int op)
    // test_wgtbuf_single(    13,     13,     32,     32,          3);
    // test_wgtbuf_single(    64,     64,     32,    128,          1);
    // test_wgtbuf_single(    52,     52,     64,     32,          7);
    // test_wgtbuf_single(    13,     13,     64,    128,          3);
    // test_wgtbuf_single(    64,     64,    256,     32,          1);
    // test_wgtbuf_single(    28,     28,    256,     32,          3);
    // test_wgtbuf_single(    10,     15,     64,    256,          3);
    // test_wgtbuf_single(    13,     13,     26,     23,          3);
    // test_wgtbuf_single(    64,     64,     28,     88,          1);
    // test_wgtbuf_single(    52,     52,     48,     17,          7);
    // test_wgtbuf_single(    13,     13,     45,     90,          3);
    // test_wgtbuf_single(    64,     64,    200,     15,          1);
    // test_wgtbuf_single(    28,     28,    176,     15,          3);
    // test_wgtbuf_single(    10,     15,     44,    176,          3);
    // test_wgtbuf_single(    10,     15,     44,    290,          3);         
    // test_wgtbuf_single(    13,     13,      8,    512,          2);        
    // test_wgtbuf_single(    13,     13,      8,    340,          3);         
    // test_wgtbuf_single(    13,     13,      8,    450,          3);         
    // test_wgtbuf_single(    13,     13,      8,    390,          3);         
    // test_wgtbuf_single(    13,     13,      8,    500,          3);         
    // test_wgtbuf_single(    13,     13,      8,    512,          3);         
    // test_wgtbuf_single(    64,     64,    200,   1024,          1);
    
    // test_wgtbuf_single(int ow, int oh, int oc, int ic, int kernel, int op)    // dw conv test
    // test_wgtbuf_single(    13,     13,     32,     32,          3,      1);
    // test_wgtbuf_single(    64,     64,    128,    128,          1,      1);
    // test_wgtbuf_single(    52,     52,     32,     32,          7,      1);
    // test_wgtbuf_single(    13,     13,    128,    128,          3,      1);
    // test_wgtbuf_single(    64,     64,     32,     32,          1,      1);
    // test_wgtbuf_single(    28,     28,     32,     32,          3,      1);
    // test_wgtbuf_single(    10,     15,    256,    256,          3,      1);
    // test_wgtbuf_single(    13,     13,   1024,   1024,          3,      1);
    // test_wgtbuf_single(    13,     13,     23,     23,          3,      1);
    // test_wgtbuf_single(    64,     64,     88,     88,          1,      1);
    // test_wgtbuf_single(    52,     52,     17,     17,          7,      1);
    // test_wgtbuf_single(    13,     13,     90,     90,          3,      1);
    // test_wgtbuf_single(    64,     64,     15,     15,          1,      1);
    // test_wgtbuf_single(    28,     28,     15,     15,          3,      1);
    // test_wgtbuf_single(    10,     15,    176,    176,          3,      1);
    // test_wgtbuf_single(    13,     13,    567,    567,          3,      1);
    // test_wgtbuf_single(    13,     13,    867,    867,          3,      1);
    // test_wgtbuf_single(    13,     13,   1024,   1024,          3,      1);

    // test_wgtbuf_single(    13,     13,      8,    567,          3);      // oversize
    // test_wgtbuf_single(    13,     13,    689,    867,          3);
    // test_wgtbuf_single(    13,     13,   1024,   1024,          3);
    // test_wgtbuf_single(    13,     13,     32,   1024,          3);

}
}
