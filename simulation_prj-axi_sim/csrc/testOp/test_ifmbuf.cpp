#include "../hw/hw_gemm.h"
#include "test_op.h"
#include "../testLayer/test_layer.h"
#include "../utils/utils.h"
#include <iostream>


namespace test{


int read_ifm(ncnn::Mat& gemm_ifm_buffer, int line_len){
    #ifdef TEST_IFMBUF
    update();
    top->io_ifm_mem_read_port0_ren = 1;
    top->io_ifm_mem_read_port0_raddr = 0;
    update();
    update();
    update();
    printf("[ log ]: line_len is %d\n",line_len);
    
    for(int depth=0; depth<line_len; depth++){  
        unsigned char* ifm_mem_ptr = gemm_ifm_buffer.channel(depth);
        for(int id=0; id<32; id++){
            unsigned char* top_ifm_mem = (unsigned char*)&top->io_ifm_mem_read_port0_rdata;
            if(top_ifm_mem[id] != *ifm_mem_ptr){
                printf("[error]: ------------------------------------------\n");
                printf("[error]: the sw ifm_mem is not equal the hw ifm_mem\n");
                printf("[error]: the i: %d;\t the id is %d\n",depth,id);
                printf("[error]: top->io_ifm_r_rdata[id]: %x;\t *ifm_mem_ptr: %x\n",top_ifm_mem[id],*ifm_mem_ptr);
				printf("maintime:%ld\n", main_time);
				update();
                // return -1;
            }

            ifm_mem_ptr++;     
        }
        top->io_ifm_mem_read_port0_raddr += 1;
        update();
    }
    GEMM_RESET;
    for(int i=0; i<100; i++)
        update();

    #endif
    return 0;
}

void test_ifmbuf_single(int layout_en, const ncnn::Mat& in){
    ncnn::Option opt;    
    ncnn::Mat gemm_ifm_buffer(IFM_BUFFER_WIDTH/8,1,IFM_BUFFER_DEPTH,1u,opt.blob_allocator);
    accel::hw_gemm inst;
    inst.layout_en = layout_en;
    inst.quant_scale = 0.5f;
    ncnn::Mat& data_in = const_cast<ncnn::Mat&>(in);

    ncnn::Mat quant_ifm;
    printf("[ log ]: the input data shape is (%d,%d,%d)\n",data_in.w,data_in.h,data_in.c);
    if(inst.layout_en == 0){ 
        inst.ifm_w = data_in.h;
        inst.ifm_h = data_in.c;
        inst.ifm_c = data_in.w;
        printf("[ log ]: the current data not use layout\n");
    }
    else{
        inst.ifm_w = data_in.w;
        inst.ifm_h = data_in.h;
        inst.ifm_c = data_in.c;
        printf("[ log ]: the current data use layout\n");
    }
    
    float mem_bytes = IFM_BUFFER_DEPTH*IFM_BUFFER_WIDTH/8;
    printf("[ log ]: the ifm_mem is %.2f MB\n",mem_bytes/1024.f/1024.f);
    float ifm_mem_size = ((float)(u_align(inst.ifm_c, 32) * inst.ifm_w * inst.ifm_h)) / 1024.f / 1024.f;
    printf("[ log ]: the ifm size is %.7f MB\n", ifm_mem_size);

    if(inst.layout_en){
        quant_ifm.create(data_in.w, data_in.h, data_in.c, 1u, opt.blob_allocator);
        mat_quant_fp32Toint8(data_in, quant_ifm, inst.quant_scale);
        assert((u_align(data_in.w*data_in.h,32)*u_align(data_in.c,32)) <= mem_bytes);
    }
    else
        assert(data_in.total() <= mem_bytes && inst.ifm_c % 32 == 0);

    #ifdef MAT_LOG
    log_mat_file<float>(data_in, (char *)"./log/ifmbuf/ifm_data.txt",-1,1,0);
    log_mat_file<unsigned char>(quant_ifm, (char *)"./log/ifmbuf/quant_data.txt",-1,1,0);
    #endif

    ncnn::Mat& ifm_in = inst.layout_en ? quant_ifm : data_in;
    int line_len = ifm_layout(ifm_in, gemm_ifm_buffer, inst);

    ncnn::Mat wgt(data_in.w, data_in.h, data_in.c, 1u, opt.blob_allocator);
    ncnn::Mat ofm(data_in.w, data_in.h, data_in.c, 1u, opt.blob_allocator);
    inst.gemm_forward_s(data_in,wgt,ofm);
        
    dma_wait();

    if(read_ifm(gemm_ifm_buffer, line_len) == 0)
        printf("\033[;32m[ log ]: TEST IFMBUF PASS!\n\n\033[0m");
    else{
        printf("\033[;31m[ log ]: TEST IFMBUF FAILED!\n\n\033[0m");
        // assert(0);
    }
}

void test_ifmbuf_batch(){
    SRAND(7767517);

    // void test_ifmbuf_single(int layout_en, const ncnn::Mat& in)
    test_ifmbuf_single(1, RandomMat(5,13,263,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(48,48,20,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(13,13,32,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(8,9,80,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(5,3,64,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(5,3,3,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(5,3,99,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(8,7,3,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(8,7,56,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(8,8,40,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(13,13,16,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(88,26,35,-300.f,300.f));
    test_ifmbuf_single(1, RandomMat(6,7,77,-300.f,300.f)); 
    test_ifmbuf_single(1, RandomMat(32,17,32,-300.f,300.f));   

    test_ifmbuf_single(0, RandomS8Mat(64,13,13));
    test_ifmbuf_single(0, RandomS8Mat(64,12,12));
    test_ifmbuf_single(0, RandomS8Mat(64,10,10));
    test_ifmbuf_single(0, RandomS8Mat(64,8,8));
    test_ifmbuf_single(0, RandomS8Mat(64,5,5));
    test_ifmbuf_single(0, RandomS8Mat(1024,13,13));
    test_ifmbuf_single(0, RandomS8Mat(512,26,26));
}

}
