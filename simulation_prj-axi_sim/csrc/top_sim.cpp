#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "Vtcp.h"
#include "verilated_vcd_c.h"
#include "veri.h"
#include "config.h"
#include <stdlib.h>
#include <time.h>
 
int main(int argc,char **argv)
{
    Verilated::commandArgs(argc,argv);
    Verilated::traceEverOn(true); //导出vcd波形需要加此语句

	// srand((unsigned)time(NULL));

    #ifdef WAVE_LOG
    top->trace(tfp, 0);
    tfp->open("wave.vcd"); //打开vcd
    #endif
    
	#if defined(TEST_IFMBUFCTL) | defined(TEST_ACCMEM) | defined(TEST_OFM)
	printf("\033[;32mDont forget to use single thread to run verilator!!\n\033[0m");
	#endif

    exec(-1);

    top->final();
    #ifdef WAVE_LOG
    tfp->close();
    #endif
    delete top;
    return 0;


    // ncnn::Option opt;
    // ncnn::Mat ifm_32x32(32,1,32,1u,opt.blob_allocator);
    // ncnn::Mat wgt_32x32(32,1,32,1u,opt.blob_allocator);
    // ncnn::Mat psum_32x32(32,1,32,4u,opt.blob_allocator);



    // int test_i = 1;
    // for(int c = 0; c < ifm_32x32.c; c++) {
    //     int* ifm_ptr = ifm_32x32.channel(c);
    //     for(int h = 0; h < ifm_32x32.h; h++){
    //         for(int w = 0; w < ifm_32x32.w; w++){
    //           ifm_ptr[(w)+(h)*ifm_32x32.w] = test_i;
    //           test_i += 1;
    //          }
    //      }
    //  }

    //  for(int c = 0; c < wgt_32x32.c; c++) {
    //     int* wgt_ptr = wgt_32x32.channel(c);
    //     for(int h = 0; h < wgt_32x32.h; h++){
    //         for(int w = 0; w < wgt_32x32.w; w++){
    //           wgt_ptr[(w)+(h)*wgt_32x32.w] = test_i;
    //           test_i += 1;
    //          }
    //      }
    //  }
    // printf("ifm_32x32\n");
    // printf_int8_mat(ifm_32x32);

    // printf("wgt_32x32\n");
    // printf_int8_mat(wgt_32x32);


    // gemm_32x32(ifm_32x32, wgt_32x32, psum_32x32);

    // printf("psum_32x32\n");
    // printf_int8_mat(psum_32x32);





}
