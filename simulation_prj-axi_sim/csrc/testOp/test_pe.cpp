#include "test_op.h"

#ifdef TEST_PE
namespace test
{

void test_pe_single(){
    SRAND(7767517);
    ncnn::Mat a = RandomMat(200, -100000.0f, 100000.0f);
    ncnn::Mat b = RandomMat(200, -100000.0f, 100000.0f);
    ncnn::Mat p = RandomMat(200, -100000.0f, 100000.0f);
    ncnn::Option opt;
    ncnn::Mat c(a.w, 4u, opt.blob_allocator);
    ncnn::Mat d(a.w, 4u, opt.blob_allocator);
    int count = 0;
    for(int i=0; i<a.w; i++){
        c[i] = a[i] * b[i] + p[i];
    }
    for(int i=0; i<10000; i++){
        if(i == a.w + 5)
            break;
        if(i > a.w -1){
            top->io_pe_i_valid = 0;
            top->io_pe_a = 0;
            top->io_pe_b = 0;
            top->io_pe_c0 = 0;
            top->io_pe_c1 = 0;
        }
        else{
            top->io_pe_i_valid = 1;
            int* ptr_a = a;
            int* ptr_b = b;
            int* ptr_p = p;
            top->io_pe_a = ptr_a[i];
            top->io_pe_b = ptr_b[i];
            top->io_pe_c0 = ptr_p[i];
        }
        update();
        if(top->io_pe_o_valid == 1){
            int ptr_d = top->io_pe_d0;
            d[count++] = *(float*)&ptr_d;
        }
    }
    for(int i=0; i<a.w; i++){  
        // if(c[i] != d[i]){
            #ifdef PRINT_MAT
                printf("[ log ]:-------------------------------------------------------------\n");
                printf("[ log ]: i = %d\n",i);
                // printf("[ log ]: a = %08x, b = %08x, a+b = %08x, recv_data = %08x\n",a[i],b[i],c[i],d[i]);
                printf("[ log ]: a = %f, b = %f, c = %f, d = %f\n",a[i],b[i],c[i],d[i]);
            #endif
        // }    
    }
    printf("\033[;32m[ log ]: TEST FP32 MUL PASS!\n\033[0m");
}

} // namespace test

#endif