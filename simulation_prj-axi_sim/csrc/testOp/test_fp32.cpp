#include "test_op.h"

#if defined(TEST_FP32_ADD) || defined(TEST_FP32_MUL) || defined(TEST_FP32_TO_INT8) || defined(TEST_INT32_TO_FP32)
namespace test
{

#ifdef TEST_FP32_ADD
void test_fp32_add(){  //0:add 1:mul
    SRAND(7767517);
    ncnn::Mat a = RandomMat(200, -10000.0f, 10000.0f);
    ncnn::Mat b = RandomMat(200, -10000.0f, 10000.0f);
    ncnn::Option opt;
    ncnn::Mat c(a.w, 1, 1, 4u, opt.blob_allocator);
    ncnn::Mat d(a.w, 1, 1, 4u, opt.blob_allocator);
    int count = 0;
    for(int i=0; i<a.w; i++){
        c[i] = a[i] + b[i];
    }


    for(int i=0; i<10000; i++){
        if(i == a.w + 5)
            break;
        if(i > a.w -1){
            top->io_fp32_en = 0;
            top->io_fp32_a = 0;
            top->io_fp32_b = 0;
        }
        else{
            top->io_fp32_en = 1;
            int* ptr_a = a;
            int* ptr_b = b;
            top->io_fp32_a = ptr_a[i];
            top->io_fp32_b = ptr_b[i];
        }
        update();
        if(top->io_fp32_test_valid == 1){
            int ptr_d = top->io_fp32_c;
            d[count++] = *(float*)&ptr_d;
        }
    }
    for(int i=0; i<a.w; i++){  
        if(c[i] != d[i]){
            #ifdef PRINT_MAT
                printf("[ log ]:-------------------------------------------------------------\n");
                printf("[ log ]: i = %d\n",i);
                // printf("[ log ]: a = %08x, b = %08x, a+b = %08x, recv_data = %08x\n",a[i],b[i],c[i],d[i]);
                printf("[ log ]: a = %f, b = %f, c = %f, d = %f\n",a[i],b[i],c[i],d[i]);
            #endif
        }    
    }
    printf("\033[;32m[ log ]: TEST FP32 ADD PASS!\n\033[0m");
}
#endif

#ifdef TEST_FP32_MUL
void test_fp32_mul(){
    SRAND(7767517);
    ncnn::Mat a = RandomMat(200, -100.0f, 100.0f);
    ncnn::Mat b = RandomMat(200, -100.0f, 100.0f);
    ncnn::Option opt;
    ncnn::Mat c(a.w, 1, 1, 4u, opt.blob_allocator);
    ncnn::Mat d(a.w, 1, 1, 4u, opt.blob_allocator);
    int count = 0;
    for(int i=0; i<a.w; i++){
        c[i] = a[i] * b[i];
    }
    for(int i=0; i<10000; i++){
        if(i == a.w + 5)
            break;
        if(i > a.w -1){
            top->io_fp32_en = 0;
            top->io_fp32_a = 0;
            top->io_fp32_b = 0;
        }
        else{
            top->io_fp32_en = 1;
            int* ptr_a = a;
            int* ptr_b = b;
            top->io_fp32_a = ptr_a[i];
            top->io_fp32_b = ptr_b[i];
        }
        update();
        if(top->io_fp32_test_valid == 1){
            int ptr_d = top->io_fp32_c;
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
#endif

#ifdef TEST_FP32_TO_INT8
signed char float2int8(float in){
    int int32 = (int)round(in);
    if(int32 > 127) return 127;
    if(int32 < -128) return -128;
    return (signed char) int32;
}

void test_fp32ToInt8(){
    SRAND(7767517);
    float a;
    signed char b;
    for(int i=0; i<10; i++){
        a = RandomFloat(-200.f,200.f);
        b = float2int8(a);
        #ifdef PRINT_MAT
        printf("[ log ]: float a = %f, hex a = %x, int8 b = %d\n",a,*((uint32_t*)&a),b);
        #endif
        top->io_a = *((uint32_t*)&a);
        top->io_i_valid = 1;
        wait();
        if(*((signed char*)&recv_data) != b){
            printf("\033[;31m[error]: recv_data is %d, standard data is %d\n\033[0m",recv_data,b);
            printf("\033[;31m[error]: TEST FP32 to INT8 ERROR!\n\033[0m");
            assert(0);
        }      
    }
    printf("\033[;32m[ log ]: TEST FP32 to INT8 PASS!\n\033[0m");
}
#endif

#ifdef TEST_INT32_TO_FP32
void test_Int32Tofp32(){
    SRAND(7767517);
    int a;
    for(int i=0; i<10; i++){
        a = RandomInt();
        float b = (float)a;
        #ifdef PRINT_MAT
        printf("[ log ]: int a = %d, float b = %f\n",a,b);
        #endif
        top->io_a = a;
        top->io_i_valid = 1;
        wait();
        printf("b = %f\n",b);
        printf("recvData = %f\n",*((float*)&recv_data));
        if(*((float*)&recv_data) != b){
            printf("b = %f\n",b);
            printf("\033[;31m[error]: recv_data is %f, standard data is %f\n\033[0m",recv_data,b);
            printf("\033[;31m[error]: TEST INT32 to FP32 ERROR!\n\033[0m");
            // assert(0);
        }      
    }
    printf("\033[;32m[ log ]: TEST INT32 to FP32 PASS!\n\033[0m");
}
#endif

} // namespace test

#endif