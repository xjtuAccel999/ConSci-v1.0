#ifndef __UTILS_H_
#define __UTILS_H_

#include "../ncnn/mat.h"
#include <cstdio>
#include <unistd.h>
#include "../hw/hw_gemm.h"
#include <string>
#include "assert.h"

#define random(x)  rand()%(x)

// log
#define ANSI_FG_BLACK   "\33[1;30m"
#define ANSI_FG_RED     "\33[1;31m"
#define ANSI_FG_GREEN   "\33[1;32m"
#define ANSI_FG_YELLOW  "\33[1;33m"
#define ANSI_FG_BLUE    "\33[1;34m"
#define ANSI_FG_MAGENTA "\33[1;35m"
#define ANSI_FG_CYAN    "\33[1;36m"
#define ANSI_FG_WHITE   "\33[1;37m"
#define ANSI_BG_BLACK   "\33[1;40m"
#define ANSI_BG_RED     "\33[1;41m"
#define ANSI_BG_GREEN   "\33[1;42m"
#define ANSI_BG_YELLOW  "\33[1;43m"
#define ANSI_BG_BLUE    "\33[1;44m"
#define ANSI_BG_MAGENTA "\33[1;35m"
#define ANSI_BG_CYAN    "\33[1;46m"
#define ANSI_BG_WHITE   "\33[1;47m"
#define ANSI_NONE       "\33[0m"

#define ANSI_FMT(str, fmt) fmt str ANSI_NONE

static int compareFloat(int a, int b){
    float f_c = *((float*)&a) - *((float*)&b);
    return abs(f_c) < 0.0001f;
}

static int compareFloat(float a, float b){
    return abs(a - b) < 0.0001f;
}

static void mat_nchw2nhwc(ncnn::Mat& mat_in, ncnn::Mat& mat_out){
    //mat_in  -> nchw mat
    //mat_out -> nhwc mat
    int size = mat_in.w * mat_in.h;
    int c = mat_in.c;
    if(mat_in.elemsize == 1u){
        memset(mat_out.data, 0, mat_out.total());
        for(int iwh=0; iwh<size; iwh++){
            char* out_ifm_ptr = mat_out.channel(iwh);
            for(int ic=0; ic<c; ic++){
                char* in_ifm_ptr = mat_in.channel(ic);
                out_ifm_ptr[ic] = in_ifm_ptr[iwh];
            }
        }
    }
    else{
        memset(mat_out.data, 0, mat_out.total()*4);
        for(int iwh=0; iwh<size; iwh++){
            float* out_ifm_ptr = mat_out.channel(iwh);
            for(int ic=0; ic<c; ic++){
                float* in_ifm_ptr = mat_in.channel(ic);
                out_ifm_ptr[ic] = in_ifm_ptr[iwh];
            }
        }
    }
}

static void mat_nhwc2nchw(ncnn::Mat& mat_in, ncnn::Mat& mat_out){
    //mat_in  -> nhwc
    //mat_out -> nchw
    assert(mat_in.elemsize == 4u);
    int size = mat_out.w * mat_out.h;
    int c = mat_in.w;
    memset(mat_out.data, 0, mat_out.total()*4);
    for(int ic=0; ic<c; ic++){
        float* out_ifm_ptr = mat_out.channel(ic);
        for(int iwh=0; iwh<size; iwh++){
            float* in_ifm_ptr = mat_in.channel(iwh);
            out_ifm_ptr[iwh] = in_ifm_ptr[ic];
        }
    }
}

static std::string uchar2string(unsigned char a){
    std::string str= "";
    str += a / 100 + 0x30;
    str += (a % 100) / 10 + 0x30;
    str += a % 10 + 0x30;
    return str;
}

static void cstr(std::string str, char* cstr){
    int len = str.length();
    for(int i=0; i<len; i++){
        cstr[i] = str[i]; 
    }
    cstr[len] = '\0';
}

static void log_origin_wgt(ncnn::Mat& wgt, accel::hw_gemm& inst, int hex_out=0){
    unsigned char* wgt_ptr = wgt;
    FILE* fp = fopen("./log/log_origin_wgt.txt","w+");
    if(fp == NULL){
        printf("[error]: log_origin_wgt.txt open is failed\n");
        assert(0);
    }
    for(int oc=0; oc<inst.ofm_c; oc++){
        fprintf(fp,"\n//----------------- oc = %d -----------------//\n",oc);
        for(int ic=0; ic<inst.ifm_c; ic++){
            fprintf(fp,"ic = %04d:\t",ic);
            for(int k=0; k<inst.kernel*inst.kernel; k++){
                if(hex_out)
                    fprintf(fp,"%02x\t",*wgt_ptr++);
                else
                    fprintf(fp,"%d\t",*((char*)wgt_ptr++));
            }
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
}

template <typename T> static void log_mat_file(ncnn::Mat& data, char* filename, unsigned int limit_c=-1, int hex_out=0, int is_float = 0){
    FILE* fp = fopen(filename,"w+");
    if(fp == NULL){
        printf("[error]: %s open is failed\n",filename);
        assert(0);
    }
    fprintf(fp,"[ log ]: mat_elemsize = %ld, mat_dims = %d\n",data.elemsize,data.dims);
    fprintf(fp,"[ log ]: mat_w = %d, mat_h = %d, mat_c = %d\n",data.w,data.h,data.c);
    for(int ic=0; ic<data.c; ic++){
        if(ic >= limit_c) 
            break;
        fprintf(fp,"\n//----------------- oc = %d -----------------//\n",ic);
        T* i_ptr = data.channel(ic);
        fprintf(fp,"oc_ptr = %p\n",i_ptr);
        for(int h=0; h<data.h; h++){
            for(int w=0; w<data.w; w++){
                if(sizeof(T) == 1)
                    if(hex_out)
                        fprintf(fp,"%02x\t",*((unsigned char*)i_ptr++));
                    else
                        fprintf(fp,"% 3d\t",*((char*)i_ptr++));
                else
                    if(hex_out)
                        fprintf(fp,"%08x\t",*((unsigned int*)i_ptr++));                   
                    else if(is_float)
                        fprintf(fp,"%05.5f\t",*((float*)i_ptr++));
                    else
                        fprintf(fp,"%d\t",*((int*)i_ptr++));
            }
            fprintf(fp,"\n");
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

//static void delete_file(char* filename){
//    if(access(filename,F_OK) == 0){
//        if(remove(filename) == 1){
//            printf("[error]: delete %s is failed\n",filename);
//            assert(0);
//        }
//    }
//}

template <typename T> static void log_block32x32_file(ncnn::Mat& data, char* filename, int ow, int oh, int oc, int owh, int kh, int kw, int ic, int hex_out=0, int is_float=0){
    FILE* fp = fopen(filename,"a+");
    if(fp == NULL){
        printf("[error]: %s open is failed\n",filename);
        assert(0);
    }
    fprintf(fp,"\n//------------------------------------------------------------//\n");
    fprintf(fp,"[ log ]: ow = %d, oh = %d, oc = %d\n",ow,oh,oc);
    fprintf(fp,"[ log ]: owh = %d, ic = %d\n",owh,ic);
    fprintf(fp,"[ log ]: kw = %d, kh = %d\n",kw,kh);

    for(int ic=0; ic<data.c; ic++){
        T* i_ptr = data.channel(ic);
        for(int wh=0; wh<data.w*data.h; wh++){
            if(sizeof(T) == 1)
                if(hex_out)
                    fprintf(fp,"%02x\t",*((unsigned char*)i_ptr++));
                else
                    fprintf(fp,"% 3d\t",*((char*)i_ptr++));
            else
                if(hex_out)
                    fprintf(fp,"%08x\t",*((unsigned int*)i_ptr++));                   
                else if(is_float)
                    fprintf(fp,"%05.5f\t",*((float*)i_ptr++));
                else
                    fprintf(fp,"%d\t",*((int*)i_ptr++));
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

#endif
