
#ifndef __DRI_SD_H_
#define __DRI_SD_H_


#include "../ncnn/net.h"

#include "xsdps.h"
#include "ff.h"

#include "sleep.h"



int sd_init();
char* load_file_align(const char* filename);
int sd_write(const char* filename,ncnn::Mat& data);
int sd_write(const char* filename,int8_t* data,size_t dataSize);
int sd_write(const char* filename);
#endif

