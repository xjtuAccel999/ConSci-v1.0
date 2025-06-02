#include "dri_sd.h"

static FATFS SD_DEV;
static const char* SD_Path="0:/";

int sd_init(){
	FRESULT result;
	result = f_mount(&SD_DEV,SD_Path,0);
	if(result != 0){
		return XST_FAILURE;
	}
	printf("[ log ]: SD INIT PASS\r\n");
	return XST_SUCCESS;
}

char* load_file_align(const char* filename){
    FIL file;
    FIL* ptr = &file;
    int status;
    uint32_t br;
    status = f_open(ptr,filename,FA_READ);
    if(status != 0)  
        printf("[error] : load_file_align => %s open is failed,status=%d\r\n",filename,status);

    size_t fsize = f_size(ptr);
    printf("[ log ]: %s is %d bytes\r\n",filename,fsize);

    // char* buffer = (char*)malloc(fsize);
    char* buffer = (char*)(ncnn::fastMalloc(fsize));
    if(buffer == NULL){
        printf("[error]: load_file_align => Memory malloc failed\r\n");
    }

    status = f_read(ptr,buffer,fsize,&br);
    if(status != 0){
        printf("[error] : load_file_align => fread is failed,status=%d\r\n",status);
    }
    if(br != fsize){
        printf("[error]: load_file_align => read param data is failed\r\n");
        printf("[error]: load_file_align => br is %d, fsize = %d\r\n",br,fsize);
    }

    f_close(ptr);
    return buffer;
}

int sd_write(const char* filename,ncnn::Mat& data){
    FIL file;
    FIL* ptr = &file;
    int status;
    uint32_t br;
    status = f_open(ptr,filename,FA_CREATE_ALWAYS | FA_WRITE);
    // void* buffer = (u8*)data.data;
    printf("data.elemsize = %d\n",data.elemsize);
    status = f_write(ptr,(void*)data.data,data.total()*data.elemsize,&br);
    if (status != FR_OK)
    {
        printf("[error]: write_mat_to_sd => f_write failed, status=%d\r\n", status);
        f_close(ptr);
        return false;
    }

    // 关闭文件
    f_close(ptr);
    return 0;
}
int sd_write(const char* filename,int8_t* data,size_t dataSize){
    FIL file;
    FIL* ptr = &file;
    int status;
    uint32_t br;
    status = f_open(ptr,filename,FA_CREATE_ALWAYS | FA_WRITE);
    // void* buffer = (u8*)data.data;
    status = f_write(ptr,(void*)data,dataSize,&br);
    if (status != FR_OK)
    {
        printf("[error]: write_mat_to_sd => f_write failed, status=%d\r\n", status);
        f_close(ptr);
        return false;
    }

    // 关闭文件
    f_close(ptr);
    return 0;
}
int sd_write(const char* filename){
    const char src_str[30] = "www.openedv.com";
    FIL file;
    FIL* ptr = &file;
    int status;
    uint32_t br;
    status = f_open(ptr,filename,FA_CREATE_ALWAYS | FA_WRITE);
    // void* buffer = (u8*)data.data;
    status = f_write(ptr,(void*)src_str,strlen(src_str),&br);
    if (status != FR_OK)
    {
        printf("[error]: write_mat_to_sd => f_write failed, status=%d\r\n", status);
        f_close(ptr);
        return false;
    }

    // 关闭文件
    f_close(ptr);
    return 0;
}