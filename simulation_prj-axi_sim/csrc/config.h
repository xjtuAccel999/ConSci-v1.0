#ifndef __CONFIG_H_
#define __CONFIG_H_

//----------------------------ncnn settings-------------------------------//

// #define FORWARD_ON_CPU
#define FORWARD_ON_NPU

// #define FORWARD_ON_NPU_CONV
// #define FORWARD_ON_NPU_POOL
// #define FORWARD_ON_NPU_ALU

// #define FORWARD_ON_CPU_CONV
// #define FORWARD_ON_CPU_POOL
// #define FORWARD_ON_CPU_ALU

#ifdef NCNN_TOOLS
    #undef FORWARD_ON_NPU
    #undef FORWARD_ON_NPU_CONV
    #undef FORWARD_ON_NPU_POOL
    #undef FORWARD_ON_NPU_ALU
    #define FORWARD_ON_CPU
#endif

#ifdef FORWARD_ON_NPU
    #define FORWARD_ON_NPU_CONV
    #define FORWARD_ON_NPU_POOL
    #define FORWARD_ON_NPU_ALU
#endif
#ifdef FORWARD_ON_CPU
    #define FORWARD_ON_CPU_CONV
    #define FORWARD_ON_CPU_POOL
    #define FORWARD_ON_CPU_ALU
#endif

//-----------------------------sim settings-------------------------------//
#define NCNN_MAT_ALIGN_BYTES 32  //256bits
// #define MASK_DMA_W
// #define USE_OPENMP
// #define COMPARE_WITH_NCNN   //only conv support
#define CHECK_ALIGN
// #define PRINT_COMPARE_ALL
#define PRINT_ALL
#define PRINT_SHAPE
// #define PRINT_SHAPE_PLUS
#define PRINT_LAYER
// #define PRINT_CONV_TIME

#ifdef PRINT_ALL
    #define PRINT_SHAPE
    #define PRINT_LAYER
    #define PRINT_SHAPE_PLUS
#endif

// #define LOG_WGT_BIAS

// #define WAVE_LOG
// #define MAT_LOG
// #define PRINT_MAT

// #define BATCH_TEST_CONV
// #define BATCH_ERROR_TEST_CONV
// #define BATCH_TEST_POOL
// #define BATCH_ERROR_TEST_POOL
#define BATCH_TEST_CYCYLES  2000

#define AXI_LITE_DATA_SUPPORT
#define DMA_SUPPORT
// #define ONLY_RESET
#define ACCMEM_GAP 1000


//----------------------------test layer----------------------------------//
// #define TEST_LAYER_CONV
// #define TEST_LAYER_POOL
// #define TEST_LAYER_ELTWISE
// #define TEST_LAYER_BINARYOP
// #define TEST_LAYER_ABSVAL
// #define TEST_LAYER_BIAS
// #define TEST_LAYER_DROPOUT
// #define TEST_LAYER_SCALE
// #define TEST_LAYER_THRESHOLD
// #define TEST_LAYER_RELU
// #define TEST_LAYER_TANH
// #define TEST_LAYER_CLIP
// #define TEST_LAYER_SIGMOID
// #define TEST_LAYER_SWISH
// #define TEST_LAYER_ELU
// #define TEST_LAYER_SELU
// #define TEST_LAYER_HARDSIGMOID
// #define TEST_LAYER_HARDSWISH
// #define TEST_LAYER_INNERPROD

//---------------------------test operator--------------------------------
// use independent test should mask all test layer test

// #define TEST_ALU_MATHFUNC
// #define TEST_ALU_MAT
// #define TEST_IFMBUF
// #define TEST_WGTBUF
// #define TEST_IFMBUFCTL    
// #define TEST_ACCMEM
// #define TEST_OFM


#if defined(TEST_IFMBUFCTL) | defined(TEST_ACCMEM) | defined(TEST_OFM)
    #ifndef TEST_LAYER_CONV
        #define TEST_LAYER_CONV
    #endif
#endif

//---------------------------test net--------------------------------//
// #define TEST_YOLOV3  //ok
// #define TEST_YOLOV3_TINY  //ok
// #define TEST_YOLOV4
// #define TEST_YOLOV4_TINY  //ok
// #define TEST_YOLOV5S  //ok
// #define TEST_YOLOV6N  //ok
#define TEST_YOLOV7  //ok
// #define TEST_YOLOV7_TINY  //ok
// #define TEST_YOLOV8
// #define TEST_YOLO_FASTER
// #define TEST_YOLO_FASTERV2

// #define TEST_MOBILENET_SSD  //ok
// #define TEST_MOBILENET_YOLO  //ok
// #define TEST_MOBILENETV2_SSDLITE
// #define TEST_MOBILENETV2_YOLO  //ok
// #define TEST_MOBILENETV3_SSDLITE

// #define TEST_RESNET18  //ok
// #define TEST_RESNET50  //ok
// #define TEST_RESNET101  //ok
// #define TEST_GOOGLENET

// #define TEST_SQUEEZENET   // ok
// #define TEST_SHUFFLENETV1
// #define TEST_SHUFFLENETV2 // ok

// #define TEST_MTCNN
// #define TEST_RETINAFACE  //opt ok
// #define TEST_SCRFD

// #define TEST_FASTERRCNN  //opt ok
// #define TEST_RFCN  //opt ok

// #define TEST_YOLACT

#if defined(TEST_YOLOV3)  || defined(TEST_YOLOV3_TINY) || defined(TEST_YOLOV4)   || defined(TEST_YOLOV4_TINY) || \
    defined(TEST_YOLOV5S) || defined(TEST_YOLOV6N)     || defined(TEST_YOLOV7)   || defined(TEST_YOLOV7_TINY) || \
    defined(TEST_YOLOV8)  || defined(TEST_YOLO_FASTER) || defined(TEST_YOLO_FASTERV2)
    #define INFERENCE_NET
#endif

#if defined(TEST_MOBILENET_SSD)    || defined(TEST_MOBILENET_YOLO)      ||  defined(TEST_MOBILENETV2_SSDLITE)   || \
    defined(TEST_MOBILENETV2_YOLO) || defined(TEST_MOBILENETV3_SSDLITE)
    #define INFERENCE_NET
#endif

#if defined(TEST_RESNET18)   || defined(TEST_RESNET50)     || defined(TEST_RESNET101)    ||  defined(TEST_GOOGLENET) || \
    defined(TEST_SQUEEZENET) || defined(TEST_SHUFFLENETV1) || defined(TEST_SHUFFLENETV2)
    #define INFERENCE_NET
#endif

#if defined(TEST_MTCNN)  || defined(TEST_RETINAFACE)     || defined(TEST_SCRFD)    ||  defined(TEST_FASTERRCNN) || \
    defined(TEST_RFCN)   || defined(TEST_YOLACT) 
    #define INFERENCE_NET
#endif


#define IMAGE_PATH "testImages/dog.jpg"

//--------------------------hareware parameters---------------------------//
#define IFM_BUFFER_WIDTH  256
#define IFM_BUFFER_DEPTH  (65536/4) //512KB
#define WGT_BUFFER_DEPTH  (9*512)
#define OSCALE_BUFFER_DEPTH  1024

#define ACCMEM_OUT_WIDTH 29
#define DMA_DATA_WIDTH 128
#define AXI_DATA_WIDTH 128
#define AXI_TRANSFER_BYTE (AXI_DATA_WIDTH/8) // axi single transfer byte num
#define DMA_DATA_NUM  (DMA_DATA_WIDTH/32)
#define HW_FREQ_MHZ  500
#if defined(INFERENCE_NET) || defined(TEST_LAYER_CONV) || \
    defined(TEST_LAYER_ABSVAL) || defined(TEST_LAYER_THRESHOLD) || defined(TEST_LAYER_DROPOUT) || \
    defined(TEST_LAYER_BIAS) || defined(TEST_LAYER_SCALE) || defined(TEST_LAYER_BINARYOP) || defined(TEST_LAYER_ELTWISE) || \
    defined(TEST_LAYER_RELU) || defined(TEST_LAYER_TANH) || defined(TEST_LAYER_SIGMOID) || defined(TEST_LAYER_SWISH) || \
    defined(TEST_LAYER_ELU) || defined(TEST_LAYER_SELU) || defined(TEST_LAYER_HARDSIGMOID) || defined(TEST_LAYER_HARDSWISH) || \
    defined(TEST_LAYER_CLIP) || defined(TEST_LAYER_INNERPROD)
    #define USE_HW_ALU
#endif

#if defined(TEST_LAYER_POOL) || defined(INFERENCE_NET)
    #define USE_HW_POOL
#endif



#if defined(TEST_LAYER_CONV) || defined(INFERENCE_NET)
    #define USE_HW_GEMM
#endif

#endif