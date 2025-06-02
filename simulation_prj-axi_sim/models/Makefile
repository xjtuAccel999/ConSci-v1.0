NET_NAME = resnet18

ORIGIN_PARAM_PATH = ./$(NET_NAME)/$(NET_NAME).param
ORIGIN_BIN_PATH   = ./$(NET_NAME)/$(NET_NAME).bin

ifeq ($(NET_NAME), yolov5s)
	MEAN = [0,0,0]
	NORM = [0.00392,0.00392,0.00392] 
	SHAPE = [640,512,3]
	IMAGELIST_PATH  =  ./imagelist_coco.txt
else ifeq ($(NET_NAME), yolov3_tiny)
	MEAN = [0,0,0]
	NORM = [0.00392,0.00392,0.00392] 
	SHAPE = [416,416,3]
	IMAGELIST_PATH  =  ./imagelist_coco.txt
else ifeq ($(NET_NAME), yolov3)
	MEAN = [0,0,0]
	NORM = [0.00392,0.00392,0.00392] 
	SHAPE = [416,416,3]
	IMAGELIST_PATH  =  ./imagelist_coco.txt
else ifeq ($(NET_NAME), yolov8s)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00784,0.00784,0.00784] 
	SHAPE = [224,224,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), yolov4)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00392,0.00392,0.00392] 
	SHAPE = [608,608,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), yolov4_tiny)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00392,0.00392,0.00392] 
	SHAPE = [416,416,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), yolov6n)
	MEAN = [0,0,0]
	NORM = [0.00392,0.00392,0.00392] 
	SHAPE = [640,480,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), yolov7_tiny)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00784,0.00784,0.00784] 
	SHAPE = [300,300,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), yolov7)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00392,0.00392,0.00392] 
	SHAPE = [640,512,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), squeezenet)
	MEAN = [104.0,117.0,123.0]
	NORM = [0.00784,0.00784,0.00784] 
	SHAPE = [227,227,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), shufflenetv2)
	MEAN = [0,0,0]
	NORM = [0.00392,0.00392,0.00392] 
	SHAPE = [224,224,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
endif


ifeq ($(NET_NAME), resnet18)
	MEAN = [123.7,116.3,103.5]
	NORM = [0.01712,0.01751,0.01743] 
	SHAPE = [224,224,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), resnet50)
	MEAN = [123.7,116.3,103.5]
	NORM = [0.01712,0.01751,0.01743]
	SHAPE = [224,224,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), resnet101)
	MEAN = [123.7,116.3,103.5]
	NORM = [0.01712,0.01751,0.01743] 
	SHAPE = [224,224,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
endif

ifeq ($(NET_NAME), mobilenet_ssd)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00784,0.00784,0.00784] 
	SHAPE = [300,300,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), mobilenet_yolo)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00784,0.00784,0.00784] 
	SHAPE = [416,416,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), mobilenetv2_ssdlite)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00784,0.00784,0.00784] 
	SHAPE = [300,300,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), mobilenetv2_yolo)
	MEAN = [127.5,127.5,127.5]
	NORM = [0.00784,0.00784,0.00784] 
	SHAPE = [352,352,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), mobilenetv3_ssdlite)
	MEAN = [123.675,116.28,103.53]
	NORM = [1.0,1.0,1.0]
	SHAPE = [300,300,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
endif

ifeq ($(NET_NAME), rfcn)
	MEAN = [102.9801,115.9465,122.7717]
	NORM = [0,0,0]
	SHAPE = [298,224,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
else ifeq ($(NET_NAME), fasterrcnn)
	MEAN = [102.9801,115.9465,122.7717]
	NORM = [0,0,0]
	SHAPE = [298,224,3]
	IMAGELIST_PATH  =  ./imagelist_voc.txt
endif

ifeq ($(NET_NAME), yolact)
	MEAN = [123.68,116.78,103.94]
	NORM = [0.01712,0.01751,0.01743] 
	SHAPE = [224,224,3]
	IMAGELIST_PATH  =  ./imagelist_coco.txt
endif

GEN_OPT_PARAM_PATH  = $(NET_NAME)/$(NET_NAME)_opt.param
GEN_OPT_BIN_PATH    = $(NET_NAME)/$(NET_NAME)_opt.bin
GEN_TABLE_PATH      = $(NET_NAME)/$(NET_NAME).table
GEN_INT8_PARAM_PATH = $(NET_NAME)/$(NET_NAME)_int8.param
GEN_INT8_BIN_PATH   = $(NET_NAME)/$(NET_NAME)_int8.bin
GEN_OPT_NCNN_PARAM_PATH  = $(NET_NAME)/$(NET_NAME)_opt.ncnn.param
GEN_OPT_NCNN_BIN_PATH    = $(NET_NAME)/$(NET_NAME)_opt.ncnn.bin


all: opt table quant

opt:
	rm -rf $(GEN_OPT_PARAM_PATH) 
	rm -rf $(GEN_OPT_BIN_PATH)
	./u_ncnnoptimize $(ORIGIN_PARAM_PATH) $(ORIGIN_BIN_PATH) $(GEN_OPT_PARAM_PATH) $(GEN_OPT_BIN_PATH) 0

table:
	rm -rf $(GEN_TABLE_PATH)
	rm -rf $(GEN_OPT_NCNN_PARAM_PATH)
	rm -rf $(GEN_OPT_NCNN_BIN_PATH)
	./ncnnoptimize $(ORIGIN_PARAM_PATH) $(ORIGIN_BIN_PATH) $(GEN_OPT_NCNN_PARAM_PATH) $(GEN_OPT_NCNN_BIN_PATH) 0
	./ncnn2table $(GEN_OPT_NCNN_PARAM_PATH) $(GEN_OPT_NCNN_BIN_PATH) $(IMAGELIST_PATH) $(GEN_TABLE_PATH) mean=$(MEAN) norm=$(NORM) shape=$(SHAPE) pixel=BGR thread=12 method=kl

quant:
	rm -rf $(GEN_INT8_PARAM_PATH)
	rm -rf $(GEN_INT8_BIN_PATH)
	./u_ncnn2int8 $(GEN_OPT_PARAM_PATH) $(GEN_OPT_BIN_PATH) $(GEN_INT8_PARAM_PATH) $(GEN_INT8_BIN_PATH) $(GEN_TABLE_PATH)

tools:
	rm -rf u_ncnnoptimize
	rm -rf u_ncnn2int8
	cd ../tools && rm -rf build && mkdir build
	cd ../tools/build && cmake .. && make -j$(shell nproc)
	cp ../tools/build/u_ncnnoptimize .
	cp ../tools/build/u_ncnn2int8 .


clean: 
	rm -rf $(GEN_OPT_PARAM_PATH)
	rm -rf $(GEN_OPT_BIN_PATH)
	rm -rf $(GEN_TABLE_PATH)
	rm -rf $(GEN_INT8_PARAM_PATH)
	rm -rf $(GEN_INT8_BIN_PATH)

.PHONY: opt table quant tools clean



