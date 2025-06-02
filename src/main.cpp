#include <stdio.h>
#include "xil_printf.h"
#include "xil_io.h"
 #include "sleep.h"
#include "net/cnn_demo.h"
#include "hw/pl_intr.h"
#include "hw/dri_sd.h"
#include "testLayer/test_layer.h"
#include "xil_cache.h"
#include "xtime_l.h"

#define AXI_LED_BASEADDR 0x80000000
int main(){
	Xil_Out32(AXI_LED_BASEADDR,1);
	sd_init();
	init_intr_sys();
	/*****************************alu test*******************************/
//    test::test_layer_absval_batch();
//    test::test_layer_bias_batch();
//    test::test_layer_dropout_batch();
//	test::test_layer_eltwise_batch();
    // test::test_layer_innerprod_batch();
//    test::test_layer_scale_batch();
//    test::test_layer_threshold();
	/*****************************pool test*******************************/
//	test::test_layer_pool_batch();

	/*****************************conv test*******************************/
//      test::test_layer_conv_batch();

	/***********************activation func test*************************/
//    test::test_layer_clip();
//    test::test_layer_hardsigmoid();
//    test::test_layer_hardswish();
//    test::test_layer_swish();
//    test::test_layer_tanh();
//    test::test_layer_sigmoid();
//    test::test_layer_elu();
//    test::test_layer_selu();
//    test::test_layer_relu();
	/*****************************net test*******************************/
	#ifdef TEST_YOLOV3_TINY
		demo::yolov3_tiny_inference();
	#endif

	#ifdef TEST_YOLOV3
		demo::yolov3_inference();
	#endif

	#ifdef TEST_YOLOV4_TINY
		demo::yolov4_tiny_inference();
	#endif

	#ifdef TEST_YOLOV5S
		demo::yolov5s_inference();
	#endif

	#ifdef TEST_YOLOV6N
		demo::yolov6n_inference();
	#endif

	#ifdef TEST_YOLOV7
		demo::yolov7_inference();
	#endif

	#ifdef TEST_YOLOV7_TINY
		demo::yolov7_tiny_inference();
	#endif

	#ifdef TEST_MOBILENET_SSD
		demo::mobilenet_ssd_inference();
	#endif

	#ifdef TEST_MOBILENET_YOLO
		demo::mobilenet_yolo_inference();
	#endif

	#ifdef TEST_MOBILENETV2_YOLO
		demo::mobilenetv2_yolo_inference();
	#endif

	#ifdef TEST_SQUEEZENET
		demo::squeezenet_inference();
	#endif

	#ifdef TEST_SHUFFLENETV2
		demo::shufflenetv2_inference();
	#endif

	#ifdef TEST_RESNET18
		demo::resnet18_inference();
	#endif

	#ifdef TEST_RESNET50
		demo::resnet50_inference();
	#endif

	#ifdef TEST_RESNET101
		demo::resnet101_inference();
	#endif

	#ifdef TEST_GOOGLENET
		demo::googlenet_inference();
	#endif

	

	#ifdef TEST_RETINAFACE
		demo::retinaface_inference();
	#endif

	while(1){
	    printf("hello world\r\n");
		Xil_Out32(AXI_LED_BASEADDR,1);
		sleep(1);
		Xil_Out32(AXI_LED_BASEADDR,2);
		sleep(1);
	}
	return 0;
}

