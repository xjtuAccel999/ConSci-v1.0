#ifndef __CNN_DEMO_H_
#define __CNN_DEMO_H_


namespace demo{
    int yolov3_inference();
    int yolov3_tiny_inference();
    int yolov4_inference();
    int yolov4_tiny_inference();
    int yolov5s_inference();
    int yolov6n_inference();
    int yolov7_inference();
    int yolov7_tiny_inference();
    int yolov8_inference();
    int yolo_faster_inference();
    int yolo_fasterv2_inference();

    int mobilenet_ssd_inference();
    int mobilenet_yolo_inference();
    int mobilenetv2_ssdlite_inference();
    int mobilenetv2_yolo_inference();
    int mobilenetv3_ssdlite_inference();

    int resnet18_inference();
    int resnet50_inference();
    int resnet101_inference();
    int googlenet_inference();

    int squeezenet_inference();
    int shufflenetv1_inference();
    int shufflenetv2_inference();

    int mtcnn_inference();
    int retinaface_inference();
    int scrfd_inference();
    int fasterrcnn_inference();
    int rfcn_inference();
    
    int yolact_inference();
}



#endif