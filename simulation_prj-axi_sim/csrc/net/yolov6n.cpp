#include "cnn_demo.h"
#include "../ncnn/net.h"
#include "../ncnn/simpleocv.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "../config.h"
#include "../veri.h"

#include <float.h>

// #define MAX_STRIDE 32

namespace demo{
struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
	int grid0;
	int grid1;
	int stride;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_grids_and_stride(const int target_w, const int target_h, const std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
	const int num_boxes = (int)strides.size();
	int i = 0, j = 0, g0 = 0, g1 = 0;
	int num_grid_w = 0, num_grid_h = 0;
	for (i = 0; i < num_boxes; i++)
	{
		num_grid_w = target_w / strides[i];
		num_grid_h = target_h / strides[i];
		for (g1 = 0; g1 < num_grid_h; g1++)
		{
			for (g0 = 0; g0 < num_grid_w; g0++)
			{
				GridAndStride& gs = grid_strides[j++];
				gs.grid0 = g0;
				gs.grid1 = g1;
				gs.stride = strides[i];
			}
		}
	}
	// The total number of boxes must be equal to (grids ¡Á anchors)
	assert(j == grid_strides.size());
}


static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
	const int num_points = grid_strides.size();
	const int num_classes = pred.w - 5;
	for (int i = 0; i < num_points; i++)
	{
		const float* objptr = pred.row(i);
		const float* scores = objptr + 5;

		// find label with max score
		int label = -1;
		float score = -FLT_MAX;
		for (int k = 0; k < num_classes; k++)
		{
			float confidence = scores[k];
			if (confidence > score)
			{
				label = k;
				score = confidence;
			}
		}
		
		if (score >= prob_threshold)
		{
			float left_offset = objptr[0];
			float top_offset = objptr[1];
			float right_offset = objptr[2];
			float bottom_offset = objptr[3];

			float x_anchor = grid_strides[i].grid0 + 0.5f;
			float y_anchor = grid_strides[i].grid1 + 0.5f;
			float stride = grid_strides[i].stride;

			float x0 = (x_anchor - left_offset) * stride;
			float y0 = (y_anchor - top_offset) * stride;
			float x1 = (x_anchor + right_offset) * stride;
			float y1 = (y_anchor + bottom_offset) * stride;
			
			Object obj;
			obj.rect.x = x0;
			obj.rect.y = y0;
			obj.rect.width = x1 - x0;
			obj.rect.height = y1 - y0;
			obj.label = label;
			obj.prob = score;
			objects.push_back(obj);
		}
	}
}

static int detect_yolov6n(const cv::Mat& bgr, std::vector<Object>& objects, const std::vector<int>& strides)
{
    ncnn::Net yolov6;

    yolov6.opt.use_int8_inference = true;
    // yolov6.opt.use_bf16_storage = true;

    // the ncnn model https://github.com/Accustomer/YOLOv6-NCNN/blob/main/models/yolov6n.param
    // if (yolov6.load_param_mem(yolov6.load_file2mem("models/yolov6n/yolov6n.param")))
    //     exit(-1);
    // if (yolov6.load_model_mem(yolov6.load_file2mem("models/yolov6n/yolov6n.bin")))
    //     exit(-1);
    // if (yolov6.load_param_mem(yolov6.load_file2mem("models/yolov6n/yolov6n_opt.param")))
    //     exit(-1);
    // if (yolov6.load_model_mem(yolov6.load_file2mem("models/yolov6n/yolov6n_opt.bin")))
    //     exit(-1);
    if (yolov6.load_param_mem(yolov6.load_file2mem("models/yolov6n/yolov6n_int8.param")))
        exit(-1);
    if (yolov6.load_model_mem(yolov6.load_file2mem("models/yolov6n/yolov6n_int8.bin")))
        exit(-1);

    const int target_size = 640;
    const float prob_threshold = 0.45f;
    const float nms_threshold = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // yolov5/models/common.py DetectMultiBackend
    const int max_stride = 32;

    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
    float scale = 1.f;

    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov6.create_extractor();

    ex.input("images", in_pad);

    ncnn::Mat out;
    ex.extract("outputs", out);

    std::vector<Object> proposals;
    std::vector<GridAndStride> grid_strides(out.h);
	generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
	generate_proposals(grid_strides, out, prob_threshold, proposals);

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

static const unsigned char colors[81][3] = {
			{56,  0,   255},
			{226, 255, 0},
			{0,   94,  255},
			{0,   37,  255},
			{0,   255, 94},
			{255, 226, 0},
			{0,   18,  255},
			{255, 151, 0},
			{170, 0,   255},
			{0,   255, 56},
			{255, 0,   75},
			{0,   75,  255},
			{0,   255, 169},
			{255, 0,   207},
			{75,  255, 0},
			{207, 0,   255},
			{37,  0,   255},
			{0,   207, 255},
			{94,  0,   255},
			{0,   255, 113},
			{255, 18,  0},
			{255, 0,   56},
			{18,  0,   255},
			{0,   255, 226},
			{170, 255, 0},
			{255, 0,   245},
			{151, 255, 0},
			{132, 255, 0},
			{75,  0,   255},
			{151, 0,   255},
			{0,   151, 255},
			{132, 0,   255},
			{0,   255, 245},
			{255, 132, 0},
			{226, 0,   255},
			{255, 37,  0},
			{207, 255, 0},
			{0,   255, 207},
			{94,  255, 0},
			{0,   226, 255},
			{56,  255, 0},
			{255, 94,  0},
			{255, 113, 0},
			{0,   132, 255},
			{255, 0,   132},
			{255, 170, 0},
			{255, 0,   188},
			{113, 255, 0},
			{245, 0,   255},
			{113, 0,   255},
			{255, 188, 0},
			{0,   113, 255},
			{255, 0,   0},
			{0,   56,  255},
			{255, 0,   113},
			{0,   255, 188},
			{255, 0,   94},
			{255, 0,   18},
			{18,  255, 0},
			{0,   255, 132},
			{0,   188, 255},
			{0,   245, 255},
			{0,   169, 255},
			{37,  255, 0},
			{255, 0,   151},
			{188, 0,   255},
			{0,   255, 37},
			{0,   255, 0},
			{255, 0,   170},
			{255, 0,   37},
			{255, 75,  0},
			{0,   0,   255},
			{255, 207, 0},
			{255, 0,   226},
			{255, 245, 0},
			{188, 255, 0},
			{0,   255, 18},
			{0,   255, 75},
			{0,   255, 151},
			{255, 56,  0},
			{245, 255, 0}
	};

    int color_index = 0;
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 80];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cc, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
    // cv::waitKey(0);
}

int yolov6n_inference()
{
    cv::Mat m = cv::imread(IMAGE_PATH, 1);
    if (m.empty())
    {
        printf("[error]: cv::imread is failed\n");
        return -1;
    }
    std::vector<int> strides = {8, 16, 32};
    std::vector<Object> objects;

    detect_yolov6n(m, objects, strides);

    draw_objects(m, objects);

    double time_ms = (main_time/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    double time_pool_ms = (time_pool/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    double time_gemm_ms = (time_gemm/2*1000.f) / (HW_FREQ_MHZ*1000000.f);
    printf("[ log ]: FREQ = %dMHz, CYCLES = %ld, TIME = %fms\n",HW_FREQ_MHZ,main_time/2,time_ms);
    printf("[ log ]: POOL_TIME = %f, %.5f%%\n", time_pool_ms, time_pool_ms/time_ms*100.f);
    printf("[ log ]: GEMM_TIME = %f, %.5f%%\n", time_gemm_ms, time_gemm_ms/time_ms*100.f);
    return 0;

}
}