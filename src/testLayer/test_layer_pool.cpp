//#include "../testOp/test_op.h"
#include "../utils/utils.h"
#include "test_layer.h"
#include <cmath>
#include "xtime_l.h"

int count_total_pool = 0;
int count_success_pool = 0;
int count_fail_pool = 0;

namespace test {

int32_t fp32_to_UInt(float num) {
    uint32_t binary = *(uint32_t*)(&num);
    return static_cast<int32_t>(binary);
}
// adjust padding
void adjust_padding(accel::hw_pool &inst, int w, int h) {
    // Helper function to adjust padding
    auto adjust = [](int size, int kernel, int stride, int& pad_low, int& pad_high) {
        int tail = (size - kernel + pad_low + pad_high) % stride;
        if (tail == 0) return;

        if (tail <= pad_low + pad_high) {
            // Decrease padding
            while (tail > 0) {
                if (pad_low >= pad_high) {
                    pad_low--;
                } else {
                    pad_high--;
                }
                tail = (size - kernel + pad_low + pad_high) % stride;
            }
        } else {
            // Increase padding
            while (tail > 0) {
                if (pad_low < pad_high) {
                    pad_low++;
                } else {
                    pad_high++;
                }
                tail = (size - kernel + pad_low + pad_high) % stride;
            }
        }
    };

    // Adjust left-right padding
    int original_pad_left = inst.pad_left;
    int original_pad_right = inst.pad_right;
    adjust(w, inst.kernel_w, inst.stride_w, inst.pad_left, inst.pad_right);
    if (inst.pad_left != original_pad_left || inst.pad_right != original_pad_right) {
        printf("\033[93m[ warn ]: Horizontal pad adjusted from [%d, %d] to [%d, %d]\033[0m\n", 
                original_pad_left, original_pad_right, inst.pad_left, inst.pad_right);
    }

    // Adjust top-bottom padding
    int original_pad_top = inst.pad_top;
    int original_pad_bottom = inst.pad_bottom;
    adjust(h, inst.kernel_w, inst.stride_w, inst.pad_top, inst.pad_bottom);
    if (inst.pad_top != original_pad_top || inst.pad_bottom != original_pad_bottom) {
        printf("\033[93m[ warn ]: Vertical pad adjusted from [%d, %d] to [%d, %d]\033[0m\n", 
                original_pad_top, original_pad_bottom, inst.pad_top, inst.pad_bottom);
    }
}
//----------------------------------------- forward_ncnn_pool ---------------------------------------------//
void forward_ncnn_pool(ncnn::Mat &ifm, ncnn::Mat &ofm, accel::hw_pool &inst) {
    // adjust_padding(inst, ifm.w, ifm.h);
    ncnn::ParamDict pd;
    pd.set(0, inst.pool_type); // pooling_type
    pd.set(1, inst.kernel_w);  // kernel_w
    pd.set(2, inst.stride_w);  // stride_w
    pd.set(3, inst.pad_left);  // pad_w
    pd.set(4, 0);              // global_pooling
    pd.set(5, inst.pad_mode);  // pad_mode
    pd.set(6, 0);              // avgpool_count_include_pad
    pd.set(7, 0);              // adaptive_pooling
    pd.set(8, ofm.w);          // out_w
    pd.set(13, inst.pad_top);
    pd.set(14, inst.pad_right);
    pd.set(15, inst.pad_bottom);
    pd.set(18, ofm.h); // out_h
    pd.set(19, inst.pad_value); // pad_value
//    pd.set(19, *((uint32_t*)&(inst.pad_value))); // pad_value
    // POOL_PAD_VALUE_SET(*((uint32_t*)&pad_value));


    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Pooling>("Pooling", pd, weights, ifm, ofm);
    if (ret != 0) {
        printf("[error]: test_pooling failed w=%d h=%d c=%d pooling_type=%d kernel=%d stride=%d\n", ifm.w, ifm.h, ifm.c, inst.pool_type, inst.kernel_w, inst.stride_w);
        assert(0);
    }
}

//----------------------------------------- forward_sim_pool ---------------------------------------------//
static float max(float a, float b) { return a >= b ? a : b; }

void forward_sim_pool(ncnn::Mat &ifm, ncnn::Mat &ofm, accel::hw_pool &inst) {
    // adjust_padding(inst, ifm.w, ifm.h);
    ncnn::Option opt;
    ncnn::Mat ifmp(ifm.w + inst.pad_left + inst.pad_right, ifm.h + inst.pad_top + inst.pad_bottom, ifm.c, 4u, opt.blob_allocator);
    for (int c = 0; c < ifm.c; c++) {
        float *ifm_ptr = ifm.channel(c);
        float *ifmp_ptr = ifmp.channel(c);
        float *ofm_ptr = ofm.channel(c);
        // Copy input matrix with padding offset
        for (int h = 0; h < ifm.h; h++) {
            memcpy(ifmp_ptr + (h + inst.pad_top) * ifmp.w + inst.pad_left, ifm_ptr + h * ifm.w, ifm.w * sizeof(float));
        }

        // Apply padding
        for (int h = 0; h < ifmp.h; h++) {
            for (int w = 0; w < ifmp.w; w++) {
                bool isPadTop = h < inst.pad_top;
                bool isPadBottom = h >= ifmp.h - inst.pad_bottom;
                bool isPadLeft = w < inst.pad_left;
                bool isPadRight = w >= ifmp.w - inst.pad_right;
                if (isPadTop || isPadBottom || isPadLeft || isPadRight) {
                    ifmp_ptr[w + h * ifmp.w] = inst.pad_mode == 1 ? ifmp_ptr[std::max(inst.pad_left, std::min(w, ifmp.w - inst.pad_right - 1)) + std::max(inst.pad_top, std::min(h, ifmp.h - inst.pad_bottom - 1)) * ifmp.w] : inst.pad_value;
                }
            }
        }
        // pooling
        for (int h = 0; h < ofm.h; h++) {
            for (int w = 0; w < ofm.w; w++) {
                float sum = inst.pool_type == POOL_AVG ? 0.0f : ifmp_ptr[(w * inst.stride_w) + (h * inst.stride_w) * ifmp.w];
                for (int kw = 0; kw < inst.kernel_w; kw++) {
                    for (int kh = 0; kh < inst.kernel_h; kh++) {
                        float akwh = ifmp_ptr[(w * inst.stride_w + kw) + (h * inst.stride_w + kh) * ifmp.w];
                        sum = inst.pool_type == POOL_AVG ? sum + akwh : std::max(sum, akwh);
                    }
                }
                ofm_ptr[(w) + (h)*ofm.w] = inst.pool_type == POOL_AVG ? sum / (inst.kernel_w * inst.kernel_h) : sum;
            }
        }
    }
}

void test_layer_pool(int iw_t, int ih_t, int ic_t, int k_t, int s_t, int type_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_down_t, int pd_mode, float pad_value) {
    
    count_total_pool ++ ;
    ncnn::Option opt;
    accel::hw_pool inst;
    inst.pool_type = type_t;
    inst.kernel_w = k_t;
    inst.kernel_h = k_t;
    inst.stride_w = s_t;
    inst.stride_h = s_t;
    inst.pad_left = pd_left_t;
    inst.pad_right = pd_right_t;
    inst.pad_top = pd_top_t;
    inst.pad_bottom = pd_down_t;
    inst.pad_mode = pd_mode;
    inst.pad_value = pad_value;

    
    adjust_padding(inst, iw_t, ih_t);
    int ofm_w = (iw_t + inst.pad_left + inst.pad_right - inst.kernel_w) / inst.stride_w + 1;
    int ofm_h = (ih_t + inst.pad_top + inst.pad_bottom - inst.kernel_h) / inst.stride_h + 1;
    ncnn::Mat ifm = RandomMat(iw_t, ih_t, ic_t, -100.0f, 100.0f);
    
      for (int c = 0; c < ifm.c; c++) {
          float i = (0.00001f + c) / 10000.0f;
          float *ifm_ptr = ifm.channel(c);
          for (int h = 0; h < ifm.h; h++) {
              for (int w = 0; w < ifm.w; w++) {
                  ifm_ptr[(w) + (h)*ifm.w] = i;
                  i += 1.0f;
              }
          }
      }

    // printf("ow=%d,oh=%d\n",ofm_w,ofm_h);
    ncnn::Mat ofm_hw(ofm_w, ofm_h, ic_t, 4u, opt.blob_allocator);
    ncnn::Mat ofm_sw(ofm_w, ofm_h, ic_t, 4u, opt.blob_allocator);

    //   printf("ifm\n");
    //   printf_float32_mat(ifm);
    #ifdef POOL_TIME
		u64 tEnd_pool_sw, tCur_pool_sw;
		u32 tUsed_pool_sw;
        XTime_GetTime(&tCur_pool_sw);
    #endif
    forward_sim_pool(ifm, ofm_sw, inst);

    #ifdef POOL_TIME
		XTime_GetTime(&tEnd_pool_sw);
		tUsed_pool_sw = ((tEnd_pool_sw-tCur_pool_sw)*1000000)/(COUNTS_PER_SECOND);
		printf("time_pool_sw elapsed is %d us\n",tUsed_pool_sw);
    #endif
    //  printf("sw is finish \n");
    //   printf("ofm_sw\n");
    //   printf_float32_mat(ofm_sw.channel(0));

    forward_ncnn_pool(ifm, ofm_hw, inst);
    //  printf("hw is finish \n");
    //   printf("ofm_hw\n");
    //   printf_float32_mat(ofm_hw.channel(0));

    // printf("ifm_addr=%p\n", (void*)&ifm.channel(0)[0]);
    // printf("ofm_sw_addr=%p\n", (void*)&ofm_sw.channel(0)[0]);
    // printf("ofm_hw_addr=%p\n", (void*)&ofm_hw.channel(0)[0]);

#ifdef MAT_LOG
    log_mat_file<float>(ofm_hw, (char *)"./log/pool/log_ofm_ncnn.txt", -1, 0, 1);
    log_mat_file<float>(ofm_sw, (char *)"./log/pool/log_ofm_sim.txt", -1, 0, 1);
#endif

    if (CompareMat(ofm_sw, ofm_hw, 0.001) == 0){
        printf("\033[;32m[ log ]: TEST LAYER POOL BETWEEN NCNN AND SIMULATION PASS\n\033[0m");
        count_success_pool++;
    }
    else {
        printf("[ log ]: ifm_w = %d, ifm_h = %d, ifm_c = %d\n", ifm.w, ifm.h, ifm.c);
        printf("[ log ]: ofm_w = %d, ofm_h = %d, ofm_c = %d\n", ofm_hw.w, ofm_hw.h, ofm_hw.c);
        printf("[ log ]: pool_type = %d, kernel = %d, stride = %d\n", type_t, k_t, s_t);
        printf("[ log ]: pad_left = %d, pad_right = %d, pad_top = %d, pad_bottom = %d\n", pd_left_t, pd_right_t, pd_top_t, pd_down_t);

        printf("\033[;31m[ log ]: TEST LAYER POOL BETWEEN NCNN AND SIMULATION FAILED\n\033[0m");
        count_fail_pool ++;
    }
    printf("[ log ]: success => test_summary [%d / %d]\n",count_success_pool,BATCH_TEST_CYCYLES);
    printf("[ log ]: fail => test_summary [%d / %d]\n",count_fail_pool,BATCH_TEST_CYCYLES);
    printf("\n");
}

int rand_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

float pool_rand_test(int rand_test[], int max_wh, int max_c) {
    int w, h, c, k, pool_type, pd_l, pd_r, pd_t, pd_b, s, t;

    // wh: 4-1024濠电偞鍨堕弻銊╊敄婢跺á娑㈠锤濡や礁鍓梺鍛婃处閸ㄩ亶顢樿ぐ鎺撶厸闁割偁鍨归崝瀣煛閸☆厽瀚�
    // c: 1-1024濠电偞鍨堕弻銊╊敄婢跺á娑㈠锤濡や礁鍓梺鍛婃处閸ㄩ亶顢樿ぐ鎺撶厸闁割偁鍨归崝瀣煛閸☆厽瀚�
    // w = rand_int(4, max_wh);
    // h = rand_int(4, max_wh);
    // c = rand_int(1, max_c);
    w = rand_int(512, max_wh);
    h = rand_int(512, max_wh);
    c = rand_int(1, max_c);
    
//    while (w * h * c > 1024 * 1024 * 4) { //whc闂備焦鐪归崝宀�锟芥凹浜炵划濠氭嚒閵堝洨鐓嬮梺璺ㄥ櫐閹凤拷 闂佽绻愮换鎰崲鐎ｎ剝濮虫い鎺戝�绘す鍐差熆鐠轰警妲洪柤鐧告嫹 1024*1024*4 (512*512*16)
//        c >>= 1; // 闂備礁鎲￠悢顒傜不閹达箑鍨傞柣銉︻殜閺屾稑鈻庤箛鏇礊濞存粣鎷�
//    } /* 闂佽楠稿﹢閬嶅磻閵堝拋鐎舵い鏍ㄧ〒閳绘棃鏌ㄩ悤鍌涘1024*1024*1024闂佽崵濮烽崕銈夊垂閼测晜宕查柛鎰ㄦ櫇椤╂煡鏌涢埄鍐剧劷婵﹪绠栧濠氬磼閵堝懏鐝濆銈忕秶閹凤拷, 濠电偠鎻徊钘壩涘▎寰綁宕ㄦ繝鍕垫祫闁荤姵浜介崝宥夊春閸岀偞鐓涢柛鎰╁妿閸╋絾淇婇銏虎妞ゆ洏鍎靛畷濂稿Ψ閿旂晫妲� */

    k = rand_int(2, 3); // kernel_size = 2 闂備胶顢婇幓顏堝箯閿燂拷 3

    if (w > 512 || h > 1020)
        s = k; // ofm_w > 512, 闂備礁鎲＄敮妤呮偡閵堝绠柨鐕傛嫹 stride = kernel_size, 闂備焦鎮堕崕铏叏瀹曞洦瀚婚柣鏃傚帶鐎氬鏌ㄩ悢璇残撶�垫澘瀚ˇ鎶芥煛娓氬洤娅嶆鐐村姈閹峰懘鎮滃Ο鍏兼 ofm_w = 512
    else
        s = rand() % k + 1; // stride = 1闂備胶顢婇幓顏堝箯閿燂拷2 (闂備胶顢婇幓顏堝箯閿燂拷3 when kernel = 3)

    pool_type = rand() % 2;
    t = rand() % 2; // pad_type, 0闂備礁鎼�氱兘宕规担鍐╋紓闂佹眹鍩勯崹浼村箟閿熺姴姹查柨鐕傛嫹,1闂備礁鎼�氱兘宕规担鍐╋紓闂佸搫顦悧鍡楋耿闁秴鐤鹃柨鐕傛嫹

    // 闂佽崵濮撮鍛村疮椤栫偛姹叉い銈囄╠ding, 濠电偠鎻徊浠嬫倶濠靛牊顐介柨鐕傛嫹 w+pd_l+pd_r-k闂備礁鎼�氼喖顪冮敓浠嬫⒑濮瑰洤濡奸悗姘煎弮瀵娊鏁愭径濠冪�梺鍝勵槸鐏忓懘骞忛悜鑺ユ櫢闁跨噦鎷�, h+pd_t+pd_b-k闂備礁鎼�氼喖顪冮敓浠嬫⒑濮瑰洤濡奸悗姘煎弮瀵娊鏁愭径濠冪�梺鍝勵槸鐏忓懘骞忛悜鑺ユ櫢闁跨噦鎷�
    do {
        pd_l = rand_int(0, 2);
        pd_r = rand_int(0, 2);
    } while ((w + pd_l + pd_r - k) % s != 0);

    do {
        pd_t = rand_int(0, 2);
        pd_b = rand_int(0, 2);
    } while ((h + pd_t + pd_b - k) % s != 0);

    rand_test[0] = w;
    rand_test[1] = h;
    rand_test[2] = c;
    rand_test[3] = k;
    rand_test[4] = s;
    rand_test[5] = t;
    rand_test[6] = pd_l;
    rand_test[7] = pd_r;
    rand_test[8] = pd_t;
    rand_test[9] = pd_b;
    rand_test[10] = pool_type;
    float v = static_cast<float>(rand()) / RAND_MAX * 2048.0f - 1024.0f;
    
    printf("----------------------------------------\nw: %d, h: %d, c: %d\n", w, h, c);
    printf("kernel: %d, stride: %d, pool_type: %s\n", k, s, pool_type == POOL_AVG ? "avg" : "max");
    printf("pad_l: %d, pad_r: %d, pad_t: %d, pad_b: %d\n", pd_l, pd_r, pd_t, pd_b);
    printf("pad_type: %d, pad_value: %.2f\n", t, v);

    return v;

}
#ifdef BATCH_TEST_POOL
void test_layer_pool_batch() {
    SRAND(114514);

    int rand_test[11];
    float rand_v;
    for(int i = 0; i < BATCH_TEST_CYCYLES; i++) {
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        rand_v = pool_rand_test(rand_test, 1024, 128);
        
        // rand_v = pool_rand_test(rand_test, 1024, 4);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 512, 16);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 256, 64);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 32, 256);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 32, 256);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 32, 256);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 32, 256);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 32, 256);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 32, 256);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 32, 1024);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
        // rand_v = pool_rand_test(rand_test, 16, 1024);
        // test_layer_pool(rand_test[0], rand_test[1], rand_test[2], rand_test[3], rand_test[4], rand_test[10], rand_test[6], rand_test[7], rand_test[8], rand_test[9], rand_test[5], rand_v);
    }

}
#else 
void test_layer_pool_batch() {
    SRAND(7767517);
    // test_layer_pool(int iw_t, int ih_t, int ic_t, int k_t, int s_t, int type_t, int pd_left_t, int pd_right_t, int pd_top_t, int pd_down_t, int pad_mode, float pad_value);
//    test_layer_pool(		416,      416,        3,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		208,      208,       32,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		104,      104,       64,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		104,      104,       64,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		416,      416,        3,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		208,      208,       32,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		104,      104,       64,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		104,      104,       64,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		 52,       52,      128,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		 26,       26,        2,       2,       2,   POOL_MAX,             0,              0,            0,             0,            1,         128.0f);
//   test_layer_pool(		 11,       11,       16,       2,       1,   POOL_AVG,             0,              0,            0,             0,            1,         128.0f);
//   test_layer_pool(		 13,       13,        2,       2,       1,   POOL_MAX,             0,              1,            0,             1,            1,         1024.0f);
//    test_layer_pool(		 13,       13,        2,       2,       1,   POOL_AVG,             0,              1,            0,             1,            1,         1024.0f);
//    test_layer_pool(		 13,       13,        2,       2,       1,   POOL_MAX,             0,              1,            0,             1,            1,         0.0f);
//    test_layer_pool(		 13,       13,        2,       2,       1,   POOL_AVG,             0,              1,            0,             1,            1,         0.0f);
//    test_layer_pool(		 20,       16,        256,       2,       2,   POOL_MAX,             0,              0,            0,             0,            0,         0.0f);
//    test_layer_pool(		 13,       13,      512,       2,       1,   POOL_MAX,             0,              1,            0,             1,            1,         -128.0f);
//    test_layer_pool(		204,      111,       116,       3,       2,   POOL_MAX,             1,              0,            0,             0,            1,         176.0f);
//    test_layer_pool(		204,      111,       116,       3,       2,   POOL_AVG,             1,              0,            0,             0,            1,         176.0f);
//    test_layer_pool(		 52,       52,      128,       5,       1,   POOL_MAX,             3,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		 52,       52,      128,       7,       1,   POOL_MAX,             0,              0,            0,             0,            1,         -128.0f);
//    test_layer_pool(		 52,       52,      128,       7,       1,   POOL_MAX,              3,              3,            3,             3 ,           1,         -128.0f);
//    test_layer_pool(		 52,       52,      128,       9,       1,   POOL_MAX,             4,              4,            4,             4 ,           1,         -128.0f);
//    test_layer_pool(		 52,       52,      128,       13,       1,   POOL_MAX,             6,              6,            6,             6,            1,         -128.0f);
//   test_layer_pool(		 28,       28,      1,       3,       2,   POOL_AVG,             0,              0,            0,             0,            1,         -12.0f);

    // test_layer_pool(		 13,       13,        2,       5,       1,   POOL_MAX,             0,              1,            0,             1,            1,         0.0f);

    test_layer_pool(		 22,       28,        2,       3,       1,   POOL_AVG,             0,              0,            1,             2,            0,         0.0f);
    test_layer_pool(		204,      111,       116,       3,       2,   POOL_AVG,             1,              0,            0,             0,            1,         176.0f);
    test_layer_pool(		 52,       42,        128,     2,       2,   POOL_AVG,             0,              0,            1,             2,            0,         0.0f);


//    test_layer_pool(52, 52, 128,2, 2, POOL_AVG, 0, 0, 0, 0,1, -128.0f);
//    test_layer_pool(26,26,256, 2, 2, POOL_AVG, 0, 0, 0, 0, 1, -128.0f) ;
//    test_layer_pool(13,13, 512, 2,1, POOL_AVG, 0, 1, 0, 1,1, -128.0f);
//    test_layer_pool(13,13,512, 2, 1, POOL_AVG, 0, 1, 0, 1,1,-128.0f) ;
//    test_layer_pool(9,9,2, 2, 1, POOL_AVG, 0, 1, 0, 1, 1,-1280.0f) ;
//     test_layer_pool(9,9,2, 2, 1, POOL_AVG, 0, 1, 0, 1,0,-1280.0f) ;
//    test_layer_pool(13,13,2, 2, 1, POOL_MAX, 0, 1, 0, 1,0,-1280.0f) ;
//    test_layer_pool(13,13,2, 2, 1, POOL_MAX, 0, 1, 0, 1, 1,-1280.0f) ;
    // test_layer_pool(13,13,1, 2, 1, POOL_MAX, 0, 1, 0, 1,0,128.0f) ;


    
    // test_layer_pool(52, 52, 128, 2, 2, POOL_AVG, 0, 0, 0, 0,0,-128.0f );
    // test_layer_pool(26, 26, 256, 2, 2, POOL_AVG, 0, 0, 0, 0,0,-128.0f );
    // test_layer_pool(13, 13, 512, 2, 1, POOL_AVG, 0, 1, 0, 1,0,-128.0f );


    // test_layer_pool(19,19,3, 3,3, POOL_MAX, 1,1,1,1,0, -128.0f);
    // test_layer_pool(18,18,3, 2,2, POOL_MAX, 0,0,0,0,0, -128.0f);
    // test_layer_pool(19,19,3, 3,3, POOL_AVG,1,1,1,1,0, -128.0f);
    // test_layer_pool(208 ,208,32, 2,2, POOL_AVG,0,0,0,0,0, -128.0f);


//     test_layer_pool(19,19,3, 3,3, POOL_MAX, 1,1,1,1,0, -128.7f);
//     test_layer_pool(18,18,3, 2,2, POOL_MAX, 0,0,0,0,0, -128.2f);
//     test_layer_pool(19,19,3, 3,3, POOL_AVG, 1,1,1,1,0, -18.5f);
    // test_layer_pool(208,208,32, 2,2, POOL_AVG, 0,0,0,0,0, -128.0f);
//    test_layer_pool(13,5,185, 3,2, POOL_MAX, 2,0,0,2,1, -18.5f);
}
#endif

} // namespace test
