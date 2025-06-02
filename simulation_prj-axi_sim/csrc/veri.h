#ifndef __VERI_H__
#define __VERI_H__


#include "verilated_vcd_c.h"
#include "Vtcp.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "ncnn/mat.h"
#include "hw/hw_gemm.h"
#include <cerrno>


extern vluint64_t main_time;
extern vluint64_t ofmbuf_congested_time;
extern uint64_t   time_pool;
extern uint64_t   time_gemm;

extern uint64_t time_dma_ch0_r;
extern uint64_t time_dma_ch1_r;
extern uint64_t time_dma_ch0_w;
extern uint64_t time_dma_ch1_w;
extern uint64_t time_ifmbuf_load;
extern uint64_t time_ofmbuf_congest;
extern uint64_t time_axisendbuf_congest;

extern Vtcp *top;
#ifndef BATCH_TEST
extern VerilatedVcdC* tfp;
#endif

extern int alu_math_data;

double sc_time_stamp();
int exec(uint64_t n);
void update();
void dma_wait();
void dma_wait(ncnn::Mat& data_res, accel::hw_gemm& inst);
void axi_lite_wait();

#endif