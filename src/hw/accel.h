#ifndef __ACCEL_H_
#define __ACCEL_H_

#include "pl_intr.h"
#include "accel_params.h"
#include "xil_cache.h"
#include "../ncnn/mat.h"

#define u_align(x,n)  ((x+n-1) & -n)  

static inline void wait_alu_mat_done(){
    while(!alu_mat_task_done);
    alu_mat_task_done = 0;
}

static inline void wait_pool_done(){
    while(!pool_task_done);
    pool_task_done = 0;
}

static inline void wait_gemm_done(){
    while(!gemm_task_done);
    gemm_task_done = 0;
}



#endif
