
#ifndef __PL_INTR_H_
#define __PL_INTR_H_

#include <stdio.h>
#include "xscugic.h"
#include "xil_exception.h"
#include "sys_intr.h"
//#include "dpdma_intr.h"


#define INT_CFG0_OFFSET 		0x00000C00
// Parameter definitions
#define SW1_INT_ID              121
#define SW2_INT_ID              122
#define SW3_INT_ID              123
#define SW4_INT_ID              124
#define SW5_INT_ID              125
#define PL_INT_TYPE_RISING_EDGE 0x03
#define PL_INT_TYPE_HIGHLEVEL   0x01
#define PL_INT_TYPE_MASK        0x03

extern int alu_mat_task_done;
extern int pool_task_done;
extern int gemm_task_done;
extern int hdmi_input_flag;

void SW1_intr_Handler(void *param);
void SW2_intr_Handler(void *param);
void SW3_intr_Handler(void *param);
void SW4_intr_Handler(void *param);
void SW5_intr_Handler(void *param);

int pl_intr_init(XScuGic * InstancePtr);
void init_intr_sys(void);

#endif
