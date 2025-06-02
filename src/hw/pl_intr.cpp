#include "pl_intr.h"
#include "sleep.h"


int alu_mat_task_done = 0;
int pool_task_done = 0;
int gemm_task_done = 0;
int hdmi_input_flag = 0;

void init_intr_sys(void)
{
	Init_Intr_System(&Intc);

	pl_intr_init(&Intc);
	Setup_Intr_Exception(&Intc);
	printf("[ log ]: INTR INIT PASS\r\n");
}


void SW1_intr_Handler(void *param)
{
    // int sw_id = (long)param;
//    printf("[ log ]: ENTER ALU_MAT INTR, ID = %d\n", sw_id);
    alu_mat_task_done = 1;
}

void SW2_intr_Handler(void *param)
{
    // int sw_id = (long)param;
//    printf("[ log ]: ENTER POOL INTR, ID = %d\n", sw_id);
    pool_task_done = 1;
}

void SW3_intr_Handler(void *param)
{
    // int sw_id = (long)param;
//    printf("[ log ]: ENTER GEMM INTR, ID = %d\n", sw_id);
    gemm_task_done = 1;
}
void SW4_intr_Handler(void *param)
{
    hdmi_input_flag = 1;
    XScuGic_Disable(&Intc, SW4_INT_ID);
    // int sw_id = (long)param;
    // printf("[ log ]: ENTER KEY INTR, ID = %d\n", sw_id);
    // int frame_id = Xil_In32(0x80010000);
    // printf("[ log ]: frame_id = %d\n",frame_id);
//    if(frame_id == 2)
//    video_buffer_update((u8*)VIDEOOUT_DBUF[(wframe1_cnt+1)%3],&RunCfg);
//     video_buffer_update((u8*)VIDEOOUT_DBUF[0],&RunCfg);
//    video_buffer_update((u8*)VIDEOOUT_DBUF[frame_id],&RunCfg);

    // video_buffer_update((u8*)VIDEOOUT_DBUF[(frame_id+2)%3],&RunCfg);
    // if(count == 151){
    //     printf("enter write pic\n");
    //     sprintf(imagename0, "frame%04u.bmp", (frame_id+0)%3);
    //     sprintf(imagename1, "frame%04u.bmp", (frame_id+1)%3);
    //     sprintf(imagename2, "frame%04u.bmp", (frame_id+2)%3);
    //     bmp_write(imagename0, (char *)&BMP_HEADER_PARAM, (char *)VIDEOOUT_DBUF[(frame_id+0)%3], LINESIZE) ;
    //     bmp_write(imagename1, (char *)&BMP_HEADER_PARAM, (char *)VIDEOOUT_DBUF[(frame_id+1)%3], LINESIZE) ;
    //     bmp_write(imagename2, (char *)&BMP_HEADER_PARAM, (char *)VIDEOOUT_DBUF[(frame_id+2)%3], LINESIZE) ;
    //     XScuGic_Disable(&Intc, SW1_INT_ID);
    //     printf("write pic finish\n");
    // }
}
// void SW5_intr_Handler(void *param)
// {
//     // int sw_id = (long)param;
// //    printf("[ log ]: ENTER GEMM INTR, ID = %d\n", sw_id);
//     resize_task_done = 1;
// }




static void pl_intr_setup(XScuGic *InstancePtr, int intId, int intType)
{
    int mask;

    intType &= PL_INT_TYPE_MASK;
    mask = XScuGic_DistReadReg(InstancePtr, INT_CFG0_OFFSET + (intId/16)*4);
    mask &= ~(PL_INT_TYPE_MASK << (intId%16)*2);
    mask |= intType << ((intId%16)*2);
    XScuGic_DistWriteReg(InstancePtr, INT_CFG0_OFFSET + (intId/16)*4, mask);
}

int pl_intr_init(XScuGic * InstancePtr)
{
	int status;

    // Connect SW1~SW4 interrupt to handler
    status = XScuGic_Connect(InstancePtr,SW1_INT_ID,(Xil_ExceptionHandler)SW1_intr_Handler,(void *)1);
    if(status != XST_SUCCESS) return XST_FAILURE;

    status = XScuGic_Connect(InstancePtr,SW2_INT_ID,(Xil_ExceptionHandler)SW2_intr_Handler,(void *)2);
    if(status != XST_SUCCESS) return XST_FAILURE;

    status = XScuGic_Connect(InstancePtr,SW3_INT_ID,(Xil_ExceptionHandler)SW3_intr_Handler,(void *)3);
    if(status != XST_SUCCESS) return XST_FAILURE;

    status = XScuGic_Connect(InstancePtr,SW4_INT_ID,(Xil_ExceptionHandler)SW4_intr_Handler,(void *)4);
    if(status != XST_SUCCESS) return XST_FAILURE;

//    status = XScuGic_Connect(InstancePtr,SW5_INT_ID,(Xil_ExceptionHandler)SW5_intr_Handler,(void *)5);
//        if(status != XST_SUCCESS) return XST_FAILURE;

    // Set interrupt type of SW1~SW4 to rising edge
    pl_intr_setup(InstancePtr, SW1_INT_ID, PL_INT_TYPE_RISING_EDGE);
    pl_intr_setup(InstancePtr, SW2_INT_ID, PL_INT_TYPE_RISING_EDGE);
    pl_intr_setup(InstancePtr, SW3_INT_ID, PL_INT_TYPE_RISING_EDGE);
    pl_intr_setup(InstancePtr, SW4_INT_ID, PL_INT_TYPE_RISING_EDGE);
//    pl_intr_setup(InstancePtr, SW5_INT_ID, PL_INT_TYPE_RISING_EDGE);

    // Enable SW1~SW2 interrupts in the controller
    XScuGic_Enable(InstancePtr, SW1_INT_ID);
    XScuGic_Enable(InstancePtr, SW2_INT_ID);
    XScuGic_Enable(InstancePtr, SW3_INT_ID);
    XScuGic_Enable(InstancePtr, SW4_INT_ID);
//    XScuGic_Enable(InstancePtr, SW5_INT_ID);


    return XST_SUCCESS;
}

