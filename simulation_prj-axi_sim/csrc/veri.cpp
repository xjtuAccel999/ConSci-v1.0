#include "veri.h"
#include "config.h"
#include "hw/accel_params.h"
#include "net/cnn_demo.h"
#include "testLayer/test_layer.h"
#include "testOp/test_op.h"
#include "utils/utils.h"
#include <queue>

vluint64_t main_time = 0;             // initial 仿真时间
vluint64_t ofmbuf_congested_time = 0; // initial 仿真时间
uint64_t time_pool = 0;
uint64_t time_gemm = 0;

uint64_t time_dma_ch0_r = 0;
uint64_t time_dma_ch1_r = 0;
uint64_t time_dma_ch0_w = 0;
uint64_t time_dma_ch1_w = 0;
uint64_t time_ifmbuf_load = 0;
uint64_t time_ofmbuf_congest = 0;
uint64_t time_axisendbuf_congest = 0;

Vtcp *top = new Vtcp("tcp");
#ifdef WAVE_LOG
VerilatedVcdC *tfp = new VerilatedVcdC(); // 导出vcd波形需要加此语句
#endif

#define SEXT(x, len)                                                                                                                                 \
    ({                                                                                                                                               \
        struct {                                                                                                                                     \
            int64_t n : len;                                                                                                                         \
        } __x = {.n = x};                                                                                                                            \
        (uint64_t) __x.n;                                                                                                                            \
    })

double sc_time_stamp() { return main_time; }

static int compare(accel::hw_gemm &inst, int a, int b) {
    if (inst.oscale_en) {
        return compareFloat(a, b);
    } else {
        return a == b;
    }
}

static int my_clock = 0;
void update() {
    for (int i = 0; i < 2; i++) {
        top->clock = my_clock;
        my_clock = !my_clock;
        top->eval();
#ifdef WAVE_LOG
        // if (main_time > 200000) {
        tfp->dump(main_time);
        if (main_time > 200000) {
            assert(0);
        }
        // }
#endif
        main_time++;
    }

    // if(top->io_ofmbuf_stop == 1)
    //     ofmbuf_congested_time ++;
}

#ifndef ONLY_RESET
void reset() {
    top->reset = 1;

    top->io_sim_alu_reg_mathfunc_ctrl_reg = 0;
    top->io_sim_alu_reg_actfunc_ctrl_reg = 0;
    top->io_sim_alu_reg_alu_veclen_ch0_reg = 0;
    top->io_sim_alu_reg_src0_addr_ch0_reg = 0;
    top->io_sim_alu_reg_src1_addr_ch0_reg = 0;
    top->io_sim_alu_reg_dst_addr_ch0_reg = 0;
    top->io_sim_alu_reg_alu_veclen_ch1_reg = 0;
    top->io_sim_alu_reg_src0_addr_ch1_reg = 0;
    top->io_sim_alu_reg_src1_addr_ch1_reg = 0;
    top->io_sim_alu_reg_dst_addr_ch1_reg = 0;
    top->io_sim_alu_reg_math_alpha_reg = 0;
    top->io_sim_alu_reg_math_beta_reg = 0;
    // top->io_sim_alu_reg_act_alpha_reg = 0;
    // top->io_sim_alu_reg_act_beta_reg = 0;
    top->io_sim_alu_reg_act_coefficient_a_reg_0 = 0;
    top->io_sim_alu_reg_act_coefficient_a_reg_1 = 0;
    top->io_sim_alu_reg_act_coefficient_a_reg_2 = 0;
    top->io_sim_alu_reg_act_coefficient_a_reg_3 = 0;
    top->io_sim_alu_reg_act_coefficient_a_reg_4 = 0;
    top->io_sim_alu_reg_act_coefficient_b_reg_0 = 0;
    top->io_sim_alu_reg_act_coefficient_b_reg_1 = 0;
    top->io_sim_alu_reg_act_coefficient_b_reg_2 = 0;
    top->io_sim_alu_reg_act_coefficient_b_reg_3 = 0;
    top->io_sim_alu_reg_act_coefficient_b_reg_4 = 0;
    top->io_sim_alu_reg_act_coefficient_c_reg_0 = 0;
    top->io_sim_alu_reg_act_coefficient_c_reg_1 = 0;
    top->io_sim_alu_reg_act_coefficient_c_reg_2 = 0;
    top->io_sim_alu_reg_act_coefficient_c_reg_3 = 0;
    top->io_sim_alu_reg_act_coefficient_c_reg_4 = 0;
    top->io_sim_alu_reg_act_range_reg_0 = 0;
    top->io_sim_alu_reg_act_range_reg_1 = 0;
    top->io_sim_alu_reg_act_range_reg_2 = 0;
    top->io_sim_alu_reg_act_range_reg_3 = 0;

    top->io_sim_pool_reg_pool_ctrl_reg = 0;
    top->io_sim_pool_reg_pool_shape_ic_reg = 0;
    top->io_sim_pool_reg_pool_shape_iwh_reg = 0;
    top->io_sim_pool_reg_pool_shape_icstep_reg = 0;
    top->io_sim_pool_reg_pool_shape_oc_reg = 0;
    top->io_sim_pool_reg_pool_shape_owh_reg = 0;
    top->io_sim_pool_reg_pool_shape_ocstep_reg = 0;
    top->io_sim_pool_reg_pool_ifm_addr_reg = 0;
    top->io_sim_pool_reg_pool_ofm_addr_reg = 0;
    top->io_sim_gemm_reg_gemm_ctrl_reg = 0;
    top->io_sim_gemm_reg_quant_data_reg = 0;
    top->io_sim_gemm_reg_requant_data_reg = 0;
    top->io_sim_gemm_reg_dequant_addr_reg = 0;
    top->io_sim_gemm_reg_bias_addr_reg = 0;
    top->io_sim_gemm_reg_ifm_shape_c_reg = 0;
    top->io_sim_gemm_reg_ifm_shape_wh_reg = 0;
    top->io_sim_gemm_reg_ifm_shape_cstep_reg = 0;
    top->io_sim_gemm_reg_ofm_shape_c_reg = 0;
    top->io_sim_gemm_reg_ofm_shape_wh_reg = 0;
    top->io_sim_gemm_reg_ofm_shape_cstep_reg = 0;
    top->io_sim_gemm_reg_wgt_len_reg = 0;
    top->io_sim_gemm_reg_ifm_baseaddr_reg = 0;
    top->io_sim_gemm_reg_wgt_baseaddr_reg = 0;
    top->io_sim_gemm_reg_ofm_baseaddr_reg = 0;

    // #ifdef DMA_SUPPORT
    //     // dma ch0
    //     top->io_dma_ch0_r_busy = 0;
    //     top->io_dma_ch0_r_valid = 0;
    //     for (int i = 0; i < DMA_DATA_WIDTH / 32; i++)
    //         top->io_dma_ch0_r_data[i] = 0;
    //     // dma ch1
    //     top->io_dma_ch1_r_busy = 0;
    //     top->io_dma_ch1_r_valid = 0;
    //     for (int i = 0; i < DMA_DATA_WIDTH / 32; i++)
    //         top->io_dma_ch1_r_data[i] = 0;
    // #endif

#ifdef TEST_IFMBUF
    top->io_ifm_mem_read_port0_ren = 0;
    top->io_ifm_mem_read_port0_raddr = 0;
    top->io_ifm_mem_read_port1_ren = 0;
    top->io_ifm_mem_read_port1_raddr = 0;
#endif

#ifdef TEST_FP32_ADD_MUL
    top->io_fp32_en = 0;
    top->io_fp32_a = 0;
    top->io_fp32_b = 0;
#endif

    update();
}
#else
void reset() {
    top->reset = 1;
    update();
}
#endif

#ifdef DMA_SUPPORT

static int align_check(uint64_t addr) {
    if (addr % (AXI_DATA_WIDTH / 8) == 0)
        return 0;
    return -1;
}

#define DELAY_VALID_R_CH0 24
#define DELAY_VALID_R_CH1 30
#define DELAY_VALID_W_CH0 24
#define DELAY_VALID_W_CH1 30
#define OUTSTANDING_DEPTH 16
static int break_flag = 0;
#define Assert(cond, format, ...)                                                                                                                    \
    do {                                                                                                                                             \
        if (!(cond)) {                                                                                                                               \
            (fflush(stdout), fprintf(stderr, ANSI_FMT(format, ANSI_FG_RED) "\n", ##__VA_ARGS__));                                                    \
            break_flag = 1;                                                                                                                          \
        }                                                                                                                                            \
    } while (0)

// static int dma_ch0_read_busy = 0;
// static uint32_t dma_ch0_read_size = 0;
// static uint64_t dma_ch0_read_addr = 0;
// static int dma_ch0_read_count = 0;

// static char flag_dma_ch0_read_valid = 0;
// static int dma_ch0_read_valid_delay = 0;

// static int dma_ch1_read_busy = 0;
// static uint32_t dma_ch1_read_size = 0;
// static uint64_t dma_ch1_read_addr = 0;
// static int dma_ch1_read_count = 0;
// static int dma_ch1_write_busy = 0;
// static uint32_t dma_ch1_write_size = 0;
// static uint64_t dma_ch1_write_addr = 0;
// static int dma_ch1_write_count = 0;

// static char flag_dma_ch1_read_valid = 0;
// static int dma_ch1_read_valid_delay = 0;
// static char flag_dma_ch1_write_valid = 0;
// static int dma_ch1_write_valid_delay = 0;

// static int dma_ch0_read_gp = 0;
// static int dma_ch1_write_gp = 0;
// static int dma_ch1_read_gp = 0;

// static int dma_ch0_write_busy = 0;
// static uint32_t dma_ch0_write_size = 0;
// static uint64_t dma_ch0_write_addr = 0;
// static int dma_ch0_write_count = 0;
// static char flag_dma_ch0_write_valid = 0;
// static int dma_ch0_write_valid_delay = 0;
// static int dma_ch0_write_gp = 0;

static std::queue<uint64_t> axi_ch0_write_addr;
static std::queue<uint32_t> axi_ch0_write_len;
static int axi_ch0_write_resp = 0;
static int axi_ch0_write_wait_cnt = 0;
static int io_axi_ch0_w_m_awready = 1;
static int io_axi_ch0_w_m_wready = 0;
static int io_axi_ch0_w_m_bvalid = 0;
void axi_ch0_w() {
    if (axi_ch0_write_len.size() == 0) { // 4. delay control
        if (top->io_axi_ch0_w_m_awvalid && top->io_axi_ch0_w_m_awready) {
            axi_ch0_write_wait_cnt = 0;
        }
    } else if (!top->io_axi_ch0_w_m_wready) {
        if (axi_ch0_write_wait_cnt <= DELAY_VALID_W_CH0) {
            axi_ch0_write_wait_cnt++;
        } else {
            io_axi_ch0_w_m_wready = 1;
        }
    }

    if (top->io_axi_ch0_w_m_awvalid && top->io_axi_ch0_w_m_awready) { // 1. transaction req start
        // axi_ch0_write_addr.push(top->io_axi_ch0_w_m_awaddr);
        axi_ch0_write_len.push(top->io_axi_ch0_w_m_awlen + 1);
        assert(top->io_axi_ch0_w_m_awsize == log2(AXI_DATA_WIDTH / 8));
        assert(top->io_axi_ch0_w_m_awburst == 1);
        for (int i = 0; i < (top->io_axi_ch0_w_m_awlen + 1); i++) {
            axi_ch0_write_addr.push((uint64_t)top->io_axi_ch0_w_m_awaddr + i * AXI_TRANSFER_BYTE);
        }
#ifdef CHECK_ALIGN
        if (align_check(top->io_axi_ch0_w_m_awaddr)) {
            printf("[ error ]: top->io_axi_ch0_w_m_awaddr = %lx, NOT ALIGN!\n", top->io_axi_ch0_w_m_awaddr);
            assert(0);
        }
#endif
        if (top->io_axi_ch0_w_m_awaddr / 4096 != (top->io_axi_ch0_w_m_awaddr + (top->io_axi_ch0_w_m_awlen + 1) * AXI_TRANSFER_BYTE - 1) / 4096) {
            printf("[ error ]: single transaction cross 4K address!\n");
        }
        if (axi_ch0_write_len.size() == OUTSTANDING_DEPTH) {
            io_axi_ch0_w_m_awready = 0;
        }
    } else if (!top->io_axi_ch0_w_m_awready && (axi_ch0_write_len.size() < OUTSTANDING_DEPTH) &&
               top->io_axi_ch0_w_m_bvalid) { // 1.1 outstanding depth not full
        io_axi_ch0_w_m_awready = 1;
    }

    if (axi_ch0_write_resp) { // 3. transaction resp
        io_axi_ch0_w_m_bvalid = 1;
        axi_ch0_write_resp = 0;
    } else if (top->io_axi_ch0_w_m_bready && top->io_axi_ch0_w_m_bvalid) {
        io_axi_ch0_w_m_bvalid = 0;
    }

    if (top->io_axi_ch0_w_m_wvalid && io_axi_ch0_w_m_wready) { // 2. write data
        // if (axi_ch0_write_len.front() == 0) {
        //     printf("[ error ]: axi_ch0_write_len.front() == 0\n");
        //     printf("[ error ]: main_time:%ld\n", main_time);
        //     printf("[ error ]: top->io_axi_ch0_w_m_wvalid:%d\n", top->io_axi_ch0_w_m_wvalid);
        //     assert(0);
        // }
        assert(axi_ch0_write_len.size() != 0);
        assert(axi_ch0_write_len.front() != 0);
        assert(top->io_axi_ch0_w_m_wstrb == 0xFFFF);
        for (int i = 0; i < AXI_DATA_WIDTH / 32; i++) {
            // printf("[ log ]:main_time=%ld, addr = %lx, data = %x\n", main_time, axi_ch0_write_addr.front(), top->io_axi_ch0_w_m_wdata[i]);
            *((int *)axi_ch0_write_addr.front() + i) = top->io_axi_ch0_w_m_wdata[i];
        }
        axi_ch0_write_addr.pop();
        if (axi_ch0_write_len.front() == 1) { // 2.1 one transaction the last transfer
            axi_ch0_write_len.pop();
            Assert(top->io_axi_ch0_w_m_wlast, "main_time:%ld, wlast=1!", main_time);
            axi_ch0_write_resp = 1;
        } else { // 2.2 not the last transfer
            axi_ch0_write_len.front()--;
            assert(!top->io_axi_ch0_w_m_wlast);
        }
    }

    top->io_axi_ch0_w_m_bid = 0;
    top->io_axi_ch0_w_m_bresp = 0;
    top->io_axi_ch0_w_m_awready = io_axi_ch0_w_m_awready;
    top->io_axi_ch0_w_m_wready = io_axi_ch0_w_m_wready;
    top->io_axi_ch0_w_m_bvalid = io_axi_ch0_w_m_bvalid;
}

static std::queue<uint64_t> axi_ch1_write_addr;
static std::queue<uint32_t> axi_ch1_write_len;
static int axi_ch1_write_resp = 0;
static int axi_ch1_write_wait_cnt = 0;
static int io_axi_ch1_w_m_awready = 1;
static int io_axi_ch1_w_m_wready = 0;
static int io_axi_ch1_w_m_bvalid = 0;
void axi_ch1_w() {
    if (axi_ch1_write_len.size() == 0) { // 4. delay control
        if (top->io_axi_ch1_w_m_awvalid && top->io_axi_ch1_w_m_awready) {
            axi_ch1_write_wait_cnt = 0;
        }
    } else if (!top->io_axi_ch1_w_m_wready) {
        if (axi_ch1_write_wait_cnt <= DELAY_VALID_W_CH1) {
            axi_ch1_write_wait_cnt++;
        } else {
            io_axi_ch1_w_m_wready = 1;
        }
    }

    if (top->io_axi_ch1_w_m_awvalid && top->io_axi_ch1_w_m_awready) { // 1. transaction req start
        // axi_ch1_write_addr.push(top->io_axi_ch1_w_m_awaddr);
        axi_ch1_write_len.push(top->io_axi_ch1_w_m_awlen + 1);
        assert(top->io_axi_ch1_w_m_awsize == log2(AXI_DATA_WIDTH / 8));
        assert(top->io_axi_ch1_w_m_awburst == 1);
        for (int i = 0; i < (top->io_axi_ch1_w_m_awlen + 1); i++) {
            axi_ch1_write_addr.push((uint64_t)top->io_axi_ch1_w_m_awaddr + i * AXI_TRANSFER_BYTE);
        }
#ifdef CHECK_ALIGN
        if (align_check(top->io_axi_ch1_w_m_awaddr)) {
            printf("[ error ]: top->io_axi_ch1_w_m_awaddr = %lx, NOT ALIGN!\n", top->io_axi_ch1_w_m_awaddr);
            assert(0);
        }
#endif
        if (top->io_axi_ch1_w_m_awaddr / 4096 != (top->io_axi_ch1_w_m_awaddr + (top->io_axi_ch1_w_m_awlen + 1) * AXI_TRANSFER_BYTE - 1) / 4096) {
            printf("[ error ]: single transaction cross 4K address!\n");
        }
        if (axi_ch1_write_len.size() == OUTSTANDING_DEPTH) {
            io_axi_ch1_w_m_awready = 0;
        }
    } else if (!top->io_axi_ch1_w_m_awready && (axi_ch1_write_len.size() < OUTSTANDING_DEPTH) &&
               top->io_axi_ch1_w_m_bvalid) { // 1.1 outstanding depth not full
        io_axi_ch1_w_m_awready = 1;
    }

    if (axi_ch1_write_resp) { // 3. transaction resp
        io_axi_ch1_w_m_bvalid = 1;
        axi_ch1_write_resp = 0;
    } else if (top->io_axi_ch1_w_m_bready && top->io_axi_ch1_w_m_bvalid) {
        io_axi_ch1_w_m_bvalid = 0;
    }

    if (top->io_axi_ch1_w_m_wvalid && io_axi_ch1_w_m_wready) { // 2. write data
        assert(axi_ch1_write_len.size() != 0);
        assert(axi_ch1_write_len.front() != 0);
        for (int i = 0; i < AXI_DATA_WIDTH / 32; i++) {
            // printf("[ log ]:main_time=%ld, addr = %lx, data = %x\n", main_time, axi_ch0_write_addr.front(), top->io_axi_ch0_w_m_wdata[i]);
            *((int *)axi_ch1_write_addr.front() + i) = top->io_axi_ch1_w_m_wdata[i];
        }
        axi_ch1_write_addr.pop();
        if (axi_ch1_write_len.front() == 1) { // 2.1 one transaction the last transfer
            axi_ch1_write_len.pop();
            assert(top->io_axi_ch1_w_m_wlast);
            axi_ch1_write_resp = 1;
        } else { // 2.2 not the last transfer
            axi_ch1_write_len.front()--;
            assert(!top->io_axi_ch1_w_m_wlast);
        }
    }

    top->io_axi_ch1_w_m_bid = 0;
    top->io_axi_ch1_w_m_bresp = 0;
    top->io_axi_ch1_w_m_awready = io_axi_ch1_w_m_awready;
    top->io_axi_ch1_w_m_wready = io_axi_ch1_w_m_wready;
    top->io_axi_ch1_w_m_bvalid = io_axi_ch1_w_m_bvalid;
}

static std::queue<uint64_t> axi_ch0_read_addr;
static std::queue<uint32_t> axi_ch0_read_len;
static int axi_ch0_read_wait_cnt = 0;
static int io_axi_ch0_r_m_arready = 1;
static int io_axi_ch0_r_m_rvalid = 0;
static int io_axi_ch0_r_m_rlast = 0;

static uint32_t io_axi_ch0_r_m_rdata[4];
void axi_ch0_r() {
    if (axi_ch0_read_len.size() == 0) { // 4. delay control
        if (top->io_axi_ch0_r_m_arvalid && top->io_axi_ch0_r_m_arready) {
            axi_ch0_read_wait_cnt = 0;
        }
    } else if (!top->io_axi_ch0_r_m_rvalid) {
        if (axi_ch0_read_wait_cnt <= DELAY_VALID_R_CH0) {
            axi_ch0_read_wait_cnt++;
        } else {
            io_axi_ch0_r_m_rvalid = 1;
            // printf("[ log ]:main_time=%ld, axi_ch0_read_wait_cnt = %d, axi_ch0_read_len.size=%d\n", main_time, axi_ch0_read_wait_cnt,
            //        axi_ch0_read_len.size());
        }
    }

    if (top->io_axi_ch0_r_m_arvalid && top->io_axi_ch0_r_m_arready) { // 1. transaction req start
        axi_ch0_read_len.push(top->io_axi_ch0_r_m_arlen + 1);
        assert(top->io_axi_ch0_r_m_arsize == log2(AXI_DATA_WIDTH / 8));
        assert(top->io_axi_ch0_r_m_arburst == 1);
		// printf("read address : %lx\n", top->io_axi_ch0_r_m_araddr);
        for (int i = 0; i < (top->io_axi_ch0_r_m_arlen + 1); i++) {
            axi_ch0_read_addr.push((uint64_t)top->io_axi_ch0_r_m_araddr + i * AXI_TRANSFER_BYTE);
        }
#ifdef CHECK_ALIGN
        if (align_check(top->io_axi_ch0_r_m_araddr)) {
            printf("[ error ]: top->io_axi_ch0_r_m_araddr = %lx, NOT ALIGN!\n", top->io_axi_ch0_r_m_araddr);
            assert(0);
        }
#endif
        if (top->io_axi_ch0_r_m_araddr / 4096 != (top->io_axi_ch0_r_m_araddr + (top->io_axi_ch0_r_m_arlen + 1) * AXI_TRANSFER_BYTE - 1) / 4096) {
            printf("[ error ]: single transaction cross 4K address!\n");
        }

        if (axi_ch0_read_len.size() == OUTSTANDING_DEPTH) {
            io_axi_ch0_r_m_arready = 0;
        }
    } else if (!top->io_axi_ch0_r_m_arready && (axi_ch0_read_len.size() < OUTSTANDING_DEPTH) &&
               (top->io_axi_ch0_r_m_rvalid && top->io_axi_ch0_r_m_rlast)) { // outstanding depth not full
        io_axi_ch0_r_m_arready = 1;
    }

    if (top->io_axi_ch0_r_m_rlast && top->io_axi_ch0_r_m_rvalid && top->io_axi_ch0_r_m_rready) { // 3. after last read
        if (axi_ch0_read_len.size() == 0) {
            io_axi_ch0_r_m_rvalid = 0;
            io_axi_ch0_r_m_rlast = 0;
        } else if (axi_ch0_read_len.front() != 1) {
            io_axi_ch0_r_m_rvalid = 0;
            io_axi_ch0_r_m_rlast = 0;
        }
    }

    if (io_axi_ch0_r_m_rvalid) { // 2. read data
        assert(axi_ch0_read_len.size() != 0);
        assert(axi_ch0_read_len.front() != 0);
        for (int i = 0; i < AXI_DATA_WIDTH / 32; i++) {
            io_axi_ch0_r_m_rdata[i] = *((int *)axi_ch0_read_addr.front() + i);
        }
        if (top->io_axi_ch0_r_m_rready) {
            axi_ch0_read_addr.pop();
            if (axi_ch0_read_len.front() == 1) { // 2.1 one transaction the last transfer
                axi_ch0_read_len.pop();
                io_axi_ch0_r_m_rlast = 1;
            } else { // 2.2 not the last transfer
                axi_ch0_read_len.front()--;
                io_axi_ch0_r_m_rlast = 0;
            }
        }
    }

    top->io_axi_ch0_r_m_arready = io_axi_ch0_r_m_arready;
    top->io_axi_ch0_r_m_rid = 0;
    top->io_axi_ch0_r_m_rresp = 0;
    top->io_axi_ch0_r_m_rvalid = io_axi_ch0_r_m_rvalid;
    for (int i = 0; i < 4; i++) {
        top->io_axi_ch0_r_m_rdata[i] = io_axi_ch0_r_m_rdata[i];
    }
    top->io_axi_ch0_r_m_rlast = io_axi_ch0_r_m_rlast;
}

static std::queue<uint64_t> axi_ch1_read_addr;
static std::queue<uint32_t> axi_ch1_read_len;
static int axi_ch1_read_wait_cnt = 0;
static int io_axi_ch1_r_m_arready = 1;
static int io_axi_ch1_r_m_rvalid = 0;
static int io_axi_ch1_r_m_rlast = 0;
static uint32_t io_axi_ch1_r_m_rdata[4];
void axi_ch1_r() {
    if (axi_ch1_read_len.size() == 0) { // 4. delay control
        if (top->io_axi_ch1_r_m_arvalid && top->io_axi_ch1_r_m_arready) {
            axi_ch1_read_wait_cnt = 0;
        }
    } else if (!top->io_axi_ch1_r_m_rvalid) {
        if (axi_ch1_read_wait_cnt <= DELAY_VALID_R_CH1) {
            axi_ch1_read_wait_cnt++;
        } else {
            io_axi_ch1_r_m_rvalid = 1;
        }
    }

    if (top->io_axi_ch1_r_m_arvalid && top->io_axi_ch1_r_m_arready) { // 1. transaction req start
        axi_ch1_read_len.push(top->io_axi_ch1_r_m_arlen + 1);
        assert(top->io_axi_ch1_r_m_arsize == log2(AXI_DATA_WIDTH / 8));
        assert(top->io_axi_ch1_r_m_arburst == 1);
        for (int i = 0; i < (top->io_axi_ch1_r_m_arlen + 1); i++) {
            axi_ch1_read_addr.push((uint64_t)top->io_axi_ch1_r_m_araddr + i * AXI_TRANSFER_BYTE);
        }
#ifdef CHECK_ALIGN
        if (align_check(top->io_axi_ch1_r_m_araddr)) {
            printf("[ error ]: top->io_axi_ch1_r_m_araddr = %lx, NOT ALIGN!\n", top->io_axi_ch1_r_m_araddr);
            assert(0);
        }
#endif
        if (top->io_axi_ch1_r_m_araddr / 4096 != (top->io_axi_ch1_r_m_araddr + (top->io_axi_ch1_r_m_arlen + 1) * AXI_TRANSFER_BYTE - 1) / 4096) {
            printf("[ error ]: single transaction cross 4K address!\n");
        }

        if (axi_ch1_read_len.size() == OUTSTANDING_DEPTH) {
            io_axi_ch1_r_m_arready = 0;
        }
    } else if (!top->io_axi_ch1_r_m_arready && (axi_ch1_read_len.size() < OUTSTANDING_DEPTH) &&
               (top->io_axi_ch1_r_m_rvalid && top->io_axi_ch1_r_m_rlast)) { // outstanding depth not full
        io_axi_ch1_r_m_arready = 1;
    }

    if (top->io_axi_ch1_r_m_rlast && top->io_axi_ch1_r_m_rvalid && top->io_axi_ch1_r_m_rready) { // 3. after last read
        if (axi_ch1_read_len.size() == 0) {
            io_axi_ch1_r_m_rvalid = 0;
            io_axi_ch1_r_m_rlast = 0;
        } else if (axi_ch1_read_len.front() != 1) {
            io_axi_ch1_r_m_rvalid = 0;
            io_axi_ch1_r_m_rlast = 0;
        }
    }

    if (io_axi_ch1_r_m_rvalid) { // 2. read data
        assert(axi_ch1_read_len.size() != 0);
        assert(axi_ch1_read_len.front() != 0);
        for (int i = 0; i < AXI_DATA_WIDTH / 32; i++) {
            io_axi_ch1_r_m_rdata[i] = *((int *)axi_ch1_read_addr.front() + i);
        }
        if (top->io_axi_ch1_r_m_rready) {
            axi_ch1_read_addr.pop();
            if (axi_ch1_read_len.front() == 1) { // 2.1 one transaction the last transfer
                axi_ch1_read_len.pop();
                io_axi_ch1_r_m_rlast = 1;
            } else { // 2.2 not the last transfer
                axi_ch1_read_len.front()--;
                io_axi_ch1_r_m_rlast = 0;
            }
        }
    }

    top->io_axi_ch1_r_m_arready = io_axi_ch1_r_m_arready;
    top->io_axi_ch1_r_m_rid = 0;
    top->io_axi_ch1_r_m_rresp = 0;
    top->io_axi_ch1_r_m_rvalid = io_axi_ch1_r_m_rvalid;
    for (int i = 0; i < 4; i++) {
        top->io_axi_ch1_r_m_rdata[i] = io_axi_ch1_r_m_rdata[i];
    }
    top->io_axi_ch1_r_m_rlast = io_axi_ch1_r_m_rlast;
}

void dma_wait() {
    while (1) {
    // for (int i = 0; i < 5000; i++) {
        // printf("main_time = %ld, i = %d\n", main_time,i);
        { // dma0 read  channels
            axi_ch0_r();
        }
        { // dma0 write channels
            axi_ch0_w();
        }
        { // dma1 read  channels
            axi_ch1_r();
        }
        { // dma1 write channels
            axi_ch1_w();
        }

#ifdef TEST_IFMBUF
        if (top->io_ifmbuf_task_done == 1) {
            break;
        }
#endif

#ifdef USE_HW_ALU
        if (top->io_alu_task_done == 1) {
            break;
        }
#endif

#ifdef USE_HW_POOL
        if (top->io_pool_task_done == 1) {
            break;
        }
#endif

#ifdef USE_HW_GEMM
        if (top->io_gemm_task_done == 1) {
            break;
        }
#endif

#ifdef PRINT_CONV_TIME
        if (top->io_dma_ch0_r_busy)
            time_dma_ch0_r++;
        if (top->io_dma_ch1_r_busy)
            time_dma_ch1_r++;
        if (top->io_dma_ch0_w_busy)
            time_dma_ch0_w++;
        if (top->io_dma_ch1_w_busy)
            time_dma_ch1_w++;
#ifndef TEST_WGTBUF
        if (top->io_ifmbuf_task_done == 0 && (top->io_sim_gemm_reg_gemm_ctrl_reg & 0x0001 == 1))
            time_ifmbuf_load++;
        if (top->io_ofmbuf_congested == 1 && (top->io_sim_gemm_reg_gemm_ctrl_reg & 0x0001 == 1))
            time_ofmbuf_congest++;
        if (top->io_axisendbuf_congested == 1)
            time_axisendbuf_congest++;
#endif
#endif
        update();
        if (break_flag)
            break;
    }
}

void dma_wait(ncnn::Mat &data_res, accel::hw_gemm &inst) {

    int x_index_cnt = 0;
    int y_index_cnt = 0;
    int ic_div32_equal = u_align(inst.ifm_c, 32) / 32;
    int k2ic_div32_equal = inst.kernel * inst.kernel * ic_div32_equal;
    int oc_div32_equal = u_align(inst.ofm_c, 32) / 32;
    int owh_div64_equal = u_align(inst.ofm_w * inst.ofm_h, 64) / 64;
    printf("[ log ]: loop begin \n");

#ifdef TEST_WGTBUF
    int wgt_raddr = 0;
    int oc_div32_cnt = 0;
    int owh_div64_cnt = 0;
    int wgt_raddr_equal = inst.kernel * inst.kernel * u_align(inst.ifm_c, 32);

    int default_addr_add = ic_div32_equal;
    if (inst.op == 1) {
        wgt_raddr_equal = inst.kernel * inst.kernel * default_addr_add;
    }
    int fixed_cnt = 0;
    int k2_idx = 0;
    int ic_idx = 0;
    int initial_addr = 0;
#endif

#ifdef TEST_OFM
    int *data_res_ch0_ptr = data_res.channel(0);
    int *data_res_ch1_ptr = data_res.channel(1);
    int count = 0;
    int ofm_wh_base = 0;
    int ofm_wh_offset = 0;
    int ofm_c_base = 0;
    int ofm_c_offset = 0;
    int ofm_wh_index_ch0 = 0;
    int ofm_wh_index_ch1 = 0;
    int ofm_c_index = 0;
    int ofm_wh_align64_div64 = u_align(inst.ofm_w * inst.ofm_h, 64) / 64;
    int ofm_size = inst.ofm_w * inst.ofm_h;
#endif

    while (1) {
        // for(int i=0; i<20000; i++){
        { // dma0 read  channels
            axi_ch0_r();
        }
        { // dma0 write channels
            axi_ch0_w();
        }
        { // dma1 read  channels
            axi_ch1_r();
        }
        { // dma1 write channels
            axi_ch1_w();
        }

#ifdef TEST_OFM
        if (top->io_gemm_odata_ch0_valid) {
            ofm_c_index = ofm_c_base * 32 + ofm_c_offset;
            ofm_wh_index_ch0 = ofm_wh_base * 64 + ofm_wh_offset * 4;
            ofm_wh_index_ch1 = ofm_wh_index_ch0 + 32;
            if (ofm_wh_offset == 7) {
                ofm_wh_offset = 0;
                if (ofm_c_offset == 31) {
                    ofm_c_offset = 0;
                    if (ofm_wh_base == ofm_wh_align64_div64 - 1) {
                        ofm_c_base++;
                        ofm_wh_base = 0;
                    } else
                        ofm_wh_base++;
                } else
                    ofm_c_offset++;
            } else
                ofm_wh_offset++;
            // printf("ofm_c_index = %d, ofm_wh_index_ch0 = %d, ofm_wh_index_ch1 = %d\n", ofm_c_index, ofm_wh_index_ch0, ofm_wh_index_ch1);
        }

        if (top->io_gemm_odata_ch0_valid) {
            // if(ofm_c_index < inst.ofm_c){
            if (!compare(inst, data_res_ch0_ptr[0], top->io_gemm_odata_ch0_data_0) & ofm_wh_index_ch0 < ofm_size) {
                printf("[error]:time: %ld, ofm_c_index = %d, ofm_wh_index_ch0 = %d, data_res_ch0_ptr[0] = %x, top->io_gemm_odata_ch0_data_0 = %x, "
                       "count = %d, data_res_ch0_ptr[0] = %f, top->io_gemm_odata_ch0_data_0 = %f\n",
                       main_time, ofm_c_index, ofm_wh_index_ch0, data_res_ch0_ptr[0], top->io_gemm_odata_ch0_data_0, count,
                       *((float *)&(data_res_ch0_ptr[0])), *((float *)&(top->io_gemm_odata_ch0_data_0)));
                // assert(0);
                break_flag = 1;
            }
            if (!compare(inst, data_res_ch0_ptr[1], top->io_gemm_odata_ch0_data_1) & ofm_wh_index_ch0 + 1 < ofm_size) {
                printf("[error]:time: %ld, ofm_c_index = %d, ofm_wh_index_ch0 = %d, data_res_ch0_ptr[1] = %x, top->io_gemm_odata_ch0_data_1 = %x, "
                       "count = %d, data_res_ch0_ptr[1] = %f, top->io_gemm_odata_ch0_data_1 = %f\n",
                       main_time, ofm_c_index, ofm_wh_index_ch0, data_res_ch0_ptr[1], top->io_gemm_odata_ch0_data_1, count,
                       *((float *)&(data_res_ch0_ptr[1])), *((float *)&(top->io_gemm_odata_ch0_data_1)));
                // assert(0);
                break_flag = 1;
            }
            if (!compare(inst, data_res_ch0_ptr[2], top->io_gemm_odata_ch0_data_2) & ofm_wh_index_ch0 + 2 < ofm_size) {
                printf("[error]:time: %ld, ofm_c_index = %d, ofm_wh_index_ch0 = %d, data_res_ch0_ptr[2] = %x, top->io_gemm_odata_ch0_data_2 = %x, "
                       "count = %d, data_res_ch0_ptr[2] = %f, top->io_gemm_odata_ch0_data_2 = %f\n",
                       main_time, ofm_c_index, ofm_wh_index_ch0, data_res_ch0_ptr[2], top->io_gemm_odata_ch0_data_2, count,
                       *((float *)&(data_res_ch0_ptr[2])), *((float *)&(top->io_gemm_odata_ch0_data_2)));
                // assert(0);
                break_flag = 1;
            }
            if (!compare(inst, data_res_ch0_ptr[3], top->io_gemm_odata_ch0_data_3) & ofm_wh_index_ch0 + 3 < ofm_size) {
                printf("[error]:time: %ld, ofm_c_index = %d, ofm_wh_index_ch0 = %d, data_res_ch0_ptr[3] = %x, top->io_gemm_odata_ch0_data_3 = %x, "
                       "count = %d, data_res_ch0_ptr[3] = %f, top->io_gemm_odata_ch0_data_3 = %f\n",
                       main_time, ofm_c_index, ofm_wh_index_ch0, data_res_ch0_ptr[3], top->io_gemm_odata_ch0_data_3, count,
                       *((float *)&(data_res_ch0_ptr[3])), *((float *)&(top->io_gemm_odata_ch0_data_3)));
                // assert(0);
                break_flag = 1;
            }
            // }
            data_res_ch0_ptr += 4;
        }
        if (top->io_gemm_odata_ch1_valid) {
            // if(ofm_c_index < inst.ofm_c){
            if (!compare(inst, data_res_ch1_ptr[0], top->io_gemm_odata_ch1_data_0) & ofm_wh_index_ch1 < ofm_size) {
                printf("[error]:time: %ld, ofm_c_index = %d, ofm_wh_index_ch1 = %d, data_res_ch1_ptr[0] = %x, top->io_gemm_odata_ch1_data_0 = %x, "
                       "count = %d, data_res_ch1_ptr[0] = %f, top->io_gemm_odata_ch1_data_0 = %f\n",
                       main_time, ofm_c_index, ofm_wh_index_ch1, data_res_ch1_ptr[0], top->io_gemm_odata_ch1_data_0, count,
                       *((float *)&(data_res_ch1_ptr[0])), *((float *)&(top->io_gemm_odata_ch1_data_0)));
                // assert(0);
                break_flag = 1;
            }
            if (!compare(inst, data_res_ch1_ptr[1], top->io_gemm_odata_ch1_data_1) & ofm_wh_index_ch1 + 1 < ofm_size) {
                printf("[error]:time: %ld, ofm_c_index = %d, ofm_wh_index_ch1 = %d, data_res_ch1_ptr[1] = %x, top->io_gemm_odata_ch1_data_1 = %x, "
                       "count = %d, data_res_ch1_ptr[1] = %f, top->io_gemm_odata_ch1_data_1 = %f\n",
                       main_time, ofm_c_index, ofm_wh_index_ch1, data_res_ch1_ptr[1], top->io_gemm_odata_ch1_data_1, count,
                       *((float *)&(data_res_ch1_ptr[1])), *((float *)&(top->io_gemm_odata_ch1_data_1)));
                // assert(0);
                break_flag = 1;
            }
            if (!compare(inst, data_res_ch1_ptr[2], top->io_gemm_odata_ch1_data_2) & ofm_wh_index_ch1 + 2 < ofm_size) {
                printf("[error]:time: %ld, ofm_c_index = %d, ofm_wh_index_ch1 = %d, data_res_ch1_ptr[2] = %x, top->io_gemm_odata_ch1_data_2 = %x, "
                       "count = %d, data_res_ch1_ptr[2] = %f, top->io_gemm_odata_ch1_data_2 = %f\n",
                       main_time, ofm_c_index, ofm_wh_index_ch1, data_res_ch1_ptr[2], top->io_gemm_odata_ch1_data_2, count,
                       *((float *)&(data_res_ch1_ptr[2])), *((float *)&(top->io_gemm_odata_ch1_data_2)));
                // assert(0);
                break_flag = 1;
            }
            if (!compare(inst, data_res_ch1_ptr[3], top->io_gemm_odata_ch1_data_3) & ofm_wh_index_ch1 + 3 < ofm_size) {
                printf("[error]:time: %ld, ofm_c_index = %d, ofm_wh_index_ch1 = %d, data_res_ch1_ptr[3] = %x, top->io_gemm_odata_ch1_data_3 = %x, "
                       "count = %d, data_res_ch1_ptr[3] = %f, top->io_gemm_odata_ch1_data_3 = %f\n",
                       main_time, ofm_c_index, ofm_wh_index_ch1, data_res_ch1_ptr[3], top->io_gemm_odata_ch1_data_3, count,
                       *((float *)&(data_res_ch1_ptr[3])), *((float *)&(top->io_gemm_odata_ch1_data_3)));
                // assert(0);
                break_flag = 1;
            }
            // }
            data_res_ch1_ptr += 4;
            count += 1;
            if (break_flag == 1) {
                printf("\033[;31m[error]: TEST OFM FAILED\n\n\033[0m");
                GEMM_RESET;
                ALU_RESET;
                POOL_RESET;
                update();
                update();
                return;
            }
            if (data_res.h == count) {
                printf("\033[;32m[ log ]: TEST OFM PASS\n\n\033[0m");
                GEMM_RESET;
                ALU_RESET;
                POOL_RESET;
                update();
                update();
                return;
            }
        }
#endif

#ifdef TEST_WGTBUF

        if (inst.op == 0) {

            if (top->io_wgt_odata_valid == 1) {
                unsigned int *wgt_data_ptr = &(top->io_wgt_odata_bits_0);
                for (int j = 0; j < 32; j++) {
                    if (*(wgt_data_ptr + j) != data_res.channel(wgt_raddr).row<unsigned char>(0)[oc_div32_cnt * 32 + j]) {
                        printf("[error]: maintime:%ld, the wgtbuf not equal the wgtbuf_res, res = %x, hw = %x\n", main_time,
                               data_res.channel(wgt_raddr).row<unsigned char>(0)[oc_div32_cnt * 32 + j], *(wgt_data_ptr + j));
                        printf("[error]: j = %d, wgt_raddr = %d, owh_div64_cnt = %d, oc_div32_cnt = %d\n", j, wgt_raddr, owh_div64_cnt, oc_div32_cnt);
                        // assert(0);
                        break_flag = 1;
                    }
                }
            }

            if (top->io_wgt_task_done == 1) {
                if (wgt_raddr == wgt_raddr_equal - 1 & owh_div64_cnt == owh_div64_equal - 1 & oc_div32_cnt == oc_div32_equal - 1) {
                    printf("[ log ]: wgt_raddr = %d, owh_div64_cnt = %d, oc_div32_cnt = %d\n", wgt_raddr, owh_div64_cnt, oc_div32_cnt);
                    printf("\033[;32m[ log ]: TEST WGTBUF PASS\n\n\033[0m");
                }

                else {
                    printf("[error]: wgt_raddr = %d, owh_div64_cnt = %d, oc_div32_cnt = %d\n", wgt_raddr, owh_div64_cnt, oc_div32_cnt);
                    printf("\033[;31m[error]: TEST WGTBUF FAILED\n\n\033[0m");
                }
                break;
            }

            if (top->io_wgt_odata_valid == 1) {
                if (wgt_raddr == wgt_raddr_equal - 1) {
                    wgt_raddr = 0;
                    if (owh_div64_cnt == owh_div64_equal - 1) {
                        owh_div64_cnt = 0;
                        oc_div32_cnt = (oc_div32_cnt == oc_div32_equal - 1) ? 0 : oc_div32_cnt + 1;
                    } else
                        owh_div64_cnt++;
                } else
                    wgt_raddr++;
            }
        } else if (inst.op == 1) {

            if (top->io_wgt_odata_valid == 1) {
                unsigned int *wgt_data_ptr = &(top->io_wgt_odata_bits_0);
                // uint8_t *i_ptr = ((uint8_t *)data_res.data)[wgt_raddr * 32];
                for (int j = 0; j < 32; j++) {
                    if (*(wgt_data_ptr + j) != data_res.channel(0).row<uint8_t>(wgt_raddr)[j]) {
                        printf("[error]: maintime:%ld, the wgtbuf not equal the wgtbuf_res, res = %x, hw = %x\n", main_time,
                               data_res.channel(0).row<uint8_t>(wgt_raddr)[j], *(wgt_data_ptr + j)); // data_res.channel(wgt_raddr)[j]
                        printf("[error]: j = %d, wgt_raddr = %d, fixed_cnt = %d, k2_idx = %d, ic_idx = %d\n", j, wgt_raddr, fixed_cnt, k2_idx,
                               ic_idx);
                        // assert(0);
                        break_flag = 1;
                    }
                }
            }
            if (top->io_wgt_task_done == 1) {
                if (wgt_raddr == wgt_raddr_equal - 1 & fixed_cnt == 31 && owh_div64_cnt == owh_div64_equal - 1 &&
                    k2_idx == (inst.kernel * inst.kernel - 1) && ic_idx == ic_div32_equal - 1) {
                    printf("[ log ]: wgt_raddr = %d, fixed_cnt = %d, k2_idx = %d, ic_idx = %d\n", wgt_raddr, fixed_cnt, k2_idx, ic_idx);
                    printf("\033[;32m[ log ]: TEST WGTBUF PASS\n\n\033[0m");
                }

                else {
                    printf("[error]: wgt_raddr = %d, fixed_cnt = %d, k2_idx = %d, ic_idx = %d\n", wgt_raddr, fixed_cnt, k2_idx, ic_idx);
                    printf("\033[;31m[error]: TEST WGTBUF FAILED\n\n\033[0m");
                }
                break;
            }
            if (top->io_wgt_odata_valid == 1) {
                if (fixed_cnt == 31) {
                    fixed_cnt = 0;
                    if (k2_idx == (inst.kernel * inst.kernel - 1)) {
                        k2_idx = 0;
                        owh_div64_cnt++;

                        if (owh_div64_cnt == owh_div64_equal) {
                            ic_idx++;
                            owh_div64_cnt = 0;
                            if (ic_idx == ic_div32_equal) {
                                ic_idx = 0;
                            }
                        }
                        wgt_raddr = ic_idx;
                    } else {
                        // printf("time: %d\n", main_time);
                        // printf("%d\n", wgt_raddr);
                        wgt_raddr = wgt_raddr + default_addr_add;
                        k2_idx++;
                    }
                } else {
                    fixed_cnt++;
                }
            }
        }

#endif

#ifdef TEST_IFMBUFCTL
        static int finish_flag = 0;
        static int error_cnt = 0;
        static int mesh_ready_cnt = 0;

        int ready_cnt = random(96) - 32;

        static int inblock_h_cnt = 0;
        static int ifm_epoch = 0; // epoch to output all ifm, max k2ic_div32_equal-1

        if (!finish_flag) {
            if (ready_cnt < 0 && top->io_ifmctl_odata_ready == 0) {
                top->io_ifmctl_odata_ready = 1;
                printf("ready_cnt = %d\n", ready_cnt);
            }
            if (top->io_ifmctl_odata_valid == 1) {
                if (ready_cnt >= 0 && top->io_ifmctl_odata_ready == 0) {
                    if (mesh_ready_cnt == ready_cnt) {
                        top->io_ifmctl_odata_ready = 1;
                        printf("ready_cnt = %d\n", ready_cnt);
                    }
                    mesh_ready_cnt++;
                }
                if (top->io_ifmctl_odata_ready == 1) {
                    // ifm buf out and check
                    unsigned int *ifm_data_ptr = &(top->io_ifmctl_odata_bits_0);
                    // if(error_cnt < 5) { printf("time: %d\n", main_time);}
                    for (int j = 0; j < 32; j++) {
                        if (*(ifm_data_ptr + j) !=
                                data_res.channel(k2ic_div32_equal * y_index_cnt + x_index_cnt).row<unsigned int>(0)[j + inblock_h_cnt * 32] &&
                            error_cnt < 5) {
                            printf("[error]: time: %ld\n", main_time);
                            printf("[error]: the ifmbuf not equal the ifmbuf_res, res = %x, hw = %x\n",
                                   data_res.channel(k2ic_div32_equal * y_index_cnt + x_index_cnt).row<unsigned int>(0)[j + inblock_h_cnt * 32],
                                   *(ifm_data_ptr + j));
                            printf("[error]: j = %d, x_index_cnt = %d, y_index_cnt = %d, "
                                   "inblock_h_cnt = %d\n",
                                   j, x_index_cnt, y_index_cnt, inblock_h_cnt);
                            error_cnt++;
                            break;
                        }
                    }

                    // change index
                    inblock_h_cnt++;
                    if (inst.op == 0) {
                        if (x_index_cnt == (k2ic_div32_equal - 1) && inblock_h_cnt == 32) { // this ofm last ifm and clk
                            inblock_h_cnt = 0;
                            x_index_cnt = 0;
                            if (y_index_cnt == (owh_div64_equal - 1)) { // this ifm epoch finish
                                y_index_cnt = 0;
                                // printf("[ log ]: ifm_epoch = %d\n", ifm_epoch);
                                ifm_epoch++;
                                if (ifm_epoch == oc_div32_equal) {
                                    if (error_cnt == 0) {
                                        printf("\033[;32m[ log ]: TEST IFMBUFCTL PASS\n\n\033[0m");
                                    } else {
                                        printf("\033[;31m[error]: TEST IFMBUFCTL FAILED\n\n\033[0m");
                                        exit(0);
                                    }
                                    finish_flag = 1;
                                    // break;
                                }
                            } else {
                                y_index_cnt++;
                            }
                            //   if (error_cnt < 5) printf("[ log ] NEXT BLOCK: x_index_cnt = %d,y_index_cnt = %d\n",
                            //   x_index_cnt, y_index_cnt);
                        } else if (inblock_h_cnt == 32) {

                            x_index_cnt++;
                            inblock_h_cnt = 0;
                            //   if (error_cnt < 5) printf("[ log ] NEXT BLOCK: x_index_cnt = %d,y_index_cnt = %d\n",
                            //   x_index_cnt, y_index_cnt);
                        }
                        // printf("[ log ]: time: %d; inblock_h_cnt = %d\n",main_time, inblock_h_cnt);
                    } else { // dw conv
                        if (inblock_h_cnt == 32) {
                            inblock_h_cnt = 0;
                            if (x_index_cnt >= (k2ic_div32_equal - ic_div32_equal) && x_index_cnt < k2ic_div32_equal) {
                                if (y_index_cnt == (owh_div64_equal - 1)) {
                                    y_index_cnt = 0;
                                    if (x_index_cnt == k2ic_div32_equal - 1) { // finish
                                        x_index_cnt = 0;
                                        if (error_cnt == 0) {
                                            printf("\033[;32m[ log ]: TEST IFMBUFCTL PASS\n\n\033[0m");
                                        } else {
                                            printf("\033[;31m[error]: TEST IFMBUFCTL FAILED\n\n\033[0m");
                                            exit(0);
                                        }
                                        finish_flag = 1;
                                        // break;
                                    } else { // next epoch
                                        x_index_cnt = x_index_cnt % ic_div32_equal + 1;
                                    }
                                } else { // next row
                                    x_index_cnt = x_index_cnt % ic_div32_equal;
                                    y_index_cnt++;
                                }
                                //   if (error_cnt < 5) printf("[ log ] NEXT BLOCK: x_index_cnt = %d,y_index_cnt = %d\n",
                                //   x_index_cnt, y_index_cnt);
                            } else {
                                x_index_cnt += (ic_div32_equal);
                                //   if (error_cnt < 5) printf("[ log ] NEXT BLOCK: x_index_cnt = %d,y_index_cnt = %d\n",
                                //   x_index_cnt, y_index_cnt);
                            }
                            // printf("[ log ]: time: %d; inblock_h_cnt = %d\n",main_time, inblock_h_cnt);}
                        }
                    }
                    // if (inblock_h_cnt == 0) {
                    //     printf("[ log ]: time: %d; inblock_h_cnt = %d\n", main_time, inblock_h_cnt);
                    //     if (error_cnt < 5)
                    //         printf("[ log ] NEXT BLOCK: x_index_cnt = %d,y_index_cnt = %d\n", x_index_cnt, y_index_cnt);
                    // }
                }
            }
        } else {
            // printf("finish_flag = %d\n", finish_flag);
            finish_flag++;
            if (finish_flag == 5) {
                printf("owh_div64_equal=%d, k2ic_div32_equal=%d, oc_div32_equal=%d\n", owh_div64_equal, k2ic_div32_equal, oc_div32_equal);
#ifndef WAVE_LOG
                // open wave log not allow batch test
                finish_flag = 0;
                error_cnt = 0;

                mesh_ready_cnt = 0;
                inblock_h_cnt = 0;
                ifm_epoch = 0;
#else
                exit(0);
#endif
                return;
            }
        }
#endif

#ifdef TEST_ACCMEM
        static int finish_flag = 0;
        static int error_cnt = 0;

        static int inblock_h_cnt[32] = {0};
        static int block_y_index_cnt[32] = {0};
        static int block_x_index_cnt[32] = {0};

        static int skip_flag = 0;

#define OUT_VALID_ADDRGAP ((uint64_t) & top->io_accmem_out_1_valid - (uint64_t) & top->io_accmem_out_0_valid)
#define OUT_DATA_ADDRGAP ((uint64_t) & top->io_accmem_out_1_bits_data0 - (uint64_t) & top->io_accmem_out_0_bits_data0)

#define TOP_OUT_VALID_COL(a) *(CData *)((uint64_t) & top->io_accmem_out_0_valid + OUT_VALID_ADDRGAP * a)
#define TOP_OUT_DATA0_COL(a) (int32_t) SEXT(*(IData *)((uint64_t) & top->io_accmem_out_0_bits_data0 + OUT_DATA_ADDRGAP * a), ACCMEM_OUT_WIDTH)
#define TOP_OUT_DATA1_COL(a) (int32_t) SEXT(*(IData *)((uint64_t) & top->io_accmem_out_0_bits_data1 + OUT_DATA_ADDRGAP * a), ACCMEM_OUT_WIDTH)
        unsigned int *gold_res;
        if (!finish_flag) {
            for (int j = 0; j < 32; j++) {
                if (TOP_OUT_VALID_COL(j) == 1) {
                    gold_res = data_res.channel(block_y_index_cnt[j] * 2 + block_x_index_cnt[j] * owh_div64_equal * 2).row<unsigned int>(0);
                    if (TOP_OUT_DATA0_COL(j) != gold_res[j + inblock_h_cnt[j] * 32] && error_cnt < 5) {
                        printf("[error]: time: %ld\n", main_time);
                        printf("[error]: data0 the hw_out not equal the out_res, res = %x, hw = %x\n", gold_res[j + inblock_h_cnt[j] * 32],
                               TOP_OUT_DATA0_COL(j));
                        printf("[error]: j = %d, block_x_index_cnt = %d, block_y_index_cnt = %d, inblock_h_cnt = %d\n", j, block_x_index_cnt[j],
                               block_y_index_cnt[j] * 2, inblock_h_cnt[j]);
                        error_cnt++;
                        skip_flag = 1;
                    }

                    // data1
                    if (!skip_flag) {
                        gold_res = data_res.channel(block_y_index_cnt[j] * 2 + 1 + block_x_index_cnt[j] * owh_div64_equal * 2).row<unsigned int>(0);
                        if (TOP_OUT_DATA1_COL(j) != gold_res[j + inblock_h_cnt[j] * 32] && error_cnt < 5) {
                            printf("[error]: time: %ld\n", main_time);
                            printf("[error]: data1 the hw_out not equal the out_res, res = %x, hw = %x\n", gold_res[j + inblock_h_cnt[j] * 32],
                                   TOP_OUT_DATA1_COL(j));
                            printf("[error]: j = %d, block_x_index_cnt = %d, block_y_index_cnt = %d, inblock_h_cnt = %d\n", j, block_x_index_cnt[j],
                                   block_y_index_cnt[j] * 2 + 1, inblock_h_cnt[j]);
                            error_cnt++;
                            break;
                        }
                    }

                    if (error_cnt == 5) {
                        finish_flag = 1;
                    }

                    // change index
                    if (inblock_h_cnt[j] == 31) { // one out block finish
                        inblock_h_cnt[j] = 0;
                        if (block_y_index_cnt[j] == owh_div64_equal - 1) { // out col finish
                            block_y_index_cnt[j] = 0;
                            if (block_x_index_cnt[j] == oc_div32_equal - 1) { // out all finish
                                block_x_index_cnt[j] = 0;
                                if (j == 31) {
                                    if (error_cnt == 0)
                                        printf("\033[;32m[ log ]: TEST ACCMEM PASS\n\n\033[0m");
                                    else
                                        printf("\033[;31m[error]: TEST ACCMEM FAILED\n\n\033[0m");
                                    finish_flag = 1;
                                }
                            } else {
                                block_x_index_cnt[j]++;
                            }
                        } else {
                            block_y_index_cnt[j]++;
                        }
                    } else {
                        inblock_h_cnt[j]++;
                    }
                    if (error_cnt < 5)
                        // printf("[ log ] time: %ld, j:%d, NEXT BLOCK: block_x_index_cnt = %d, block_y_index_cnt = %d\n", main_time, j,
                        //        block_x_index_cnt[j], block_y_index_cnt[j]);

                        if (skip_flag) {
                            skip_flag = 0;
                            break;
                        }
                }
            }
        } else {
            finish_flag++;
            if (finish_flag == 500) {
#ifndef WAVE_LOG
                finish_flag = 0;
                error_cnt = 0;

                for (int i = 0; i < 32; i++) {
                    block_x_index_cnt[i] = 0;
                    block_y_index_cnt[i] = 0;
                    inblock_h_cnt[i] = 0;
                }
#else
                exit(0);
#endif
                return;
            }
        }

#endif

        update();
        if (break_flag)
            break;
    }
}

#endif

int exec(uint64_t n) {
    for (uint64_t i = 0; i < n && !Verilated::gotFinish(); i++) {
        if (main_time < 4)
            reset();
        else {
            top->reset = 0;
#ifdef BATCH_TEST
#ifdef TEST_LAYER_CONV
            printf("[ log ]: the task is testing for layer conv batch\r\n");
            test::test_layer_conv_batch();
#endif
#ifdef TEST_LAYER_POOL
            printf("[ log ]: the task is testing for layer pool batch\r\n");
            test::test_layer_pool_batch();
#endif
#else
#ifdef TEST_FP32_ADD
            printf("[ log ]: the task is testing for fp32_add\r\n");
            test::test_fp32_add();
#endif
#ifdef TEST_FP32_MUL
            printf("[ log ]: the task is testing for fp32_mul\r\n");
            test::test_fp32_mul();
#endif
#ifdef TEST_PE
            printf("[ log ]: the task is testing for pe\r\n");
            test::test_pe_single();
#endif
#ifdef TEST_FP32_TO_INT8
            printf("[ log ]: the task is testing for fp32_to_int8\r\n");
            test::test_fp32ToInt8();
#endif
#ifdef TEST_INT32_TO_FP32
            printf("[ log ]: the task is testing for int32_to_fp32\r\n");
            test::test_Int32Tofp32();
#endif
#ifdef TEST_ALU_MATHFUNC
            printf("[ log ]: the task is testing for alu_math_single\r\n");
            test::test_alu_math_single();
#endif
#ifdef TEST_ALU_MAT
            printf("[ log ]: the task is testing for alu_metric_single\r\n");
            test::test_alu_metric_single();
#endif
#ifdef TEST_IFMBUF
            printf("[ log ]: the task is testing for ifmbuf batch\r\n");
            test::test_ifmbuf_batch();
#endif
#ifdef TEST_WGTBUF
            printf("[ log ]: the task is testing for wgtbuf batch\r\n");
            test::test_wgtbuf_batch();
#endif

#ifdef TEST_LAYER_CONV
            printf("[ log ]: the task is testing for layer conv batch\r\n");
            test::test_layer_conv_batch();
#endif
#ifdef TEST_LAYER_POOL
            printf("[ log ]: the task is testing for layer pool batch\r\n");
            test::test_layer_pool_batch();
#endif
#ifdef TEST_LAYER_ELTWISE
            test::test_layer_eltwise_batch();
#endif
#ifdef TEST_LAYER_BINARYOP
            test::test_layer_binaryop_batch();
#endif
#ifdef TEST_LAYER_ABSVAL
            test::test_layer_absval_batch();
#endif
#ifdef TEST_LAYER_BIAS
            test::test_layer_bias_batch();
#endif
#ifdef TEST_LAYER_DROPOUT
            test::test_layer_dropout_batch();
#endif
#ifdef TEST_LAYER_THRESHOLD
            test::test_layer_threshold();
#endif
#ifdef TEST_LAYER_SCALE
            test::test_layer_scale_batch();
#endif
#ifdef TEST_LAYER_RELU
            test::test_layer_relu();
#endif
#ifdef TEST_LAYER_CLIP
            test::test_layer_clip();
#endif
#ifdef TEST_LAYER_TANH
            test::test_layer_tanh();
#endif
#ifdef TEST_LAYER_SIGMOID
            test::test_layer_sigmoid();
#endif
#ifdef TEST_LAYER_SWISH
            test::test_layer_swish();
#endif
#ifdef TEST_LAYER_ELU
            test::test_layer_elu();
#endif
#ifdef TEST_LAYER_SELU
            test::test_layer_selu();
#endif
#ifdef TEST_LAYER_HARDSIGMOID
            test::test_layer_hardsigmoid();
#endif
#ifdef TEST_LAYER_HARDSWISH
            test::test_layer_hardswish();
#endif
#ifdef TEST_LAYER_INNERPROD
            test::test_layer_innerprod_batch();
#endif

#ifdef TEST_YOLOV3
            demo::yolov3_inference();
#endif
#ifdef TEST_YOLOV3_TINY
            demo::yolov3_tiny_inference();
#endif
#ifdef TEST_YOLOV4
            demo::yolov4_inference();
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
#ifdef TEST_YOLOV8
            demo::yolov8_inference();
#endif
#ifdef TEST_YOLO_FASTER
            demo::yolo_faster_inference();
#endif
#ifdef TEST_YOLO_FASTERV2
            demo::yolo_fasterv2_inference();
#endif

#ifdef TEST_MOBILENET_SSD
            demo::mobilenet_ssd_inference();
#endif
#ifdef TEST_MOBILENET_YOLO
            demo::mobilenet_yolo_inference();
#endif
#ifdef TEST_MOBILENETV2_SSDLITE
            demo::mobilenetv2_ssdlite_inference();
#endif
#ifdef TEST_MOBILENETV2_YOLO
            demo::mobilenetv2_yolo_inference();
#endif
#ifdef TEST_MOBILENETV3_SSDLITE
            demo::mobilenetv3_ssdlite_inference();
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

#ifdef TEST_SQUEEZENET
            demo::squeezenet_inference();
#endif
#ifdef TEST_SHUFFLENETV1
            demo::shufflenetv1_inference();
#endif
#ifdef TEST_SHUFFLENETV2
            demo::shufflenetv2_inference();
#endif

#ifdef TEST_MTCNN
            demo::mtcnn_inference();
#endif
#ifdef TEST_RETINAFACE
            demo::retinaface_inference();
#endif
#ifdef TEST_SCRFD
            demo::scrfd_inference();
#endif
#ifdef TEST_FASTERRCNN
            demo::fasterrcnn_inference();
#endif
#ifdef TEST_RFCN
            demo::rfcn_inference();
#endif

#ifdef TEST_YOLACT
            demo::yolact_inference();
#endif

#endif
            break;
        }
    }
    return 0;
}