import chisel3._
import chisel3.util._
import util_function._

class whc extends Bundle {
  val w = Input(UInt(12.W))
  val h = Input(UInt(12.W))
  val c = Input(UInt(12.W))
}

class wgt_8bit_axi extends Module with axi_config with buffer_config {
  val io = IO(new Bundle() {
    //ctrl
    val op             = Input(UInt(2.W))
    val start          = Input(Bool()) //edge
    val start_one_task = Input(Bool()) //start one task
    val clr            = Input(Bool())
    val oc_align       = Input(UInt(12.W))
    val k2ic_align     = Input(UInt(log2Ceil(WGT_BUFFER_DEPTH * 32).W))
    val burst_size     = Input(UInt(log2Up(WGT_BUFFER_DEPTH + 1).W))
    val wgt_baseaddr   = Input(UInt(AXI_ADDR_WIDTH.W))
    //axi
    val axi = new axi_r
    //buffer
    val task_one_done = Output(Bool())
    val task_all_done = Output(Bool())
  })

  val start_t0 = RegNext(io.start,0.B)
  val en       = RegInit(false.B)
  en := Mux(io.start_one_task, true.B, Mux(io.task_one_done, false.B, en))

  val oc_align_div32    = io.oc_align(11, 5)
  val dw_op             = io.op === 1.U
  val total_addr_offset = Mux(dw_op, io.k2ic_align, Cat(io.k2ic_align, 0.U(5.W)))

  val axi_en         = RegInit(false.B)
  val axi_addr       = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_areq       = RegInit(false.B)
  val axi_burst_size = io.burst_size //k2ic*32*8/128
  //  val axi_burst_size      = Mux(dw_op, io.k2ic_align >> 4.U, Cat(io.k2ic_align, 0.U(1.W))) //k2ic*32*8/128
  val axi_busy_down       = io.axi.data.last && en && io.axi.id === RID("gemm_wgt").U
  val axi_areq_down       = fallEdge(axi_areq)
  val start_one_task_keep = RegInit(false.B)
  start_one_task_keep := Mux(io.start_one_task, true.B, Mux(axi_areq, false.B, start_one_task_keep))
  axi_en              := Mux(axi_busy_down, false.B, Mux(start_one_task_keep, true.B, axi_en))
  axi_addr            := Mux(io.start, io.wgt_baseaddr, Mux(axi_areq_down, axi_addr + total_addr_offset, axi_addr))
  axi_areq := Mux(
    axi_en && io.axi.id === RID("gemm_wgt").U && start_one_task_keep,
    true.B,
    Mux(axi_areq, false.B, axi_areq)
  )
  axi_areq            := Mux(axi_areq, false.B, Mux(axi_en && io.axi.id === RID("gemm_wgt").U && start_one_task_keep, true.B, axi_areq))
  io.axi.areq.axiEn   := axi_en
  io.axi.areq.axiSize := axi_burst_size
  io.axi.areq.axiAddr := axi_addr
  io.axi.areq.axiAreq := axi_areq

  val oc_align_div32_cnt = RegInit(0.U(7.W))
  oc_align_div32_cnt := Mux(start_t0, 0.U, Mux(axi_areq_down, oc_align_div32_cnt + 1.U, oc_align_div32_cnt))
  val task_done = RegInit(false.B)
  task_done := Mux(
    io.clr | io.start,
    false.B,
    Mux((oc_align_div32_cnt === oc_align_div32 || dw_op) && axi_busy_down, true.B, task_done)
  )

  io.task_one_done := axi_busy_down & en
  io.task_all_done := task_done
}

class weightBuffer extends Module with axi_config with buffer_config {
  val io = IO(new Bundle() {
    val wgt_en   = Input(Bool())
    val cfg_gemm = Input(new cfg_gemm_io)
    //axi
    val axi_ch0 = new axi_r
    val axi_ch1 = new axi_r
    //gemm inference
    val o_data       = Decoupled(Vec(32, UInt(32.W)))
    val finish       = Output(Bool())
    val w_axi_finish = Output(Bool())
  })

  val cfg_gemm = RegEnable(io.cfg_gemm, 0.U.asTypeOf(new cfg_gemm_io),dualEdge(io.cfg_gemm.en))
  val wgt_en   = RegNext(io.wgt_en,0.B)

  val wgt_buffer = Seq.fill(2)(SPRAM_WGT_WRAP(WGT_BUFFER_WIDTH, WGT_BUFFER_DEPTH, "ultra", true))

  val wgt_8bit_axi_cell0 = Module(new wgt_8bit_axi).io
  val wgt_8bit_axi_cell1 = Module(new wgt_8bit_axi).io

  val start_t0 = riseEdge(wgt_en)
  val start_t1 = RegNext(start_t0,0.B)
  val en       = RegInit(false.B) // buffer is pingpong filling, wgt maybe fetch when unset
  en := Mux(start_t1, true.B, Mux(wgt_8bit_axi_cell0.task_all_done, false.B, en))

  // ################ const ################
  val wgt_baseaddr    = RegEnable(cfg_gemm.wgt_addr, 0.U, start_t0)
  val wgt_baseaddr1   = RegEnable(cfg_gemm.wgt_addr + (WGT_BUFFER_DEPTH / 2 * 32).U, 0.U, start_t0)
  val ic_align        = RegEnable(align(cfg_gemm.ic, 12, 32), 0.U, start_t0)
  val k2              = RegEnable(cfg_gemm.kernel * cfg_gemm.kernel, 0.U, start_t0)
  val k2ic_align      = RegInit(0.U(log2Ceil(WGT_BUFFER_DEPTH * 32).W))
  val owh             = RegInit(0.U(24.W))
  val owh_align       = RegEnable(align(owh, 64), start_t1)
  val owh_align_div64 = owh_align(23, 6)
  val oc_align        = RegInit(0.U(12.W))
  val oc_align_div32  = oc_align(11, 5)
  val ic_align_div32  = ic_align(11, 5)
  val dw_op           = cfg_gemm.op === 1.U
  k2ic_align := Mux(start_t1, k2 * ic_align, k2ic_align)
  owh        := Mux(start_t0, cfg_gemm.ow * cfg_gemm.oh, owh)
  oc_align   := RegEnable(align(cfg_gemm.oc, 32), 0.U, start_t0)
  val half_wgt_enough = RegNext(Mux(dw_op, ((k2ic_align >> 5) <= (WGT_BUFFER_DEPTH / 2).U), (k2ic_align <= (WGT_BUFFER_DEPTH / 2).U)),0.B)

  // ################ buffer state ################
  val wgt_buffer0_full = RegInit(false.B) // buffer data is valid
  val wgt_buffer1_full = RegInit(false.B) // buffer data is valid
  val wgtbuf_w_sel     = RegInit(false.B) // write select
  wgt_buffer0_full := Mux(
    !wgt_en,
    false.B,
    Mux(!wgtbuf_w_sel & wgt_one_axi_done, true.B, Mux(wgtbuf_w_sel & wgt_one_output_done, false.B, wgt_buffer0_full))
  )
  wgt_buffer1_full := Mux(
    !wgt_en,
    false.B,
    Mux(wgtbuf_w_sel & wgt_one_axi_done, true.B, Mux(!wgtbuf_w_sel & wgt_one_output_done, false.B, wgt_buffer1_full))
  )
  wgtbuf_w_sel := Mux(
    !wgt_en,
    false.B,
    Mux(riseEdge(wgt_buffer0_full ^ wgt_buffer1_full), !wgtbuf_w_sel, wgtbuf_w_sel)
  )

  // ################ axi ctrl ################
  val wgt_start_one_task = RegInit(false.B) // set any buffer empty and unset when axi begin to fill buffer
  val task_start_cnt     = RegInit(0.U(3.W))
  when(wgt_start_one_task || task_start_cnt === 5.U || wgt_one_axi_done) {
    task_start_cnt := 0.U
  }.elsewhen((!wgt_buffer0_full || !wgt_buffer1_full) & !io.axi_ch0.areq.axiEn & !io.axi_ch1.areq.axiEn) {
    task_start_cnt := task_start_cnt + 1.U
  }
  wgt_start_one_task := (task_start_cnt === 5.U) & en & !wgt_8bit_axi_cell0.task_all_done


  lazy val axi0_done_wait = RegInit(0.B)
  lazy val axi1_done_wait = RegInit(0.B)
  when(wgt_8bit_axi_cell0.task_one_done && wgt_8bit_axi_cell1.task_one_done) {
    axi1_done_wait := 0.B
  }.elsewhen(wgt_8bit_axi_cell0.task_one_done && !axi1_done_wait) {
    axi0_done_wait := 1.B
  }.elsewhen(wgt_8bit_axi_cell1.task_one_done) {
    axi0_done_wait := 0.B
  }
  when(wgt_8bit_axi_cell0.task_one_done && wgt_8bit_axi_cell1.task_one_done) {
    axi1_done_wait := 0.B
  }.elsewhen(wgt_8bit_axi_cell1.task_one_done && !axi0_done_wait || half_wgt_enough && wgt_start_one_task) {
    axi1_done_wait := 1.B
  }.elsewhen(wgt_8bit_axi_cell0.task_one_done) {
    axi1_done_wait := 0.B
  }
  lazy val axi0_done        = axi0_done_wait || wgt_8bit_axi_cell0.task_one_done
  lazy val axi1_done        = axi1_done_wait || wgt_8bit_axi_cell1.task_one_done
  lazy val wgt_one_axi_done = riseEdge((axi0_done && axi1_done)) // one buffer fill complete

  wgt_8bit_axi_cell0.start_one_task := wgt_start_one_task
  wgt_8bit_axi_cell0.op             := cfg_gemm.op
  wgt_8bit_axi_cell0.start          := start_t1
  wgt_8bit_axi_cell0.clr            := ~wgt_en
  wgt_8bit_axi_cell0.oc_align       := oc_align
  wgt_8bit_axi_cell0.k2ic_align     := k2ic_align
  wgt_8bit_axi_cell0.burst_size     := Mux(half_wgt_enough, Mux(dw_op, k2ic_align >> 4, k2ic_align << 1), WGT_BUFFER_DEPTH.U)
  wgt_8bit_axi_cell0.wgt_baseaddr   := wgt_baseaddr
  wgt_8bit_axi_cell0.axi            <> io.axi_ch0

  wgt_8bit_axi_cell1.start_one_task := wgt_start_one_task && !half_wgt_enough
  wgt_8bit_axi_cell1.op             := cfg_gemm.op
  wgt_8bit_axi_cell1.start          := ShiftRegister(start_t1, 2, 0.B, 1.B) && !half_wgt_enough
  wgt_8bit_axi_cell1.clr            := ~wgt_en
  wgt_8bit_axi_cell1.oc_align       := oc_align
  wgt_8bit_axi_cell1.k2ic_align     := k2ic_align
  wgt_8bit_axi_cell1.burst_size     := Mux(dw_op, k2ic_align >> 4, k2ic_align << 1) - WGT_BUFFER_DEPTH.U
  wgt_8bit_axi_cell1.wgt_baseaddr   := wgt_baseaddr1
  wgt_8bit_axi_cell1.axi            <> io.axi_ch1

  // ################ buffer ctl write ################
  val axi0_rdata_t        = RegEnable(io.axi_ch0.data.data, io.axi_ch0.data.valid)
  val axi0_rvalid_t       = RegNext(io.axi_ch0.data.valid,0.B)
  val wgt_buffer_wen0     = axi0_rvalid_t & axi0_valid_cnt
  lazy val axi0_valid_cnt = RegInit(false.B) // two transfer make one valid and one buffer
  axi0_valid_cnt := Mux(wgt_start_one_task, false.B, Mux(axi0_rvalid_t, ~axi0_valid_cnt, axi0_valid_cnt))

  val axi1_rdata_t        = RegEnable(io.axi_ch1.data.data, io.axi_ch1.data.valid)
  val axi1_rvalid_t       = RegNext(io.axi_ch1.data.valid,0.B)
  val wgt_buffer_wen1     = axi1_rvalid_t & axi1_valid_cnt
  lazy val axi1_valid_cnt = RegInit(false.B) // two transfer make one valid and one buffer
  axi1_valid_cnt := Mux(wgt_start_one_task, false.B, Mux(axi1_rvalid_t, ~axi1_valid_cnt, axi1_valid_cnt))

  wgt_buffer(0).wen0 := !wgtbuf_w_sel && wgt_buffer_wen0
  wgt_buffer(0).wen1 := !wgtbuf_w_sel && wgt_buffer_wen1
  wgt_buffer(1).wen0 := wgtbuf_w_sel && wgt_buffer_wen0
  wgt_buffer(1).wen1 := wgtbuf_w_sel && wgt_buffer_wen1

  val wgtbuf_waddr0 = RegInit(0.U(log2Ceil(WGT_BUFFER_DEPTH).W))
  wgtbuf_waddr0 := Mux(wgt_start_one_task, 0.U, Mux(wgt_buffer_wen0, wgtbuf_waddr0 + 1.U, wgtbuf_waddr0))
  val wgtbuf_waddr1 = RegInit(0.U(log2Ceil(WGT_BUFFER_DEPTH).W))
  wgtbuf_waddr1 := Mux(wgt_start_one_task, 0.U, Mux(wgt_buffer_wen1, wgtbuf_waddr1 + 1.U, wgtbuf_waddr1))

  wgt_buffer(0).waddr0 := wgtbuf_waddr0
  wgt_buffer(0).waddr1 := wgtbuf_waddr1
  wgt_buffer(1).waddr0 := wgtbuf_waddr0
  wgt_buffer(1).waddr1 := wgtbuf_waddr1

  lazy val axi0_rdata_reg = RegInit(0.U(AXI_DATA_WIDTH.W))
  axi0_rdata_reg := Mux(axi0_rvalid_t, axi0_rdata_t, axi0_rdata_reg)
  lazy val axi1_rdata_reg = RegInit(0.U(AXI_DATA_WIDTH.W))
  axi1_rdata_reg := Mux(axi1_rvalid_t, axi1_rdata_t, axi1_rdata_reg)

  wgt_buffer(0).wdata0 := Cat(axi0_rdata_t, axi0_rdata_reg)
  wgt_buffer(0).wdata1 := Cat(axi1_rdata_t, axi1_rdata_reg)
  wgt_buffer(1).wdata0 := Cat(axi0_rdata_t, axi0_rdata_reg)
  wgt_buffer(1).wdata1 := Cat(axi1_rdata_t, axi1_rdata_reg)

  // ################ buffer read ################
  val pre_load_flag       = dualEdge(wgtbuf_w_sel) & !RegNext(finish,0.B)
  val wgt_buffer0_r_valid = wgtbuf_w_sel & wgt_buffer0_full
  val wgt_buffer1_r_valid = !wgtbuf_w_sel & wgt_buffer1_full
  val wgt_buffer_ren      = Wire(Bool())
  wgt_buffer_ren := (wgt_en & (wgt_buffer0_r_valid | wgt_buffer1_r_valid) & w_out_ready) | pre_load_flag

  wgt_buffer(0).ren := wgtbuf_w_sel && wgt_buffer_ren
  wgt_buffer(1).ren := !wgtbuf_w_sel && wgt_buffer_ren

  val wgtbuf_raddr = RegInit(0.U(log2Ceil(WGT_BUFFER_DEPTH).W))
  wgt_buffer(0).raddr := wgtbuf_raddr
  wgt_buffer(1).raddr := wgtbuf_raddr

  val wgt_buffer_rdata         = Wire(UInt(WGT_BUFFER_WIDTH.W))
  lazy val wgt_one_output_done = RegInit(false.B) // one buffer use done
  wgt_one_output_done := Mux(
    !wgt_en,
    false.B,
    !dw_op & wgtbuf_repeat_equal & wgtbuf_raddr === k2ic_align - 2.U || dw_op && wgtbuf_dwconv_icblock_equal && wgtbuf_repeat_equal && wgtbuf_dwconv_k2_idx_equal && wgtbuf_dwconv_fixed_cnt === 30.U
  )
  wgt_buffer_rdata := Mux(RegNext(wgtbuf_w_sel,0.B), wgt_buffer(0).rdata, wgt_buffer(1).rdata)

  // >>>>>>>>>>>>>> conv <<<<<<<<<<<<<<
  // this epoch the last read, this ofmblock the last read in dwconv
  lazy val wgtbuf_raddr_equal  = Mux(dw_op, wgtbuf_dwconv_k2_idx_equal && wgtbuf_dwconv_change_addr_p, wgtbuf_raddr === k2ic_align - 1.U)
  lazy val wgtbuf_repeat_cnt   = RegInit(0.U(18.W)) // single buffer content epoch
  lazy val wgtbuf_repeat_equal = (wgtbuf_repeat_cnt === owh_align_div64 - 1.U) // this buffer the last epoch
  val wgtbuf_oc_cnt            = RegInit(0.U(7.W)) // wgt oc block idx
  val wgtbuf_oc_equal          = wgtbuf_oc_cnt === oc_align_div32 - 1.U // wgt the last buffer
  wgtbuf_repeat_cnt := Mux(wgt_buffer_ren && wgtbuf_raddr_equal, Mux(wgtbuf_repeat_equal, 0.U, wgtbuf_repeat_cnt + 1.U), wgtbuf_repeat_cnt)
  wgtbuf_oc_cnt     := Mux(wgt_buffer_ren & wgtbuf_raddr_equal & wgtbuf_repeat_equal, Mux(wgtbuf_oc_equal, 0.U, wgtbuf_oc_cnt + 1.U), wgtbuf_oc_cnt)

  // >>>>>>>>>>>>>> depthwise conv <<<<<<<<<<<<<<
  val wgtbuf_dwconv_next_addr      = Wire(UInt(log2Ceil(WGT_BUFFER_DEPTH).W))
  lazy val wgtbuf_dwconv_fixed_cnt = RegInit(0.U(5.W))
  lazy val wgtbuf_dwconv_change_addr_p =
    wgtbuf_dwconv_fixed_cnt === 31.U && dw_op && wgt_buffer_ren && w_out_ready
  lazy val wgtbuf_dwconv_k2_idx        = RegInit(0.U(6.W))
  lazy val wgtbuf_dwconv_k2_idx_equal  = wgtbuf_dwconv_k2_idx === k2 - 1.U
  lazy val wgtbuf_dwconv_owh_equal     = wgtbuf_repeat_equal
  lazy val wgtbuf_dwconv_icblock_idx   = RegInit(0.U(6.W))
  lazy val wgtbuf_dwconv_icblock_equal = wgtbuf_dwconv_icblock_idx === ic_align_div32 - 1.U

  wgtbuf_dwconv_next_addr := Mux(
    wgtbuf_dwconv_k2_idx_equal,
    Mux(wgtbuf_dwconv_owh_equal, Mux(wgtbuf_dwconv_icblock_equal, 0.U, wgtbuf_dwconv_icblock_idx + 1.U), wgtbuf_dwconv_icblock_idx),
    wgtbuf_raddr + ic_align_div32
  )
  when(dw_op && wgt_buffer_ren) {
    wgtbuf_dwconv_fixed_cnt := wgtbuf_dwconv_fixed_cnt + 1.U
    when(wgtbuf_dwconv_fixed_cnt === 31.U) {
      wgtbuf_dwconv_fixed_cnt := 0.U
    }
  }
  when(wgtbuf_dwconv_change_addr_p && wgt_buffer_ren) {
    wgtbuf_dwconv_k2_idx := wgtbuf_dwconv_k2_idx + 1.U
    when(wgtbuf_dwconv_k2_idx_equal) {
      wgtbuf_dwconv_k2_idx := 0.U
      when(wgtbuf_dwconv_owh_equal) {
        wgtbuf_dwconv_icblock_idx := wgtbuf_dwconv_icblock_idx + 1.U
        when(wgtbuf_dwconv_icblock_idx === (ic_align_div32 - 1.U)) {
          wgtbuf_dwconv_icblock_idx := 0.U
        }
      }
    }
  }

  wgtbuf_raddr := Mux(
    wgt_en,
    Mux(
      wgt_buffer_ren,
      Mux(
        dw_op,
        Mux(wgtbuf_dwconv_change_addr_p, wgtbuf_dwconv_next_addr, wgtbuf_raddr),
        Mux(wgtbuf_raddr_equal, 0.U, wgtbuf_raddr + 1.U)
      ),
      wgtbuf_raddr
    ),
    0.U
  )

  // >>>>>>>>>>>>>> Register Slice <<<<<<<<<<<<<<
  lazy val finish      = Wire(Bool())
  val w_out_reg        = Reg(Vec(32, UInt(32.W)))
  lazy val w_out_valid = RegInit(0.B)
  lazy val w_out_ready = io.o_data.ready || !w_out_valid
  val w_valid          = RegNext(wgt_en & (wgt_buffer0_r_valid | wgt_buffer1_r_valid),0.B)

  finish := RegNext(
    (!dw_op & wgtbuf_oc_equal ||
      dw_op && (wgtbuf_dwconv_icblock_idx === (ic_align_div32 - 1.U)) && wgtbuf_dwconv_change_addr_p) && wgtbuf_raddr_equal && wgtbuf_repeat_equal && wgt_buffer_ren
  ,0.B)
  val finish_out = RegEnable(finish, 0.B, io.o_data.ready)
  io.finish := finish_out

  when(io.o_data.ready || !w_out_valid) {
    w_out_valid := w_valid
    when(w_valid) {
      for (i <- 0 until 32) {
        w_out_reg(i) := Cat(0.U(24.W), wgt_buffer_rdata(8 * i + 7, 8 * i))
      }
    }
  }

  io.o_data.valid := w_out_valid
  for (i <- 0 until 32) {
    io.o_data.bits(i) := w_out_reg(i)
  }

  val w_axi_finish = RegInit(0.B)
  when(wgt_en & Mux(oc_align_div32 === 1.U || dw_op, wgt_buffer0_full, wgt_buffer1_full)) {
    w_axi_finish := 1.B
  }
  io.w_axi_finish <> w_axi_finish

}
