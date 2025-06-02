import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util._
import util_function._

trait pool_config {
  val max_ifm_w = 1024 // 416
  val max_ifm_h = 1024 // 416
  val max_ifm_c = 1024 // 512
  val max_ofm_w = 512 // 208
  val max_ofm_h = 512 // 208
  val axi_send_size = math.pow(2, 7).toInt // 128
  val w_len = log2Ceil(max_ifm_w + 3) // 11
  val ow_len = log2Ceil(max_ofm_w + 3) // 10
  val h_len = log2Ceil(max_ifm_h + 3) // 11
  val oh_len = log2Ceil(max_ofm_h + 3) // 10
  val c_len = log2Ceil(max_ifm_c + 3) // 11
  val POOL_MAX = 0.U(1.W)
  val POOL_AVG = 1.U(1.W)
  val PAD_CONST = 0.U(1.W)
  val PAD_BORDER = 1.U(1.W)
  val KERNEL_2 = 0.U(1.W)
  val KERNEL_3 = 1.U(1.W)
  val SP_WRITE = false.B //0:W 1:R
  val SP_READ = true.B
  val FP32_DIV9_CYCLES = 3
  val DIV9_METHOD = "shift_add" // or "mult"
}

class pool extends Module with axi_config with pool_config {
  val io = IO(new Bundle {
    /*---- General Registers ----*/
    val pool_io = Input(new cfg_pool_io)

    /*-- axi interface --*/
    val axi_ch0 = new axi_r
    val axi_ch1 = new axi_r
    val axi_send_ch0 = Flipped(new data_axiSend)
    val axi_send_ch1 = Flipped(new data_axiSend)

    val axiSend_congested = Input(Bool())

    /*-- control signal --*/
    val task_done = Output(Bool())
  })
  val cfg_pool = RegEnable(io.pool_io,0.U.asTypeOf(new cfg_pool_io),dualEdge(io.pool_io.en))

  val pool_unit_0 = Module(new Pool_Unit)
  val pool_unit_1 = Module(new Pool_Unit)

  val src0_addr0 = cfg_pool.ifm_addr
  val dst_addr0 = cfg_pool.ofm_addr

  val src0_addr1 = Cat(
    cfg_pool.ifm_addr(AXI_ADDR_WIDTH - 1, 4) + cfg_pool
      .ic(c_len - 1, 1) * cfg_pool.icstep((w_len + h_len) - 1, 2),
    cfg_pool.ifm_addr(3, 0)
  ).asUInt
  val dst_addr1 = Cat(
    cfg_pool.ofm_addr(AXI_ADDR_WIDTH - 1, 4) + cfg_pool
      .ic(c_len - 1, 1) * cfg_pool.ocstep((ow_len + oh_len) - 1, 2),
    cfg_pool.ofm_addr(3, 0)
  ).asUInt

  val iwh = (cfg_pool.iw(2, 0) * cfg_pool.ih(2, 0))(2, 0)
  val iwh_fix_8l4 = (0.U(3.W) - iwh(2, 0))(2) === 1.U // 要补>=4个数到对齐256
  val icstep_is_fix_8 = cfg_pool.icstep(2) === 0.U // 对齐到了256

  val owh = (cfg_pool.ow(2, 0) * cfg_pool.oh(2, 0))(2, 0)
  val owh_fix_8l4 = (0.U(3.W) - owh(2, 0))(2) === 1.U // 要补>=4个数到对齐256
  val ocstep_is_fix_8 = cfg_pool.ocstep(2) === 0.U // 对齐到了256

  val i_ex_fix =
    iwh_fix_8l4 && icstep_is_fix_8 // 补>=4对齐到256 && 对齐到了256 -> 每个c的输入需要额外的128位地址偏移
  val o_ex_fix =
    owh_fix_8l4 && ocstep_is_fix_8 // 补>=4对齐到256 && 对齐到了256 -> 每个c的输出时, 多输出4个额外的数据补齐256

  val pool_0_c = cfg_pool.ic(c_len - 1, 1)
  val pool_1_c = cfg_pool.ic(c_len - 1, 1) + cfg_pool.ic(0)
  /*-- Pool 0 --*/
  /*---- Pool Registers ----*/
  pool_unit_0.io.pool_start := riseEdge(cfg_pool.en)
  pool_unit_0.io.pool_type := cfg_pool.op(0)
  pool_unit_0.io.pool_kernel_w := cfg_pool.kernel_w(0)
  pool_unit_0.io.pool_kernel_h := cfg_pool.kernel_h(0)
  pool_unit_0.io.stride_w := cfg_pool.stride_w
  pool_unit_0.io.stride_h := cfg_pool.stride_h
  pool_unit_0.io.pad_mode := cfg_pool.pad_mode
  pool_unit_0.io.pad_value := cfg_pool.pad_value
  pool_unit_0.io.pad_left := cfg_pool.pool_left
  pool_unit_0.io.pad_right := cfg_pool.pool_right
  pool_unit_0.io.pad_top := cfg_pool.pool_top
  pool_unit_0.io.pad_bottom := cfg_pool.pool_bottom
  /*---- General Registers ----*/
  pool_unit_0.io.ifm_w := cfg_pool.iw(w_len - 1, 0)
  pool_unit_0.io.ifm_h := cfg_pool.ih(h_len - 1, 0)
  pool_unit_0.io.ifm_c := pool_0_c
  pool_unit_0.io.ofm_w := cfg_pool.ow(ow_len - 1, 0)
  pool_unit_0.io.ofm_h := cfg_pool.oh(oh_len - 1, 0)
  pool_unit_0.io.src_addr := src0_addr0
  pool_unit_0.io.dst_addr := dst_addr0
  /*---- Faxi interface ----*/
  pool_unit_0.io.rid <> io.axi_ch0.id
  pool_unit_0.io.rareq <> io.axi_ch0.areq
  pool_unit_0.io.rdata <> io.axi_ch0.data

  pool_unit_0.io.axi_send <> io.axi_send_ch0
  pool_unit_0.io.axiSend_congested <> io.axiSend_congested
  pool_unit_0.io.i_ex_fix := RegNext(i_ex_fix,0.U)
  pool_unit_0.io.o_ex_fix := RegNext(o_ex_fix,0.U)

  /*-- Pool 1 --*/
  /*---- Pool Registers ----*/
  pool_unit_1.io.pool_start := riseEdge(cfg_pool.en)
  pool_unit_1.io.pool_type := cfg_pool.op(0)
  pool_unit_1.io.pool_kernel_w := cfg_pool.kernel_w(0)
  pool_unit_1.io.pool_kernel_h := cfg_pool.kernel_h(0)
  pool_unit_1.io.stride_w := cfg_pool.stride_w
  pool_unit_1.io.stride_h := cfg_pool.stride_h
  pool_unit_1.io.pad_mode := cfg_pool.pad_mode
  pool_unit_1.io.pad_value := cfg_pool.pad_value
  pool_unit_1.io.pad_left := cfg_pool.pool_left
  pool_unit_1.io.pad_right := cfg_pool.pool_right
  pool_unit_1.io.pad_top := cfg_pool.pool_top
  pool_unit_1.io.pad_bottom := cfg_pool.pool_bottom
  /*---- General Registers ----*/
  pool_unit_1.io.ifm_w := cfg_pool.iw(w_len - 1, 0)
  pool_unit_1.io.ifm_h := cfg_pool.ih(h_len - 1, 0)
  pool_unit_1.io.ifm_c := pool_1_c
  pool_unit_1.io.ofm_w := cfg_pool.ow(ow_len - 1, 0)
  pool_unit_1.io.ofm_h := cfg_pool.oh(oh_len - 1, 0)
  pool_unit_1.io.src_addr := src0_addr1
  pool_unit_1.io.dst_addr := dst_addr1
  /*---- Faxi interface ----*/
  pool_unit_1.io.rid <> io.axi_ch1.id
  pool_unit_1.io.rareq <> io.axi_ch1.areq
  pool_unit_1.io.rdata <> io.axi_ch1.data

  pool_unit_1.io.axi_send <> io.axi_send_ch1
  pool_unit_1.io.axiSend_congested <> io.axiSend_congested
  pool_unit_1.io.i_ex_fix := RegNext(i_ex_fix,0.U)
  pool_unit_1.io.o_ex_fix := RegNext(o_ex_fix,0.U)

  //  io.task_done := io.en && riseEdge(pool_unit_0.io.task_done && pool_unit_1.io.task_done)
  io.task_done := ShiftRegister(
    cfg_pool.en,
    1
  ) && (pool_unit_0.io.task_done || pool_0_c === 0.U) && (pool_unit_1.io.task_done || pool_1_c === 0.U)
}

class Pool_Unit extends Module with axi_config with pool_config {
  val io = IO(new Bundle {
    /*---- Pool Registers ----*/
    val pool_start = Input(Bool())
    val pool_type = Input(UInt(1.W))
    val pool_kernel_w = Input(UInt(1.W))
    val pool_kernel_h = Input(UInt(1.W))
    val pad_value = Input(UInt(32.W))

    val stride_w = Input(UInt(2.W))
    val stride_h = Input(UInt(2.W))
    val pad_mode = Input(UInt(1.W))
    val pad_left = Input(UInt(2.W))
    val pad_right = Input(UInt(2.W))
    val pad_top = Input(UInt(2.W))
    val pad_bottom = Input(UInt(2.W))
    /*---- General Registers ----*/
    val ifm_w = Input(UInt(w_len.W))
    val ifm_h = Input(UInt(h_len.W))
    val ifm_c = Input(UInt(c_len.W))
    val ofm_w = Input(UInt(w_len.W))
    val ofm_h = Input(UInt(h_len.W))
    val src_addr = Input(UInt(AXI_ADDR_WIDTH.W))
    val dst_addr = Input(UInt(AXI_ADDR_WIDTH.W))
    /*---- Faxi interface ----*/
    //axi read channel
    val rid = Input(UInt(RID.values.toList.max.W))
    val rareq = new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH)
    val rdata = new axiRData_io(AXI_DATA_WIDTH)

    val axi_send = Flipped(new data_axiSend)

    /*---- control signal ----*/
    val task_done = Output(Bool())
    val i_ex_fix = Input(Bool())
    val o_ex_fix = Input(Bool())

    val axiSend_congested = Input(Bool())
  })

  val started = RegInit(false.B)
  val task_done = RegInit(false.B)
  when(io.pool_start) {
    started := true.B
    task_done := false.B
  }
  io.task_done := task_done

  // pib
  val pib = Module(new Pool_In_Buf)
  pib.io.ifm_w <> io.ifm_w
  pib.io.ifm_h <> io.ifm_h
  pib.io.ifm_c <> io.ifm_c
  pib.io.src_addr <> io.src_addr
  pib.io.rid <> io.rid
  pib.io.rareq <> io.rareq
  pib.io.rdata <> io.rdata
  pib.io.pad_top <> io.pad_top
  pib.io.pad_bottom <> io.pad_bottom
  pib.io.pad_value <> io.pad_value
  pib.io.pad_mode <> io.pad_mode
  pib.io.pad_r2 := io.pad_right === 2.U // 如果要支持pad=3,这里改成=2或=3 @PAD3
  pib.io.pool_start <> io.pool_start
  pib.io.congested <> io.axiSend_congested
  pib.io.i_ex_fix <> io.i_ex_fix

  // input_trim
  val input_trim = Module(new Pool_X_Trim)
  input_trim.io.ifm_w <> io.ifm_w
  input_trim.io.data_in <> pib.io.data_out
  input_trim.io.valid_in <> pib.io.valid_out
  input_trim.io.line_offset <> pib.io.offset
  input_trim.io.pad_left <> io.pad_left
  input_trim.io.pad_right <> io.pad_right
  input_trim.io.pad_value <> io.pad_value
  input_trim.io.pad_mode <> io.pad_mode

  // pool_x
  val pool_x = Module(new Pool_X)
  pool_x.io.data_in <> input_trim.io.data_out
  pool_x.io.valid_in <> input_trim.io.valid_out
  pool_x.io.pool_type <> io.pool_type
  pool_x.io.kernel_w <> io.pool_kernel_w
  pool_x.io.stride_w <> io.stride_w

  pool_x.io.line_size := ((3.U(
    3.W
  ) + io.pad_left + io.pad_right + io.ifm_w) >> 2).asUInt
  // prb
  val prb_x = Module(new Pool_Y_in_Buf)
  prb_x.io.data_in <> pool_x.io.data_out
  prb_x.io.valid_in <> pool_x.io.valid_out
  prb_x.io.line_end_in <> pool_x.io.line_end

  // pool_y
  val pool_y = Module(new Pool_Y)
  pool_y.io.valid_in <> prb_x.io.valid_out
  pool_y.io.data_in <> prb_x.io.data_out
  pool_y.io.line_end_in <> prb_x.io.line_end_out
  pool_y.io.ofm_h <> io.ofm_h
  pool_y.io.pool_type <> io.pool_type
  pool_y.io.kernel_h <> io.pool_kernel_h
  pool_y.io.stride_h <> io.stride_h

  // pool_y trim
  val pool_y_trim = Module(new Pool_Y_Trim)
  pool_y_trim.io.data_in <> pool_y.io.data_out
  pool_y_trim.io.valid_in <> pool_y.io.valid_out
  pool_y_trim.io.line_end_in <> pool_y.io.line_end_out
  pool_y_trim.io.c_finish <> pool_y.io.c_finish
  pool_y_trim.io.ofm_w <> io.ofm_w
  pool_y_trim.io.ofm_c <> io.ifm_c
  pool_y_trim.io.o_ex_fix <> io.o_ex_fix

  // pmb
  val pmb = Module(new Pool_Mult_Buf)
  pmb.io.data_in <> pool_y_trim.io.data_out
  pmb.io.valid_in <> pool_y_trim.io.valid_out
  pmb.io.task_done_in <> pool_y_trim.io.task_done
  pmb.io.kernel <> io.pool_kernel_w
  pmb.io.pool_type <> io.pool_type

  // pob
  val pob = Module(new Pool_Out_Buf)
  pob.io.start <> io.pool_start
  pob.io.data_in <> pmb.io.data_out
  pob.io.valid_in <> pmb.io.valid_out
  pob.io.task_done_in <> pmb.io.task_done_out
  pob.io.dst_addr <> io.dst_addr
  pob.io.axi_send <> io.axi_send

  when(riseEdge(pob.io.task_done_out)) {
    task_done := true.B
  }
}

object Pool_Test_Gen extends App {
  new (chisel3.stage.ChiselStage)
    .execute(
      Array("--target-dir", "./verilog/test"),
      Seq(ChiselGeneratorAnnotation(() => new Pool_Unit))
    )
}

class Pool_In_Buf extends Module with axi_config with pool_config {
  val io = IO(new Bundle {
    val ifm_w = Input(UInt(w_len.W))
    val ifm_h = Input(UInt(h_len.W))
    val ifm_c = Input(UInt(c_len.W))

    // axi
    val src_addr = Input(UInt(AXI_ADDR_WIDTH.W))
    val rid = Input(UInt(RID.values.toList.max.W))
    val rareq = new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH)
    val rdata = new axiRData_io(AXI_DATA_WIDTH)
    val data_out = Output(UInt(128.W))
    val valid_out = Output(Bool())
    val offset = Output(UInt(2.W))

    val pad_top = Input(UInt(2.W))
    val pad_bottom = Input(UInt(2.W))
    val pad_r2 = Input(Bool())
    val pad_value = Input(UInt(32.W))
    val pad_mode = Input(UInt(1.W))

    val pool_start = Input(Bool()) // 一个脉冲信号
    val congested = Input(Bool())
    val i_ex_fix = Input(Bool())
  })
  io.data_out := DontCare
  io.valid_out := false.B
  io.rareq.axiEn := false.B
  io.rareq.axiAreq := false.B
  io.rareq.axiAddr := DontCare
  io.rareq.axiSize := DontCare

  val buf_size = max_ifm_w / 4
  val addr_width = log2Ceil(buf_size)

  val ram = TPRAM_WRAP(128, buf_size, "block")
  val in_ptr = RegInit(0.U(addr_width.W))
  val out_ptr = RegInit(0.U(addr_width.W))
  // default
  ram.waddr := in_ptr
  ram.raddr := out_ptr
  ram.ren := true.B
  ram.wen := io.rdata.valid
  ram.wdata := io.rdata.data

  // val last_valid = Mux(io.ifm_w(1, 0) === 0.U, 4.U, io.ifm_w(1, 0)) // 一行的最后一次输出有几个有效数字，例如w=13时，last_valid=1
  val offset = RegInit(
    0.U(2.W)
  ) // 一行起始处有几个无效数据， 第一行的offset为0，w=13时，第2行应该是1，第3行是2，第4行是3，第5行是0
  io.offset := offset
  val buf_offset = RegInit(0.U(2.W)) // 输入的offset, 和输出是分开的
  val offset_next = offset + io.ifm_w(1, 0)
  // 每输出完一行, out_ptr 减去 (offset + last_valid(1, 0)) =/= 0.U 作为修正
  val addr_start = RegInit(0.U((AXI_ADDR_WIDTH - 4).W))

  val max_line_buf = Wire(UInt(7.W))
  when(io.ifm_w(10, 9) =/= 0.U) { // >= 512
    max_line_buf := 1.U
  }.elsewhen(io.ifm_w(8) =/= 0.U) { // >= 256
    max_line_buf := 2.U
  }.elsewhen(io.ifm_w(7) =/= 0.U) { // >= 128
    max_line_buf := 4.U
  }.elsewhen(io.ifm_w(6) =/= 0.U) { // >= 64
    max_line_buf := 8.U
  }.elsewhen(io.ifm_w(5) =/= 0.U) { // >= 32
    max_line_buf := 16.U
  }.elsewhen(io.ifm_w(4) =/= 0.U) { // >= 16
    max_line_buf := 32.U
  }.otherwise { // < 16
    max_line_buf := 64.U
  }

  val started = RegInit(false.B)
  val pad_top_count = RegInit(0.U(2.W))
  val pad_bottom_count = RegInit(0.U(2.W))

  val is_pad_top = RegInit(false.B)
  val is_pad_bottom = RegInit(false.B)
  val is_pad = is_pad_top | is_pad_bottom
  val valid_out = RegInit(false.B)
  io.data_out := Mux(
    is_pad,
    Mux(
      io.pad_mode === PAD_CONST,
      Cat(io.pad_value, io.pad_value, io.pad_value, io.pad_value),
      ram.rdata
    ),
    ram.rdata
  )
  when(valid_out) {
    io.valid_out := true.B
  }

  val pause = RegInit(false.B) // 输出完一行等一下

  val h_buffered_count = RegInit(0.U(h_len.W)) // 缓存了多少行(当前c)
  val c_buffered_count = RegInit(0.U(c_len.W)) // 缓存了多少c
  val out_count = RegInit(0.U((w_len - 2).W)) // 输出了多少个
  val h_out_count = RegInit(0.U(h_len.W)) // 输出了多少行(当前c)
  val c_out_count = RegInit(0.U(c_len.W)) // 输出了多少c
  val buffered_count = RegInit(0.U(log2Ceil(buf_size + 1).W)) // 缓存了多少个128位
  val r_ready = RegInit(false.B)
  val w_ready = RegInit(false.B)
  val r0 :: r1 :: r2 :: nil = Enum(3)
  val r_state = RegInit(r0)
  val axi_lines = RegInit(0.U(7.W))

  when(io.pool_start) {
    started := true.B
    pad_top_count := 0.U
    pad_bottom_count := 0.U
    addr_start := io.src_addr(AXI_ADDR_WIDTH - 1, 4)
    h_buffered_count := 0.U
    c_buffered_count := 0.U
    out_count := 0.U
    h_out_count := 0.U
    c_out_count := 0.U
    buffered_count := 0.U
  }

  when(started) {
    when(
      buffered_count <= (buf_size / 4).U && c_buffered_count < io.ifm_c && !io.congested && r_state === r0
    ) {
      r_ready := true.B
      axi_lines := Mux(
        max_line_buf > (io.ifm_h - h_buffered_count),
        io.ifm_h - h_buffered_count,
        max_line_buf
      ) // 这步拆开防止时序违例
    }
    when(r_ready) {
      io.rareq.axiEn := true.B
      val axi_length = (axi_lines * io.ifm_w)(w_len + 5, 0)
      val axi_size = ((axi_length - (0.U(2.W) - buf_offset) + (0.U(
        2.W
      ) - (buf_offset + axi_length(1, 0)))) >> 2).asUInt
      switch(r_state) {
        is(r0) { // axi 发送axiAreq
          when(io.rid === RID("pool").U) {
            io.rareq.axiAreq := true.B
            io.rareq.axiAddr := Cat(addr_start, 0.U(4.W))
            io.rareq.axiSize := axi_size
            r_state := r2
          }
        }
        is(r2) { // axi开始读
          when(io.rdata.valid) {
            ram.wen := io.rdata.valid
            ram.wdata := io.rdata.data
            ram.waddr := in_ptr
            in_ptr := in_ptr + 1.U
            buffered_count := buffered_count + 1.U
          }
          when(io.rdata.last) { // finish
            r_state := r0
            when(h_buffered_count + axi_lines === io.ifm_h) { // 读完了一个c
              h_buffered_count := 0.U
              c_buffered_count := c_buffered_count + 1.U
              buf_offset := 0.U
              addr_start := addr_start + (RegNext(
                axi_size
              ,0.U) + io.i_ex_fix.asUInt)
            }.otherwise {
              addr_start := addr_start + RegNext(
                axi_size
              ,0.U) // 用RegNext防止时序违例，这里前一个周期axi_size肯定不变
              h_buffered_count := h_buffered_count + axi_lines
              buf_offset := buf_offset + axi_length(1, 0)
            }
            r_ready := false.B
          }
        }
      }
    }
    // output: 至少缓存了一行, 就开始输出一行
    val line_size =
      ((io.ifm_w + offset + (0.U(2.W) - (offset + io.ifm_w(1, 0)))) >> 2).asUInt
    when(buffered_count >= line_size && !io.congested && !pause) {
      w_ready := true.B // 准备输出
      when(pad_top_count < io.pad_top) {
        is_pad_top := true.B
        is_pad_bottom := false.B
      }.elsewhen(h_out_count === io.ifm_h && pad_bottom_count < io.pad_bottom) {
        is_pad_top := false.B
        is_pad_bottom := true.B
      }.otherwise {
        is_pad_top := false.B
        is_pad_bottom := false.B
      }
    }
    when(pause) {
      pause := false.B
    }
    when(w_ready) {
      when(out_count < line_size) {
        valid_out := true.B
        ram.ren := true.B
        ram.raddr := out_ptr
        out_count := out_count + 1.U
        out_ptr := out_ptr + 1.U
      }.otherwise { // finish
        pause := io.pad_r2 // 如果pad_right是2，每输出一行要等一个周期(如果要支持pad_3也要等)，因为cutter有可能需要额外的周期来做padding，pad_right<2不用等 // @PAD3
        valid_out := false.B
        ram.ren := false.B
        w_ready := false.B
        out_count := 0.U

        when(is_pad_top) { // 读完了返回去
          out_ptr := out_ptr - line_size
          pad_top_count := pad_top_count + 1.U
        }.elsewhen(is_pad_bottom) {
          when(pad_bottom_count + 1.U < io.pad_bottom) { // pad_bottom还没完,读完了返回去
            out_ptr := out_ptr - line_size
            pad_bottom_count := pad_bottom_count + 1.U
          }.otherwise { // 完成了一整个c
            h_out_count := 0.U
            c_out_count := c_out_count + 1.U
            pad_bottom_count := 0.U
            pad_top_count := 0.U
            offset := 0.U
            buffered_count := buffered_count - (line_size - (io.rdata.valid =/= 0.U).asUInt)
          }
        }.elsewhen(h_out_count === io.ifm_h - 1.U && io.pad_bottom =/= 0.U) { // 最后一行,如果要pad_bottom,读完了返回去
          out_ptr := out_ptr - line_size
          h_out_count := h_out_count + 1.U
        }.otherwise { // 缓存中的一个有效行应该用完了
          offset := offset_next
          val delta_buffered_count =
            Mux(offset_next =/= 0.U, line_size - 1.U, line_size)
          val buffered_count_pre_sub =
            Mux(io.rdata.valid, buffered_count + 1.U, buffered_count)
          when(h_out_count === io.ifm_h - 1.U) { // 完成了一整个c
            buffered_count := buffered_count_pre_sub - line_size
            h_out_count := 0.U
            c_out_count := c_out_count + 1.U
            pad_top_count := 0.U
            offset := 0.U
          }.otherwise { // c 还没结束
            buffered_count := buffered_count_pre_sub - delta_buffered_count
            h_out_count := h_out_count + 1.U
            when(offset_next =/= 0.U) { // 下一行offset不为0, 要将out_ptr-1 // 完成了一整个c就不要后面几个数了(c是对齐到128位的)
              out_ptr := out_ptr - 1.U
            }
          }
        }
      }
    }

    when(c_out_count === io.ifm_c) {
      started := false.B
    }
  }

}

class Pool_X_Trim extends Module with axi_config with pool_config {
  val io = IO(new Bundle {

    // const
    val ifm_w = Input(UInt(w_len.W))

    // io
    val data_in = Input(UInt(128.W))
    val valid_in = Input(Bool())
    val data_out = Output(UInt(128.W))
    val valid_out = Output(Bool())

    val line_offset = Input(
      UInt(2.W)
    ) // 一行起始处有几个无效数据， 第一行的offset为0，w=13时，第2行应该是1，第3行是2，第4行是3，第5行是0

    val pad_left = Input(UInt(2.W))
    val pad_right = Input(UInt(2.W))
    val pad_value = Input(UInt(32.W))
    val pad_mode = Input(UInt(1.W))

  })

  val (r0, r1, r2) =
    (RegInit(0.U(32.W)), RegInit(0.U(32.W)), RegInit(0.U(32.W)))
  val w_buffered_count = RegInit(0.U(w_len.W))
  val valid_count = RegInit(0.U(2.W)) // 0,1,2,3

  val data_in_r = RegNext(io.data_in,0.U)
  val data_in0_r = data_in_r(31, 0)
  val data_in1_r = data_in_r(63, 32)
  val data_in2_r = data_in_r(95, 64)
  val data_in3_r = data_in_r(127, 96)
  val valid_in_r = RegInit(0.U(3.W))

  val offset_acivated = RegInit(false.B)
  when(io.valid_in) {
    when(w_buffered_count === 0.U && !offset_acivated) {
      valid_in_r := 4.U(3.W) - io.line_offset
      offset_acivated := true.B
    }.otherwise {
      valid_in_r := 4.U
    }
  }.otherwise {
    valid_in_r := 0.U
  }

  // input 0 with pad
  when(w_buffered_count === 0.U && riseEdge(io.valid_in)) {
    val pad_left_value = Wire(UInt(32.W))
    when(io.pad_mode === PAD_CONST) {
      pad_left_value := io.pad_value
    }.otherwise {
      when(io.line_offset === 0.U) {
        pad_left_value := data_in0_r
      }.elsewhen(io.line_offset === 1.U) {
        pad_left_value := data_in1_r
      }.elsewhen(io.line_offset === 2.U) {
        pad_left_value := data_in2_r
      }.otherwise {
        pad_left_value := data_in3_r
      }
    }
    r0 := pad_left_value
    r1 := pad_left_value
    valid_count := io.pad_left
  }

  io.data_out := DontCare
  io.valid_out := false.B
  val do_pad_right = RegInit(false.B)
  val last_valid = RegInit(0.U(32.W))
  when(valid_in_r =/= 0.U) {
    when(w_buffered_count + valid_in_r < io.ifm_w) { // no pad
      valid_count := valid_count + valid_in_r
      w_buffered_count := w_buffered_count + valid_in_r
      offset_acivated := false.B
      when(valid_count === 0.U) { // 已有0个的情况 、（1，2，3）
        when(valid_in_r === 1.U) { // 输入1个
          r0 := data_in3_r
        }.elsewhen(valid_in_r === 2.U) { // 输入2个
          r0 := data_in2_r
          r1 := data_in3_r
        }.elsewhen(valid_in_r === 3.U) { // 输入3个
          r0 := data_in1_r
          r1 := data_in2_r
          r2 := data_in3_r
        }.otherwise { // 输入4个
          io.data_out := data_in_r
          io.valid_out := true.B
        }
      }.elsewhen(valid_count === 1.U) { // 已有1个的情况
        when(valid_in_r === 1.U) { // 输入1个
          r1 := data_in3_r
        }.elsewhen(valid_in_r === 2.U) { // 输入2个
          r1 := data_in2_r
          r2 := data_in3_r
        }.elsewhen(valid_in_r === 3.U) { // 输入3个
          io.data_out := Cat(data_in3_r, data_in2_r, data_in1_r, r0)
          io.valid_out := true.B
        }.otherwise { // 输入4个
          io.data_out := Cat(data_in2_r, data_in1_r, data_in0_r, r0)
          r0 := data_in3_r
          io.valid_out := true.B
        }
      }.elsewhen(valid_count === 2.U) { // 已有2个的情况
        when(valid_in_r === 1.U) { // 输入1个
          r2 := data_in3_r
        }.elsewhen(valid_in_r === 2.U) { // 输入2个
          io.data_out := Cat(data_in3_r, data_in2_r, r1, r0)
          io.valid_out := true.B
        }.elsewhen(valid_in_r === 3.U) { // 输入3个
          io.data_out := Cat(data_in2_r, data_in1_r, r1, r0)
          r0 := data_in3_r
          io.valid_out := true.B
        }.otherwise { // 输入4个
          io.data_out := Cat(data_in1_r, data_in0_r, r1, r0)
          r0 := data_in2_r
          r1 := data_in3_r
          io.valid_out := true.B
        }
      }.otherwise { // 已有3个的情况
        /* 注释掉是因为pad不可能是3，这种情况下 valid_in_r肯定是4， 如果支持pad_left=3, 把注释掉的内容打开 */ // @PAD3
        //        when(valid_in_r === 1.U) { // 输入1个
        //          io.data_out  := Cat(data_in3_r, r2, r1, r0)
        //          io.valid_out := true.B
        //        }.elsewhen(valid_in_r === 2.U) { // 输入2个
        //          io.data_out  := Cat(data_in2_r, r2, r1, r0)
        //          r0           := data_in3_r
        //          io.valid_out := true.B
        //        }.elsewhen(valid_in_r === 3.U) { // 输入3个
        //          io.data_out  := Cat(data_in1_r, r2, r1, r0)
        //          r0           := data_in2_r
        //          r1           := data_in3_r
        //          io.valid_out := true.B
        //        }.otherwise { // 输入4个
        io.data_out := Cat(data_in0_r, r2, r1, r0)
        r0 := data_in1_r
        r1 := data_in2_r
        r2 := data_in3_r
        io.valid_out := true.B
        //        }
      }
    }.otherwise { // when(w_buffered_count + valid_in_r >= io.ifm_w)
      val act_valid = (io.ifm_w - w_buffered_count)(2, 0)
      last_valid := MuxLookup(
        act_valid,
        data_in0_r,
        Array(
          // 1.U -> data_in0_r, //by default
          2.U -> data_in1_r,
          3.U -> data_in2_r,
          4.U -> data_in3_r
        )
      )
      do_pad_right := true.B
      w_buffered_count := io.ifm_w
      valid_count := valid_count + act_valid
      when(valid_count === 0.U) { // 已有0个的情况
        // 注意这里判断的是act_valid而不是offset影响的valid_count (valid_count<4是高位有效,act_valid<4是低位有效，只能重写v_v)
        when(act_valid <= 3.U) { // 输入3个 (2个/1个多覆盖一次r0 r1也无所谓,减少判断次数)
          r0 := data_in0_r
          r1 := data_in1_r
          r2 := data_in2_r
        }.otherwise { // 输入4个
          io.data_out := data_in_r
          io.valid_out := true.B
        }
      }.elsewhen(valid_count === 1.U) { // 已有1个的情况
        when(act_valid <= 2.U) { // 输入2个 (1个多覆盖一次r0也无所谓,减少判断次数)
          r1 := data_in0_r
          r2 := data_in1_r
        }.otherwise { // 输入4个 (3个多覆盖一次r0也无所谓,减少判断次数)
          io.data_out := Cat(data_in2_r, data_in1_r, data_in0_r, r0)
          r0 := data_in3_r
          io.valid_out := true.B
        }
      }.elsewhen(valid_count === 2.U) { // 已有2个的情况
        when(act_valid === 1.U) { // 输入1个
          r2 := data_in0_r
        }.otherwise { // 输入4/3/2 个
          io.data_out := Cat(data_in1_r, data_in0_r, r1, r0)
          r0 := data_in2_r
          r1 := data_in3_r
          io.valid_out := true.B
        }
      }.otherwise { // 已有3个的情况
        io.data_out := Cat(data_in0_r, r2, r1, r0)
        r0 := data_in1_r
        r1 := data_in2_r
        r2 := data_in3_r
        io.valid_out := true.B
      }
    }
  }.otherwise { // valid_in_r === 0, do_pad_right
    val pad_right_value =
      Mux(io.pad_mode === PAD_CONST, io.pad_value, last_valid)
    when(do_pad_right) {
      when(valid_count === 0.U) { // pad_r = 1或者2, 都能一个周期结束
        do_pad_right := false.B
        w_buffered_count := 0.U
        when(io.pad_right =/= 0.U) { // pad 1 也可以按照 pad2处理
          io.data_out := Cat(0.U(64.W), pad_right_value, pad_right_value)
          io.valid_out := true.B
        }
      }.elsewhen(valid_count === 1.U) { // pad_r = 1或者2, 都能一个周期结束
        do_pad_right := false.B
        w_buffered_count := 0.U
        // 不管pad等于几，全按pad = 2处理即可
        io.data_out := Cat(0.U(32.W), pad_right_value, pad_right_value, r0)
        io.valid_out := true.B
      }.elsewhen(valid_count === 2.U) { // pad_r = 1或者2, 都能一个周期结束
        do_pad_right := false.B
        w_buffered_count := 0.U
        // 不管pad等于几，全按pad = 2处理即可
        io.data_out := Cat(pad_right_value, pad_right_value, r1, r0)
        io.valid_out := true.B
      }.otherwise { // valid_count === 3.U;// pad_r=1:一个周期结束, pad_r=2:需要2个周期结束（3+2=5，一个周期只能输出4，第二个周期递归到valid_count === 1.U的情况）
        when(io.pad_right =/= 2.U) { // pad_r = 0或1，一个周期结束
          do_pad_right := false.B
          w_buffered_count := 0.U
          io.data_out := Cat(pad_right_value, r2, r1, r0)
          io.valid_out := true.B
        }.otherwise { // pad_r = 2
          // do_pad_right := true.B // 还是true，下个周期递归到 valid_count === 1.U, 注释掉是因为这里寄存器不设置false就不会变
          valid_count := 1.U
          w_buffered_count := 0.U
          io.data_out := Cat(pad_right_value, r2, r1, r0)
          r0 := pad_right_value
          io.valid_out := true.B
        } // 没有考虑pad_right=3的情况 @PAD3
      }
    }
  }
}

class Pool_X extends Module with pool_config with cal_cell_params {
  val io = IO(new Bundle {
    val data_in = Input(UInt(128.W))
    val valid_in = Input(Bool())
    val data_out = Output(UInt(128.W))
    val valid_out = Output(UInt(3.W))

    val pool_type = Input(UInt(1.W))
    val kernel_w = Input(UInt(1.W))
    val stride_w = Input(UInt(2.W))

    val line_size = Input(
      UInt((w_len - 2).W)
    ) // line_size = (w + pad_left + pad_right + 3) >> 2
    val line_end = Output(Bool())
  })

  val (in0, in1, in2, in3) = (
    io.data_in(31, 0),
    io.data_in(63, 32),
    io.data_in(95, 64),
    io.data_in(127, 96)
  )

  val b0000 :: b0101 :: b0111 :: b1111 :: b0001 :: b1001 :: b0011 :: nil = Enum(
    7
  )

  val w_in4_count = RegInit(0.U((w_len - 2).W))
  val (r0, r1) = (Reg(UInt(32.W)), Reg(UInt(32.W)))
  val valid_count = RegInit(0.U(2.W))
  val valid_out = RegInit(b0000)
  val line_end = Wire(Bool())

  val adder_array = Module(new Pool_X_Adder_Array)
  adder_array.io.pool_type <> io.pool_type
  adder_array.io.kernel <> io.kernel_w
  val ai = adder_array.io.in
  val ao = adder_array.io.out
  adder_array.io.valid_in := io.valid_in

  ai(0) := DontCare
  ai(1) := DontCare
  ai(2) := DontCare
  ai(3) := DontCare
  ai(4) := DontCare
  ai(5) := DontCare
  line_end := false.B // 默认，只有在手动设置true的时候激活一下

  val valid_shift = Mux(
    io.pool_type === POOL_AVG,
    Mux(
      io.kernel_w === KERNEL_2,
      ShiftRegister(valid_out, FP32_ADD_CYCLES - 1),
      ShiftRegister(valid_out, FP32_ADD_CYCLES * 2 - 1)
    ),
    Mux(io.kernel_w === KERNEL_2, valid_out, ShiftRegister(valid_out, 1))
  )
  io.line_end := Mux(
    io.pool_type === POOL_AVG,
    Mux(
      io.kernel_w === KERNEL_2,
      ShiftRegister(line_end, FP32_ADD_CYCLES - 1),
      ShiftRegister(line_end, FP32_ADD_CYCLES * 2 - 1)
    ),
    Mux(io.kernel_w === KERNEL_2, line_end, ShiftRegister(line_end, 1))
  )

  // output = Cat(valid_out, data_out)
  val output = MuxLookup(
    valid_shift,
    0.U(131.W),
    Array(
      b0101 -> Cat(2.U(3.W), 0.U(64.W), ao(2), ao(0)),
      b0111 -> Cat(3.U(3.W), 0.U(32.W), ao(2), ao(1), ao(0)),
      b1111 -> Cat(4.U(3.W), ao(3), ao(2), ao(1), ao(0)),
      b0001 -> Cat(1.U(3.W), 0.U(96.W), ao(0)),
      b1001 -> Cat(2.U(3.W), 0.U(64.W), ao(3), ao(0)),
      b0011 -> Cat(2.U(3.W), 0.U(64.W), ao(1), ao(0))
    )
  )
  io.valid_out := output(130, 128)
  io.data_out := output(127, 0)

  valid_out := b0000
  when(io.valid_in) {
    when(w_in4_count =/= io.line_size - 1.U) {
      w_in4_count := w_in4_count + 1.U
    }.otherwise {
      w_in4_count := 0.U
      line_end := true.B
    }
    when(io.kernel_w === KERNEL_2) { // kernel_w = 2
      when(io.stride_w === 2.U) { // k=2, s=2 // 这种情况太简单了= =
        // valid_count: 0 -> 1 -> 0 -> ...
        ai(0) := in0
        ai(1) := in1
        ai(2) := in2
        ai(3) := in3
        valid_out := b0101
      }.otherwise { // k=2, s=1
        // valid_count: 0 -> 1 -> 1 -> ...
        when(valid_count === 0.U) { // 没有有效缓存值(一行的第一个)
          ai(0) := in0
          ai(1) := in1
          ai(2) := in2
          ai(3) := in3
          valid_out := b0111
        }.otherwise { // 这种情况下 valid_count 只可能是 1
          ai(0) := r0
          ai(1) := in0
          ai(2) := in1
          ai(3) := in2
          ai(4) := in3
          valid_out := b1111
        }
        when(w_in4_count =/= io.line_size - 1.U) {
          r0 := in3
          valid_count := 1.U // 0 -> 1, 1 -> 1
        }.otherwise { // 算上当前输入正好一行
          valid_count := 0.U
        }
      }
    }.otherwise { // kernel_w = 3 // 晚点再写
      when(io.stride_w === 3.U) { // k=3, s=3
        // valid_count: 0 -> 1 -> 2 -> 0 -> ...
        when(valid_count === 0.U) { // 没有有效缓存值
          ai(0) := in0
          ai(1) := in1
          ai(2) := in2
          r0 := in3
          valid_out := b0001
        }.elsewhen(valid_count === 1.U) {
          ai(0) := r0
          ai(1) := in0
          ai(2) := in1
          r0 := in2
          r1 := in3
          valid_out := b0001
        }.otherwise { // 有两个缓存
          ai(0) := r0
          ai(1) := r1
          ai(2) := in0
          ai(3) := in1
          ai(4) := in2
          ai(5) := in3
          valid_out := b1001
        }
        when(w_in4_count =/= io.line_size - 1.U) {
          when(valid_count === 0.U) {
            valid_count := 1.U
          }.elsewhen(valid_count === 1.U) {
            valid_count := 2.U
          }.otherwise { // 2 -> 0
            valid_count := 0.U
          }
        }.otherwise { // 算上当前输入正好一行
          valid_count := 0.U
        }
      }.elsewhen(io.stride_w === 2.U) { // k=3, s=2
        // valid_count: 0 -> 2 -> 2 -> ...
        r0 := in2
        r1 := in3
        when(valid_count === 0.U) { // 没有有效缓存值
          ai(0) := in0
          ai(1) := in1
          ai(2) := in2
          valid_out := b0001
        }.elsewhen(valid_count === 2.U) { // 只有0和2两种情况
          ai(0) := r0
          ai(1) := r1
          ai(2) := in0
          ai(3) := in1
          ai(4) := in2
          valid_out := b0101
        }
        when(w_in4_count =/= io.line_size - 1.U) {
          valid_count := 2.U
        }.otherwise { // 算上当前输入正好一行
          valid_count := 0.U
        }
      }.otherwise { // k=3, s=1
        // valid_count: 0 -> 2 -> 2 -> ...
        r0 := in2
        r1 := in3
        when(valid_count === 0.U) { // 没有有效缓存值
          ai(0) := in0
          ai(1) := in1
          ai(2) := in2
          ai(3) := in3
          valid_out := b0011
        }.otherwise { // 只有0和2两种情况
          ai(0) := r0
          ai(1) := r1
          ai(2) := in0
          ai(3) := in1
          ai(4) := in2
          ai(5) := in3
          valid_out := b1111
        }
        when(w_in4_count =/= io.line_size - 1.U) {
          valid_count := 2.U
        }.otherwise { // 算上当前输入正好一行
          valid_count := 0.U
        }
      }
    }
  }
}

class Add_Max(use_valid_out: Boolean = false)
    extends Module
    with pool_config
    with Convert
    with cal_cell_params {
  val io = IO(new Bundle {
    val in_0 = Input(UInt(32.W))
    val in_1 = Input(UInt(32.W))
    val valid_in = Input(Bool())
    val valid_out = if (use_valid_out) Some(Output(Bool())) else None
    val out = Output(UInt(32.W))
    val pool_type = Input(UInt(1.W))
  })

  val adder = FP32_Adder(use_valid_out)
  val (a, b) = (FP32(io.in_0), FP32(io.in_1))
  val max_ab: UInt = RegNext(Mux(a > b, a, b),0.U)
  adder.io.x := a
  adder.io.y := b
  adder.io.valid_in := io.valid_in && io.pool_type === POOL_AVG
  val add_ab = adder.io.z
  io.out := Mux(io.pool_type === POOL_AVG, add_ab, max_ab)
  if (use_valid_out) {
    io.valid_out.get := Mux(
      io.pool_type === POOL_AVG,
      adder.io.valid_out.get,
      RegNext(io.valid_in,0.U)
    )
  }
}

/* in:(0)(1)(2)(3)(4)(5) ---
 *                          |
 *                          V
 * out:(0)(4) <---- (0+1)(0+1+2)
 *     (1)(5) <---- (1+2)(1+2+3)
 *     (2)(6) <---- (2+3)(2+3+4)
 *     (3)(7) <---- (3+4)(3+4+5) */
class Pool_X_Adder_Array extends Module with pool_config with cal_cell_params {
  val io = IO(new Bundle {
    val in = Input(Vec(6, UInt(32.W)))
    val out = Output(Vec(4, UInt(32.W)))
    val pool_type = Input(UInt(1.W))
    val kernel = Input(UInt(1.W))
    val valid_in = Input(Bool())
  })

  val adder_array = Array(
    Module(new Add_Max(true)),
    Module(new Add_Max),
    Module(new Add_Max),
    Module(new Add_Max),
    Module(new Add_Max),
    Module(new Add_Max),
    Module(new Add_Max),
    Module(new Add_Max)
  )
  val out_2 = Wire(Vec(4, UInt(32.W)))
  val out_3 = Wire(Vec(4, UInt(32.W)))

  val valid_in_r = RegNext(io.valid_in,0.B)
  val valid_in_r2 = RegNext(valid_in_r,0.B)
  val valid_in_r3 = RegNext(valid_in_r2,0.B)
  val valid_in_r4 = RegNext(valid_in_r3,0.B)

  for (i <- 0 until 4) {
    adder_array(i).io.in_0 := io.in(i)
    adder_array(i).io.in_1 := io.in(i + 1)
    adder_array(i).io.pool_type := io.pool_type
    adder_array(i).io.valid_in := io.valid_in
    out_2(i) := adder_array(i).io.out

    adder_array(i + 4).io.in_0 := adder_array(i).io.out
    val sh1 = RegEnable(io.in(i + 2), io.valid_in)
    val sh2 = RegEnable(sh1, valid_in_r)
    val sh3 = RegEnable(sh2, valid_in_r2)
    val sh4 = RegEnable(sh3, valid_in_r3)
    val sh5 = RegEnable(sh4, valid_in_r4)

    adder_array(i + 4).io.in_1 := Mux(io.pool_type === POOL_AVG, sh5, sh1)
    adder_array(i + 4).io.pool_type := io.pool_type
    adder_array(i + 4).io.valid_in := adder_array(0).io.valid_out.get
    out_3(i) := adder_array(i + 4).io.out
  }

  io.out := Mux(io.kernel === KERNEL_2, out_2, out_3)
}

class Pool_Y_in_Buf extends Module with pool_config with cal_cell_params {
  val io = IO(new Bundle {
    val data_in = Input(UInt(128.W))
    val valid_in = Input(UInt(3.W))
    val line_end_in = Input(Bool())
    val data_out = Output(UInt(128.W))
    val valid_out = Output(Bool())
    val line_end_out = Output(Bool())
  })

  val (r0, r1, r2) =
    (RegInit(0.U(32.W)), RegInit(0.U(32.W)), RegInit(0.U(32.W)))
  val valid_count = RegInit(0.U(2.W)) // 0,1,2,3
  val (in0, in1, in2, in3) = (
    io.data_in(31, 0),
    io.data_in(63, 32),
    io.data_in(95, 64),
    io.data_in(127, 96)
  )

  io.data_out := DontCare
  io.valid_out := false.B
  io.line_end_out := false.B

  val line_end = RegInit(false.B)
  when(line_end) {
    line_end := false.B
    io.line_end_out := true.B
  }

  when(ShiftRegister(io.line_end_in, 2)) {
    when(valid_count =/= 0.U) {
      io.data_out := Cat(r2, r1, r0)
      io.valid_out := true.B
    }
    valid_count := 0.U
  }.otherwise {
    valid_count := valid_count + io.valid_in
  }

  when(ShiftRegister(io.line_end_in, 1)) { // 提前一个周期给出 line_end_out
    when((valid_count + io.valid_in)(1, 0) =/= 0.U) {
      line_end := true.B
    }.otherwise {
      io.line_end_out := true.B
    }
  }

  when(io.valid_in =/= 0.U) {
    when(valid_count === 0.U) { // 已有0个的情况
      when(io.valid_in === 4.U) { // 输入4个
        io.data_out := io.data_in
        io.valid_out := true.B
      }.otherwise { // 输入1-3个
        r0 := in0
        r1 := in1
        r2 := in2
      }
    }.elsewhen(valid_count === 1.U) { // 已有1个的情况
      when(io.valid_in >= 3.U) { // 输入3或4个
        io.data_out := Cat(in2, in1, in0, r0)
        r0 := in3
        io.valid_out := true.B
      }.otherwise { // 输入1或2个
        r1 := in0
        r2 := in1
      }
    }.elsewhen(valid_count === 2.U) { // 已有2个的情况
      when(io.valid_in >= 2.U) { // 输入2-4个
        io.data_out := Cat(in1, in0, r1, r0)
        r1 := in3
        r0 := in2
        io.valid_out := true.B
      }.otherwise { // 输入1个
        r2 := in0
      }
    }.otherwise { // 已有3个的情况
      io.data_out := Cat(in0, r2, r1, r0)
      r2 := in3
      r1 := in2
      r0 := in1
      io.valid_out := true.B
    }
  }
}

class Pool_Y extends Module with pool_config with cal_cell_params {
  val io = IO(new Bundle {
    val ofm_h = Input(UInt(oh_len.W))
    val line_end_in = Input(Bool())
    val line_end_out = Output(Bool())

    val data_in = Input(UInt(128.W))
    val valid_in = Input(Bool())
    val data_out = Output(UInt(128.W))
    val valid_out = Output(Bool())

    val pool_type = Input(UInt(1.W))
    val kernel_h = Input(UInt(1.W))
    val stride_h = Input(UInt(2.W))

    val c_finish = Output(Bool())
  })

  io.c_finish := false.B

  val ram_dep: Int = (max_ofm_w + 3) >> 2
  val (r0, r1, r2) = (
    SPRAM_WRAP(128, ram_dep, "block"),
    SPRAM_WRAP(128, ram_dep, "block"),
    SPRAM_WRAP(128, ram_dep, "block")
  )
  val vr_count = RegInit(0.U(2.W)) // valid_regs_count: 0,1,2,3
  val vr_count_r = RegNext(vr_count,0.U)
  val last_reg = RegInit(0.U(2.W)) // 0,1,2
  val last_reg_r = RegNext(last_reg,0.U)
  val ptr = RegInit(0.U((ow_len - 2).W))

  val valid_in_r = RegNext(io.valid_in,0.B)
  val data_in_r = RegNext(io.data_in,0.U)
  val line_end_in_r = RegNext(io.line_end_in,0.U)
  val h_out_count = RegInit(0.U(oh_len.W))
  val c_finish = Wire(Bool())

  r0.en := false.B
  r1.en := false.B
  r2.en := false.B
  r0.wr := DontCare
  r1.wr := DontCare
  r2.wr := DontCare
  r0.addr := ptr
  r1.addr := ptr
  r2.addr := ptr
  r0.wdata := io.data_in
  r1.wdata := io.data_in
  r2.wdata := io.data_in

  val adder_array = Module(new Pool_Y_Adder_Array)
  adder_array.io.pool_type := io.pool_type
  adder_array.io.kernel_h := io.kernel_h
  val aio = adder_array.io
  aio.in_0 := DontCare
  aio.in_1 := DontCare
  aio.in_2 := DontCare

  adder_array.io.valid_in := false.B
  c_finish := false.B

  io.data_out := aio.out
  io.valid_out := adder_array.io.valid_out
  io.line_end_out := Mux(
    io.pool_type === POOL_AVG,
    Mux(
      io.kernel_h === KERNEL_2,
      ShiftRegister(line_end_in_r, FP32_ADD_CYCLES),
      ShiftRegister(line_end_in_r, FP32_ADD_CYCLES * 2)
    ),
    Mux(
      io.kernel_h === KERNEL_2,
      ShiftRegister(line_end_in_r, 1),
      ShiftRegister(line_end_in_r, 2)
    )
  )
  io.c_finish := Mux(
    io.pool_type === POOL_AVG,
    Mux(
      io.kernel_h === KERNEL_2,
      ShiftRegister(c_finish, FP32_ADD_CYCLES + 1),
      ShiftRegister(c_finish, FP32_ADD_CYCLES * 2 + 1)
    ),
    Mux(
      io.kernel_h === KERNEL_2,
      ShiftRegister(c_finish, 2),
      ShiftRegister(c_finish, 3)
    )
  )

  when(io.valid_in) {
    when(io.line_end_in) {
      ptr := 0.U
    }.otherwise {
      ptr := ptr + 1.U
    }
    when(io.kernel_h === KERNEL_2) { // k = 2
      when(io.stride_h === 2.U) { // k=2, s=2
        when(vr_count === 0.U) { // 没有存储的数据，应该先存一行，存在r0
          r0.en := true.B
          r0.wr := SP_WRITE
        }.otherwise { // 已经存了一行，可以输出了
          r0.en := true.B
          r0.wr := SP_READ
          when(io.line_end_in) {
            when(h_out_count + 1.U =/= io.ofm_h) { // 还没输出完一个c
              h_out_count := h_out_count + 1.U
            }.otherwise { // 输出完了一整个c
              h_out_count := 0.U
              c_finish := true.B
            }
          }
        }
        when(io.line_end_in) {
          vr_count := Cat(vr_count(1), ~vr_count(0)) // 0->1, 1->0
        }
      }.otherwise { // k=2, s=1
        when(vr_count === 0.U) { // 没有存储的数据，应该先存一行
          r0.en := true.B // 就存到r0
          r0.wr := SP_WRITE
          when(io.line_end_in) {
            vr_count := 1.U
            last_reg := 0.U
          }
        }.otherwise { // 已经存了一行，可以输出了
          when(last_reg === 0.U) { // r0 有数据，读r0，写r1
            r1.en := true.B
            r1.wr := SP_WRITE
            r0.en := true.B
            r0.wr := SP_READ
          }.otherwise { // r1 有数据，读r1，写r0
            r0.en := true.B
            r0.wr := SP_WRITE
            r1.en := true.B
            r1.wr := SP_READ
          }
          when(io.line_end_in) {
            when(h_out_count + 1.U =/= io.ofm_h) { // 还没输出完一个c
              h_out_count := h_out_count + 1.U
              vr_count := 1.U
              last_reg := Cat(last_reg(1), ~last_reg(0)) // 0->1, 1->0
            }.otherwise { // 输出完了一整个c
              c_finish := true.B
              h_out_count := 0.U
              vr_count := 0.U
            }
          }
        }
      }
    }.otherwise { // k = 3
      // vr_count=0或1, 不管s是几都只能写入, 不会读
      when(vr_count === 0.U) { // 没有存储的数据，应该先存一行，存在r0
        r0.en := true.B
        r0.wr := SP_WRITE
        when(io.line_end_in) {
          vr_count := 1.U
          last_reg := 0.U // 显然这行必须写
        }
      }.elsewhen(vr_count === 1.U) { // 存了1行, 再存第二行
        when(last_reg === 0.U) { // 上一行存在r0
          r1.en := true.B
          r1.wr := SP_WRITE
        }.elsewhen(last_reg === 1.U) { // 上一行存在r1
          r2.en := true.B
          r2.wr := SP_WRITE
        }.otherwise { // 上一行存在r2
          r0.en := true.B
          r0.wr := SP_WRITE
        }
        when(io.line_end_in) {
          vr_count := 2.U
          last_reg := MuxLookup(last_reg, 0.U, Array(0.U -> 1.U, 1.U -> 2.U))
        }
      }.otherwise { // 存了两行了, 应该输出了
        r0.en := true.B
        r1.en := true.B
        r2.en := true.B
        r0.wr := Mux(
          last_reg === 2.U,
          SP_WRITE,
          SP_READ
        ) // 上一行存在r2, 读r1 r2, 写r0
        r1.wr := Mux(
          last_reg === 0.U,
          SP_WRITE,
          SP_READ
        ) // 上一行存在r0, 读r0 r2, 写r1
        r2.wr := Mux(
          last_reg === 1.U,
          SP_WRITE,
          SP_READ
        ) // 上一行存在r1, 读r0 r1, 写r2
        when(io.line_end_in) {
          when(h_out_count + 1.U =/= io.ofm_h) { // 还没输出完一个c
            h_out_count := h_out_count + 1.U
            vr_count := 3.U ^ io.stride_h // 3->0, 2->1, 1->2, 正好是异或
            last_reg := MuxLookup(last_reg, 0.U, Array(0.U -> 1.U, 1.U -> 2.U))
          }.otherwise { // 输出完了一整个c
            c_finish := true.B
            vr_count := 0.U
            h_out_count := 0.U
          }
        }
      }
    }
  }

  // 延后一个周期来匹配 rdata 的延迟
  when(valid_in_r && vr_count_r === Mux(io.kernel_h === KERNEL_2, 1.U, 2.U)) {
    aio.in_0 := data_in_r
    adder_array.io.valid_in := true.B
    when(io.kernel_h === KERNEL_2) { // k = 2
      when(last_reg_r === 0.U || io.stride_h === 2.U) { // r0有数据，读r0
        aio.in_1 := r0.rdata
      }.otherwise { // 那就是r1有数据了
        aio.in_1 := r1.rdata
      }
    }.otherwise { // k = 3
      aio.in_1 := Mux(last_reg_r === 2.U, r2.rdata, r0.rdata)
      aio.in_2 := Mux(last_reg_r === 0.U, r2.rdata, r1.rdata)
    }
    when(h_out_count === io.ofm_h) {
      io.c_finish := true.B
      vr_count := 0.U
    }
  }

  when(ptr(ow_len - 3) =/= 0.U) { // 防止访问溢出 (w=512, k=2, s=1, pad_l=0, pad_r=1的情况), 超过限制关闭ram阻止写入
    r0.en := false.B
    r1.en := false.B
    r2.en := false.B
  }

}

/*  in_2 ()()()()
 *  in_1 ()()()()
 *  in_0 ()()()()
 *       ↓ ↓ ↓ ↓
 * out_3 ()()()() = in_2 + in_1 + in_0
 * out_2 ()()()() = in_1 + in_0
 * */
class Pool_Y_Adder_Array extends Module with pool_config with cal_cell_params {
  val io = IO(new Bundle {
    val in_0 = Input(UInt(128.W))
    val in_1 = Input(UInt(128.W))
    val in_2 = Input(UInt(128.W))
    val out = Output(UInt(128.W))
    val kernel_h = Input(UInt(1.W))
    val pool_type = Input(UInt(1.W))
    val valid_in = Input(Bool())
    val valid_out = Output(Bool())
  })

  val a0 = Module(new Add_Max(true))
  val a1 = Module(new Add_Max)
  val a2 = Module(new Add_Max)
  val a3 = Module(new Add_Max)
  a0.io.pool_type := io.pool_type
  a1.io.pool_type := io.pool_type
  a2.io.pool_type := io.pool_type
  a3.io.pool_type := io.pool_type
  a0.io.in_0 := io.in_0(31, 0)
  a1.io.in_0 := io.in_0(63, 32)
  a2.io.in_0 := io.in_0(95, 64)
  a3.io.in_0 := io.in_0(127, 96)
  a0.io.in_1 := io.in_1(31, 0)
  a1.io.in_1 := io.in_1(63, 32)
  a2.io.in_1 := io.in_1(95, 64)
  a3.io.in_1 := io.in_1(127, 96)
  a0.io.valid_in := io.valid_in
  a1.io.valid_in := io.valid_in
  a2.io.valid_in := io.valid_in
  a3.io.valid_in := io.valid_in

  val a4 = Module(new Add_Max(true))
  val a5 = Module(new Add_Max)
  val a6 = Module(new Add_Max)
  val a7 = Module(new Add_Max)
  a4.io.pool_type := io.pool_type
  a5.io.pool_type := io.pool_type
  a6.io.pool_type := io.pool_type
  a7.io.pool_type := io.pool_type
  a4.io.in_0 := a0.io.out
  a5.io.in_0 := a1.io.out
  a6.io.in_0 := a2.io.out
  a7.io.in_0 := a3.io.out
  val in_2_r = RegEnable(io.in_2, io.valid_in)
  val valid_in_r = RegNext(io.valid_in,0.B)
  val in_2_r2 = RegEnable(in_2_r, valid_in_r)
  val valid_in_r2 = RegNext(valid_in_r,0.B)
  val in_2_r3 = RegEnable(in_2_r2, valid_in_r2)
  val valid_in_r3 = RegNext(valid_in_r2,0.B)
  val in_2_r4 = RegEnable(in_2_r3, valid_in_r3)
  val valid_in_r4 = RegNext(valid_in_r3,0.B)
  val in_2_r5 = RegEnable(in_2_r4, valid_in_r4)
  a4.io.in_1 := Mux(io.pool_type === POOL_AVG, in_2_r5(31, 0), in_2_r(31, 0))
  a5.io.in_1 := Mux(io.pool_type === POOL_AVG, in_2_r5(63, 32), in_2_r(63, 32))
  a6.io.in_1 := Mux(io.pool_type === POOL_AVG, in_2_r5(95, 64), in_2_r(95, 64))
  a7.io.in_1 := Mux(
    io.pool_type === POOL_AVG,
    in_2_r5(127, 96),
    in_2_r(127, 96)
  )
  a4.io.valid_in := a0.io.valid_out.get
  a5.io.valid_in := a0.io.valid_out.get
  a6.io.valid_in := a0.io.valid_out.get
  a7.io.valid_in := a0.io.valid_out.get

  io.out := Mux(
    io.kernel_h === KERNEL_2,
    Cat(a3.io.out, a2.io.out, a1.io.out, a0.io.out),
    Cat(a7.io.out, a6.io.out, a5.io.out, a4.io.out)
  )
  io.valid_out := Mux(
    io.kernel_h === KERNEL_2,
    a0.io.valid_out.get,
    a4.io.valid_out.get
  )
}

class Pool_Y_Trim extends Module with pool_config with cal_cell_params {
  val io = IO(new Bundle {
    val data_in = Input(UInt(128.W))
    val valid_in = Input(Bool())
    val line_end_in = Input(Bool())
    val c_finish = Input(Bool())
    val ofm_w = Input(UInt(ow_len.W))
    val ofm_c = Input(UInt(c_len.W))

    val data_out = Output(UInt(128.W))
    val valid_out = Output(Bool())
    val task_done = Output(Bool())
    val o_ex_fix = Input(Bool())
  })

  io.task_done := false.B
  io.valid_out := false.B
  io.data_out := DontCare

  val (r0, r1, r2) =
    (RegInit(0.U(32.W)), RegInit(0.U(32.W)), RegInit(0.U(32.W)))
  val valid_count = RegInit(0.U(2.W)) // 0,1,2,3
  val (in0, in1, in2, in3) = (
    io.data_in(31, 0),
    io.data_in(63, 32),
    io.data_in(95, 64),
    io.data_in(127, 96)
  )

  val w_count = RegInit(0.U(ow_len.W))
  val c_count = RegInit(0.U(c_len.W))
  val ex_output = RegInit(false.B) // 有时候不得不多一个周期来输出

  val act_valid = (io.ofm_w - w_count)(2, 0)
  when(io.valid_in) {
    when(valid_count === 0.U) {
      when(w_count + 4.U < io.ofm_w) {
        w_count := w_count + 4.U
        io.data_out := io.data_in
        io.valid_out := true.B
      }.otherwise { // 有效的数量应该是 ofm_w(1,0) 或 4
        when(io.line_end_in) {
          w_count := 0.U
        }.otherwise {
          w_count := io.ofm_w
        }
        when(io.c_finish || act_valid === 4.U) {
          io.data_out := io.data_in
          io.valid_out := act_valid =/= 0.U
        }.otherwise {
          r0 := in0
          r1 := in1
          r2 := in2
          valid_count := valid_count + act_valid
        }
      }
    }.elsewhen(valid_count === 1.U) {
      when(w_count + 4.U < io.ofm_w) {
        w_count := w_count + 4.U
        io.data_out := Cat(in2, in1, in0, r0)
        r0 := in3
        io.valid_out := true.B
      }.otherwise { // 有效的数量应该是 ofm_w(1,0) 或 4
        when(io.line_end_in) {
          w_count := 0.U
        }.otherwise {
          w_count := io.ofm_w
        }
        when(!io.c_finish) {
          valid_count := valid_count + act_valid
          when(act_valid < 3.U) {
            r1 := in0
            r2 := in1
          }.otherwise {
            io.data_out := Cat(in2, in1, in0, r0)
            r0 := in3
            io.valid_out := true.B
          }
        }.otherwise {
          io.data_out := Cat(in2, in1, in0, r0)
          r0 := in3
          io.valid_out := true.B
          when(act_valid === 4.U) {
            valid_count := 1.U
            ex_output := true.B
          }.otherwise {
            valid_count := 0.U
          }
        }
      }
    }.elsewhen(valid_count === 2.U) {
      when(w_count + 4.U < io.ofm_w) {
        w_count := w_count + 4.U
        io.data_out := Cat(in1, in0, r1, r0)
        r0 := in2
        r1 := in3
        io.valid_out := true.B
      }.otherwise { // 有效的数量应该是 ofm_w(1,0) 或 4
        when(io.line_end_in) {
          w_count := 0.U
        }.otherwise {
          w_count := io.ofm_w
        }
        when(!io.c_finish) {
          valid_count := valid_count + act_valid
          when(act_valid < 2.U) {
            r2 := in0
          }.otherwise {
            io.data_out := Cat(in1, in0, r1, r0)
            io.valid_out := true.B
            r0 := in2
            r1 := in3
          }
        }.otherwise { // c_finish !!
          io.data_out := Cat(in1, in0, r1, r0)
          r0 := in2
          r1 := in3
          io.valid_out := true.B
          when(act_valid < 3.U) {
            valid_count := 0.U
          }.otherwise {
            valid_count := valid_count + act_valid
            ex_output := true.B
          }
        }
      }
    }.otherwise { // valid_count === 3.U
      when(w_count + 4.U < io.ofm_w) {
        w_count := w_count + 4.U
        io.data_out := Cat(in0, r2, r1, r0)
        r0 := in1
        r1 := in2
        r2 := in3
        io.valid_out := true.B
      }.otherwise { // 有效的数量应该是 ofm_w(1,0) 或 4
        when(io.line_end_in) {
          w_count := 0.U
        }.otherwise {
          w_count := io.ofm_w
        }
        when(!io.c_finish) {
          valid_count := valid_count + act_valid
          when(act_valid =/= 0.U) {
            io.data_out := Cat(in0, r2, r1, r0)
            io.valid_out := true.B
            r0 := in1
            r1 := in2
            r2 := in3
          }
        }.otherwise { // c_finish !!
          io.data_out := Cat(in0, r2, r1, r0)
          r0 := in1
          r1 := in2
          r2 := in3
          io.valid_out := true.B
          when(act_valid < 2.U) {
            valid_count := 0.U
          }.otherwise {
            valid_count := valid_count + act_valid
            ex_output := true.B
          }
        }
      }
    }
  }
  when(ex_output) { // io.valid_in 一定是false(一定是), 不然就是出错了!!
    io.data_out := Cat(r2, r1, r0)
    io.valid_out := true.B
    ex_output := false.B
    valid_count := 0.U
  }

  val c_finish = Mux(io.o_ex_fix, ShiftRegister(io.c_finish, 2), io.c_finish)

  // 统计c以输出task_done
  when(c_finish) {
    c_count := c_count + 1.U
  }
  when(ShiftRegister(io.c_finish, 2)) {
    io.valid_out := io.o_ex_fix // 额外的valid, 如果需要对齐到256
  }
  when(c_count === io.ofm_c && io.ofm_c =/= 0.U) {
    c_count := 0.U
    io.task_done := true.B
  }
}

class FP32_DIV4 extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(32.W))
    val out = Output(UInt(32.W))
  })
  when(io.in(30, 23) < 2.U) {
    io.out := Cat(io.in(31), 0.U(8.W), 0.U(23.W)) // underflow
  }.otherwise {
    io.out := Cat(io.in(31), io.in(30, 23) - 2.U, io.in(22, 0))
  }
}

class FP32_DIV9(use_valid_out: Boolean = false)
    extends Module
    with pool_config {
  val io = IO(new Bundle {
    val in = Input(UInt(32.W))
    val out = Output(UInt(32.W))
    val valid_in = Input(Bool())
    val valid_out = if (use_valid_out) Some(Output(Bool())) else None
  })

  if (DIV9_METHOD == "shift_add") {
    val (xs, xe, xf) = (io.in(31), io.in(30, 23), Cat(1.U(1.W), io.in(22, 0)))

    val msb_will_0 = RegEnable(
      xf(22, 20) === 0.U,
      io.valid_in
    ) // io.in(22, 0) < 1048576 (2^20)
    val xf_is_9 = RegEnable(
      xf(22, 0) === 1048576.U,
      io.valid_in
    ) // Cat(1.U(1.W), io.in(22, 0)) = 0x900000
    val valid_in_r = RegNext(io.valid_in,0.B)
    val underflow1 = RegEnable(
      (msb_will_0 && RegEnable(xe, io.valid_in) < 4.U) || RegEnable(
        xe,
        io.valid_in
      ) < 3.U,
      valid_in_r
    )
    val xe0 = RegEnable(xe, io.valid_in)
    val xe1 = RegEnable(xe0, valid_in_r) - Mux(
      RegEnable(msb_will_0, valid_in_r),
      4.U,
      3.U
    )
    val xf0 = RegEnable(xf - (xf >> 3).asUInt, io.valid_in)
    val xf1 = RegEnable(xf0 + (xf0 >> 6).asUInt, valid_in_r)
    val valid_in_r2 = RegNext(valid_in_r,0.B)
    val xf2 = RegEnable((xf1 + (xf1 >> 12).asUInt)(22, 0), valid_in_r2)

    val out_s0 = RegEnable(xs, io.valid_in)
    val out_s1 = RegEnable(out_s0, valid_in_r)
    val out_s2 = RegEnable(out_s1, valid_in_r2)

    val xf_is_9_1 = RegEnable(xf_is_9, valid_in_r)
    val xf_is_9_2 = RegEnable(xf_is_9_1, valid_in_r2)
    val msb_will_0_1 = RegEnable(msb_will_0, valid_in_r)
    val msb_will_0_2 = RegEnable(msb_will_0_1, valid_in_r2)

    val out_e =
      Mux(RegEnable(underflow1, valid_in_r2), 0.U, RegEnable(xe1, valid_in_r2))
    val out_f = Mux(
      xf_is_9_2,
      0.U,
      Mux(msb_will_0_2, Cat(xf2(21, 0), 0.U(1.W)), xf2(22, 0))
    )

    io.out := Cat(out_s2, out_e, out_f)
    if (use_valid_out) {
      io.valid_out.get := RegNext(valid_in_r2,0.U)
    }

  } else { // DIV9_METHOD == "mult"
    val mult = FP32_Mult(use_valid_out)
    mult.io.x := io.in
    mult.io.y := java.lang.Float.floatToRawIntBits(1.0f / 9.0f).U(32.W)
    mult.io.valid_in := io.valid_in
    io.out := mult.io.z
    if (use_valid_out) {
      io.valid_out.get := mult.io.valid_out.get
    }
  }

}

class Pool_Mult_Buf extends Module with pool_config with cal_cell_params {

  val io = IO(new Bundle {
    /*--- input ---*/
    val data_in = Input(UInt(128.W))
    val valid_in = Input(Bool())
    val task_done_in = Input(Bool())
    val kernel = Input(UInt(1.W))
    val pool_type = Input(UInt(1.W))

    /*--- output ---*/
    val data_out = Output(UInt(128.W))
    val valid_out = Output(Bool())
    val task_done_out = Output(Bool())
  })

  val div4 = Array.fill(4)(Module(new FP32_DIV4))
  val div9 = Array(
    Module(new FP32_DIV9(true)),
    Module(new FP32_DIV9),
    Module(new FP32_DIV9),
    Module(new FP32_DIV9)
  )

  div4(0).io.in := io.data_in(31, 0)
  div4(1).io.in := io.data_in(63, 32)
  div4(2).io.in := io.data_in(95, 64)
  div4(3).io.in := io.data_in(127, 96)
  div9(0).io.in := io.data_in(31, 0)
  div9(1).io.in := io.data_in(63, 32)
  div9(2).io.in := io.data_in(95, 64)
  div9(3).io.in := io.data_in(127, 96)
  val div9_en =
    io.valid_in && io.pool_type === POOL_AVG && io.kernel === KERNEL_3
  div9(0).io.valid_in := div9_en
  div9(1).io.valid_in := div9_en
  div9(2).io.valid_in := div9_en
  div9(3).io.valid_in := div9_en

  when(io.pool_type === POOL_MAX) {
    io.data_out := io.data_in
    io.valid_out := io.valid_in
    io.task_done_out := io.task_done_in
  }.otherwise {
    when(io.kernel === KERNEL_2) {
      io.data_out := Cat(
        div4(3).io.out,
        div4(2).io.out,
        div4(1).io.out,
        div4(0).io.out
      )
      io.valid_out := io.valid_in
      io.task_done_out := io.task_done_in
    }.otherwise { // kernel=3
      io.data_out := Cat(
        div9(3).io.out,
        div9(2).io.out,
        div9(1).io.out,
        div9(0).io.out
      )
      io.valid_out := div9(0).io.valid_out.get
      io.task_done_out := ShiftRegister(io.task_done_in, FP32_DIV9_CYCLES)
    }
  }
}

class Pool_Out_Buf
    extends Module
    with pool_config
    with axi_config
    with cal_cell_params {
  val io = IO(new Bundle {
    /*--- input ---*/
    val start = Input(Bool())
    val data_in = Input(UInt(128.W))
    val valid_in = Input(Bool())
    val task_done_in = Input(Bool())
    val dst_addr = Input(UInt(AXI_ADDR_WIDTH.W))

    val axi_send = Flipped(new data_axiSend)
    val task_done_out = Output(Bool())
  })

  val count = RegInit(0.U(log2Ceil(axi_send_size + 1).W))
  val dst_addr = RegInit(0.U((AXI_ADDR_WIDTH - 4).W))
  val task_done = RegInit(false.B)
  io.task_done_out := task_done

  io.axi_send.data := io.data_in
  io.axi_send.data_valid := io.valid_in
  io.axi_send.addr := Cat(dst_addr, 0.U(4.W))

  when(io.valid_in) {
    count := count + 1.U
  }

  when(io.task_done_in) {
    task_done := true.B
  }

  //axi_send_size = 128
  when(count(7) === 1.U) {
    count := count - Mux(io.valid_in, (axi_send_size - 1).U, axi_send_size.U)
    io.axi_send.size := axi_send_size.U
    io.axi_send.size_valid := true.B
    io.axi_send.addr_valid := true.B
    dst_addr := Cat(dst_addr(AXI_ADDR_WIDTH - 5, 7) + 1.U, dst_addr(6, 0))
  }.elsewhen(task_done) {
    count := 0.U // no more
    io.axi_send.size := count
    io.axi_send.size_valid := count =/= 0.U
    io.axi_send.addr_valid := count =/= 0.U
  }.otherwise {
    io.axi_send.size := DontCare
    io.axi_send.size_valid := false.B
    io.axi_send.addr_valid := false.B
  }

  when(io.start) {
    dst_addr := io.dst_addr(AXI_ADDR_WIDTH - 1, 4)
    count := 0.U
    task_done := false.B
    io.task_done_out := false.B // 防止被上一次未重置的task_done影响
  }

}
