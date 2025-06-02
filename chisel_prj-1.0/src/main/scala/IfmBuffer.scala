import chisel3._
import chisel3.util._
import util_function._

class ifm_w_io(IFM_BUFFER_DEPTH: Int, IFM_BUFFER_WIDTH: Int) extends Bundle {
  val wen   = Output(Bool())
  val waddr = Output(UInt(log2Ceil(IFM_BUFFER_DEPTH).W))
  val wdata = Output(UInt(IFM_BUFFER_WIDTH.W))
}

class ifm_r_io(IFM_BUFFER_DEPTH: Int, IFM_BUFFER_WIDTH: Int) extends Bundle {
  val ren   = Input(Bool())
  val raddr = Input(UInt(log2Ceil(IFM_BUFFER_DEPTH).W))
  val rdata = Output(UInt((IFM_BUFFER_WIDTH).W))
}

class ifm_param_fifo_io extends Bundle with axi_config {
  val iwh_base = Decoupled(UInt(24.W))
  val ic_base  = Decoupled(UInt(12.W))
  val ic_pad   = Decoupled(UInt(5.W))
  val axi_addr = Decoupled(UInt(AXI_ADDR_WIDTH.W))
  val axi_size = Decoupled(UInt(AXI_SIZE_WIDTH.W))
}

class ifm_param_io extends Bundle with axi_config {
  val iwh_base = UInt(24.W)
  val ic_base  = UInt(12.W)
  val ic_pad   = UInt(5.W)
  val axi_addr = UInt(AXI_ADDR_WIDTH.W)
  val axi_size = UInt(AXI_SIZE_WIDTH.W)
  val valid    = Bool()
}

class quant_cell extends Module {
  val io = IO(new Bundle() {
    val en     = Input(Bool())
    val i_data = Input(UInt(32.W))
    val scale  = Input(UInt(32.W))
    val o_data = Output(UInt(8.W))
  })
  val i_data_t = RegEnable(io.i_data, 0.U, io.en)
  val quant_data = Float.FloatMul(Float(i_data_t), Float(io.scale), io.en)
  io.o_data := Float.FloatToSInt(quant_data, 8, io.en).asUInt
}

class quant_unit extends Module with axi_config with cal_cell_params {
  val io = IO(new Bundle() {
    val i_en    = Input(Bool())
    val i_data  = Flipped(Valid(UInt(AXI_DATA_WIDTH.W)))
    val i_scale = Input(UInt(32.W))
    val o_data  = Output(new data_gp(4, 8))
  })
  val quant = Seq.fill(AXI_DATA_WIDTH / 32)(Module(new quant_cell))
  for (i <- 0 until AXI_DATA_WIDTH / 32) {
    quant(i).io.i_data := io.i_data.bits(i * 32 + 31, i * 32)
    quant(i).io.scale  := io.i_scale
    quant(i).io.en     := io.i_en
  }
  for (i <- 0 until 4) {
    io.o_data.data(i) := quant(3 - i).io.o_data
  }
  io.o_data.valid := ShiftRegister(io.i_data.valid, FP32_TO_SINT_CYCLES + FP32_MUL_CYCLES + 1, 0.U, io.i_en)
}

class ifm_param_cal extends Module with axi_config {
  val io = IO(new Bundle() {
    val cfg_gemm         = Input(new cfg_gemm_io)
    val param_fifo_o     = new ifm_param_fifo_io
    val param_fifo_empty = Output(Bool())
    val param_cal_finish = Output(Bool())
  })

  val param_cal_finish = RegInit(false.B)
  val ic_align32       = RegEnable(align(io.cfg_gemm.ic, 12, 32), io.cfg_gemm.en)
  val iwh              = RegEnable(io.cfg_gemm.iw * io.cfg_gemm.ih, io.cfg_gemm.en)
  io.param_cal_finish := param_cal_finish
  val en = ShiftRegister(io.cfg_gemm.en, 3, 0.U, 1.B)

  val iwh_base_fifo = Module(new Queue(UInt(24.W), entries = 4))
  val ic_base_fifo  = Module(new Queue(UInt(12.W), entries = 4))
  val ic_pad_fifo   = Module(new Queue(UInt(5.W), entries = 4))
  val axi_addr_fifo = Module(new Queue(UInt(AXI_ADDR_WIDTH.W), entries = 4))
  val axi_size_fifo = Module(new Queue(UInt(AXI_SIZE_WIDTH.W), entries = 4))

  io.param_fifo_o.iwh_base   <> iwh_base_fifo.io.deq
  io.param_fifo_o.ic_base    <> ic_base_fifo.io.deq
  io.param_fifo_o.ic_pad     <> ic_pad_fifo.io.deq
  io.param_fifo_o.axi_addr   <> axi_addr_fifo.io.deq
  io.param_fifo_o.axi_size   <> axi_size_fifo.io.deq
  iwh_base_fifo.io.enq.valid := false.B
  ic_base_fifo.io.enq.valid  := false.B
  ic_pad_fifo.io.enq.valid   := false.B
  axi_addr_fifo.io.enq.valid := false.B
  axi_size_fifo.io.enq.valid := false.B
  iwh_base_fifo.io.enq.bits  := 0.U
  ic_base_fifo.io.enq.bits   := 0.U
  ic_pad_fifo.io.enq.bits    := 0.U
  axi_addr_fifo.io.enq.bits  := 0.U
  axi_size_fifo.io.enq.bits  := 0.U

  val fifo_not_full = RegNext(axi_addr_fifo.io.enq.ready,0.B)
  io.param_fifo_empty := axi_addr_fifo.io.count === 0.U
  val iwh_base         = RegInit(0.U(24.W))
  val iwh_base_plus    = RegInit(0.U(24.W))
  val ic_base          = RegInit(0.U(12.W))
  val ic_base_plus     = RegInit(0.U(12.W))
  val ic_base_next     = RegInit(0.U(12.W))
  val ic_pad_base_next = RegInit(0.U(12.W))
  val ic_pad           = RegInit(0.U(5.W))
  val axi_addr         = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_addr_base    = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_size         = RegInit(0.U(AXI_SIZE_WIDTH.W))
  val axi_addr_offset  = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val first_write      = RegInit(true.B)
  val isFp32Mat        = io.cfg_gemm.en & io.cfg_gemm.layout_en

  val ic_limit      = ic_base_plus >= io.cfg_gemm.ic
  val iwh_limit     = iwh_base_plus >= iwh
  val int8_next_iwh = RegInit(0.U(6.W))
  val int8_cur_iwh  = iwh - iwh_base_plus

  val sIdle :: sFp32Init :: sFp32Init_t :: sFp32axi :: sInt8Init :: sWrite :: sFinish :: sJump :: Nil = Enum(8)

  val state = RegInit(sIdle)

  switch(state) {
    is(sIdle) {
      when(param_cal_finish | !io.cfg_gemm.en || !en) {
        state := sIdle
      }.otherwise {
        state := Mux(isFp32Mat, sFp32Init, sInt8Init)
      }
      axi_addr         := io.cfg_gemm.ifm_addr
      axi_addr_base    := 0.U
      axi_size         := Mux(isFp32Mat, 16.U, 0.U)
      iwh_base         := 0.U
      iwh_base_plus    := 0.U
      ic_base          := 0.U
      ic_base_plus     := 0.U
      ic_base_next     := 0.U
      ic_pad           := Mux(isFp32Mat, Mux(io.cfg_gemm.ic < 32.U, 32.U - io.cfg_gemm.ic, 0.U), 0.U)
      ic_pad_base_next := 0.U
      int8_next_iwh    := Mux(iwh > 32.U, 32.U, iwh)
    }

    is(sFp32Init) {
      when(first_write) {
        state := sWrite
      }.elsewhen(fifo_not_full) {
        when(ic_pad =/= 0.U || ic_base_plus >= io.cfg_gemm.ic) {
          ic_base      := 0.U
          ic_base_plus := 0.U
          iwh_base     := iwh_base + 64.U
        }.otherwise {
          ic_base := ic_base_plus
        }
        state := sFp32Init_t
      }.otherwise {
        state := sFp32Init
      }
    }

    is(sFp32Init_t) { //state 2
      ic_base_next    := ic_base + 32.U
      axi_addr_offset := ic_base * Cat(io.cfg_gemm.icstep, 0.U(2.W))
      axi_addr_base   := io.cfg_gemm.ifm_addr + Cat(iwh_base, 0.U(2.W))
      state           := sFp32axi
    }

    is(sFp32axi) { //state 3
      ic_pad   := Mux(ic_base_next > io.cfg_gemm.ic, ic_base_next - io.cfg_gemm.ic, 0.U)
      axi_addr := axi_addr_offset + axi_addr_base
      state    := sWrite
    }

    is(sInt8Init) {
      when(fifo_not_full) {
        when(!first_write) {
          iwh_base := iwh_base + 32.U
          axi_addr := axi_addr + Cat(Mux(io.cfg_gemm.div_ifm_c_en, io.cfg_gemm.div_ifm_c, ic_align32), 0.U(5.W))
        }
        axi_size := Mux(io.cfg_gemm.div_ifm_c_en, ic_align32(11, 4), ic_align32(11, 4) * int8_next_iwh)
//        axi_size := Mux(cfg_gemm.div_ifm_c_en, cfg_gemm.div_ifm_c(11,4), ic_align32(11,4)*int8_next_iwh)
        state := sWrite
      }.otherwise {
        state := sInt8Init
      }
    }

    is(sWrite) {
      first_write                := false.B
      iwh_base_fifo.io.enq.bits  := iwh_base
      ic_base_fifo.io.enq.bits   := ic_base
      ic_pad_fifo.io.enq.bits    := ic_pad
      axi_addr_fifo.io.enq.bits  := axi_addr
      axi_size_fifo.io.enq.bits  := axi_size
      ic_base_plus               := Mux(isFp32Mat, ic_base_plus + 32.U, ic_base_plus)
      iwh_base_plus              := Mux(isFp32Mat, iwh_base + 64.U, iwh_base_plus + 32.U)
      iwh_base_fifo.io.enq.valid := true.B
      ic_base_fifo.io.enq.valid  := true.B
      ic_pad_fifo.io.enq.valid   := true.B
      axi_addr_fifo.io.enq.valid := true.B
      axi_size_fifo.io.enq.valid := true.B
      state                      := sFinish
    }

    is(sFinish) {
      int8_next_iwh    := Mux(isFp32Mat, 0.U, Mux(int8_cur_iwh < 32.U, int8_cur_iwh, 32.U))
      param_cal_finish := (isFp32Mat & ic_limit & iwh_limit) | (!isFp32Mat & iwh_limit)
      state            := sJump
    }

    is(sJump) {
      state := Mux(param_cal_finish, sIdle, Mux(isFp32Mat, sFp32Init, sInt8Init))
    }
  }
}

class ifm_param_arbit extends Module with axi_config {
  val io = IO(new Bundle() {
    val param_fifo_i     = Flipped(new ifm_param_fifo_io)
    val param_ch0        = Output(new ifm_param_io)
    val param_ch1        = Output(new ifm_param_io)
    val param_areq_ch0   = Input(Bool())
    val param_areq_ch1   = Input(Bool())
    val param_fifo_empty = Input(Bool())
  })

  io.param_ch0 <> 0.U.asTypeOf(new ifm_param_io)
  io.param_ch1 <> 0.U.asTypeOf(new ifm_param_io)
  val param_fifo_rd_en = RegInit(false.B)
  io.param_fifo_i.iwh_base.ready := param_fifo_rd_en
  io.param_fifo_i.ic_base.ready  := param_fifo_rd_en
  io.param_fifo_i.ic_pad.ready   := param_fifo_rd_en
  io.param_fifo_i.axi_addr.ready := param_fifo_rd_en
  io.param_fifo_i.axi_size.ready := param_fifo_rd_en

  val sIdle :: sWait_ch0 :: sWait_ch1 :: Nil = Enum(3)
  val state                                  = RegInit(sIdle)

  switch(state) {
    is(sIdle) {
      when(io.param_areq_ch0 & !io.param_fifo_empty) {
        state := sWait_ch0
      }.elsewhen(io.param_areq_ch1 & !io.param_fifo_empty) {
        state := sWait_ch1
      }.otherwise {
        state := sIdle
      }
      when((io.param_areq_ch0 | io.param_areq_ch1) & !io.param_fifo_empty) {
        param_fifo_rd_en := true.B
      }
    }

    is(sWait_ch0) {
      param_fifo_rd_en      := false.B
      io.param_ch0.iwh_base := io.param_fifo_i.iwh_base.bits
      io.param_ch0.ic_base  := io.param_fifo_i.ic_base.bits
      io.param_ch0.ic_pad   := io.param_fifo_i.ic_pad.bits
      io.param_ch0.axi_addr := io.param_fifo_i.axi_addr.bits
      io.param_ch0.axi_size := io.param_fifo_i.axi_size.bits
      io.param_ch0.valid    := io.param_fifo_i.axi_addr.valid
      state                 := Mux(io.param_fifo_i.axi_addr.valid, sIdle, sWait_ch0)
    }

    is(sWait_ch1) {
      param_fifo_rd_en      := false.B
      io.param_ch1.iwh_base := io.param_fifo_i.iwh_base.bits
      io.param_ch1.ic_base  := io.param_fifo_i.ic_base.bits
      io.param_ch1.ic_pad   := io.param_fifo_i.ic_pad.bits
      io.param_ch1.axi_addr := io.param_fifo_i.axi_addr.bits
      io.param_ch1.axi_size := io.param_fifo_i.axi_size.bits
      io.param_ch1.valid    := io.param_fifo_i.axi_addr.valid
      state                 := Mux(io.param_fifo_i.axi_addr.valid, sIdle, sWait_ch1)
    }
  }
}

class int8_ifm_cell extends Module with axi_config with buffer_config {
  val io = IO(new Bundle() {
    //ctrl signal
    val cfg_gemm = Input(new cfg_gemm_io)
    //axi read
    val axi = new axi_r
    //param_fifo
    val param       = Input(new ifm_param_io)
    val param_areq  = Output(Bool())
    val param_empty = Input(Bool())
    //ifm mem write
    val ifm_w          = new ifm_w_io(IFM_BUFFER_DEPTH, IFM_BUFFER_WIDTH)
    val int8_cell_idle = Output(Bool())
  })

  val burst_num_cnt  = RegInit(0.U(6.W))
  val iwh_base       = RegEnable(io.param.iwh_base, 0.U, io.param.valid)
  val axi_addr_base  = RegEnable(io.param.axi_addr, 0.U, io.param.valid)
  val axi_size       = RegEnable(io.param.axi_size, 0.U, io.param.valid)
  val axi_addr       = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_en         = RegInit(false.B)
  val axi_areq       = RegInit(false.B)
  val param_areq     = RegInit(false.B)
  val en             = io.cfg_gemm.en & !io.cfg_gemm.layout_en
  val burst_num      = RegEnable(Mux(io.cfg_gemm.div_ifm_c_en, 32.U, 1.U), io.cfg_gemm.en)
  val ifm_waddr      = RegInit(0.U((log2Ceil(IFM_BUFFER_DEPTH) + 1).W))
  val ifm_waddr_base = RegInit(0.U((log2Ceil(IFM_BUFFER_DEPTH) + 1).W))
  val ic_align32     = RegEnable(align(io.cfg_gemm.ic, 12, 32), io.cfg_gemm.en)

  io.axi.areq.axiEn   := axi_en
  io.axi.areq.axiAreq := axi_areq
  io.axi.areq.axiSize := axi_size
  io.axi.areq.axiAddr := axi_addr
  io.param_areq       := param_areq

  val sIdle :: sAreqParam :: sCfgaxi :: sArbit :: sWait :: sFinish :: Nil = Enum(6)

  val state = RegInit(sIdle)
  io.int8_cell_idle := state === sIdle

  switch(state) {
    is(sIdle) {  //state0
      axi_en        := false.B
      axi_areq      := false.B
      burst_num_cnt := 0.U
      state         := Mux(en & !io.param_empty, sAreqParam, sIdle)
    }
    is(sAreqParam) {  //state1
      param_areq    := true.B
      burst_num_cnt := 0.U
      when(io.param_empty) {
        param_areq := false.B
        state      := sIdle
      }.elsewhen(io.param.valid) {
        param_areq := false.B
        state      := sCfgaxi
      }.otherwise {
        state := sAreqParam
      }
    }
    is(sCfgaxi) {  //state2
      ifm_waddr_base := iwh_base * ic_align32(11, 5)
      when(burst_num_cnt === 0.U) {
        axi_addr := axi_addr_base
      }.otherwise {
        axi_addr := axi_addr + io.cfg_gemm.div_ifm_c
      }
      burst_num_cnt := burst_num_cnt + 1.U
      state         := sArbit
    }
    is(sArbit) {  //state3
      axi_en := true.B
      when(axi_en & io.axi.id === RID("gemm_ifm").U) {
        state    := sWait
        axi_areq := true.B
      }.otherwise {
        state := sArbit
      }
    }
    is(sWait) {  //state4
      axi_areq := false.B
      state := Mux(io.axi.data.last, sFinish, sWait)
    }
    is(sFinish) {
      when(burst_num_cnt === burst_num) {
        state := Mux(io.param_empty, sIdle, sAreqParam)
      }.otherwise {
        state := sCfgaxi
      }
    }
  }

  val axi_data   = RegEnable(io.axi.data.data, io.axi.data.valid)
  val axi_valid  = RegNext(io.axi.data.valid,0.B)
  val axi_data_t = RegEnable(axi_data, 0.U, axi_valid)
  val ifm_cnt    = RegInit(0.U(1.W))
  ifm_cnt        := Mux(axi_valid, !ifm_cnt, ifm_cnt)
  ifm_waddr      := Mux(!en, 0.U, Mux(state === sArbit && burst_num_cnt === 1.U, ifm_waddr_base, Mux(io.ifm_w.wen, ifm_waddr + 1.U, ifm_waddr)))
  io.ifm_w.wen   := RegEnable(axi_valid & ifm_cnt & !ifm_waddr(log2Ceil(IFM_BUFFER_DEPTH)), 0.U, io.cfg_gemm.en)
  io.ifm_w.wdata := RegEnable(Cat(axi_data, axi_data_t), io.cfg_gemm.en)
  io.ifm_w.waddr := RegEnable(ifm_waddr(log2Ceil(IFM_BUFFER_DEPTH) - 1, 0), io.cfg_gemm.en)

}

class fp32_ifm_cell extends Module with axi_config with buffer_config with cal_cell_params {
  val io = IO(new Bundle() {
    //ctrl signal
    val cfg_gemm = Input(new cfg_gemm_io)
    //axi read
    val axi = new axi_r
    //param_fifo
    val param       = Input(new ifm_param_io)
    val param_areq  = Output(Bool())
    val param_empty = Input(Bool())
    //ifm mem write
    val ifm_w          = new ifm_w_io(IFM_BUFFER_DEPTH, IFM_BUFFER_WIDTH)
    val fp32_cell_idle = Output(Bool())
  })

  val intr_num_cnt  = RegInit(0.U(6.W))
  val burst_num_cnt  = RegInit(0.U(6.W))
  val dma_trans_finish = RegInit(false.B)
  val ic_pad         = RegEnable(io.param.ic_pad, 0.U, io.param.valid)
  val axi_addr_base  = RegEnable(io.param.axi_addr, 0.U, io.param.valid)
  val axi_size       = RegEnable(io.param.axi_size, 0.U, io.param.valid)
  val axi_addr       = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_en         = RegInit(false.B)
  val axi_areq       = RegInit(false.B)
  val param_areq     = RegInit(false.B)
  val en             = io.cfg_gemm.en & io.cfg_gemm.layout_en
  val burst_num      = RegInit(0.U(6.W))
  val pad_zero_cnt   = RegInit(0.U(10.W))
  val pad_zero_valid = RegInit(false.B)
  val ifm_w_busy     = RegInit(false.B)

  //cal ifmbuffer addr
  val iwh_base       = RegEnable(io.param.iwh_base, 0.U, io.param.valid)
  val ic_base        = RegEnable(io.param.ic_base, 0.U, io.param.valid)
  val ic_align32     = RegEnable(align(io.cfg_gemm.ic, 12, 32), io.cfg_gemm.en)
  val ifm_waddr_step = ic_align32(11, 5)
  val ifm_waddr_base = RegInit(0.U((log2Ceil(IFM_BUFFER_DEPTH) + 1).W))

  io.axi.areq.axiEn   := axi_en
  io.axi.areq.axiAreq := axi_areq
  io.axi.areq.axiSize := axi_size
  io.axi.areq.axiAddr := axi_addr
  io.param_areq       := param_areq

  val sIdle :: sAreqParam :: sArbit :: sCfgaxi :: sWait :: sPadZero :: Nil = Enum(6)

  val state = RegInit(sIdle)
  io.fp32_cell_idle := state === sIdle & !ifm_w_busy

  switch(state) {
    is(sIdle) {  //state0
      axi_en        := false.B
      axi_areq      := false.B
      burst_num     := 0.U
      intr_num_cnt := 0.U
      state         := Mux(en & !io.param_empty, sAreqParam, sIdle)
    }
    is(sAreqParam) {  //state1
      when(ifm_w_busy) {
        state := sAreqParam
      }.otherwise {
        param_areq    := true.B
        intr_num_cnt := 0.U
        when(io.param_empty) {
          param_areq := false.B
          state      := sIdle
        }.elsewhen(io.param.valid) {
          param_areq := false.B
          state      := sArbit
        }.otherwise {
          state := sAreqParam
        }
      }
    }
    is(sArbit) {  //state2
      axi_en := true.B
      when(axi_en & io.axi.id === RID("gemm_ifm").U) {
        state    := sCfgaxi
      }.otherwise {
        state := sArbit
      }
      ifm_waddr_base := (iwh_base * ifm_waddr_step)(ifm_waddr_base.getWidth - 1, 0) + ic_base(11, 5)
      burst_num      := 32.U - ic_pad
    }
    is(sCfgaxi) {  //state3
      when(intr_num_cnt === 0.U) {
        axi_addr := axi_addr_base
      }.elsewhen(!io.axi.outstanding.outstanding_almost_full) {
        axi_addr := axi_addr + Cat(io.cfg_gemm.icstep, 0.U(2.W))
      }
      when(intr_num_cnt === burst_num){
        intr_num_cnt := 0.U
        axi_areq := false.B
        state := sWait
      }.elsewhen(!io.axi.outstanding.outstanding_almost_full){
        intr_num_cnt := intr_num_cnt + 1.U
        axi_areq := true.B
      }.otherwise{
        axi_areq := false.B
      }
    }
    is(sWait) {  //state4
      when(dma_trans_finish){
        when(ic_pad === 0.U) {
          state := Mux(io.param_empty, sIdle, sAreqParam)
        }.otherwise {
          state := sPadZero
        }
      }.otherwise{
        state := sWait
      }
    }
    is(sPadZero) {
      when(pad_zero_cnt === Cat(ic_pad, 0.U(4.W))) {
        pad_zero_cnt   := 0.U
        pad_zero_valid := false.B
        state          := Mux(io.param_empty, sIdle, sAreqParam)
      }.otherwise {
        pad_zero_cnt   := pad_zero_cnt + 1.U
        pad_zero_valid := true.B
      }
    }
  }

  when(burst_num_cnt === burst_num && burst_num =/= 0.U){
    burst_num_cnt := 0.U
    dma_trans_finish := true.B
  }.otherwise{
    burst_num_cnt := Mux(io.axi.data.last, burst_num_cnt + 1.U, burst_num_cnt)
    dma_trans_finish := false.B
  }

  val quant = Module(new quant_unit)
  quant.io.i_data.bits  := io.axi.data.data
  quant.io.i_data.valid := io.axi.data.valid
  quant.io.i_scale      := io.cfg_gemm.quant_data
  quant.io.i_en         := io.cfg_gemm.layout_en
  val quant_data = quant.io.o_data

  val pad_zero_valid_t = ShiftRegister(pad_zero_valid, FP32_MUL_CYCLES + 5, 0.B, io.cfg_gemm.en)

  //write to ifm mem
  val layout_array_clr   = param_areq
  val layout_array       = Seq.fill(32)(SPRAM_WRAP(32, 16, "distribute"))
  val layout_array_waddr = RegInit(0.U(4.W))
  val layout_array_cnt   = RegInit(0.U(5.W))
  val load_done          = RegInit(false.B) //layout array load finish
  when(layout_array_clr) {
    layout_array_waddr := 0.U
  }.elsewhen(quant_data.valid | pad_zero_valid_t) {
    layout_array_waddr := layout_array_waddr + 1.U
  }
  when(layout_array_clr) {
    layout_array_cnt := 0.U
  }.elsewhen(layout_array_waddr === 15.U && (quant_data.valid | pad_zero_valid_t)) {
    layout_array_cnt := layout_array_cnt + 1.U
  }
  when(layout_array_clr) {
    load_done := false.B
  }.elsewhen(layout_array_waddr === 15.U && layout_array_cnt === 31.U) {
    load_done := true.B
  }

  val layout_array_raddr = RegInit(0.U(4.W))
  val layout_array_sel   = RegInit(0.U(2.W))
  for (i <- 0 until 32) {
    layout_array(i).en    := en
    layout_array(i).wr    := ~((quant_data.valid | pad_zero_valid_t) & layout_array_cnt === i.U)
    layout_array(i).wdata := Mux(pad_zero_valid_t, 0.U, Cat(quant_data.data))
    layout_array(i).addr  := Mux(layout_array(i).wr, layout_array_raddr, layout_array_waddr)
  }
  val store_done = layout_array_raddr === 15.U && layout_array_sel === 3.U //ifm buffer store finish
  ifm_w_busy := Mux(layout_array_waddr =/= 0.U, true.B, Mux(store_done, false.B, ifm_w_busy))
  val ifm_wen = RegInit(false.B)
  ifm_wen            := Mux(riseEdge(load_done), true.B, Mux(store_done, false.B, ifm_wen))
  layout_array_raddr := Mux(layout_array_clr, 0.U, Mux(ifm_wen && layout_array_sel === 3.U, layout_array_raddr + 1.U, layout_array_raddr))
  layout_array_sel   := Mux(layout_array_clr, 0.U, Mux(ifm_wen, layout_array_sel + 1.U, layout_array_sel))

  //ifm_mem write abrit
  val ifm_mem_waddr    = RegInit(0.U((log2Ceil(IFM_BUFFER_DEPTH) + 1).W))
  val ifm_waddr_offset = Wire(UInt((log2Ceil(IFM_BUFFER_DEPTH) + 1).W))

  ifm_waddr_offset := Cat(layout_array_raddr, layout_array_sel) * ifm_waddr_step
  ifm_mem_waddr    := ifm_waddr_base + ifm_waddr_offset

  val ifm_wen_t = RegNext(ifm_wen,0.B) && !ifm_mem_waddr(log2Ceil(IFM_BUFFER_DEPTH))

  io.ifm_w.wen   := RegEnable(ifm_wen_t, 0.U, io.cfg_gemm.en)
  io.ifm_w.waddr := RegEnable(ifm_mem_waddr(log2Ceil(IFM_BUFFER_DEPTH) - 1, 0), io.cfg_gemm.en)
  val layout_array_sel_t = RegEnable(layout_array_sel, io.cfg_gemm.en)
  val ifm_wdata          = Wire(UInt(IFM_BUFFER_WIDTH.W))
  ifm_wdata := 0.U
  for (j <- 0 until 4) {
    when(layout_array_sel_t === j.U) {
      ifm_wdata := (for (i <- 0 until 32) yield { layout_array(i).rdata(8 * j + 7, 8 * j) }).reverse.reduce(Cat(_, _))
      //ifm_mem: high bit cached high channel data
    }
  }
  io.ifm_w.wdata := RegEnable(ifm_wdata, io.cfg_gemm.en)
}

class IfmBuffer extends Module with axi_config with buffer_config {
  val io = IO(new Bundle() {
    //cfg
    val cfg_gemm = Input(new cfg_gemm_io)
    //axi
    val axi_ch0 = new axi_r
    val axi_ch1 = new axi_r
    //ifm
    val ifm_r_ch0 = new ifm_r_io(IFM_BUFFER_DEPTH, IFM_BUFFER_WIDTH)
    val ifm_r_ch1 = new ifm_r_io(IFM_BUFFER_DEPTH, IFM_BUFFER_WIDTH)
    val task_done = Output(Bool())
  })

  val cfg_gemm = RegEnable(io.cfg_gemm, 0.U.asTypeOf(new cfg_gemm_io),dualEdge(io.cfg_gemm.en))
  val ifm_mem  = DPRAM_IFM(IFM_BUFFER_WIDTH, IFM_BUFFER_DEPTH, "ultra")

  val param_cal      = Module(new ifm_param_cal)
  val param_arbit    = Module(new ifm_param_arbit)
  val int8_ifm_cell0 = Module(new int8_ifm_cell)
  val int8_ifm_cell1 = Module(new int8_ifm_cell)
  val fp32_ifm_cell0 = Module(new fp32_ifm_cell)
  val fp32_ifm_cell1 = Module(new fp32_ifm_cell)

  param_cal.io.cfg_gemm           <> cfg_gemm
  param_cal.io.param_fifo_o       <> param_arbit.io.param_fifo_i
  param_arbit.io.param_fifo_empty <> param_cal.io.param_fifo_empty

  int8_ifm_cell0.io.cfg_gemm    <> cfg_gemm
  int8_ifm_cell0.io.param_empty <> param_cal.io.param_fifo_empty
  int8_ifm_cell1.io.cfg_gemm    <> cfg_gemm
  int8_ifm_cell1.io.param_empty <> param_cal.io.param_fifo_empty
  fp32_ifm_cell0.io.cfg_gemm    <> cfg_gemm
  fp32_ifm_cell0.io.param_empty <> param_cal.io.param_fifo_empty
  fp32_ifm_cell1.io.cfg_gemm    <> cfg_gemm
  fp32_ifm_cell1.io.param_empty <> param_cal.io.param_fifo_empty

  val isFp32Mat = cfg_gemm.layout_en && cfg_gemm.en

  io.task_done := RegNext(
    cfg_gemm.en & param_cal.io.param_cal_finish & param_cal.io.param_fifo_empty & ((int8_ifm_cell0.io.int8_cell_idle & int8_ifm_cell1.io.int8_cell_idle & !isFp32Mat) |
      (fp32_ifm_cell0.io.fp32_cell_idle & fp32_ifm_cell1.io.fp32_cell_idle & isFp32Mat))
    ,0.U
  )

  when(isFp32Mat) {
    param_arbit.io.param_areq_ch0 <> fp32_ifm_cell0.io.param_areq
    param_arbit.io.param_areq_ch1 <> fp32_ifm_cell1.io.param_areq
    int8_ifm_cell0.io.param       <> 0.U.asTypeOf(new ifm_param_io)
    int8_ifm_cell1.io.param       <> 0.U.asTypeOf(new ifm_param_io)
    fp32_ifm_cell0.io.param       <> param_arbit.io.param_ch0
    fp32_ifm_cell1.io.param       <> param_arbit.io.param_ch1
  }.otherwise {
    param_arbit.io.param_areq_ch0 <> int8_ifm_cell0.io.param_areq
    param_arbit.io.param_areq_ch1 <> int8_ifm_cell1.io.param_areq
    int8_ifm_cell0.io.param       <> param_arbit.io.param_ch0
    int8_ifm_cell1.io.param       <> param_arbit.io.param_ch1
    fp32_ifm_cell0.io.param       <> 0.U.asTypeOf(new ifm_param_io)
    fp32_ifm_cell1.io.param       <> 0.U.asTypeOf(new ifm_param_io)
  }

  int8_ifm_cell0.io.axi <> io.axi_ch0
  int8_ifm_cell1.io.axi <> io.axi_ch1
  fp32_ifm_cell0.io.axi <> io.axi_ch0
  fp32_ifm_cell1.io.axi <> io.axi_ch1
  when(isFp32Mat) {
    io.axi_ch0.areq                  <> fp32_ifm_cell0.io.axi.areq
    io.axi_ch1.areq                  <> fp32_ifm_cell1.io.axi.areq
    fp32_ifm_cell0.io.axi.data       <> io.axi_ch0.data
    fp32_ifm_cell1.io.axi.data       <> io.axi_ch1.data
    int8_ifm_cell0.io.axi.data.data  <> 0.U
    int8_ifm_cell0.io.axi.data.valid <> 0.U
    int8_ifm_cell1.io.axi.data.data  <> 0.U
    int8_ifm_cell1.io.axi.data.valid <> 0.U
  }.otherwise {
    io.axi_ch0.areq                  <> int8_ifm_cell0.io.axi.areq
    io.axi_ch1.areq                  <> int8_ifm_cell1.io.axi.areq
    int8_ifm_cell0.io.axi.data       <> io.axi_ch0.data
    int8_ifm_cell1.io.axi.data       <> io.axi_ch1.data
    fp32_ifm_cell0.io.axi.data.data  <> 0.U
    fp32_ifm_cell0.io.axi.data.valid <> 0.U
    fp32_ifm_cell1.io.axi.data.data  <> 0.U
    fp32_ifm_cell1.io.axi.data.valid <> 0.U
  }

  val ifm_w_ch0 = Mux(isFp32Mat, fp32_ifm_cell0.io.ifm_w, int8_ifm_cell0.io.ifm_w)
  val ifm_w_ch1 = Mux(isFp32Mat, fp32_ifm_cell1.io.ifm_w, int8_ifm_cell1.io.ifm_w)

  //ifm_mem connect
  ifm_mem.en_a       := cfg_gemm.en
  ifm_mem.en_b       := cfg_gemm.en
  ifm_mem.wr_a       := ~ifm_w_ch0.wen
  ifm_mem.wr_b       := ~ifm_w_ch1.wen
  ifm_mem.addr_a     := Mux(ifm_mem.wr_a, io.ifm_r_ch0.raddr, ifm_w_ch0.waddr)
  ifm_mem.addr_b     := Mux(ifm_mem.wr_b, io.ifm_r_ch1.raddr, ifm_w_ch1.waddr)
  io.ifm_r_ch0.rdata := ifm_mem.rdata_a
  io.ifm_r_ch1.rdata := ifm_mem.rdata_b
  ifm_mem.wdata_a    := ifm_w_ch0.wdata
  ifm_mem.wdata_b    := ifm_w_ch1.wdata

}
