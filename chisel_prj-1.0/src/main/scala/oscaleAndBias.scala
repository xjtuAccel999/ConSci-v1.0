import chisel3._
import chisel3.util._
import util_function._

class mem_oscale_bias extends Bundle with buffer_config {
  val oscale_addr = Input(UInt(log2Ceil(OSCALE_BIAS_BUFFER_DEPTH * 4).W))
  val bias_addr   = Input(UInt(log2Ceil(OSCALE_BIAS_BUFFER_DEPTH * 4).W))
  val oscale_data = Output(UInt(32.W))
  val bias_data   = Output(UInt(32.W))
}

class oscaleAndBiasMem extends Module with buffer_config with axi_config {
  val io = IO(new Bundle() {
    val cfg   = Input(new cfg_gemm_io)
    val axi   = new axi_r
    val mem_r = new mem_oscale_bias
    val ready = Output(Bool())
  })

  val axi_data_t  = RegEnable(io.axi.data.data,io.axi.data.valid)
  val axi_valid_t = RegNext(io.axi.data.valid,0.B)
  val axi_last_t = RegNext(io.axi.data.last,0.B)

  val en = RegInit(false.B)
  when(io.ready) {
    en := false.B
  }.elsewhen(io.cfg.en) {
    en := true.B
  }.otherwise {
    en := false.B
  }
  val c_align           = RegEnable(align(io.cfg.oc, 12, 4), 0.U, en)
//  val axi_busy_falledge = fallEdge(axi_busy_t)

  val sIdle :: sArbit :: sSetOscale :: sWaitOscale :: sSetBias :: sWaitBias :: sFinish :: Nil = Enum(7)

  val state            = RegInit(sIdle)
  val oscale_mem_waddr = RegInit(0.U(log2Ceil(OSCALE_BIAS_BUFFER_DEPTH).W))
  val bias_mem_waddr   = RegInit(0.U(log2Ceil(OSCALE_BIAS_BUFFER_DEPTH).W))

  val axi_en   = RegInit(false.B)
  val axi_areq = RegInit(false.B)
  val axi_addr = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_size = RegInit(0.U(32.W))
  io.axi.areq.axiEn   := axi_en
  io.axi.areq.axiAreq := axi_areq
  io.axi.areq.axiAddr := axi_addr
  io.axi.areq.axiSize := axi_size

  val oscale_mem = SPRAM_WRAP(128, OSCALE_BIAS_BUFFER_DEPTH, "block")
  val bias_mem   = SPRAM_WRAP(128, OSCALE_BIAS_BUFFER_DEPTH, "block")

  io.ready      := false.B
  oscale_mem.wr := true.B
  bias_mem.wr   := true.B
  switch(state) {
    is(sIdle) {
      axi_en           := false.B
      io.ready         := false.B
      oscale_mem_waddr := 0.U
      bias_mem_waddr   := 0.U
      when((io.cfg.bias_en | io.cfg.oscale_en) & en) {
        state := sArbit
      }.elsewhen(io.cfg.bias_en === false.B & io.cfg.oscale_en === false.B & en) {
        state := sFinish
      }.otherwise {
        state := sIdle
      }
    }
    is(sArbit) { //generate axi req
      axi_en := true.B
      when(axi_en & io.axi.id === RID("opfusion").U) {
        state := Mux(io.cfg.oscale_en, sSetOscale, sSetBias)
      }
    }
    is(sSetOscale) {
      axi_size := c_align(11, 2)
      axi_addr := io.cfg.dequant_addr
      axi_areq := true.B
      state    := sWaitOscale
    }
    is(sWaitOscale) {
      axi_areq := false.B
      when(axi_last_t) {
        state := Mux(io.cfg.bias_en, sSetBias, sFinish)
      }
      oscale_mem_waddr := Mux(axi_valid_t, oscale_mem_waddr + 1.U, oscale_mem_waddr)
      oscale_mem.wr    := !axi_valid_t
    }
    is(sSetBias) {
      axi_size := c_align(11, 2)
      axi_addr := io.cfg.bias_addr
      axi_areq := true.B
      state    := sWaitBias
    }
    is(sWaitBias) {
      axi_areq := false.B
      when(axi_last_t) {
        state := sFinish
      }
      bias_mem_waddr := Mux(axi_valid_t, bias_mem_waddr + 1.U, bias_mem_waddr)
      bias_mem.wr    := !axi_valid_t
    }
    is(sFinish) {
      oscale_mem_waddr := 0.U
      bias_mem_waddr   := 0.U
      axi_en           := false.B
      io.ready         := true.B
      when(io.cfg.en) {
        state := sFinish
      }.otherwise {
        io.ready := false.B
        state    := sIdle
      }
    }
  }

  val c_remain = c_align - io.cfg.oc

  oscale_mem.en := io.cfg.oscale_en
  when(oscale_mem_waddr === axi_size - 1.U) {
    oscale_mem.wdata := axi_data_t
    switch(c_remain) {
      is(1.U) {
        oscale_mem.wdata := Cat(0.U(32.U), axi_data_t(95, 0))
      }
      is(2.U) {
        oscale_mem.wdata := Cat(0.U(64.U), axi_data_t(63, 0))
      }
      is(3.U) {
        oscale_mem.wdata := Cat(0.U(96.U), axi_data_t(31, 0))
      }
    }
  }.otherwise {
    oscale_mem.wdata := axi_data_t
  }

  bias_mem.en := io.cfg.bias_en
  when(bias_mem_waddr === axi_size - 1.U) {
    bias_mem.wdata := axi_data_t
    switch(c_remain) {
      is(1.U) {
        bias_mem.wdata := Cat(0.U(32.U), axi_data_t(95, 0))
      }
      is(2.U) {
        bias_mem.wdata := Cat(0.U(64.U), axi_data_t(63, 0))
      }
      is(3.U) {
        bias_mem.wdata := Cat(0.U(96.U), axi_data_t(31, 0))
      }
    }
  }.otherwise {
    bias_mem.wdata := axi_data_t
  }

  oscale_mem.addr := Mux(oscale_mem.wr, io.mem_r.oscale_addr(log2Ceil(OSCALE_BIAS_BUFFER_DEPTH) + 1, 2), oscale_mem_waddr)
  bias_mem.addr   := Mux(bias_mem.wr, io.mem_r.bias_addr(log2Ceil(OSCALE_BIAS_BUFFER_DEPTH) + 1, 2), bias_mem_waddr)

  io.mem_r.oscale_data := 0.U
  io.mem_r.bias_data   := 0.U
  val oscale_sel = RegInit(0.U(2.W))
  val bias_sel   = RegInit(0.U(2.W))
  oscale_sel := io.mem_r.oscale_addr(1, 0)
  bias_sel   := io.mem_r.bias_addr(1, 0)
  for (i <- 0 until 4) {
    when(oscale_sel === i.U) {
      io.mem_r.oscale_data := oscale_mem.rdata(32 * i + 31, 32 * i)
    }
    when(bias_sel === i.U) {
      io.mem_r.bias_data := bias_mem.rdata(32 * i + 31, 32 * i)
    }
  }
}

class oscaleAndBiasDataCal extends Module with cal_cell_params {
  val io = IO(new Bundle() {
    val i_en        = Input(Bool())
    val i_data      = Input(UInt(32.W))
    val i_oscale    = Input(UInt(32.W))
    val i_oscale_en = Input(Bool())
    val i_bias      = Input(UInt(32.W))
    val i_bias_en   = Input(Bool())
    val o_data      = Output(UInt(32.W))
  })
  val convert_data = Float.SIntToFloat(io.i_data.asSInt, 32, io.i_en)
  val convert_data_t = ShiftRegister(Float(io.i_data), FP32_MUL_CYCLES + SINT_TO_FLOAT_CYCYLES, io.i_en)
  val oscale_data   = Mux(io.i_oscale_en, Float.FloatMul(convert_data, Float(io.i_oscale), io.i_en), convert_data_t)
  val oscale_data_t = ShiftRegister(oscale_data, FP32_ADD_CYCLES, io.i_en)
  val bias_data     = Mux(io.i_bias_en, Float.FloatAdd(Float(io.i_bias), oscale_data, io.i_en), oscale_data_t)
  io.o_data := bias_data.bits
}

class oscaleAndBias extends Module with buffer_config with cal_cell_params {
  val io = IO(new Bundle() {
    val cfg_gemm     = Input(new cfg_gemm_io)
    val axi          = new axi_r
    val ofm_data_ch0 = Input(new data_gp(4, 32))
    val ofm_data_ch1 = Input(new data_gp(4, 32))
    val bias_ch0     = Output(new data_gp(4, 32))
    val bias_ch1     = Output(new data_gp(4, 32))
    val ready        = Output(Bool())
  })

  val cfg_gemm = RegEnable(io.cfg_gemm, 0.U.asTypeOf(new cfg_gemm_io),dualEdge(io.cfg_gemm.en))

  val ofm_data_ch0_t = RegEnable(io.ofm_data_ch0, io.cfg_gemm.en)
  val ofm_data_ch1_t = RegEnable(io.ofm_data_ch1, io.cfg_gemm.en)

  val mem = Module(new oscaleAndBiasMem)
  mem.io.cfg <> cfg_gemm
  mem.io.axi <> io.axi
  io.ready   := mem.io.ready

  val data_cal0 = Seq.fill(4)(Module(new oscaleAndBiasDataCal))
  val data_cal1 = Seq.fill(4)(Module(new oscaleAndBiasDataCal))
  val bias_data = Mux(RegEnable(mem.io.mem_r.bias_addr, cfg_gemm.en) < cfg_gemm.oc, mem.io.mem_r.bias_data, 0.U)

  for (i <- 0 until 4) {
    data_cal0(i).io.i_oscale_en := cfg_gemm.oscale_en
    data_cal0(i).io.i_bias_en   := cfg_gemm.bias_en
    data_cal0(i).io.i_oscale    := mem.io.mem_r.oscale_data
    data_cal0(i).io.i_bias      := bias_data
    data_cal0(i).io.i_data      := ofm_data_ch0_t.data(i)
    data_cal0(i).io.i_en        := cfg_gemm.en

    data_cal1(i).io.i_oscale_en := cfg_gemm.oscale_en
    data_cal1(i).io.i_bias_en   := cfg_gemm.bias_en
    data_cal1(i).io.i_oscale    := mem.io.mem_r.oscale_data
    data_cal1(i).io.i_bias      := bias_data
    data_cal1(i).io.i_data      := ofm_data_ch1_t.data(i)
    data_cal1(i).io.i_en        := cfg_gemm.en

    io.bias_ch0.data(i) := data_cal0(i).io.o_data
    io.bias_ch1.data(i) := data_cal1(i).io.o_data
  }

  val bias_valid_t2 = ShiftRegister(ofm_data_ch0_t.valid, FP32_MUL_CYCLES + FP32_ADD_CYCLES + SINT_TO_FLOAT_CYCYLES, 0.U, cfg_gemm.en)
  io.bias_ch0.valid := bias_valid_t2
  io.bias_ch1.valid := bias_valid_t2

  val en                = cfg_gemm.oscale_en | cfg_gemm.bias_en
  val owh               = RegEnable(cfg_gemm.ow * cfg_gemm.oh, 0.U, en)
  val owh_align64_div64 = RegEnable(align(owh, 24, 64), 0.U, ShiftRegister(en, 2))(23, 6)

  val cnt       = RegInit(0.U(8.W))
  val owh_index = RegInit(0.U(18.W))
  val oc_index  = RegInit(0.U(6.W))
  when(!en) {
    cnt       := 0.U
    owh_index := 0.U
    oc_index  := 0.U
  }.elsewhen(ofm_data_ch0_t.valid) {
    cnt := cnt + 1.U
    when(cnt === 255.U) {
      when(owh_index === owh_align64_div64 - 1.U) {
        oc_index  := oc_index + 1.U
        owh_index := 0.U
      }.otherwise {
        owh_index := owh_index + 1.U
      }
    }
  }

  mem.io.mem_r.oscale_addr := ShiftRegister(Cat(oc_index(4, 0), cnt(7, 3)), SINT_TO_FLOAT_CYCYLES - 1, cfg_gemm.oscale_en)
  when(en) {
    mem.io.mem_r.bias_addr := ShiftRegister(mem.io.mem_r.oscale_addr, FP32_MUL_CYCLES, cfg_gemm.bias_en)
  }.otherwise {
    mem.io.mem_r.bias_addr := 0.U
  }
}
