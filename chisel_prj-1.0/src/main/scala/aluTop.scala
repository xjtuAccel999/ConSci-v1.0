import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util._
import util_function._

class data_gp(group: Int, width: Int) extends Bundle {
  val data  = Vec(group, UInt(width.W))
  val valid = Bool()
}

class aluCell(alu_id: Int) extends Module with alu_config {
  val io = IO(new Bundle() {
    //input
    val cfg       = Input(new cfg_alu_io)
    val axi_recv  = new axi_r
    val bias_data = Input(new data_gp(4, 32))
    val congested = Input(Bool())
    //output
    val task_done = Output(Bool())
    val axi_send  = Flipped(new data_axiSend)
    val act_odata = Output(new data_gp(4, 32))
    //innerproduct
    val innerprod_o          = Valid(UInt(32.W))
    val axi_lite_innerprod_i = if (alu_id == 0) Some(Flipped(Valid(UInt(32.W)))) else None
    val axi_lite_innerprod_o = if (alu_id == 0) Some(Valid(UInt(32.W))) else None
  })
  val axi_recv_t_data  = RegEnable(io.axi_recv.data.data,io.axi_recv.data.valid)
  val axi_recv_t_valid = RegNext(io.axi_recv.data.valid,0.B)
  val src_mem          = SPRAM_WRAP(AXI_DATA_WIDTH, ALU_SRC_DEPTH, "block")
  src_mem.en    := io.cfg.math_en && io.cfg.math_src_num(1)
  src_mem.wdata := axi_recv_t_data
  src_mem.wr    := 1.B
  src_mem.addr  := 0.U

  val sIdle :: sWriteSramCfg :: sWaitSram :: saxiCfg :: sWaitaxi :: sIsFinish :: sWaitFinish :: Nil = Enum(7)

  val state = RegInit(sIdle)

  //innerprodFunc
  val innerprodModule = Module(new innerprodFunc(alu_id))
  innerprodModule.io.cfg <> io.cfg
  io.innerprod_o         <> 0.U.asTypeOf(new ValidIO(UInt(32.W)))
  val innerprod_valid_i = RegNext(axi_recv_t_valid && src_mem.en && src_mem.wr,0.B)
  when(io.cfg.innerprod_en) {
    io.innerprod_o.bits              <> innerprodModule.io.o_data.bits.asUInt
    io.innerprod_o.valid             <> innerprodModule.io.o_data.valid
    innerprodModule.io.i_data0.bits  <> RegEnable(axi_recv_t_data, 0.U, io.cfg.innerprod_en)
    innerprodModule.io.i_data0.valid <> innerprod_valid_i
    innerprodModule.io.i_data1.bits  <> src_mem.rdata
    innerprodModule.io.i_data1.valid <> innerprod_valid_i
  }.otherwise {
    innerprodModule.io.i_data0 <> 0.U.asTypeOf(new ValidIO(UInt(32.W)))
    innerprodModule.io.i_data1 <> 0.U.asTypeOf(new ValidIO(UInt(32.W)))
  }

  //mathFunc
  val mathFuncModule = Module(new mathFunc(alu_id))
  mathFuncModule.io.cfg := io.cfg
  val mathFunc_valid = RegNext(axi_recv_t_valid & state === sWaitaxi & io.cfg.math_en,0.U)
  when(io.cfg.innerprod_en) {
    for (i <- 1 until 4) {
      mathFuncModule.io.i_data0(i) := 0.U.asTypeOf(new ValidIO(UInt(32.W)))
      mathFuncModule.io.i_data1(i) := 0.U.asTypeOf(new ValidIO(UInt(32.W)))
    }
    mathFuncModule.io.i_data1.head := 0.U.asTypeOf(new ValidIO(UInt(32.W)))
    mathFuncModule.io.i_data0.head := (if (alu_id == 0) io.axi_lite_innerprod_i.get else 0.U.asTypeOf(new ValidIO(UInt(32.W))))
  }.otherwise {
    for (i <- 0 until 4) {
      mathFuncModule.io.i_data0(i).valid := mathFunc_valid
      mathFuncModule.io.i_data0(i).bits  := RegEnable(axi_recv_t_data(i * 32 + 31, i * 32), 0.U, io.cfg.math_en)
      mathFuncModule.io.i_data1(i).valid := mathFunc_valid
      mathFuncModule.io.i_data1(i).bits  := src_mem.rdata(i * 32 + 31, i * 32)
    }
  }

  //actFunc
  val actFuncModule = Module(new activationFunc)
  actFuncModule.io.cfg := io.cfg
  when(io.cfg.innerprod_en) { //from innerprod
    actFuncModule.io.i_data.valid     := RegNext(mathFuncModule.io.o_data.head.valid,0.U)
    actFuncModule.io.i_data.data.head := RegEnable(mathFuncModule.io.o_data.head.bits, io.cfg.act_en)
    for (i <- 1 until 4) {
      actFuncModule.io.i_data.data(i) := 0.U
    }
  }.otherwise {
    when(io.cfg.act_src_sel(0)) { //from axi
      actFuncModule.io.i_data.valid := RegNext(axi_recv_t_valid & io.cfg.act_en,0.U)
      for (i <- 0 until 4) {
        actFuncModule.io.i_data.data(i) := RegEnable(axi_recv_t_data(i * 32 + 31, i * 32), io.cfg.act_en)
      }
    }.otherwise { //from oscaleAndBias
      actFuncModule.io.i_data.data := RegEnable(io.bias_data.data, io.cfg.act_en)
      actFuncModule.io.i_data.valid := RegEnable(io.bias_data.valid, 0.U, io.cfg.act_en)
    }
  }

  //to opfusion
  io.act_odata := Mux(io.cfg.act_dst_sel(1) & !io.cfg.innerprod_en, actFuncModule.io.o_data, 0.U.asTypeOf(new data_gp(4, 32)))

  //to axiSend
  when(io.cfg.innerprod_en) {
    io.axi_send.data_valid := false.B
    io.axi_send.data       := 0.U
  }.otherwise {
    when(io.cfg.act_dst_sel(0)) {
      io.axi_send.data_valid := actFuncModule.io.o_data.valid
      io.axi_send.data := (for (i <- 0 until 4) yield {
        actFuncModule.io.o_data.data(i)
      }).reverse.reduce(Cat(_, _))
    }.otherwise {
      io.axi_send.data_valid := mathFuncModule.io.o_data(0).valid
      io.axi_send.data := (for (i <- 0 until 4) yield {
        mathFuncModule.io.o_data(i).bits
      }).reverse.reduce(Cat(_, _))
    }
  }

  //to axi lite
  if (alu_id == 0) {
    when(io.cfg.innerprod_en) {
      io.axi_lite_innerprod_o.get.valid <> actFuncModule.io.o_data.valid
      io.axi_lite_innerprod_o.get.bits  <> actFuncModule.io.o_data.data.head
    }.otherwise {
      io.axi_lite_innerprod_o.get <> 0.U.asTypeOf(new ValidIO(UInt(32.W)))
    }
  }

  io.axi_recv.areq.axiEn := io.cfg.math_en | (io.cfg.act_en & io.cfg.act_src_sel(0)) | io.cfg.innerprod_en
//  val axi_rbusy_falledge = fallEdge(io.axi_recv.busy)
  val axi_alu            = io.axi_recv.areq.axiEn & io.axi_recv.id === RID("alu").U
//  val trans_finish       = RegNext(axi_rbusy_falledge & axi_alu)
  val trans_finish       = RegNext(io.axi_recv.data.last & axi_alu,0.B)

  val sram_waddr      = RegInit(0.U((log2Ceil(ALU_SRC_DEPTH) + 1).W))
  val sram_raddr      = RegInit(0.U(log2Ceil(ALU_SRC_DEPTH).W))
  val axi_src0_addr   = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_src1_addr   = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_dst_addr    = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val remain_len      = RegInit(0.U((AXI_SIZE_WIDTH - 2).W))
  val remain_len_flag = remain_len > ALU_SRC_DEPTH.U

  val task_done = RegInit(false.B)
  io.task_done := task_done
  val axi_areq = RegInit(false.B)
  val axi_addr = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val axi_size = RegInit(0.U(32.W))
  io.axi_recv.areq.axiAreq := axi_areq
  io.axi_recv.areq.axiAddr := axi_addr
  io.axi_recv.areq.axiSize := axi_size

  io.axi_send.addr_valid := false.B
  io.axi_send.size_valid := false.B
  io.axi_send.addr       := 0.U
  io.axi_send.size       := 0.U

  val delay_cnt   = RegInit(0.U(5.W))
  val delay_value = RegInit(0.U(5.W))
  delay_value := Mux(io.cfg.act_en, ACTIVATION_FUNC_CYCLES.U, (FP32_ADD_CYCLES + FP32_MUL_CYCLES).U)

  //three phase state machine
  switch(state) { //situation: use axi
    is(sIdle) {
      if (alu_id == 0) {
        axi_src0_addr := io.cfg.src0_addr_ch0
        axi_src1_addr := io.cfg.src1_addr_ch0
        axi_dst_addr  := io.cfg.dst_addr_ch0
        remain_len    := io.cfg.veclen_ch0(31, 2)
      } else {
        axi_src0_addr := io.cfg.src0_addr_ch1
        axi_src1_addr := io.cfg.src1_addr_ch1
        axi_dst_addr  := io.cfg.dst_addr_ch1
        remain_len    := io.cfg.veclen_ch1(31, 2)
      }
      task_done := false.B
      axi_size  := 0.U
      axi_addr  := 0.U
      axi_areq  := 0.U
      delay_cnt := 0.U
//      when(axi_alu & !io.axi_recv.busy) {
      when(axi_alu) {
        when(io.cfg.math_en & io.cfg.math_src_num(0) | io.cfg.act_en & io.cfg.act_src_sel(0)) { //single src num
          state := saxiCfg
        }.elsewhen(io.cfg.math_en & io.cfg.math_src_num(1)) { //double src num
          state := sWriteSramCfg
        }.otherwise {
          state := sIdle
        }
      }
    }

    is(sWriteSramCfg) {
      axi_addr  := axi_src1_addr
      axi_size  := Mux(remain_len_flag, ALU_SRC_DEPTH.U, remain_len)
      axi_areq  := true.B
      state     := sWaitSram
      delay_cnt := 0.U
    }

    is(sWaitSram) {
      axi_areq := false.B
      src_mem.wr   := sram_waddr(log2Ceil(ALU_SRC_DEPTH)) //write
      src_mem.addr := sram_waddr(log2Ceil(ALU_SRC_DEPTH) - 1, 0)
      sram_waddr   := Mux(axi_recv_t_valid, sram_waddr + 1.U, sram_waddr)
      state        := Mux(trans_finish, saxiCfg, sWaitSram)
    }

    is(saxiCfg) {
      //set axi read
      axi_addr := axi_src0_addr
      axi_size := Mux(remain_len_flag, ALU_SRC_DEPTH.U, remain_len)
      when(io.congested) {
        state    := saxiCfg
        axi_areq := false.B
      }.otherwise {
        state    := sWaitaxi
        axi_areq := true.B
      }
      delay_cnt := 0.U
    }

    is(sWaitaxi) {
      axi_areq := false.B
      state := Mux(trans_finish, sIsFinish, sWaitaxi)
      //sram read address control
      when(io.cfg.math_en & io.cfg.math_src_num(1)) {
        src_mem.wr   := true.B
        src_mem.addr := sram_raddr
        sram_raddr   := Mux(axi_recv_t_valid, sram_raddr + 1.U, sram_raddr)
      }
    }

    is(sIsFinish) {
      sram_waddr := 0.U
      sram_raddr := 0.U
      delay_cnt  := delay_cnt + 1.U
      state      := sIsFinish
      when(delay_cnt === delay_value) {
        when(remain_len_flag) { //remain len > ALU_SRC_DEPTH
          //update axi read
          remain_len    := remain_len - ALU_SRC_DEPTH.U
          axi_src0_addr := (ALU_SRC_DEPTH * AXI_DATA_WIDTH / 8).U + axi_src0_addr
          axi_src1_addr := (ALU_SRC_DEPTH * AXI_DATA_WIDTH / 8).U + axi_src1_addr
          //update axi send buffer
          axi_dst_addr     := (ALU_SRC_DEPTH * AXI_DATA_WIDTH / 8).U + axi_dst_addr
          io.axi_send.size := ALU_SRC_DEPTH.U
          state            := Mux(io.cfg.math_en & io.cfg.math_src_num(1), sWriteSramCfg, saxiCfg)
        }.otherwise {
          remain_len       := 0.U
          io.axi_send.size := remain_len
          task_done        := true.B
          state            := sWaitFinish
        }
        io.axi_send.addr       := axi_dst_addr
        io.axi_send.addr_valid := !io.cfg.innerprod_en
        io.axi_send.size_valid := !io.cfg.innerprod_en
      }
    }

    is(sWaitFinish) {
      io.axi_send.size_valid := false.B
      io.axi_send.addr_valid := false.B
      sram_raddr             := 0.U
      sram_waddr             := 0.U
      delay_cnt              := 0.U
      when(io.cfg.math_en | io.cfg.act_en) {
        state := sWaitFinish
      }.otherwise {
        state     := sIdle
        task_done := false.B
      }
    }
  }
}

class aluTop extends Module with alu_config with cal_cell_params {
  val io = IO(new Bundle() {
    //ctrl registers
    val cfg = Input(new cfg_alu_io)
    //data stream
    val bias_ch0 = Input(new data_gp(4, 32))
    val bias_ch1 = Input(new data_gp(4, 32))
    val act_ch0  = Output(new data_gp(4, 32))
    val act_ch1  = Output(new data_gp(4, 32))
    //axi
    val axi_r_ch0              = new axi_r
    val axi_r_ch1              = new axi_r
    val axi_send_ch0           = Flipped(new data_axiSend)
    val axi_send_ch1           = Flipped(new data_axiSend)
    val axi_send_congested_ch0 = Input(Bool())
    val axi_send_congested_ch1 = Input(Bool())
    //axi lite innerproduct
    val axi_lite_innerprod_o = Output(UInt(32.W))
    //intr
    val task_done = Output(Bool())
  })

 val cfg = RegEnable(io.cfg, 0.U.asTypeOf(new cfg_alu_io),dualEdge(io.cfg.innerprod_en)|dualEdge(io.cfg.act_en)|dualEdge(io.cfg.math_en))
  // val cfg = RegNext(io.cfg)

  val alu_cell0 = Module(new aluCell(0)).io
  val alu_cell1 = Module(new aluCell(1)).io

  alu_cell0.cfg       <> cfg
  alu_cell0.bias_data <> io.bias_ch0
  alu_cell0.axi_recv  <> io.axi_r_ch0
  alu_cell0.axi_send  <> io.axi_send_ch0
  alu_cell0.act_odata <> io.act_ch0
  alu_cell0.congested <> io.axi_send_congested_ch0

  alu_cell1.cfg       <> cfg
  alu_cell1.bias_data <> io.bias_ch1
  alu_cell1.axi_recv  <> io.axi_r_ch1
  alu_cell1.axi_send  <> io.axi_send_ch1
  alu_cell1.act_odata <> io.act_ch1
  alu_cell1.congested <> io.axi_send_congested_ch1

  val innerprod_value0 = RegInit(0.S(32.W))
  val innerprod_value1 = RegInit(0.S(32.W))
  when(cfg.innerprod_en) {
    innerprod_value0 := Mux(alu_cell0.innerprod_o.valid, alu_cell0.innerprod_o.bits.asSInt, innerprod_value0)
    innerprod_value1 := Mux(alu_cell1.innerprod_o.valid, alu_cell1.innerprod_o.bits.asSInt, innerprod_value1)
  }.otherwise {
    innerprod_value0 := 0.S
    innerprod_value1 := 0.S
  }
  val innerprod_valid = ShiftRegister(alu_cell0.innerprod_o.valid & alu_cell1.innerprod_o.valid, 2, 0.U, 1.B)
  val innerprod_data  = RegInit(0.S(32.W))
  innerprod_data                           := innerprod_value0 + innerprod_value1
  alu_cell0.axi_lite_innerprod_i.get.valid <> ShiftRegister(innerprod_valid, SINT_TO_FLOAT_CYCYLES, 0.U, 1.B)
  alu_cell0.axi_lite_innerprod_i.get.bits  <> Float.SIntToFloat(innerprod_data, 32, cfg.innerprod_en).bits
  val axi_lite_innerprod_o = RegEnable(alu_cell0.axi_lite_innerprod_o.get.bits, 0.U, alu_cell0.axi_lite_innerprod_o.get.valid)
  io.axi_lite_innerprod_o <> axi_lite_innerprod_o

  val axi_lite_taskdone = RegInit(false.B)
  when(!cfg.innerprod_en) {
    axi_lite_taskdone := false.B
  }.elsewhen(alu_cell0.axi_lite_innerprod_o.get.valid) {
    axi_lite_taskdone := true.B
  }

  when(cfg.innerprod_en) {
    io.task_done <> axi_lite_taskdone
  }.otherwise {
    io.task_done := RegNext(alu_cell0.task_done & alu_cell1.task_done,0.U)
  }
}

object alu_gen extends App {
  new (chisel3.stage.ChiselStage)
    .execute(Array("--target-dir", "./verilog/alu"), Seq(ChiselGeneratorAnnotation(() => new aluTop)))
}
