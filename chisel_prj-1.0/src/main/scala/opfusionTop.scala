import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util._
import util_function._

class requantCell extends Module with cal_cell_params {
  val io = IO(new Bundle() {
    val cfg_gemm = Input(new cfg_gemm_io)
    val i_data   = Input(new data_gp(4, 32))
    val o_data   = Output(new data_gp(4, 8))
  })

  val i_data_t = RegEnable(io.i_data, io.cfg_gemm.requant_en)

  when(io.cfg_gemm.requant_en) {
    io.o_data.valid := ShiftRegister(io.i_data.valid, FP32_MUL_CYCLES + FP32_TO_SINT_CYCLES + 1, 0.U, 1.B)
    for (i <- 0 until 4) {
      val requant_data = Float.FloatMul(Float(io.cfg_gemm.requant_data), Float(i_data_t.data(i)), io.cfg_gemm.requant_en)
      io.o_data.data(i) := Float.FloatToSInt(requant_data, 8, io.cfg_gemm.requant_en).asUInt
    }
  }.otherwise {
    io.o_data.valid := false.B
    for (i <- 0 until 4) {
      io.o_data.data(i) := 0.U
    }
  }
}

class nchw_fp32_axi_ctrl(block_offset_bytes: Int) extends Module with axi_config {
  val io = IO(new Bundle() {
    //input
    val cfg_gemm = Input(new cfg_gemm_io)
    val i_data   = Flipped(Valid(UInt(128.W)))
    //output
    val axi       = Flipped(new data_axiSend)
    val task_done = Output(Bool())
  })

  val en       = RegNext(io.cfg_gemm.en & !io.cfg_gemm.requant_en,0.B)

  val start             = riseEdge(en)
  val owh               = RegEnable(io.cfg_gemm.ow * io.cfg_gemm.oh, 0.U, start)
  val owh_align64_div64 = align(owh, 24, 64)(23, 6)
  val owh_align_4       = align(owh, 24, 4)
  val oc_align32_div32  = RegEnable(align(io.cfg_gemm.oc, 12, 32)(11, 5), 0.U, start)

  val owh_offset = RegInit(0.U(3.W))
  val oc_offset  = RegInit(0.U(5.W))
  val owh_base   = RegInit(0.U(18.W))
  val oc_base    = RegInit(0.U(7.W))

  val task_done = RegInit(false.B)
  io.task_done := task_done

  when(en) {
    when(io.i_data.valid) {
      owh_offset := owh_offset + 1.U
      when(owh_offset === 7.U) {
        oc_offset := oc_offset + 1.U
        when(oc_offset === 31.U) {
          when(owh_base === owh_align64_div64 - 1.U) {
            owh_base := 0.U
            oc_base  := oc_base + 1.U
            when(oc_base === oc_align32_div32 - 1.U) {
              task_done := true.B
            }
          }.otherwise {
            owh_base := owh_base + 1.U
          }
        }
      }
    }
  }.otherwise {
    owh_base   := 0.U
    owh_offset := 0.U
    oc_base    := 0.U
    oc_offset  := 0.U
    task_done  := false.B
  }

  val owh_t = RegInit(0.U(24.W))
  val oc_t  = RegInit(0.U(12.W))
  owh_t := (Cat(owh_offset, 0.U(2.W)) + block_offset_bytes.U) + Cat(owh_base, 0.U(6.W))
  oc_t  := Cat(oc_base, 0.U(5.W)) + oc_offset

  io.axi.data_valid := RegNext(io.i_data.valid,0.B) & owh_t < owh_align_4 & oc_t < io.cfg_gemm.oc
  io.axi.data       := RegEnable(io.i_data.bits, 0.U, io.cfg_gemm.en)

  val data_oc_valid = RegNext(io.i_data.valid,0.B) & oc_t < io.cfg_gemm.oc

  val state0 = data_oc_valid & RegNext(owh_offset === 0.U,0.B)
  val state1 = RegNext(state0,0.B)
  val state2 = RegNext(state1,0.B)
  val state3 = RegNext(state2,0.B)

//  val oc_addr = RegEnable(oc_t * io.cstep, 0.U, state1)
  val oc_addr = RegInit(0.U(32.W))
  when(!en) {
    oc_addr := 0.U
  }.elsewhen(state1) {
    oc_addr := oc_t * io.cfg_gemm.ocstep
  }

  val owh_base_addr       = RegInit(0.U(24.W))
  val owh_base_addr_limit = RegInit(0.U(24.W))
  owh_base_addr       := Mux(state1, Cat(owh_base, 0.U(6.W)) + block_offset_bytes.U, owh_base_addr)
  owh_base_addr_limit := Mux(state2, Mux(owh_base_addr > owh_align_4, owh_align_4, owh_base_addr), owh_base_addr_limit)
  val owh_base_addr_t = RegEnable(Cat((oc_addr + owh_base_addr)(29, 0), 0.U(2.W)), 0.U, state2)
  io.axi.addr := RegEnable(owh_base_addr_t + io.cfg_gemm.ofm_addr, 0.U, state3)

  val len_remain = RegEnable((owh_align_4 - owh_base_addr_limit)(23, 2), 0.U, state3)
  io.axi.size := Mux(len_remain > 7.U, 8.U, len_remain)

  io.axi.addr_valid := data_oc_valid & RegNext(owh_offset === 7.U,0.U) & len_remain =/= 0.U
  io.axi.size_valid := io.axi.addr_valid
}

class nhwc_int8_axi_ctrl(block_offset_bytes: Int) extends Module with axi_config {
  val io = IO(new Bundle() {
    //input
    val cfg_gemm = Input(new cfg_gemm_io)
    val i_data   = Flipped(Valid(UInt(128.W)))
    //output
    val axi       = Flipped(new data_axiSend)
    val task_done = Output(Bool())
  })

  val en       = RegNext(io.cfg_gemm.en & io.cfg_gemm.requant_en,0.B)

  val whstep = io.cfg_gemm.ocstep(11, 0)

  val start    = riseEdge(en)
  val start_t1 = RegNext(start,0.B)
  val start_t2 = RegNext(start_t1,0.B)

  val oc_align16_div16    = RegEnable(align(io.cfg_gemm.oc, 12, 16)(11, 4), 0.U, start)
  val owh                 = RegEnable(io.cfg_gemm.ow * io.cfg_gemm.oh, 0.U, start)
  val owh_align64         = RegEnable(align(owh, 24, 64), 0.U, start_t1)
  val owh_align64_eq_base = RegEnable(owh_align64 - (64 - block_offset_bytes).U, 0.U, start_t2)

  val oc_offest  = RegInit(0.U(1.W))
  val oc_base    = RegInit(0.U(8.W))
  val owh_offset = RegInit(0.U(5.W))
  val owh_base   = RegInit(0.U(24.W))

  val task_done = RegInit(false.B)
  io.task_done := task_done

  when(en) {
    when(io.i_data.valid) {
      oc_offest := oc_offest + 1.U
      when(oc_offest === 1.U) {
        owh_offset := owh_offset + 1.U
        when(owh_offset === 31.U) {
          when(owh_base === owh_align64_eq_base) {
            oc_base  := oc_base + 2.U
            owh_base := block_offset_bytes.U
          }.otherwise {
            owh_base := owh_base + 64.U
          }
        }
      }
    }
  }.otherwise {
    owh_base   := block_offset_bytes.U
    owh_offset := 0.U
    oc_base    := 0.U
    oc_offest  := 0.U
  }
  when(!en) {
    task_done := false.B
  }.elsewhen(oc_base >= oc_align16_div16 && oc_base =/= 0.U) {
    task_done := true.B
  }

  val owh_t = RegInit(0.U(24.W))
  val oc_t  = RegInit(0.U(8.W))
  owh_t := owh_base + owh_offset
  oc_t  := oc_base + oc_offest

  io.axi.data_valid := RegNext(io.i_data.valid,0.B) & owh_t < owh
  io.axi.data       := RegEnable(io.i_data.bits, 0.U, io.cfg_gemm.en)
  io.axi.size       := 2.U
  io.axi.size_valid := RegNext(oc_offest === 1.U && io.axi.data_valid,0.B)

  val state        = RegNext(oc_offest === 0.U,0.B) & io.axi.data_valid
  val state1       = RegNext(state,0.B)
  val owh_baseaddr = RegInit(0.U(32.W))
  when(!en) {
    owh_baseaddr := 0.U
  }.elsewhen(state1) {
    owh_baseaddr := owh_t * whstep
  }
  val owh_addr = RegInit(0.U(32.W))
  when(!en) {
    owh_addr := 0.U
  }.otherwise {
    owh_addr := owh_baseaddr + Cat(ShiftRegister(oc_base, 2, 0.U, io.cfg_gemm.en), 0.U(4.W))
  }
  io.axi.addr       := RegEnable(owh_addr + io.cfg_gemm.ofm_addr, 0.U, io.cfg_gemm.en)
  io.axi.addr_valid := ShiftRegister(io.axi.size_valid, 3, 0.U, 1.B)
}

class opfusionCell(block_offset: Int) extends Module {
  val io = IO(new Bundle() {
    //input
    val cfg_gemm = Input(new cfg_gemm_io)
    val act_data = Input(new data_gp(4, 32))
    //output
    val axi_send       = Flipped(new data_axiSend)
    val gemm_task_done = Output(Bool())
  })

  val requantUnit = Module(new requantCell)
  requantUnit.io.cfg_gemm := io.cfg_gemm
  when(io.cfg_gemm.requant_en) {
    requantUnit.io.i_data := io.act_data
  }.otherwise {
    requantUnit.io.i_data := 0.U.asTypeOf(new data_gp(4, 32))
  }

  val transposeUnit = Module(new transpose)
  transposeUnit.io.cfg_gemm := io.cfg_gemm
  transposeUnit.io.i_data   := requantUnit.io.o_data

  val nchw_ctrl = Module(new nchw_fp32_axi_ctrl(block_offset))
  when(!io.cfg_gemm.requant_en) {
    nchw_ctrl.io.cfg_gemm     := io.cfg_gemm
    nchw_ctrl.io.i_data.valid := io.act_data.valid
    nchw_ctrl.io.i_data.bits := (for (i <- 0 until 4) yield {
      io.act_data.data(i)
    }).reverse.reduce(Cat(_, _))
  }.otherwise {
    nchw_ctrl.io.cfg_gemm     := 0.U.asTypeOf(new cfg_gemm_io)
    nchw_ctrl.io.i_data.valid := 0.U
    nchw_ctrl.io.i_data.bits  := 0.U
  }

  val nhwc_ctrl = Module(new nhwc_int8_axi_ctrl(block_offset))
  nhwc_ctrl.io.i_data := transposeUnit.io.o_data
  when(io.cfg_gemm.requant_en) {
    nhwc_ctrl.io.cfg_gemm := io.cfg_gemm
  }.otherwise {
    nhwc_ctrl.io.cfg_gemm := 0.U.asTypeOf(new cfg_gemm_io)
  }

  val nhwc_ctrl_en = RegNext(io.cfg_gemm.en & io.cfg_gemm.requant_en,0.B)
  val nchw_ctrl_en = RegNext(io.cfg_gemm.en & !io.cfg_gemm.requant_en,0.B)

  when(nhwc_ctrl_en) {
    io.gemm_task_done <> nhwc_ctrl.io.task_done
    io.axi_send       <> nhwc_ctrl.io.axi
  }.elsewhen(nchw_ctrl_en) {
    io.gemm_task_done <> nchw_ctrl.io.task_done
    io.axi_send       <> nchw_ctrl.io.axi
  }.otherwise {
    io.gemm_task_done <> 0.U
    io.axi_send       <> 0.U.asTypeOf(new data_axiSend)
  }
}

class opfusionTop extends Module {
  val io = IO(new Bundle() {
    //input
    val cfg_gemm     = Input(new cfg_gemm_io)
    val act_data_ch0 = Input(new data_gp(4, 32))
    val act_data_ch1 = Input(new data_gp(4, 32))
    //output
    val axi_send_ch0   = Flipped(new data_axiSend)
    val axi_send_ch1   = Flipped(new data_axiSend)
    val gemm_task_done = Output(Bool())
  })

  val cfg_gemm = RegEnable(io.cfg_gemm, 0.U.asTypeOf(new cfg_gemm_io),dualEdge(io.cfg_gemm.en))

  val cell0 = Module(new opfusionCell(0))
  val cell1 = Module(new opfusionCell(32))

  cell0.io.cfg_gemm := cfg_gemm
  cell1.io.cfg_gemm := cfg_gemm
  cell0.io.act_data <> io.act_data_ch0
  cell1.io.act_data <> io.act_data_ch1
  cell0.io.axi_send <> io.axi_send_ch0
  cell1.io.axi_send <> io.axi_send_ch1

  io.gemm_task_done := cell0.io.gemm_task_done & cell1.io.gemm_task_done
}

object opfusion_gen extends App {
  new (chisel3.stage.ChiselStage)
    .execute(Array("--target-dir", "./verilog/opfusion"), Seq(ChiselGeneratorAnnotation(() => new opfusionTop)))
}
