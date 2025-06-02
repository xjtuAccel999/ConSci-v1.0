import chisel3._
import chisel3.util._
import util_function._

class mathFuncCell extends Module with alu_config {
  val io = IO(new Bundle() {
    //input
    val i_data0 = Flipped(Valid(UInt(32.W)))
    val i_data1 = Flipped(Valid(UInt(32.W)))
    val cfg     = Input(new cfg_alu_io)
    //output
    val axi_data = Valid(UInt(32.W))
  })

  val mul_src0 = io.i_data0.bits
  val mul_src1 = Wire(UInt(32.W))
  val mul_o    = Wire(UInt(32.W))

  mul_src1 := 0.U
  switch(io.cfg.math_mul_src1_sel) {
    is(1.U) {
      mul_src1 := io.i_data1.bits
    }
    is(2.U) {
      mul_src1 := io.cfg.math_alpha
    }
  }

  val add_src0      = Wire(UInt(32.W))
  val add_src1      = Wire(UInt(32.W))
  val add_src0_t    = Mux(io.cfg.math_add_en, ShiftRegister(add_src0, FP32_ADD_CYCLES, io.cfg.math_en), 0.U)
  val add_src1_t    = Mux(io.cfg.math_add_en, ShiftRegister(io.i_data1.bits, FP32_ADD_CYCLES, io.cfg.math_en), 0.U)
  val add_o         = Wire(UInt(32.W))
  val add_compare_o = Wire(UInt(32.W))
  add_src0 := 0.U
  switch(io.cfg.math_add_src0_sel) {
    is(1.U) {
      add_src0 := io.i_data0.bits
    }
    is(2.U) {
      add_src0 := mul_o
    }
  }

  add_src1 := 0.U
  switch(io.cfg.math_add_src1_sel) {
    is(1.U) {
      add_src1 := Cat(Mux(io.cfg.math_sub_en, ~io.i_data1.bits(31), io.i_data1.bits(31)), io.i_data1.bits(30, 0))
    }
    is(2.U) {
      add_src1 := Cat(Mux(io.cfg.math_sub_en, ~io.cfg.math_beta(31), io.cfg.math_beta(31)), io.cfg.math_beta(30, 0))
    }
  }

  val valid_t1      = RegNext(io.i_data0.valid,0.B)
  val valid_mul     = ShiftRegister(valid_t1, FP32_MUL_CYCLES - 1, 0.B, 1.B)
  val valid_add     = ShiftRegister(valid_t1, FP32_ADD_CYCLES - 1, 0.B, 1.B)
  val valid_mul_add = ShiftRegister(valid_t1, FP32_ADD_CYCLES + FP32_MUL_CYCLES - 1, 0.B, 1.B)

  mul_o := Mux(io.cfg.math_mul_en && valid_mul, Float.FloatMul(Float(mul_src0), Float(mul_src1), io.cfg.math_en).bits, 0.U)
  val add_o_t = Float.FloatAdd(Float(add_src0), Float(add_src1), io.cfg.math_en).bits
  add_o := 0.U
  when(io.cfg.math_add_en && io.cfg.math_mul_en) {
    add_o := Mux(valid_mul_add, add_o_t, 0.U)
  }.elsewhen(io.cfg.math_add_en) {
    add_o := Mux(valid_add, add_o_t, 0.U)
  }
  //if op == min | max | add | sub, add_en must be true

  val (big_flag, small_flag, equal_flag) = (!add_o(31), add_o(31), add_o.andR)
  add_compare_o := add_o
  when(io.cfg.math_max_en) {
    add_compare_o := Mux(big_flag, add_src0_t, add_src1_t)
  }.elsewhen(io.cfg.math_min_en) {
    add_compare_o := Mux(small_flag, add_src0_t, add_src1_t)
  }

  io.axi_data.bits  := 0.U
  io.axi_data.valid := false.B
  switch(Cat(io.cfg.math_add_en, io.cfg.math_mul_en)) {
    is(0.U) {
      switch(io.cfg.math_op) {
        is(ABS_ID) {
          io.axi_data.bits  := RegEnable(Cat(0.U(1.W), io.i_data0.bits(30, 0)), 0.U, io.i_data0.valid)
          io.axi_data.valid := valid_t1
        }
      }
    }
    is(1.U) {
      io.axi_data.bits  := mul_o
      io.axi_data.valid := valid_mul
    }
    is(2.U) {
      io.axi_data.bits  := add_compare_o
      io.axi_data.valid := valid_add
      switch(io.cfg.math_op) {
        is(THRESHOLD_ID) {
          io.axi_data.bits  := Mux(big_flag, 0x3f800000.asUInt, 0.U(32.W))
          io.axi_data.valid := valid_add
        }
        is(EQUAL_ID) {
          io.axi_data.bits  := Mux(equal_flag, 0x3f800000.asUInt, 0.U(32.W))
          io.axi_data.valid := valid_add
        }
      }
    }
    is(3.U) {
      io.axi_data.bits  := add_o
      io.axi_data.valid := valid_mul_add
    }
  }
}

class mathFunc(alu_id: Int) extends Module with cal_cell_params {
  val io = IO(new Bundle() {
    //input
    val cfg     = Input(new cfg_alu_io)
    val i_data0 = Flipped(Vec(4, Valid(UInt(32.W))))
    val i_data1 = Flipped(Vec(4, Valid(UInt(32.W))))
    val o_data  = Vec(4, Valid(UInt(32.W)))
  })

  val mathFuncUnit = Seq.fill(4)(Module(new mathFuncCell).io)

  for (i <- 0 until 4) {
    mathFuncUnit(i).i_data0  <> io.i_data0(i)
    mathFuncUnit(i).i_data1  <> io.i_data1(i)
    mathFuncUnit(i).cfg      <> io.cfg
    mathFuncUnit(i).axi_data <> io.o_data(i)
  }
}
