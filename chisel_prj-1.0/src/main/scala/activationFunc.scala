import chisel3._
import chisel3.util._
import util_function._

class MuxOut extends Module {
  val io = IO(new Bundle() {
    //input
    val i_data          = Input(UInt(32.W))
    val cfg             = Input(new cfg_alu_io)
    val act_func_prop   = Input(UInt(2.W))
    val cfg_coefficient = Input(Vec(5, UInt(32.W)))
    val x_range         = Input(Vec(4, UInt(32.W)))
    //output
    val act_coefficient = Output(UInt(32.W))
  })

  val act_prop = ((io.act_func_prop === 1.U) || (io.act_func_prop === 2.U)) && (io.i_data(31) === 1.U)
  val i_data   = FP32(Mux(act_prop, Cat(0.U, io.i_data(30, 0)), io.i_data))
  val x0_range = FP32(io.x_range(0))
  val x1_range = FP32(io.x_range(1))
  val x2_range = FP32(io.x_range(2))
  val x3_range = FP32(io.x_range(3))
  val o_act_coefficient = Mux(
    i_data < x0_range,
    io.cfg_coefficient(0),
    Mux(
      i_data < x1_range,
      io.cfg_coefficient(1),
      Mux(i_data < x2_range, io.cfg_coefficient(2), Mux(i_data < x3_range, io.cfg_coefficient(3), io.cfg_coefficient(4)))
    )
  )
  val o_act_coefficient_t = RegEnable(o_act_coefficient, 0.U, io.cfg.act_en)

  io.act_coefficient := o_act_coefficient_t
}

class activationFuncDataCell extends Module with alu_config {
  val io = IO(new Bundle() {
    //input
    val i_data = Input(UInt(32.W))
    val cfg    = Input(new cfg_alu_io)
    //output
    val o_data = Output(UInt(32.W))
  })
  val MuxOut = Seq.fill(3)(Module(new MuxOut).io)

  for (i <- 0 until 3) {
    MuxOut(i).i_data        <> io.i_data
    MuxOut(i).cfg           <> io.cfg
    MuxOut(i).act_func_prop <> io.cfg.act_func_prop
    MuxOut(i).x_range       <> io.cfg.act_range
  }
  MuxOut(0).cfg_coefficient <> io.cfg.act_coefficient_a
  MuxOut(1).cfg_coefficient <> io.cfg.act_coefficient_b
  MuxOut(2).cfg_coefficient <> io.cfg.act_coefficient_c
  io.o_data                 := 0.U
  //mult
  val x_x   = Wire(UInt(32.W))
  val x_b   = Wire(UInt(32.W))
  val a_x_x = Wire(UInt(32.W))

  //x*x
  x_x := Float.FloatMul(Float(io.i_data), Float(io.i_data), io.cfg.act_en).bits
  //x*b
  val i_data_t = RegInit(0.U(32.W))
  when(io.cfg.act_en){
    when(((io.cfg.act_func_prop === 1.U) || (io.cfg.act_func_prop === 2.U)) && (io.i_data(31) === 1.U)){
      i_data_t := Cat(0.U, io.i_data(30, 0))
    }.otherwise{
      i_data_t := io.i_data
    }
  }
  x_b := Float.FloatMul(Float(i_data_t), Float(MuxOut(1).act_coefficient), io.cfg.act_en).bits
  //a*x*x
  a_x_x := Float.FloatMul(Float(ShiftRegister(MuxOut(0).act_coefficient, FP32_MUL_CYCLES - 1, io.cfg.act_en)), Float(x_x), io.cfg.act_en).bits
  //adder
  val x_b_and_c = Wire(UInt(32.W))
  x_b_and_c := Float.FloatAdd(Float(ShiftRegister(MuxOut(2).act_coefficient, FP32_MUL_CYCLES, io.cfg.act_en)), Float(x_b), io.cfg.act_en).bits

  val o_data_t = Wire(UInt(32.W))
  val i_sign_t = ShiftRegister(io.i_data(31), ACTIVATION_FUNC_CYCLES - FP32_ADD_CYCLES, io.cfg.act_en)
  val o_data_prop = i_sign_t && ((io.cfg.act_func_prop === 1.U) || (io.cfg.act_func_prop === 2.U))

  if (FP32_MUL_CYCLES * 2 > FP32_ADD_CYCLES + FP32_MUL_CYCLES + 1) {
    val delay_cycles = FP32_MUL_CYCLES - FP32_ADD_CYCLES - 1
    val axx_bx_c = Float.FloatAdd(Float(a_x_x), Float(ShiftRegister(x_b_and_c, delay_cycles, io.cfg.act_en)), io.cfg.act_en)
    o_data_t := Mux(o_data_prop, Cat(~axx_bx_c.bits(31), axx_bx_c.bits(30, 0)), axx_bx_c.bits)
  } else {
    val delay_cycles = FP32_ADD_CYCLES - FP32_MUL_CYCLES + 1
    val axx_bx_c = Float.FloatAdd(Float(ShiftRegister(a_x_x, delay_cycles, io.cfg.act_en)), Float(x_b_and_c), io.cfg.act_en)
    o_data_t := Mux(o_data_prop, Cat(~axx_bx_c.bits(31), axx_bx_c.bits(30, 0)), axx_bx_c.bits)
  }

  when(io.cfg.act_op === 0.U) {
    io.o_data := io.i_data
  }.elsewhen((ShiftRegister(io.i_data(31), ACTIVATION_FUNC_CYCLES, io.cfg.act_en) === 1.U) && (io.cfg.act_func_prop === 2.U)) {
    io.o_data := Float.FloatAdd(Float(o_data_t), Float(0x3f800000.U(32.W)), io.cfg.act_en).bits
  }.otherwise {
    io.o_data := ShiftRegister(o_data_t, FP32_ADD_CYCLES, io.cfg.act_en)
  }
}

class activationFuncValidCell extends Module with alu_config {
  val io = IO(new Bundle() {
    //input
    val i_valid = Input(Bool())
    val cfg     = Input(new cfg_alu_io)
    //output
    val o_valid = Output(Bool())
  })

  io.o_valid := false.B
  when(io.cfg.act_en) {
    when(io.cfg.act_op === 0.U) {
      io.o_valid := io.i_valid
    }.otherwise {
      io.o_valid := ShiftRegister(io.i_valid, ACTIVATION_FUNC_CYCLES, 0.U, 1.B)
    }
  }
}
//
class activationFunc extends Module {
  val io = IO(new Bundle() {
    //input
    val cfg    = Input(new cfg_alu_io)
    val i_data = Input(new data_gp(4, 32))
    val o_data = Output(new data_gp(4, 32))
  })

  val cfg = RegEnable(io.cfg, 0.U.asTypeOf(new cfg_alu_io),dualEdge(io.cfg.act_en))

//    io.o_data := io.i_data
  val activationFuncDataUnit  = Seq.fill(4)(Module(new activationFuncDataCell).io)
  val activationFuncValidUnit = Module(new activationFuncValidCell)

  activationFuncValidUnit.io.cfg     <> cfg
  activationFuncValidUnit.io.i_valid <> io.i_data.valid
  activationFuncValidUnit.io.o_valid <> io.o_data.valid

  for (i <- 0 until 4) {
    activationFuncDataUnit(i).i_data <> io.i_data.data(i)
    activationFuncDataUnit(i).cfg    <> cfg
    activationFuncDataUnit(i).o_data <> io.o_data.data(i)
  }
}
