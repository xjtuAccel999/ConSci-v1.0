import chisel3._
import chisel3.util._
import util_function._

class innerprodFuncCell extends Module with alu_config {
  val io = IO(new Bundle() {
    //input
    val i_data0 = Flipped(Valid(SInt(8.W)))
    val i_data1 = Flipped(Valid(SInt(8.W)))
    val cfg     = Input(new cfg_alu_io)
    //output
    val o_data = Output(SInt(32.W))
  })
  val innerprod_en_t1 = RegNext(io.cfg.innerprod_en,0.B)
  val mul_o           = RegInit(0.S(16.W))
  val add_o           = RegInit(0.S(32.W))
  val valid_i         = io.i_data0.valid & io.i_data1.valid
  when(innerprod_en_t1) {
    when(valid_i) {
      mul_o := io.i_data0.bits * io.i_data1.bits
    }
  }.otherwise {
    mul_o := 0.S
  }

  when(innerprod_en_t1) {
    when(RegNext(valid_i,0.B)) {
      add_o := mul_o + add_o
    }
  }.otherwise {
    add_o := 0.S
  }
  io.o_data := add_o
}

class innerprodFunc(innerprod_id: Int) extends Module {
  val io = IO(new Bundle() {
    //input
    val cfg     = Input(new cfg_alu_io)
    val i_data0 = Flipped(Valid(UInt(128.W)))
    val i_data1 = Flipped(Valid(UInt(128.W)))
    val o_data  = Valid(SInt(32.W))
  })

  val innerprodUnit   = Seq.fill(16)(Module(new innerprodFuncCell))
  val innerprod_en_t1 = RegNext(io.cfg.innerprod_en,0.B)
  for (i <- 0 until 16) {
    innerprodUnit(i).io.cfg           <> io.cfg
    innerprodUnit(i).io.i_data0.valid <> io.i_data0.valid
    innerprodUnit(i).io.i_data1.valid <> io.i_data1.valid
    innerprodUnit(i).io.i_data0.bits  <> io.i_data0.bits(8 * i + 7, 8 * i).asSInt
    innerprodUnit(i).io.i_data1.bits  <> io.i_data1.bits(8 * i + 7, 8 * i).asSInt
  }
  val innerprod_valid_o = RegInit(false.B)
  val vec_len           = (if (innerprod_id == 0) io.cfg.veclen_ch0 else io.cfg.veclen_ch1)(31, 2) - 1.U
  val cnt               = RegInit(0.U(30.W))
  when(innerprod_en_t1) {
    when(io.i_data0.valid & io.i_data1.valid) {
      when(cnt === vec_len) {
        cnt               := 0.U
        innerprod_valid_o := true.B
      }.otherwise {
        cnt := cnt + 1.U
      }
    }
  }.otherwise {
    cnt               := 0.U
    innerprod_valid_o := false.B
  }

  val add_tree_level1 = Wire(Vec(8, SInt(32.W)))
  for (i <- 0 until 8) {
    add_tree_level1(i) := RegEnable(innerprodUnit(i).io.o_data + innerprodUnit(i + 8).io.o_data, io.cfg.innerprod_en)
  }
  val add_tree_level2 = Wire(Vec(4, SInt(32.W)))
  for (i <- 0 until 4) {
    add_tree_level2(i) := RegEnable(add_tree_level1(i) + add_tree_level1(i + 4), io.cfg.innerprod_en)
  }
  val add_tree_level3 = Wire(Vec(2, SInt(32.W)))
  for (i <- 0 until 2) {
    add_tree_level3(i) := RegEnable(add_tree_level2(i) + add_tree_level2(i + 2), io.cfg.innerprod_en)
  }
  io.o_data.bits  := RegEnable(add_tree_level3.head + add_tree_level3(1), io.cfg.innerprod_en)
  io.o_data.valid := ShiftRegister(innerprod_valid_o, 5, 0.U, 1.B)
}
