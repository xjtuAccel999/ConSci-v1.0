import chisel3._
import chisel3.util._
import util_function._

class transposeCell extends Module {
  val io = IO(new Bundle() {
    val i_data0 = Input(UInt(8.W))
    val i_data1 = Input(UInt(8.W))
    val i_valid = Input(Bool())
    val i_sel   = Input(Bool())
    val o_data0 = Output(UInt(8.W))
    val o_data1 = Output(UInt(8.W))
  })

  val i_data   = Mux(io.i_sel, io.i_data1, io.i_data0)
  val i_data_t = ShiftRegister(i_data, 1, 0.U, io.i_valid)
  io.o_data0 := i_data_t
  io.o_data1 := i_data_t
}

class transpose extends Module {
  val io = IO(new Bundle() {
    val cfg_gemm = Input(new cfg_gemm_io)
    val i_data   = Input(new data_gp(4, 8))
    val o_data   = Valid(UInt(128.W))
  })

  val cal_state1        = RegNext(io.cfg_gemm.en,0.B)
  val cal_state2        = RegNext(cal_state1,0.B)
  val cal_state3        = RegNext(cal_state2,0.B)
  val owh               = RegEnable(io.cfg_gemm.ow * io.cfg_gemm.oh, 0.U, cal_state1)
  val owh_align64_div64 = RegEnable(align(owh, 24, 64)(23, 6), 0.U, cal_state2)
  val oc_align32_div32  = RegEnable(align(io.cfg_gemm.oc, 12, 32)(11, 5), 0.U, cal_state2)
  val block_num         = RegInit(0.U(25.W))
  block_num := Mux(cal_state3, owh_align64_div64 * oc_align32_div32, block_num)

  val data_gp  = (for (i <- 0 until 4) yield { io.i_data.data(i) }).reverse.reduce(Cat(_, _))
  val datagp_t = RegInit(0.U(256.W))
  when(!io.cfg_gemm.en) {
    datagp_t := 0.U
  }.elsewhen(io.i_data.valid) {
    datagp_t := Cat(data_gp, datagp_t(255, 32)) //transpose data input
  }

  val cnt = RegInit(0.U(3.W))
  cnt := Mux(io.i_data.valid, cnt + 1.U, cnt)
  val valid_in = RegNext(cnt === 7.U && io.i_data.valid,0.B) //transpose valid input
  val sel      = RegInit(0.U(6.W))
  when(!io.cfg_gemm.en) {
    sel := 0.U
  }.elsewhen(valid_in) {
    sel := sel + 1.U
  }

  //block cal
  val block_num_cnt  = RegInit(0.U(25.W))
  val add_zero_cnt   = RegInit(0.U(6.W))
  val add_zero_valid = RegInit(false.B)
  when(!io.cfg_gemm.en) {
    block_num_cnt := 0.U
  }.elsewhen(sel(4, 0) === 31.U && valid_in) {
    block_num_cnt := block_num_cnt + 1.U
  }.elsewhen(add_zero_valid) {
    block_num_cnt := 0.U
  }

  when(!io.cfg_gemm.en) {
    add_zero_cnt := 0.U
  }.elsewhen(add_zero_valid) {
    add_zero_cnt := add_zero_cnt + 1.U
  }
  when(!io.cfg_gemm.en) {
    add_zero_valid := false.B
  }.elsewhen(block_num_cnt === block_num && block_num_cnt =/= 0.U) {
    add_zero_valid := true.B
  }.elsewhen(add_zero_cnt === 63.U) {
    add_zero_valid := false.B
  }

  val memArray          = Seq.fill(32, 32)(Module(new transposeCell))
  val memArray_valid_in = valid_in | (add_zero_valid & add_zero_cnt(0))

  for (j <- 0 until 32) {
    for (i <- 0 until 32) { //horizontal
      memArray(i)(j).io.i_valid := memArray_valid_in
      memArray(i)(j).io.i_sel   := sel(5)
    }
  }

  for (j <- 1 until 32) {
    for (i <- 1 until 32) { //horizontal
      memArray(i)(j).io.i_data0 := memArray(i - 1)(j).io.o_data0
      memArray(i)(j).io.i_data1 := memArray(i)(j - 1).io.o_data1
    }
  }

  val h_out = Wire(Vec(32, UInt(8.W)))
  for (i <- 0 until 32) {
    memArray(i).head.io.i_data1 := Mux(add_zero_valid, 0.U, datagp_t(255 - 8 * i, 248 - 8 * i))
    if (i > 0) {
      memArray(i).head.io.i_data0 := memArray(i - 1).head.io.o_data0
    }
    h_out(i) := memArray(i)(31).io.o_data1
  }

  val v_out = Wire(Vec(32, UInt(8.W)))
  for (j <- 0 until 32) {
    memArray.head(j).io.i_data0 := Mux(add_zero_valid, 0.U, datagp_t(255 - 8 * j, 248 - 8 * j))
    if (j > 0) {
      memArray.head(j).io.i_data1 := memArray.head(j - 1).io.o_data1
    }
    v_out(j) := memArray(31)(j).io.o_data0
  }

  val out     = Mux(sel(5), h_out, v_out)
  val out_cat = Cat(out)
  val o_ready = RegInit(false.B)
  when(!io.cfg_gemm.en) {
    o_ready := false.B
  }.elsewhen(sel(5)) {
    o_ready := true.B
  }

  val out_data_t = RegInit(0.U(256.W))
  val out_valid  = o_ready && memArray_valid_in
  when(!io.cfg_gemm.en) {
    out_data_t := 0.U
  }.elsewhen(memArray_valid_in) {
    out_data_t := out_cat
  }
  val out_valid_t = RegNext(out_valid,0.B)
  io.o_data.valid := out_valid_t | RegNext(out_valid_t,0.B)
  io.o_data.bits  := Mux(out_valid_t, out_data_t(127, 0), out_data_t(255, 128))
}

class ifm_transpose extends Module {
  val io = IO(new Bundle() {
    val en             = Input(Bool())
    val add_zero_valid = Input(Bool())
    val i_data         = Input(new data_gp(4, 8))
    val o_data         = Valid(UInt(128.W))
  })

  val data_gp  = (for (i <- 0 until 4) yield { io.i_data.data(i) }).reverse.reduce(Cat(_, _))
  val datagp_t = RegInit(0.U(256.W))
  when(!io.en) {
    datagp_t := 0.U
  }.elsewhen(io.i_data.valid) {
    datagp_t := Cat(data_gp, datagp_t(255, 32)) //transpose data input
  }

  val cnt = RegInit(0.U(3.W))
  cnt := Mux(io.i_data.valid, cnt + 1.U, cnt)
  val valid_in = RegNext(cnt === 7.U && io.i_data.valid,0.B) //transpose valid input
  val sel      = RegInit(0.U(6.W))
  when(!io.en) {
    sel := 0.U
  }.elsewhen(valid_in) {
    sel := sel + 1.U
  }

  val memArray          = Seq.fill(32, 32)(Module(new transposeCell))
  val memArray_valid_in = valid_in | io.add_zero_valid

  for (j <- 0 until 32) {
    for (i <- 0 until 32) { //horizontal
      memArray(i)(j).io.i_valid := memArray_valid_in
      memArray(i)(j).io.i_sel   := sel(5)
    }
  }

  for (j <- 1 until 32) {
    for (i <- 1 until 32) { //horizontal
      memArray(i)(j).io.i_data0 := memArray(i - 1)(j).io.o_data0
      memArray(i)(j).io.i_data1 := memArray(i)(j - 1).io.o_data1
    }
  }

  val h_out = Wire(Vec(32, UInt(8.W)))
  for (i <- 0 until 32) {
    memArray(i).head.io.i_data1 := Mux(io.add_zero_valid, 0.U, datagp_t(255 - 8 * i, 248 - 8 * i))
    if (i > 0) {
      memArray(i).head.io.i_data0 := memArray(i - 1).head.io.o_data0
    }
    h_out(i) := memArray(i)(31).io.o_data1
  }

  val v_out = Wire(Vec(32, UInt(8.W)))
  for (j <- 0 until 32) {
    memArray.head(j).io.i_data0 := Mux(io.add_zero_valid, 0.U, datagp_t(255 - 8 * j, 248 - 8 * j))
    if (j > 0) {
      memArray.head(j).io.i_data1 := memArray.head(j - 1).io.o_data1
    }
    v_out(j) := memArray(31)(j).io.o_data0
  }

  val out     = Mux(sel(5), h_out, v_out)
  val out_cat = Cat(out)
  val o_ready = RegInit(false.B)
  when(!io.en) {
    o_ready := false.B
  }.elsewhen(sel(5)) {
    o_ready := true.B
  }

  val out_data_t = RegInit(0.U(256.W))
  val out_valid  = o_ready && memArray_valid_in
  when(!io.en) {
    out_data_t := 0.U
  }.elsewhen(memArray_valid_in) {
    out_data_t := out_cat
  }
  val out_valid_t = RegNext(out_valid,0.B)
  io.o_data.valid := out_valid_t | RegNext(out_valid_t,0.B)
  io.o_data.bits  := Mux(out_valid_t, out_data_t(127, 0), out_data_t(255, 128))
}
