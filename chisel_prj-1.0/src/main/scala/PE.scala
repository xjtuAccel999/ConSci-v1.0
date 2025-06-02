import chisel3._
import chisel3.util._
import util_function._

class PEControl extends Bundle {
  val cal_type  = Bool() // 1 -- dot product, 0 -- gemm
  val datatype  = UInt(3.W) // 001 -- INT8, 010 -- INT32, 100 -- FL32
  val propagate = Bool() // propagate: Which register should be propagated (and which should be accumulated)?
  val sel       = Bool() // which b to compute
}

class PE(row: Int = 0) extends Module with pe_config with hw_config {

  val io = IO(new Bundle {
    val in_a       = Input(UInt(pe_data_w.W)) // ifm
    val in_b       = Input(UInt(pe_data_w.W)) // w
    val in_c0      = Input(UInt(pe_data_w.W)) // part sum
    val in_c1      = Input(UInt(pe_data_w.W)) // part sum
    val in_b_valid = Input(Bool())
    val en         = Input(Bool())

    val ctl = Input(new PEControl)

    val out_a       = Output(UInt(pe_data_w.W)) // ifm
    val out_b       = Output(UInt(pe_data_w.W)) // w output
    val out_d0      = Output(UInt(pe_data_w.W)) // part sum
    val out_d1      = Output(UInt(pe_data_w.W)) // part sum
    val out_b_valid = Output(Bool())

  })
  assert(io.ctl.datatype === 1.U)
  val c_width = if (row == 0) 0 else log2Floor(row) + 16
  val c0      = if (row == 0) 0.S else io.in_c0(c_width - 1, 0).asSInt
  val c1      = if (row == 0) 0.S else io.in_c1(c_width - 1, 0).asSInt

  val b = RegInit(Vec(2, SInt(pe_data_w.W)), 0.B.asTypeOf(Vec(2, SInt(pe_data_w.W))))

  val a0  = io.in_a(7, 0).asSInt
  val a1  = io.in_a(23, 16).asSInt
  val p   = io.ctl.propagate
  val sel = io.ctl.sel

  val use_b = Wire(SInt(wgt_data_w.W))
  use_b := (if (row == 0) Mux(io.ctl.cal_type, io.in_b.asSInt, b(sel)) else b(sel))
  when(io.in_b_valid && io.en) {
    b(p) := (if (row == 0) Mux(io.ctl.cal_type, 0.S, io.in_b.asSInt) else io.in_b.asSInt)
  }
  io.out_b := b(p).asUInt

  val b_invalid = Wire(Vec(2, Bool()))
  b_invalid(0) := riseEdge(io.ctl.sel, io.en) // sel posedge
  b_invalid(1) := fallEdge(io.ctl.sel, io.en) // sel negedge
  val out_b_valid = !b_invalid(p) && !(io.ctl.propagate === io.ctl.sel)
  io.out_b_valid := out_b_valid
  //  || RegEnable(out_b_valid, dualEdge(io.en))

  if (row == 0) {
    io.out_a := RegEnable(io.in_a, 0.U, io.en && !io.ctl.cal_type)
  } else {
    io.out_a := RegEnable(io.in_a, 0.U, io.en)
  }

  if (FPGA_MODE) {
    if (row == 0) {
      io.out_d0 := RegEnable(a0 * use_b + 0.S(pe_data_w.W), 0.S, io.en).asUInt
      io.out_d1 := RegEnable(a1 * use_b + 0.S(pe_data_w.W), 0.S, io.en).asUInt
    } else {
      io.out_d0 := RegEnable(a0 * use_b + io.in_c0.asSInt, 0.S, io.en).asUInt
      io.out_d1 := RegEnable(a1 * use_b + io.in_c1.asSInt, 0.S, io.en).asUInt
    }
  } else {
    if (row == 0) {
      io.out_d0 := RegEnable((a0 * use_b).pad(pe_data_w), 0.S, io.en).asUInt
      io.out_d1 := RegEnable((a1 * use_b).pad(pe_data_w), 0.S, io.en).asUInt
    } else {
      io.out_d0 := RegEnable((a0 * use_b + c0).pad(pe_data_w), 0.S, io.en).asUInt
      io.out_d1 := RegEnable((a1 * use_b + c1).pad(pe_data_w), 0.S, io.en).asUInt
    }
  }
}
