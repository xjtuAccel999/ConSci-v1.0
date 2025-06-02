import chisel3._
import chisel3.util._
import chisel3.stage.ChiselGeneratorAnnotation

trait Convert {
  implicit def AnyToInt(a: Any): Int = Integer.parseInt(a.toString)

  implicit def AnyToBoolean(a: Any): Boolean = a == true

  implicit def AnyToString(a: Any): String = a.toString

  implicit def FP32ToUInt(f: FP32): UInt = f.asUInt()

  implicit def UIntToFP32(f: UInt): FP32 = FP32(f)

  implicit def BooleanToInt(b: Boolean): Int = if (b) 1 else 0

  implicit def BoolToUInt(b: Bool): UInt = b.asUInt

  implicit def UIntToBool(u: UInt): Bool = u.asBool
}

class FP32 {
  var bits: Bits = _

  def this(bits: Bits) {
    this()
    this.bits = bits
  }

  def this(fp: FP32) {
    this()
    this.bits = fp.bits
  }

  def asUInt(): UInt = {
    this.bits.asUInt
  }

  /* xs != ys -> ys
   * xs == ys -> ((xe,xf) > (ys,yf)) ^ xs - */
  def >(that: FP32): Bool = {
    Mux(this.bits(31) ^ that.bits(31), that.bits(31), (this.bits.asUInt(30, 0) > that.bits.asUInt(30, 0)) ^ this.bits(31))
  }

  def <(that: FP32): Bool = {
    Mux(this.bits(31) ^ that.bits(31), this.bits(31), (this.bits.asUInt(30, 0) < that.bits.asUInt(30, 0)) ^ this.bits(31))
  }

  def *(that: FP32)(implicit config: Map[String, Any] = Map()): FP32 = {
    val mult = Module(new FP32_Mult(true))
    mult.io.x := this.bits.asUInt
    mult.io.y := that.bits.asUInt
    new FP32(mult.io.z)
  }

}

object FP32 {
  def apply(bits: Bits): FP32 = {
    new FP32(bits)
  }
}

class FP32_Adder(use_valid_out: Boolean = false) extends Module with Convert with cal_cell_params {

  val io = IO(new Bundle {
    val x         = Input(UInt(32.W))
    val y         = Input(UInt(32.W))
    val z         = Output(UInt(32.W))
    val valid_in  = Input(Bool())
    val valid_out = if (use_valid_out) Some(Output(Bool())) else None
  })

  val (xs_0, xe_0, xf_0, ys_0, ye_0, yf_0) = (io.x(31), io.x(30, 23), io.x(22, 0), io.y(31), io.y(30, 23), io.y(22, 0))

  /*-------- p0 --------*/
  val y_gt_x_0 = io.y(30, 0) > io.x(30, 0)
  val valid_0  = io.valid_in
  val zs_0     = Mux(y_gt_x_0, ys_0, xs_0)
  val ze_0     = Mux(y_gt_x_0, ye_0, xe_0)

  val xf1_0   = Mux(y_gt_x_0, Cat(ye_0 =/= 0.U, yf_0), Cat(xe_0 =/= 0.U, xf_0))
  val yf1_0   = Mux(y_gt_x_0, Cat(xe_0 =/= 0.U, xf_0), Cat(ye_0 =/= 0.U, yf_0))
  val xe_ye_0 = Mux(y_gt_x_0, ye_0, xe_0) - Mux(y_gt_x_0, xe_0, ye_0)

  val zs_a    = RegEnable(zs_0, valid_0)
  val ze_a    = RegEnable(ze_0, valid_0)
  val xf1_a   = RegEnable(xf1_0, valid_0)
  val yf1_a   = RegEnable(yf1_0, valid_0)
  val xe_ye_a = RegEnable(xe_ye_0, valid_0)
  val xs_ys_a = RegEnable(xs_0 === ys_0, valid_0)

  val xe_ye_reduced = xe_ye_a(4, 0)
  val xe_ye_32      = xe_ye_a(7, 5) =/= 0.U
  val yf2_a0        = Mux(xe_ye_32, 0.U, yf1_a)
  val yf2_a1        = Mux(xe_ye_reduced(4), yf2_a0 >> 16, yf2_a0)
  val yf2_a2        = Mux(xe_ye_reduced(3), yf2_a1 >> 8, yf2_a1)
  val yf2_a3        = Mux(xe_ye_reduced(2), yf2_a2 >> 4, yf2_a2)
  val yf2_a4        = Mux(xe_ye_reduced(1), yf2_a3 >> 2, yf2_a3)
  val yf2_a5        = Mux(xe_ye_reduced(0), yf2_a4 >> 1, yf2_a4).asUInt
  /*-------- p1 --------*/
  val valid_1 = RegNext(valid_0,0.U)
  val xs_ys_1 = RegEnable(xs_ys_a, valid_1)
  val xf1_1   = RegEnable(xf1_a, valid_1)
  val yf2_1   = RegEnable(yf2_a5, valid_1)
  val ze_1    = RegEnable(ze_a, valid_1)
  val zs_1    = RegEnable(zs_a, valid_1)

  val zf_1 = Mux(xs_ys_1, xf1_1 +& yf2_1, xf1_1 - yf2_1)
  /*-------- p2 --------*/
  val valid_2 = RegNext(valid_1,0.U)
  val ze_2    = RegEnable(ze_1, valid_2)
  val zs_2    = RegEnable(zs_1, valid_2)
  val zf_2    = RegEnable(zf_1, valid_2)
  val off0    = Calc_offset_0()
  off0.io.frac := zf_2(23, 1)
  val offset_2 = off0.io.off

  /*-------- p3 --------*/
  val valid_3  = RegNext(valid_2,0.U)
  val ze_3     = RegEnable(ze_2, valid_3)
  val zs_3     = RegEnable(zs_2, valid_3)
  val zf_3     = RegEnable(zf_2, valid_3)
  val offset_3 = RegEnable(offset_2, valid_3)

  val underflow_3   = ze_3 < offset_3
  val shift_right_3 = zf_3(24) === 1.U
  val ze1_3 = PriorityMux(
    Seq(
      underflow_3 -> 0.U(8.W),
      shift_right_3 -> (ze_3 + 1.U),
      true.B -> (ze_3 - offset_3)
    )
  )
  val zf1_3 = PriorityMux(
    Seq(
      underflow_3 -> 0.U(23.W),
      shift_right_3 -> zf_3(23, 1),
      true.B -> (zf_3 << offset_3)(22, 0)
    )
  )

  /*-------- out --------*/
  val valid_4 = RegNext(valid_3,0.U)
  io.z := RegEnable(Cat(zs_3, ze1_3, zf1_3), valid_4)
  if (use_valid_out) {
    io.valid_out.get := RegNext(valid_4,0.U)
  }
}

object FP32_Adder {
  def apply(use_valid_out: Boolean = false) = Module(new FP32_Adder(use_valid_out))
}

class Calc_offset_0 extends Module {
  val io = IO(new Bundle {
    val frac = Input(UInt(23.W))
    val off  = Output(UInt(5.W))
  })
  io.off := 22.U - Log2(io.frac)
}

object Calc_offset_0 {
  def apply() = Module(new Calc_offset_0)
}

class FP32_Mult(use_valid_out: Boolean=false) extends Module with Convert with cal_cell_params {

  val io = IO(new Bundle {
    val x = Input(UInt(32.W))
    val y = Input(UInt(32.W))
    val z = Output(UInt(32.W))
    val valid_in = Input(Bool())
    val valid_out = if (use_valid_out) Some(Output(Bool())) else None
  })

  val zs = io.x(31) ^ io.y(31)
  val xe = io.x(30, 23)
  val ye = io.y(30, 23)
  val xf = io.x(22, 0)
  val yf = io.y(22, 0)

  val is_zero = RegEnable(xe === 0.U || ye === 0.U, io.valid_in)

  val zf_0 = (Cat(1.U, xf) * Cat(1.U, yf))(47, 22)

  val zf_1  = RegEnable(zf_0(24, 0), io.valid_in)
  val valid_in_r = RegNext(io.valid_in,0.U)

  val carry = RegEnable(zf_0(25), io.valid_in) || (zf_1 === 0x1ffffff.U)
  val zf    = Mux(carry, zf_1(24, 2) + zf_1(1), zf_1(23, 1) + zf_1(0))
  val ze    = RegEnable(xe + ye, io.valid_in) - Mux(carry, 126.U, 127.U)

  val out_z = Cat(
    RegEnable(zs, io.valid_in),
    PriorityMux(
      Seq(
        is_zero -> 0.U(31.W), // zero
        // (ze(8) =/= 0.U) -> Mux(ze(9), 0.U(31.W), 0x7f800000.U(31.W)), // underflow, overflow
        true.B -> Cat(ze(7, 0), zf)
      )
    )
  )

  io.z := RegEnable(out_z, valid_in_r)
  if(use_valid_out) {
    io.valid_out.get := RegNext(valid_in_r,0.U)
  }
}

object FP32_Mult {
  def apply(use_valid_out: Boolean=false) = Module(new FP32_Mult(use_valid_out))
}