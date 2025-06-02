//package hardfloat
import chisel3._
import chisel3.util._
import hardfloat._

import scala.language.implicitConversions
import util_function._

abstract class Arithmetic[T <: Data] {
  implicit def cast(a: T): ArithmeticOps[T]
}

abstract class ArithmeticOps[T <: Data](a: T) {
  def +(b: T): T
  def *(b: T): T
  def abs: T
}

class sint2float extends Module {
  val io = IO(new Bundle() {
    val i_valid = Input(Bool())
    val i_data  = Input(UInt(32.W))
    val o_data  = Output(UInt(32.W))
  })

  val i_sign = RegInit(0.U(1.W))
  val i_uint = RegInit(0.U(31.W))
  when(io.i_valid) {
    i_sign := io.i_data(31)
    i_uint := Mux(io.i_data(31), (~io.i_data(30, 0)).asUInt + 1.U, io.i_data(30, 0))
  }

  val float_sign = RegInit(0.U(1.W))
  val float_exp  = RegInit(0.U(8.W))
  val float_frac = RegInit(0.U(23.W))
  when(RegNext(io.i_valid,0.B)){
    float_sign := i_sign
    float_exp  := 0.U
    float_frac := 0.U
    val data_extend = Cat(i_uint, 0.U(23.W))
    for (i <- 0 until 31) {
      when(i_uint(i).asBool) {
        float_frac := data_extend(22 + i, i)
        float_exp  := (127 + i).U
      }
    }
  }

  io.o_data := Cat(float_sign, float_exp, float_frac)
}


case class Float(val exp_width: Int, val sig_width: Int) extends Bundle {
  val bits = UInt((exp_width + sig_width).W)
}

object Float extends cal_cell_params {

  def FloatToSInt(a: Float, intWidth: Int, valid:Bool): SInt = {
    val convert = Module(new RecFNToIN(a.exp_width, a.sig_width, intWidth))
    convert.io.in           := recFNFromFN(a.exp_width, a.sig_width, a.bits)
    convert.io.roundingMode := consts.round_near_maxMag
    convert.io.signedOut    := true.B
    val res = RegEnable(convert.io.out.asSInt, 0.S, valid)
    RegEnable(Mux(res === -128.S(8.W), -127.S(8.W), res), RegNext(valid,0.B))
  }

  def SIntToFloat(a: SInt, intWidth: Int = 32, valid:Bool, exp_width: Int = 8, sig_width: Int = 24): Float = {
    if (USE_HARDFLOAT) {
      val convert = Module(new INToRecFN(intWidth, exp_width, sig_width))
      convert.io.signedIn       := true.B
      convert.io.in             := a.asUInt
      convert.io.roundingMode   := consts.round_near_even
      convert.io.detectTininess := consts.tininess_afterRounding
      Float(RegNext(fNFromRecFN(exp_width, sig_width, convert.io.out),0.U))
    } else {
      val convert = Module(new sint2float)
      convert.io.i_data := a.asUInt
      convert.io.i_valid   := valid
      Float(convert.io.o_data)
    }
  }

  def FloatMul(a:Float, b:Float, valid:Bool): Float = {
    if (USE_HARDFLOAT) {
      val a_rec = recFNFromFN(a.exp_width, a.sig_width, a.bits)
      val b_rec = recFNFromFN(b.exp_width, b.sig_width, b.bits)
      val muler = Module(new MulRecFN(a.exp_width, a.sig_width)).io
      muler.a              := RegNext(a_rec,0.U)
      muler.b              := RegNext(b_rec,0.U)
      muler.roundingMode   := consts.round_near_even
      muler.detectTininess := consts.tininess_afterRounding
      Float(RegNext(fNFromRecFN(a.exp_width, a.sig_width, muler.out),0.U))
    } else {
      val muler = Module(new FP32_Mult)
      muler.io.x := a.bits
      muler.io.y := b.bits
      muler.io.valid_in := valid
      Float(muler.io.z)
    }
  }

  def FloatAdd(a:Float, b:Float, valid:Bool):Float = {
    if (USE_HARDFLOAT) {
      val a_rec = recFNFromFN(a.exp_width, a.sig_width, a.bits)
      val b_rec = recFNFromFN(b.exp_width, b.sig_width, b.bits)
      val adder = Module(new AddRecFN(a.exp_width, a.sig_width)).io
      adder.subOp          := false.B
      adder.a              := RegNext(a_rec,0.U)
      adder.b              := RegNext(b_rec,0.U)
      adder.roundingMode   := consts.round_near_even
      adder.detectTininess := consts.tininess_afterRounding
      Float(RegNext(fNFromRecFN(a.exp_width, a.sig_width, adder.out),0.U))
    } else {
      val adder = Module(new FP32_Adder)
      adder.io.x := a.bits
      adder.io.y := b.bits
      adder.io.valid_in := valid
      Float(adder.io.z)
    }
  }

  def apply(a: UInt, exp_width: Int = 8, sig_width: Int = 24): Float = {
    val result = Wire(Float(exp_width, sig_width))
    result.bits := a
    result
  }
  implicit def Float2Bits(x: Float): UInt = x.bits
}
