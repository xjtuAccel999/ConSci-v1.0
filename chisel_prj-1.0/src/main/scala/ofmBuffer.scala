import chisel3._
import chisel3.util._
import util_function._

class ofmBufferCell extends Module with buffer_config {
  val io = IO(new Bundle() {
    //input
    val i_data = Flipped(Valid(UInt(32.W)))
    //output
    val o_data    = Decoupled(UInt(128.W))
    val congested = Output(Bool())
  })

  val i_valid_t = RegNext(io.i_data.valid,0.B)
  val i_data_t  = RegEnable(io.i_data.bits,io.i_data.valid)

  val data_t = RegInit(0.U(128.W))
  data_t := Mux(i_valid_t, Cat(i_data_t, data_t(127, 32)), data_t)

  val cnt = RegInit(0.U(2.W))
  cnt := Mux(i_valid_t, cnt + 1.U, cnt)

//  val fifo = Module(new Queue(UInt(128.W), entries = OFM_BUFFER_DEPTH))
  val fifo = Module(new standard_fifo(128, OFM_BUFFER_DEPTH, "block"))
//  val fifo = Module(new standard_fifo_delay(128, OFM_BUFFER_DEPTH, "block"))
  fifo.io.enq.valid := RegNext(i_valid_t & cnt === 3.U,0.U)
  fifo.io.enq.bits  := data_t
  fifo.io.deq       <> io.o_data

  val congested_top_limit  = fifo.io.count > (OFM_BUFFER_DEPTH - 8 - 2).U
  val congested_down_limit = fifo.io.count < 16.U
  val congested            = RegInit(false.B)
  congested    := Mux(congested_top_limit, true.B, Mux(congested_down_limit, false.B, congested))
  io.congested := congested
}

class ofmBuffer extends Module with buffer_config with mesh_config {
  val io = IO(new Bundle() {
    //input
    val i_data            = Flipped(Vec(mesh_columns, Valid(new acc_data)))
    val axiSend_congested = Input(Bool())
    //output
    val o_data_ch0          = Output(new data_gp(4, 32))
    val o_data_ch1          = Output(new data_gp(4, 32))
    val ofmBuffer_congested = Output(Bool())
  })

  val buffer0 = Seq.fill(32)(Module(new ofmBufferCell))
  val buffer1 = Seq.fill(32)(Module(new ofmBufferCell))

  io.ofmBuffer_congested := buffer0.head.io.congested

  for (i <- 0 until 32) {
    buffer0(i).io.i_data.valid := io.i_data(i).valid
    buffer0(i).io.i_data.bits  := io.i_data(i).bits.data0.asSInt.pad(32).asUInt
    buffer1(i).io.i_data.valid := io.i_data(i).valid
    buffer1(i).io.i_data.bits  := io.i_data(i).bits.data1.asSInt.pad(32).asUInt
  }

  val fifo_valid = Wire(Vec(32, Bool()))
  val cnt        = RegInit(0.U(8.W))

  for (i <- 0 until 32) {
    fifo_valid(i)              := !io.axiSend_congested & buffer0(i).io.o_data.valid
    buffer0(i).io.o_data.ready := cnt(7, 3) === i.U && fifo_valid(i)
    buffer1(i).io.o_data.ready := buffer0(i).io.o_data.ready
  }

  io.o_data_ch0.valid := false.B
  io.o_data_ch1.valid := false.B
  for (j <- 0 until 4) {
    io.o_data_ch0.data(j) := 0.U
    io.o_data_ch1.data(j) := 0.U
  }

  for (i <- 0 until 32) {
//    when(buffer0(i).io.o_data.valid & buffer0(i).io.o_data.ready) {
    when(RegNext(buffer0(i).io.o_data.valid & buffer0(i).io.o_data.ready,0.B)) {
      io.o_data_ch0.valid := true.B
      io.o_data_ch1.valid := true.B
      for (j <- 0 until 4) {
        io.o_data_ch0.data(j) := buffer0(i).io.o_data.bits(32 * j + 31, 32 * j)
        io.o_data_ch1.data(j) := buffer1(i).io.o_data.bits(32 * j + 31, 32 * j)
      }
    }
  }

  for (i <- 0 until 32) {
    when(cnt(7, 3) === i.U) {
      cnt := Mux(fifo_valid(i), cnt + 1.U, cnt)
    }
  }
}
