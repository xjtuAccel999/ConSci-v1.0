import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util._
import util_function._

class AccMemUnit extends Module with mesh_config {
  val io = IO(new Bundle {
    val ofm         = Flipped(Valid(new ofm_data))
    val out         = Valid(new acc_data)
    val ofmbuf_stop = Input(Bool())
  })
  val mem = Seq.fill(2)(Module(new TPRAM_WRAP(pe_data_w, mesh_size, "block")))
  val en  = !(io.ofmbuf_stop || ofm_in_stage && !io.ofm.valid)

  lazy val sNone :: sStart :: sProcess :: sEnd :: Nil = Enum(4)
  lazy val state                                      = RegInit(sNone)
  val end_read_addr                                   = RegInit(0.U(log2Ceil(mesh_size).W))

  when(en) {
    switch(state) {
      is(sNone) {
        when(io.ofm.valid) {
          state := sStart
        }
      }
      is(sStart) {
        when(io.ofm.bits.finish) {
          state := sEnd
        }.elsewhen(io.ofm.bits.addr === (mesh_size - 1).U && en) {
          state := sProcess
        }
      }
      is(sProcess) {
        when(io.ofm.bits.finish) {
          state := sEnd
        }
      }
      is(sEnd) {
        when(end_read_addr === 0.U) {
          state := sNone
        }
      }
    }
  }

  when(en) {
    when(io.ofm.bits.finish || state === sEnd && end_read_addr =/= 0.U) {
      end_read_addr := Mux(end_read_addr === (mesh_size - 1).U, 0.U, end_read_addr + 1.U)
    }
  }

  // out
  io.out.valid := ShiftRegister(io.ofm.bits.acc_last && io.ofm.valid, mesh_size, 0.B, en) && en && state =/= sNone
  when(io.out.valid) {
    io.out.bits.data0 := mem(0).io.rdata
    io.out.bits.data1 := mem(1).io.rdata
  }.otherwise {
    io.out.bits.data0 := 0.U
    io.out.bits.data1 := 0.U
  }

  // ofm in
  lazy val ofm_in_stage = state === sStart || state === sProcess || riseEdge(io.ofm.valid)

  val add_zero = state === sStart || riseEdge(io.ofm.valid, en) && state === sNone || io.out.valid

  // READ
  val read_buffer    = Wire(Vec(2, UInt(pe_data_w.W)))
  val mem_raddr_temp = RegEnable(mem(0).io.raddr, 0.U, en)
  for (i <- 0 to 1) {
    when(en) {
      when(io.ofm.valid) {
        mem(i).io.ren   := 1.B
        mem(i).io.raddr := Mux(io.ofm.bits.addr =/= (mesh_size - 1).U, io.ofm.bits.addr + 1.U, 0.U)
      }.elsewhen(state === sEnd) { // finish stage
        mem(i).io.ren   := 1.B
        mem(i).io.raddr := end_read_addr
      }.otherwise {
        mem(i).io.ren   := 0.B
        mem(i).io.raddr := 0.U
      }
    }.elsewhen(state =/= sNone) {
      mem(i).io.ren   := 1.B
      mem(i).io.raddr := mem_raddr_temp
    }.otherwise {
      mem(i).io.ren   := 0.B
      mem(i).io.raddr := 0.U
    }

    when(add_zero) {
      read_buffer(i) := 0.U
    }.otherwise {
      read_buffer(i) := mem(i).io.rdata
    }
  }

  // WRITE
  when(io.ofm.valid) {
    mem(0).io.waddr := io.ofm.bits.addr
    mem(0).io.wdata := io.ofm.bits.data0 + read_buffer(0)
    mem(0).io.wen   := en
    mem(1).io.waddr := io.ofm.bits.addr
    mem(1).io.wdata := io.ofm.bits.data1 + read_buffer(1)
    mem(1).io.wen   := en
  }.otherwise {
    mem(0).io.waddr := 0.U
    mem(0).io.wdata := 0.U
    mem(0).io.wen   := 0.B
    mem(1).io.waddr := 0.U
    mem(1).io.wdata := 0.U
    mem(1).io.wen   := 0.B
  }
}

class AccMem extends Module with mesh_config {
  val io = IO(new Bundle {
    val cfg_gemm    = Input(new cfg_gemm_io)
    val ofm         = Flipped(Vec(mesh_columns, Valid(new ofm_data)))
    val ofmbuf_stop = Input(Bool())

    val out = Vec(mesh_columns, Valid(new acc_data))
  })
  val cfg_gemm = RegNext(io.cfg_gemm,0.U.asTypeOf(new cfg_gemm_io))

  val acc_mem = Seq.fill(mesh_columns)(Module(new AccMemUnit))
  for (i <- 0 until mesh_columns) {
    acc_mem(i).io.ofm         <> io.ofm(i)
    acc_mem(i).io.ofmbuf_stop <> Mux(cfg_gemm.op === 1.U, RegNext(io.ofmbuf_stop,0.U), io.ofmbuf_stop)
    io.out(i)                 <> acc_mem(i).io.out
  }
}
