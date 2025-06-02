import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util._
import util_function._

class acc_data extends Bundle with mesh_config {
  val data0 = UInt(pe_data_w.W)
  val data1 = UInt(pe_data_w.W)
}

class ofm_data extends acc_data {
  val addr     = UInt(ofm_buffer_addr_w.W)
  val acc_last = Bool()
  val finish   = Bool()
}

class Mesh() extends Module with mesh_config with hw_config {
  val io = IO(new Bundle {
    val w           = Flipped(Decoupled(Vec(mesh_columns, UInt(pe_data_w.W))))
    val ifm         = Flipped(Decoupled(Vec(mesh_rows, UInt(pe_data_w.W)))) // each has 2 int8 or 1 int32/fp32
    val ofm         = Vec(mesh_columns, Valid(new ofm_data))
    val cfg_gemm    = Input(new cfg_gemm_io)
    val ofmbuf_stop = Input(Bool())
    val w_finish    = Input(Bool())
    val last_in     = Input(Bool())
  })
  val cfg_gemm = RegEnable(io.cfg_gemm, 0.U.asTypeOf(new cfg_gemm_io),dualEdge(io.cfg_gemm.en))
//  val mesh  = Seq.fill(mesh_rows, mesh_columns)(Module(new PE))
//  val meshT = mesh.transpose

  val mesh  = Array.tabulate(32, 32)((row, col) => Module(new PE(row)))
  val meshT = mesh.transpose

  val sNone :: sOnlyW :: sWandIFM :: sOnlyIFM :: sOFM :: sDot :: Nil = Enum(6)
  val state                                                          = RegInit(sNone)

  val w_state   = state === sOnlyW || state === sWandIFM // w input
  val ifm_state = state === sWandIFM || state === sOnlyIFM // ifm input

  val dw_op      = cfg_gemm.op === 1.U
  val w_block    = w_state && !io.w.valid
  val ifm_block  = ifm_state && !io.ifm.valid
  val en         = !(io.ofmbuf_stop || w_block)
  val w_finish   = io.w_finish
  val ifm_finish = ShiftRegister(w_finish, mesh_size, 0.B, en)

  // ic * k * k <= 32 ...
  val w_only_once = RegNext(
    cfg_gemm.ic * cfg_gemm.kernel * cfg_gemm.kernel <= 32.U && cfg_gemm.ow * cfg_gemm.oh <= 64.U && cfg_gemm.oc <= 32.U
  ,0.B)

  val cnt = RegInit(0.U(log2Ceil(mesh_size).W))
  when(en) {
    switch(state) {
      is(sNone) {
        when(io.w.valid && io.ifm.valid) {
          state := Mux(dw_op, sDot, sOnlyW)
        }
      }
      is(sOnlyW) {
        when(cnt === (mesh_size - 1).U) {
          when(w_only_once) {
            state := sOnlyIFM
          }.otherwise {
            state := sWandIFM
          }
        }
      }
      is(sWandIFM) {
        when(io.w_finish) {
          state := sOnlyIFM
        }
      }
      is(sOnlyIFM) {
        when(ifm_finish) {
          state := sOFM
        }
      }
      is(sOFM) {
        when(io.ofm(mesh_size - 1).bits.finish) {
          state := sNone
        }
      }
      is(sDot) {
        when(io.w_finish) {
          state := sNone
        }
      }
    }
  }

  // cnt
  when(en) {
    when(state =/= sNone) {
      cnt := Mux(cnt === (mesh_size - 1).U, 0.U, cnt + 1.U)
    }.otherwise {
      when(io.w.valid && io.w.ready) {
        cnt := cnt + 1.U
      }.elsewhen(io.w_finish) {
        cnt := 0.U
      }
    }
  }

  io.ifm.ready := (ifm_state || dw_op && io.w.valid) && en
  val ready_tmp           = ShiftRegister(io.ifm.valid & io.w.valid, 5, 1.B)
  val ifm_valid_first_reg = riseEdge(io.ifm.valid) && !io.w.valid
  val ifm_valid_first     = RegInit(false.B)
  when(ifm_valid_first_reg) {
    ifm_valid_first := true.B
  }
  io.w.ready := io.ifm.valid && en
  val ifm_handshake = io.ifm.valid && io.ifm.ready

  val propagate = RegInit(0.B)
  when(dw_op) {
    propagate := 0.B
  }.elsewhen(cnt === (mesh_rows - 1).U && en) {
    propagate := !propagate
  }

  // TODO: maybe optimize this
  // pipeline propagate across each row
  for (r <- 0 until mesh_rows) {
    for (c <- 0 until mesh_columns) {
      mesh(r)(c).io.ctl.propagate := ShiftRegister(propagate, c, 0.B, en)
    }
  }

  // pipeline sel across each row
  for (c <- 0 until mesh_columns) {
    meshT(c).foldLeft(!mesh(0)(c).io.ctl.propagate) {
      case (w, pe) =>
        pe.io.ctl.sel := w
        RegEnable(pe.io.ctl.sel, 0.B, en)
    }
  }

  // broadcast en & datatype & cal_type
  for (r <- 0 until mesh_rows) {
    for (c <- 0 until mesh_columns) {
      mesh(r)(c).io.en           := en
      mesh(r)(c).io.ctl.datatype := 1.B
      if (r == 0) {
        mesh(r)(c).io.ctl.cal_type := dw_op
      } else {
        mesh(r)(c).io.ctl.cal_type := 0.U
      }
    }
  }

  // pipeline ifm across each row
  for (r <- 0 until mesh_rows) {
    mesh(r).foldLeft(ShiftRegister(io.ifm.bits(r), r, 0.U, en)) {
      case (ifm, pe) =>
        pe.io.in_a := ifm
        pe.io.out_a
    }
  }

  // pipeline w across each column
  for (c <- 0 until mesh_columns) {
    meshT(c).foldLeft(ShiftRegister(io.w.bits(c), c, 0.U, en)) {
      case (w, pe) =>
        pe.io.in_b := w
        pe.io.out_b
    }
    meshT(c).foldLeft(1.B) {
      case (in, pe) =>
        pe.io.in_b_valid := in
        pe.io.out_b_valid
    }
  }

  // pipeline part sum across each column
  for (r <- 0 until mesh_rows) {
    for (c <- 0 until mesh_columns) {
      if (r != 0) {
        mesh(r)(c).io.in_c0 := mesh(r - 1)(c).io.out_d0
        mesh(r)(c).io.in_c1 := mesh(r - 1)(c).io.out_d1
      } else {
        mesh(r)(c).io.in_c0 := 0.U
        mesh(r)(c).io.in_c1 := 0.U
      }
    }
  }

  // ofm valid
  val ofm_valid = RegInit(0.U((mesh_rows * 2).W))
  when(en) {
    ofm_valid := ofm_valid ## ifm_handshake
  }
  for (c <- 0 until mesh_columns) {
    io.ofm(c).valid := Mux(
      dw_op,
      RegEnable(io.ifm.valid & io.ifm.ready & io.w.valid & io.w.ready, en),
      ofm_valid(mesh_rows + c - 1) && en
    )
  }

  // ofm acc_last
  val ofm_acc_last = RegInit(0.U((mesh_rows * 2).W))
  when(en) {
    ofm_acc_last := ofm_acc_last ## io.last_in
  }
  for (c <- 0 until mesh_columns) {
    io.ofm(c).bits.acc_last := Mux(
      dw_op,
      RegNext(io.last_in,0.U),
      ofm_acc_last(mesh_rows + c - 1) && io.ofm(c).valid
    )
  }

  // ofm ofm_finish
  val ofm_finish = RegInit(0.U((mesh_rows * 2).W))
  when(en) {
    ofm_finish := ofm_finish ## ifm_finish
  }
  for (c <- 0 until mesh_columns) {
    io.ofm(c).bits.finish := Mux(dw_op, RegNext(io.w_finish,0.U), ofm_finish(mesh_rows + c - 1))
  }

  // ofm addr
  val addr_cnt_sr = RegInit(
    Vec(mesh_rows, UInt(ofm_buffer_addr_w.W)),
    0.B.asTypeOf(Vec(mesh_rows, UInt(ofm_buffer_addr_w.W)))
  ) // shift reg
  when(ofm_valid(mesh_rows - 1) && en) {
    addr_cnt_sr(0) := addr_cnt_sr(0) + 1.U
    when(addr_cnt_sr(0) === (mesh_rows - 1).U) {
      addr_cnt_sr(0) := 0.U
    }
  }
  when(en) {
    for (i <- 1 until mesh_rows) {
      addr_cnt_sr(i) := addr_cnt_sr(i - 1)
    }
  }
  for (c <- 0 until mesh_columns) {
    io.ofm(c).bits.addr := Mux(dw_op, RegNext(cnt,0.U), addr_cnt_sr(c))
  }

  // ofm bits
  for (c <- 0 until mesh_columns) {
    io.ofm(c).bits.data0 := mesh(mesh_rows - 1)(c).io.out_d0
    io.ofm(c).bits.data1 := mesh(mesh_rows - 1)(c).io.out_d1
  }

  when(dw_op) {
    for (c <- 0 until mesh_columns) {
      mesh(0)(c).io.in_a   := io.ifm.bits(c)
      mesh(0)(c).io.in_b   := io.w.bits(c)
      io.ofm(c).bits.data0 := mesh(0)(c).io.out_d0
      io.ofm(c).bits.data1 := mesh(0)(c).io.out_d1
    }
  }

}

object mesh_gen extends App {
  new (chisel3.stage.ChiselStage)
    .execute(Array("--target-dir", "./verilog/mesh"), Seq(ChiselGeneratorAnnotation(() => new Mesh)))
}

object MeshGen extends App {
  (new chisel3.stage.ChiselStage)
    .execute(args, Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new Mesh)))
}
