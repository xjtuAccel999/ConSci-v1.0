import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.util._
import util_function._

class IfmBufferCtl extends Module with mesh_config with buffer_config {
  val io = IO(new Bundle {
    //axi-lite reg
    val cfg_gemm = Input(new cfg_gemm_io)
    //ifm buffer
    val ifm_read_port0 = Flipped(new ifm_r_io(IFM_BUFFER_DEPTH, IFM_BUFFER_WIDTH))
    val ifm_read_port1 = Flipped(new ifm_r_io(IFM_BUFFER_DEPTH, IFM_BUFFER_WIDTH))
    val task_done      = Input(Bool())
    // mesh
    val ifm     = Decoupled(Vec(mesh_rows, UInt(pe_data_w.W)))
    val last_in = Output(Bool())
  })
  val cfg_gemm = RegEnable(io.cfg_gemm, 0.U.asTypeOf(new cfg_gemm_io),dualEdge(io.cfg_gemm.en))
//  assert(io.im2col_format === 3.U, "currently im2col_format only support 3(ifm_int8)")
//  assert(io.cfg_gemm.ic % 32.U === 0.U, "ifm channel must be 32 aligned")
  // ################ const ################
  // ifm
  val low_w    = cfg_gemm.pad_left
  val high_w   = cfg_gemm.pad_left + cfg_gemm.iw
  val low_h    = cfg_gemm.pad_top
  val high_h   = cfg_gemm.pad_top + cfg_gemm.ih
  val ifm_pd_w = cfg_gemm.iw + cfg_gemm.pad_left + cfg_gemm.pad_right
  val ifm_pd_h = cfg_gemm.ih + cfg_gemm.pad_top + cfg_gemm.pad_bottom
  // im2col
  val ic_align = align(cfg_gemm.ic, 12, 32)
  val oc_align = align(cfg_gemm.oc, 12, 32)

  val ic_align_div32 = ic_align(11, 5) // / 32.U  11bit
  val oc_align_div32 = oc_align(11, 5) // / 32.U  11bit
  val dw_op          = cfg_gemm.op === 1.U
  val ifm_epoch      = oc_align_div32

  withReset(riseEdge(io.task_done) || reset.asBool) {
    // ################ def common ################
    val kw_cnt                   = RegInit(0.U(3.W))
    val kh_cnt                   = RegInit(0.U(3.W))
    val ifm_buffer_addr_offset   = RegInit(0.U(11.W)) // block_cnt_in_ic
    val ifm_buffer_addr_offset_r = ShiftRegister(ifm_buffer_addr_offset, 3, 0.U, ifm_out_ready)
    val block_h_cnt              = RegInit(0.U(5.W))
    val ofmblock_first_ifmblock  = RegInit(1.B)
    val ofm_col_cnt              = RegInit(0.U(11.W))

    // ################ def addr 0 ################
    val ow_cnt0            = RegInit(0.U(12.W))
    val oh_cnt0            = RegInit(0.U(12.W))
    val ow_cnt0_block_row0 = RegInit(0.U(12.W))
    val oh_cnt0_block_row0 = RegInit(0.U(12.W))

    // ################ def addr 1 ################
    val ow_cnt1_initial         = Wire(UInt(12.W))
    val ow_cnt1_initial_lut     = VecInit(for (i <- 1 to 32) yield (32 % i).U) // index 0 to 31
    val oh_cnt1_initial         = Wire(UInt(12.W))
    val oh_cnt1_initial_lut     = VecInit(for (i <- 1 to 32) yield (32 / i).U) // index 0 to 31
    val ow_cnt1                 = RegInit(ow_cnt1_initial)
    val oh_cnt1                 = RegInit(oh_cnt1_initial)
    val ow_cnt1_block_row0      = RegInit(ow_cnt1_initial)
    val oh_cnt1_block_row0      = RegInit(oh_cnt1_initial)
    val ow_cnt1_block_row0_next = RegInit(ow_cnt1_initial)
    val oh_cnt1_block_row0_next = RegInit(oh_cnt1_initial)

    // ################ conv condition ################
    // default && last_clk_of_one_block_p
    val last_clk_of_one_block_p       = (block_h_cnt === 31.U)
    val kernel_w_move_l               = ifm_buffer_addr_offset === (ic_align_div32 - 1.U) // ic_index ===
    val kernel_h_move_l               = (kw_cnt === cfg_gemm.kernel - 1.U) // && kernel_w_move_l
    val kernel_h_get_top_l            = (kh_cnt === cfg_gemm.kernel - 1.U) // && kernel_h_move_l
    val this_ofmblock_last_ifmblock_l = kernel_w_move_l && kernel_h_move_l && kernel_h_get_top_l
    val ifm_last_row_l                = oh_cnt1 >= cfg_gemm.oh || oh_cnt1 === (cfg_gemm.oh - 1.U) && ow_cnt1 === (cfg_gemm.ow - 1.U)
    val this_ofmcol_last_ifmblock_p   = last_clk_of_one_block_p && this_ofmblock_last_ifmblock_l && ifm_last_row_l

    // ################ depthwise conv condition ################
    // default && last_clk_of_one_block_p
    // val dw_kernel_w_move = 1
    val dw_kernel_h_move_l             = kernel_h_move_l
    val dw_kernel_h_get_top_l          = kernel_h_get_top_l
    val dw_ofm_row_last_ofm_block_l    = kernel_w_move_l
    val dw_compute_next_row_ofmblock_l = dw_kernel_h_get_top_l
    val dw_ofmblcok_last_ifmblock_l    = kernel_h_move_l && kernel_h_get_top_l
    val dw_ofmblcok_last_ifmblock_p    = dw_ofmblcok_last_ifmblock_l && last_clk_of_one_block_p
    val dw_next_ofm_col_p              = dw_ofmblcok_last_ifmblock_p && ifm_last_row_l
    val dw_last_ifmblock_p             = this_ofmcol_last_ifmblock_p
    val dw_last_ifmblock_level         = RegInit(0.B)
    when(dw_last_ifmblock_p) {
      dw_last_ifmblock_level := 1.B
    }

    // ################ compute common ################
    when(ifm_out_ready) {
      when(last_clk_of_one_block_p) {
        when(kernel_w_move_l || dw_op) {
          kw_cnt := kw_cnt + 1.U
          when(kernel_h_move_l) { // dw_kernel_h_move_l
            kw_cnt := 0.U
          }
        }
      }
    }
    when(ifm_out_ready) {
      when(last_clk_of_one_block_p && (kernel_w_move_l || dw_op) && kernel_h_move_l) {
        kh_cnt := kh_cnt + 1.U
        when(kernel_h_get_top_l) {
          kh_cnt := 0.U
        }
      }
    }

    when(ifm_out_ready) {
      when(dw_op) {
        when(last_clk_of_one_block_p && dw_kernel_h_move_l && dw_kernel_h_get_top_l && ifm_last_row_l) {
          ifm_buffer_addr_offset := ifm_buffer_addr_offset + 1.U
          when(dw_ofm_row_last_ofm_block_l) {
            ifm_buffer_addr_offset := 0.U
          }
        }
      }.elsewhen(last_clk_of_one_block_p) {
        ifm_buffer_addr_offset := ifm_buffer_addr_offset + 1.U
        when(kernel_w_move_l) {
          ifm_buffer_addr_offset := 0.U
        }
      }
    }

    when(ifm_out_ready) {
      block_h_cnt := block_h_cnt + 1.U
      when(last_clk_of_one_block_p) {
        block_h_cnt := 0.U
      }
    }

    // [Todo] set for 32 cycle to cnt ow_cnt1
    when(ifm_out_ready) {
      when(last_clk_of_one_block_p) {
        ofmblock_first_ifmblock := 0.B
        when(this_ofmblock_last_ifmblock_l || dw_op && dw_ofmblcok_last_ifmblock_l) {
          ofmblock_first_ifmblock := 1.B
        }
      }
    }

    when(ifm_out_ready) {
      when(this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p) {
        ofm_col_cnt := ofm_col_cnt + 1.U
      }
    }

    // ################ compute addr 0 ################
    val ow_cnt0_block_row0_temp =
      Mux(this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p, 0.U, Mux(ow_cnt1 === cfg_gemm.ow - 1.U, 0.U, ow_cnt1 + 1.U))
    when(ifm_out_ready) {
      ow_cnt0 := ow_cnt0 + 1.U
      when(last_clk_of_one_block_p) {
        ow_cnt0 := ow_cnt0_block_row0
        when(this_ofmblock_last_ifmblock_l || dw_op && dw_ofmblcok_last_ifmblock_l) { // when fetch last one clk in this block row
          ow_cnt0_block_row0 := ow_cnt0_block_row0_temp
          ow_cnt0            := ow_cnt0_block_row0_temp
        }
      }.elsewhen(ow_cnt0 === cfg_gemm.ow - 1.U) {
        ow_cnt0 := 0.U
      }
    }

    val oh_cnt0_block_row0_temp =
      Mux(this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p, 0.U, Mux(ow_cnt1 === cfg_gemm.ow - 1.U, oh_cnt1 + 1.U, oh_cnt1))
    when(ifm_out_ready) {
      when(last_clk_of_one_block_p) {
        oh_cnt0 := oh_cnt0_block_row0
        when(this_ofmblock_last_ifmblock_l || dw_op && dw_ofmblcok_last_ifmblock_l) {
          oh_cnt0_block_row0 := oh_cnt0_block_row0_temp
          oh_cnt0            := oh_cnt0_block_row0_temp
        }
      }.elsewhen(ow_cnt0 === cfg_gemm.ow - 1.U) {
        oh_cnt0 := oh_cnt0 + 1.U
      }
    }

    val iw_pd_cnt0              = RegEnable(ow_cnt0 * cfg_gemm.stride + kw_cnt, 0.U, ifm_out_ready)
    val ih_pd_cnt0              = RegEnable(oh_cnt0 * cfg_gemm.stride + kh_cnt, 0.U, ifm_out_ready)
    val iw_cnt0                 = RegEnable(Mux(iw_pd_cnt0 < low_w || iw_pd_cnt0 >= high_w, 0.U, iw_pd_cnt0 - low_w), 0.U, ifm_out_ready)
    val ih_cnt0                 = RegEnable(Mux(ih_pd_cnt0 < low_h || ih_pd_cnt0 >= high_h, 0.U, ih_pd_cnt0 - low_h), 0.U, ifm_out_ready)
    val ifm_buffer_addr_base0_p = RegEnable(ih_cnt0 * cfg_gemm.iw + iw_cnt0, 0.U, ifm_out_ready)
    val ifm_buffer_addr_base0   = ifm_buffer_addr_base0_p * ic_align_div32
    val ifm_buffer_addr0        = RegEnable(ifm_buffer_addr_base0 + ifm_buffer_addr_offset_r, 0.U, ifm_out_ready)

    // ################ compute addr 1 ################
    ow_cnt1_initial := Mux(cfg_gemm.ow > 32.U, 32.U, ow_cnt1_initial_lut(cfg_gemm.ow - 1.U))
    val ow_cnt1_block_row0_next_temp = Wire(UInt(12.W))
    ow_cnt1_block_row0_next_temp := ow_cnt1_block_row0_next
    when(ifm_out_ready) {
      when(this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p) {
        ow_cnt1_block_row0_next      := ow_cnt1_initial
        ow_cnt1_block_row0_next_temp := ow_cnt1_initial
      }.elsewhen(ofmblock_first_ifmblock) {
        ow_cnt1_block_row0_next      := ow_cnt1_block_row0_next + 2.U
        ow_cnt1_block_row0_next_temp := ow_cnt1_block_row0_next + 2.U
        when(cfg_gemm.ow === 1.U) {
          ow_cnt1_block_row0_next      := 0.U
          ow_cnt1_block_row0_next_temp := 0.U
        }.elsewhen(ow_cnt1_block_row0_next === cfg_gemm.ow - 1.U) {
          ow_cnt1_block_row0_next      := 1.U
          ow_cnt1_block_row0_next_temp := 1.U
        }.elsewhen(ow_cnt1_block_row0_next === cfg_gemm.ow - 2.U) {
          ow_cnt1_block_row0_next      := 0.U
          ow_cnt1_block_row0_next_temp := 0.U
        }
      }

    }
    when(ifm_out_ready) {
      ow_cnt1 := ow_cnt1 + 1.U
      when(last_clk_of_one_block_p) {
        ow_cnt1 := ow_cnt1_block_row0
        when(this_ofmblock_last_ifmblock_l || dw_op && dw_ofmblcok_last_ifmblock_l) {
          ow_cnt1_block_row0 := Mux(
            this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p,
            ow_cnt1_initial,
            Mux(cfg_gemm.kernel > 1.U || (!dw_op && ic_align_div32 > 1.U), ow_cnt1_block_row0_next, ow_cnt1_block_row0_next_temp)
          )
          ow_cnt1 := Mux(
            this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p,
            ow_cnt1_initial,
            Mux(cfg_gemm.kernel > 1.U || (!dw_op && ic_align_div32 > 1.U), ow_cnt1_block_row0_next, ow_cnt1_block_row0_next_temp)
          )
        }
      }.elsewhen(ow_cnt1 === cfg_gemm.ow - 1.U) {
        ow_cnt1 := 0.U
      }
    }

    oh_cnt1_initial := Mux(cfg_gemm.ow > 32.U, 0.U, oh_cnt1_initial_lut(cfg_gemm.ow - 1.U))
    val oh_cnt1_block_row0_next_temp = Wire(UInt(12.W))
    oh_cnt1_block_row0_next_temp := oh_cnt1_block_row0_next
    when(ifm_out_ready) {
      when(this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p) {
        oh_cnt1_block_row0_next      := oh_cnt1_initial
        oh_cnt1_block_row0_next_temp := oh_cnt1_initial
      }.elsewhen(ofmblock_first_ifmblock) {
        when(cfg_gemm.ow === 1.U) {
          oh_cnt1_block_row0_next      := oh_cnt1_block_row0_next + 2.U
          oh_cnt1_block_row0_next_temp := oh_cnt1_block_row0_next + 2.U
        }.elsewhen(
          ow_cnt1_block_row0_next === cfg_gemm.ow - 1.U ||
            ow_cnt1_block_row0_next === cfg_gemm.ow - 2.U
        ) {
          oh_cnt1_block_row0_next      := oh_cnt1_block_row0_next + 1.U
          oh_cnt1_block_row0_next_temp := oh_cnt1_block_row0_next + 1.U
        }
      }
    }
    when(ifm_out_ready) {
      when(last_clk_of_one_block_p) {
        oh_cnt1 := oh_cnt1_block_row0
        when(this_ofmblock_last_ifmblock_l || dw_op && dw_ofmblcok_last_ifmblock_l) {
          oh_cnt1_block_row0 := Mux(
            this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p,
            oh_cnt1_initial,
            Mux(cfg_gemm.kernel > 1.U || (!dw_op && ic_align_div32 > 1.U), oh_cnt1_block_row0_next, oh_cnt1_block_row0_next_temp)
          )
          oh_cnt1 := Mux(
            this_ofmcol_last_ifmblock_p || dw_op && dw_next_ofm_col_p,
            oh_cnt1_initial,
            Mux(cfg_gemm.kernel > 1.U || (!dw_op && ic_align_div32 > 1.U), oh_cnt1_block_row0_next, oh_cnt1_block_row0_next_temp)
          )
        }
      }.elsewhen(ow_cnt1 === cfg_gemm.ow - 1.U) {
        oh_cnt1 := oh_cnt1 + 1.U
      }
    }

    val iw_pd_cnt1              = RegEnable(ow_cnt1 * cfg_gemm.stride + kw_cnt, 0.U, ifm_out_ready)
    val ih_pd_cnt1              = RegEnable(oh_cnt1 * cfg_gemm.stride + kh_cnt, 0.U, ifm_out_ready)
    val iw_cnt1                 = RegEnable(Mux(iw_pd_cnt1 < low_w || iw_pd_cnt1 >= high_w, 0.U, iw_pd_cnt1 - low_w), 0.U, ifm_out_ready)
    val ih_cnt1                 = RegEnable(Mux(ih_pd_cnt1 < low_h || ih_pd_cnt1 >= high_h, 0.U, ih_pd_cnt1 - low_h), 0.U, ifm_out_ready)
    val ifm_buffer_addr_base1_p = RegEnable(ih_cnt1 * cfg_gemm.iw + iw_cnt1, 0.U, ifm_out_ready)
    val ifm_buffer_addr_base1   = ifm_buffer_addr_base1_p * ic_align_div32
    val ifm_buffer_addr1        = RegEnable(ifm_buffer_addr_base1 + ifm_buffer_addr_offset_r, 0.U, ifm_out_ready)

    // ################ ifm buffer read ################
    val ifm_buffer_addr0_reg = RegEnable(ifm_buffer_addr0, 0.U, ifm_out_ready)
    val ifm_buffer_addr1_reg = RegEnable(ifm_buffer_addr1, 0.U, ifm_out_ready)
    when(ifm_out_ready) {
      io.ifm_read_port0.raddr := ifm_buffer_addr0
      io.ifm_read_port1.raddr := ifm_buffer_addr1
    }.otherwise {
      io.ifm_read_port0.raddr := ifm_buffer_addr0_reg
      io.ifm_read_port1.raddr := ifm_buffer_addr1_reg
    }
    //    io.ifm_read_port0.raddr := ifm_buffer_addr0
    //    io.ifm_read_port1.raddr := ifm_buffer_addr1
    io.ifm_read_port0.ren := io.task_done
    io.ifm_read_port1.ren := io.task_done

    // ################ mesh write ################

    val ifm_all_finish  = ShiftRegister(ofm_col_cnt > ifm_epoch, 5, 0.B, ifm_out_ready)
    val oh_cnt0_surplus = ShiftRegister(oh_cnt0 >= cfg_gemm.oh, 5, 0.B, ifm_out_ready)
    val oh_cnt1_surplus = ShiftRegister(oh_cnt1 >= cfg_gemm.oh, 5, 0.B, ifm_out_ready)

    val write_padding_0 = ShiftRegister(
      iw_pd_cnt0 < low_w || iw_pd_cnt0 >= high_w || ih_pd_cnt0 < low_h || ih_pd_cnt0 >= high_h,
      4,
      0.B,
      ifm_out_ready
    )
    val write_padding_1 = ShiftRegister(
      iw_pd_cnt1 < low_w || iw_pd_cnt1 >= high_w || ih_pd_cnt1 < low_h || ih_pd_cnt1 >= high_h,
      4,
      0.B,
      ifm_out_ready
    )

    val read_zero_0 = write_padding_0 || oh_cnt0_surplus || ifm_all_finish
    val read_zero_1 = write_padding_1 || oh_cnt1_surplus || ifm_all_finish

    io.last_in := ShiftRegister(
      Mux(dw_op, dw_ofmblcok_last_ifmblock_l, this_ofmblock_last_ifmblock_l),
      ifm_delay,
      0.B,
      ifm_out_ready
    )
    dontTouch(io.last_in)

    val ifm_out_reg = Reg(Vec(mesh_rows, UInt(pe_data_w.W)))
    when(ifm_valid && (!ifm_out_valid || io.ifm.ready)) {
      for (i <- 0 until mesh_rows) {
        ifm_out_reg(i) := (io.ifm_read_port1.rdata(i * 8 + 7, i * 8) & Fill(8, !read_zero_1)) ## 0.U(8.W) ##
          (io.ifm_read_port0.rdata(i * 8 + 7, i * 8) & Fill(8, !read_zero_0))
      }
    }
    for (i <- 0 until mesh_rows) {
      io.ifm.bits(i) := ifm_out_reg(i)
    }

  }

  lazy val ifm_out_valid = RegInit(0.B)
  when(io.ifm.ready || !ifm_out_valid) {
    ifm_out_valid := ifm_valid
  }

  lazy val ifm_out_ready = (!ifm_out_valid || io.ifm.ready) && io.task_done
  lazy val ifm_valid     = ShiftRegister(io.task_done, ifm_delay - 1, 0.B, ifm_out_ready)

  io.ifm.valid := ifm_out_valid
}

object IfmBufferCtl_gen extends App {
  new (chisel3.stage.ChiselStage)
    .execute(Array("--target-dir", "./verilog/ifmbufferctl"), Seq(ChiselGeneratorAnnotation(() => new IfmBufferCtl)))
}
