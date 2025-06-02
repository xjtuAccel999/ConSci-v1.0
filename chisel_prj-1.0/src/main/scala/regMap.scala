import chisel3._
import chisel3.util._
import util_function._

class cfg_alu_io extends Bundle with axi_config {
  val math_en           = Bool()
  val math_src_num      = UInt(2.W)
  val math_mul_src1_sel = UInt(3.W)
  val math_add_src0_sel = UInt(3.W)
  val math_add_src1_sel = UInt(3.W)
  val math_sub_en       = Bool()
  val math_add_en       = Bool()
  val math_mul_en       = Bool()
  val math_max_en       = Bool()
  val math_min_en       = Bool()
  val math_op           = UInt(4.W)
  val act_en            = Bool()
  val act_src_sel       = UInt(2.W)
  val act_dst_sel       = UInt(2.W)
  val veclen_ch0        = UInt(32.W)
  val src0_addr_ch0     = UInt(AXI_ADDR_WIDTH.W)
  val src1_addr_ch0     = UInt(AXI_ADDR_WIDTH.W)
  val dst_addr_ch0      = UInt(AXI_ADDR_WIDTH.W)
  val veclen_ch1        = UInt(32.W)
  val src0_addr_ch1     = UInt(AXI_ADDR_WIDTH.W)
  val src1_addr_ch1     = UInt(AXI_ADDR_WIDTH.W)
  val dst_addr_ch1      = UInt(AXI_ADDR_WIDTH.W)
  val math_alpha        = UInt(32.W)
  val math_beta         = UInt(32.W)
  val innerprod_en      = Bool()
  //activation function parameter
  val act_coefficient_a = Vec(5, UInt(32.W))
  val act_coefficient_b = Vec(5, UInt(32.W))
  val act_coefficient_c = Vec(5, UInt(32.W))
  val act_range         = Vec(4, UInt(32.W))
  val act_func_prop     = UInt(2.W) // 00:monotonic function 01:odd function  10:even function
  val act_op            = UInt(8.W)

}

class cfg_pool_io extends Bundle with axi_config {
  val en          = Bool()
  val op          = UInt(2.W)
  val kernel_w    = UInt(2.W)
  val kernel_h    = UInt(2.W)
  val stride_w    = UInt(2.W)
  val stride_h    = UInt(2.W)
  val src_sel     = UInt(2.W)
  val pool_bottom = UInt(2.W)
  val pool_top    = UInt(2.W)
  val pool_right  = UInt(2.W)
  val pool_left   = UInt(2.W)
  val pad_mode    = Bool()
  val iw          = UInt(12.W)
  val ih          = UInt(12.W)
  val ic          = UInt(12.W)
  val icstep      = UInt(26.W)
  val ow          = UInt(12.W)
  val oh          = UInt(12.W)
  val oc          = UInt(12.W)
  val ocstep      = UInt(26.W)
  val ifm_addr    = UInt(AXI_ADDR_WIDTH.W)
  val ofm_addr    = UInt(AXI_ADDR_WIDTH.W)
  val pad_value   = UInt(32.W)
}

class cfg_gemm_io extends Bundle with axi_config {
  val en           = Bool()
  val op           = UInt(2.W) // now support 0=conv, 1=dw
  val kernel       = UInt(3.W)
  val stride       = UInt(3.W)
  val pad_mode     = UInt(2.W)
  val pad_left     = UInt(2.W)
  val pad_right    = UInt(2.W)
  val pad_top      = UInt(2.W)
  val pad_bottom   = UInt(2.W)
  val bias_en      = Bool()
  val oscale_en    = Bool()
  val requant_en   = Bool()
  val layout_en    = Bool()
  val div_ifm_c_en = Bool()
  val quant_data   = UInt(32.W)
  val requant_data = UInt(32.W)
  val dequant_addr = UInt(AXI_ADDR_WIDTH.W)
  val bias_addr    = UInt(AXI_ADDR_WIDTH.W)
  val iw           = UInt(12.W)
  val ih           = UInt(12.W)
  val ic           = UInt(12.W)
  val icstep       = UInt(26.W)
  val ow           = UInt(12.W)
  val oh           = UInt(12.W)
  val oc           = UInt(12.W)
  val ocstep       = UInt(26.W)
  val wgt_len      = UInt(32.W)
  val ifm_addr     = UInt(AXI_ADDR_WIDTH.W)
  val wgt_addr     = UInt(AXI_ADDR_WIDTH.W)
  val ofm_addr     = UInt(AXI_ADDR_WIDTH.W)
  val div_ifm_c    = UInt(12.W)
}


class cfg_reset_io extends Bundle with axi_config {
  val soft_reset           = Bool()
}

class reg_alu_io extends Bundle with axi_config {
  val mathfunc_ctrl_reg  = UInt(32.W)
  val actfunc_ctrl_reg   = UInt(32.W)
  val alu_veclen_ch0_reg = UInt(32.W)
  val src0_addr_ch0_reg  = UInt(AXI_ADDR_WIDTH.W)
  val src1_addr_ch0_reg  = UInt(AXI_ADDR_WIDTH.W)
  val dst_addr_ch0_reg   = UInt(AXI_ADDR_WIDTH.W)
  val alu_veclen_ch1_reg = UInt(32.W)
  val src0_addr_ch1_reg  = UInt(AXI_ADDR_WIDTH.W)
  val src1_addr_ch1_reg  = UInt(AXI_ADDR_WIDTH.W)
  val dst_addr_ch1_reg   = UInt(AXI_ADDR_WIDTH.W)
  val math_alpha_reg     = UInt(32.W)
  val math_beta_reg      = UInt(32.W)
//  val act_alpha_reg      = UInt(32.W)
//  val act_beta_reg       = UInt(32.W)
  val act_coefficient_a_reg = Vec(5, UInt(32.W))
  val act_coefficient_b_reg = Vec(5, UInt(32.W))
  val act_coefficient_c_reg = Vec(5, UInt(32.W))
  val act_range_reg         = Vec(4, UInt(32.W))
  val innerprod_ctrl_reg    = UInt(32.W)
}

class reg_pool_io extends Bundle with axi_config {
  val pool_ctrl_reg         = UInt(32.W)
  val pool_shape_ic_reg     = UInt(32.W)
  val pool_shape_iwh_reg    = UInt(32.W)
  val pool_shape_icstep_reg = UInt(32.W)
  val pool_shape_oc_reg     = UInt(32.W)
  val pool_shape_owh_reg    = UInt(32.W)
  val pool_shape_ocstep_reg = UInt(32.W)
  val pool_ifm_addr_reg     = UInt(AXI_ADDR_WIDTH.W)
  val pool_ofm_addr_reg     = UInt(AXI_ADDR_WIDTH.W)
  val pool_pad_value_reg    = UInt(32.W)
}

class reg_gemm_io extends Bundle with axi_config {
  val gemm_ctrl_reg       = UInt(32.W)
  val quant_data_reg      = UInt(32.W)
  val requant_data_reg    = UInt(32.W)
  val dequant_addr_reg    = UInt(AXI_ADDR_WIDTH.W)
  val bias_addr_reg       = UInt(AXI_ADDR_WIDTH.W)
  val ifm_shape_c_reg     = UInt(32.W)
  val ifm_shape_wh_reg    = UInt(32.W)
  val ifm_shape_cstep_reg = UInt(32.W)
  val ofm_shape_c_reg     = UInt(32.W)
  val ofm_shape_wh_reg    = UInt(32.W)
  val ofm_shape_cstep_reg = UInt(32.W)
  val wgt_len_reg         = UInt(32.W)
  val ifm_baseaddr_reg    = UInt(AXI_ADDR_WIDTH.W)
  val wgt_baseaddr_reg    = UInt(AXI_ADDR_WIDTH.W)
  val ofm_baseaddr_reg    = UInt(AXI_ADDR_WIDTH.W)
  val div_ifm_c_reg       = UInt(32.W)
}

class reg_reset_io extends Bundle with axi_config {
  val reset_reg       = UInt(32.W)
}
class regMap_sel extends Module with hw_config {
  val io = IO(new Bundle() {
    //input
    val reg_alu_i    = if (SIM_MODE) Some(Input(new reg_alu_io)) else None
    val reg_pool_i   = if (SIM_MODE) Some(Input(new reg_pool_io)) else None
    val reg_gemm_i   = if (SIM_MODE) Some(Input(new reg_gemm_io)) else None
    val reg_reset_i  = if (SIM_MODE) Some(Input(new reg_reset_io)) else None
    val axi_lite_i   = if (!SIM_MODE) Some(Input(Vec(128, UInt(ACCEL_AXI_LITE_DATA_WIDTH.W)))) else None
    //output
    val reg_alu_o    = Output(new reg_alu_io)
    val reg_pool_o   = Output(new reg_pool_io)
    val reg_gemm_o   = Output(new reg_gemm_io)
    val reg_reset_o   = Output(new reg_reset_io)
  })
  if (SIM_MODE) {
    io.reg_alu_o    <> io.reg_alu_i.get
    io.reg_pool_o   <> io.reg_pool_i.get
    io.reg_gemm_o   <> io.reg_gemm_i.get
    io.reg_reset_o  <> io.reg_reset_i.get
  } else {
    io.reg_alu_o.mathfunc_ctrl_reg        <> io.axi_lite_i.get(0)
    io.reg_alu_o.actfunc_ctrl_reg         <> io.axi_lite_i.get(1)
    io.reg_alu_o.alu_veclen_ch0_reg       <> io.axi_lite_i.get(2)
    io.reg_alu_o.src0_addr_ch0_reg        <> io.axi_lite_i.get(3)
    io.reg_alu_o.src1_addr_ch0_reg        <> io.axi_lite_i.get(4)
    io.reg_alu_o.dst_addr_ch0_reg         <> io.axi_lite_i.get(5)
    io.reg_alu_o.alu_veclen_ch1_reg       <> io.axi_lite_i.get(6)
    io.reg_alu_o.src0_addr_ch1_reg        <> io.axi_lite_i.get(7)
    io.reg_alu_o.src1_addr_ch1_reg        <> io.axi_lite_i.get(8)
    io.reg_alu_o.dst_addr_ch1_reg         <> io.axi_lite_i.get(9)
    io.reg_alu_o.math_alpha_reg           <> io.axi_lite_i.get(10)
    io.reg_alu_o.math_beta_reg            <> io.axi_lite_i.get(11)
    io.reg_alu_o.act_range_reg(0)         <> io.axi_lite_i.get(12)
    io.reg_alu_o.act_range_reg(1)         <> io.axi_lite_i.get(13)
    io.reg_alu_o.act_range_reg(2)         <> io.axi_lite_i.get(14)
    io.reg_alu_o.act_range_reg(3)         <> io.axi_lite_i.get(15)
    io.reg_alu_o.act_coefficient_a_reg(0) <> io.axi_lite_i.get(16)
    io.reg_alu_o.act_coefficient_a_reg(1) <> io.axi_lite_i.get(17)
    io.reg_alu_o.act_coefficient_a_reg(2) <> io.axi_lite_i.get(18)
    io.reg_alu_o.act_coefficient_a_reg(3) <> io.axi_lite_i.get(19)
    io.reg_alu_o.act_coefficient_a_reg(4) <> io.axi_lite_i.get(20)
    io.reg_alu_o.act_coefficient_b_reg(0) <> io.axi_lite_i.get(21)
    io.reg_alu_o.act_coefficient_b_reg(1) <> io.axi_lite_i.get(22)
    io.reg_alu_o.act_coefficient_b_reg(2) <> io.axi_lite_i.get(23)
    io.reg_alu_o.act_coefficient_b_reg(3) <> io.axi_lite_i.get(24)
    io.reg_alu_o.act_coefficient_b_reg(4) <> io.axi_lite_i.get(25)
    io.reg_alu_o.act_coefficient_c_reg(0) <> io.axi_lite_i.get(26)
    io.reg_alu_o.act_coefficient_c_reg(1) <> io.axi_lite_i.get(27)
    io.reg_alu_o.act_coefficient_c_reg(2) <> io.axi_lite_i.get(28)
    io.reg_alu_o.act_coefficient_c_reg(3) <> io.axi_lite_i.get(29)
    io.reg_alu_o.act_coefficient_c_reg(4) <> io.axi_lite_i.get(30)
    io.reg_alu_o.innerprod_ctrl_reg       <> io.axi_lite_i.get(31)

    io.reg_pool_o.pool_ctrl_reg         <> io.axi_lite_i.get(40)
    io.reg_pool_o.pool_shape_ic_reg     <> io.axi_lite_i.get(41)
    io.reg_pool_o.pool_shape_iwh_reg    <> io.axi_lite_i.get(42)
    io.reg_pool_o.pool_shape_icstep_reg <> io.axi_lite_i.get(43)
    io.reg_pool_o.pool_shape_oc_reg     <> io.axi_lite_i.get(44)
    io.reg_pool_o.pool_shape_owh_reg    <> io.axi_lite_i.get(45)
    io.reg_pool_o.pool_shape_ocstep_reg <> io.axi_lite_i.get(46)
    io.reg_pool_o.pool_ifm_addr_reg     <> io.axi_lite_i.get(47)
    io.reg_pool_o.pool_ofm_addr_reg     <> io.axi_lite_i.get(48)
    io.reg_pool_o.pool_pad_value_reg    <> io.axi_lite_i.get(49)

    io.reg_gemm_o.gemm_ctrl_reg       <> io.axi_lite_i.get(60)
    io.reg_gemm_o.quant_data_reg      <> io.axi_lite_i.get(61)
    io.reg_gemm_o.requant_data_reg    <> io.axi_lite_i.get(62)
    io.reg_gemm_o.dequant_addr_reg    <> io.axi_lite_i.get(63)
    io.reg_gemm_o.bias_addr_reg       <> io.axi_lite_i.get(64)
    io.reg_gemm_o.ifm_shape_c_reg     <> io.axi_lite_i.get(65)
    io.reg_gemm_o.ifm_shape_wh_reg    <> io.axi_lite_i.get(66)
    io.reg_gemm_o.ifm_shape_cstep_reg <> io.axi_lite_i.get(67)
    io.reg_gemm_o.ofm_shape_c_reg     <> io.axi_lite_i.get(68)
    io.reg_gemm_o.ofm_shape_wh_reg    <> io.axi_lite_i.get(69)
    io.reg_gemm_o.ofm_shape_cstep_reg <> io.axi_lite_i.get(70)
    io.reg_gemm_o.wgt_len_reg         <> io.axi_lite_i.get(71)
    io.reg_gemm_o.ifm_baseaddr_reg    <> io.axi_lite_i.get(72)
    io.reg_gemm_o.wgt_baseaddr_reg    <> io.axi_lite_i.get(73)
    io.reg_gemm_o.ofm_baseaddr_reg    <> io.axi_lite_i.get(74)
    io.reg_gemm_o.div_ifm_c_reg       <> io.axi_lite_i.get(75)

    io.reg_reset_o.reset_reg          <> io.axi_lite_i.get(76)

  }
}

class regMap extends Module with axi_config with hw_config {
  val io = IO(new Bundle() {
    val reg_alu    = if (SIM_MODE) Some(Input(new reg_alu_io)) else None
    val reg_pool   = if (SIM_MODE) Some(Input(new reg_pool_io)) else None
    val reg_gemm   = if (SIM_MODE) Some(Input(new reg_gemm_io)) else None
    val reg_reset  = if (SIM_MODE) Some(Input(new reg_reset_io)) else None
    val axi_lite   = if (!SIM_MODE) Some(Input(Vec(128, UInt(ACCEL_AXI_LITE_DATA_WIDTH.W)))) else None

    val cfg_alu    = Output(new cfg_alu_io)
    val cfg_pool   = Output(new cfg_pool_io)
    val cfg_gemm   = Output(new cfg_gemm_io)
    val cfg_reset  = Output(new cfg_reset_io)
  })

  val reg_in = Module(new regMap_sel)
  if (SIM_MODE) {
    reg_in.io.reg_alu_i.get    <> io.reg_alu.get
    reg_in.io.reg_pool_i.get   <> io.reg_pool.get
    reg_in.io.reg_gemm_i.get   <> io.reg_gemm.get
    reg_in.io.reg_reset_i.get  <> io.reg_reset.get
  } else {
    reg_in.io.axi_lite_i.get <> io.axi_lite.get
  }

  //ALU CFG
  io.cfg_alu.math_en           := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(0)
  io.cfg_alu.math_src_num      := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(2, 1)
  io.cfg_alu.math_mul_src1_sel := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(5, 3)
  io.cfg_alu.math_add_src0_sel := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(8, 6)
  io.cfg_alu.math_add_src1_sel := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(11, 9)
  io.cfg_alu.math_sub_en       := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(12)
  io.cfg_alu.math_add_en       := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(13)
  io.cfg_alu.math_mul_en       := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(14)
  io.cfg_alu.math_max_en       := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(15)
  io.cfg_alu.math_min_en       := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(16)
  io.cfg_alu.math_op           := reg_in.io.reg_alu_o.mathfunc_ctrl_reg(26, 23)

  io.cfg_alu.act_en        := reg_in.io.reg_alu_o.actfunc_ctrl_reg(0)
  io.cfg_alu.act_func_prop := reg_in.io.reg_alu_o.actfunc_ctrl_reg(2, 1) // 00:monotonic function 01:odd function  10:even function
  io.cfg_alu.act_src_sel   := reg_in.io.reg_alu_o.actfunc_ctrl_reg(4, 3)
  io.cfg_alu.act_dst_sel   := reg_in.io.reg_alu_o.actfunc_ctrl_reg(6, 5)
  io.cfg_alu.act_op        := reg_in.io.reg_alu_o.actfunc_ctrl_reg(23, 16)

  io.cfg_alu.veclen_ch0    := reg_in.io.reg_alu_o.alu_veclen_ch0_reg
  io.cfg_alu.src0_addr_ch0 := reg_in.io.reg_alu_o.src0_addr_ch0_reg
  io.cfg_alu.src1_addr_ch0 := reg_in.io.reg_alu_o.src1_addr_ch0_reg
  io.cfg_alu.dst_addr_ch0  := reg_in.io.reg_alu_o.dst_addr_ch0_reg
  io.cfg_alu.veclen_ch1    := reg_in.io.reg_alu_o.alu_veclen_ch1_reg
  io.cfg_alu.src0_addr_ch1 := reg_in.io.reg_alu_o.src0_addr_ch1_reg
  io.cfg_alu.src1_addr_ch1 := reg_in.io.reg_alu_o.src1_addr_ch1_reg
  io.cfg_alu.dst_addr_ch1  := reg_in.io.reg_alu_o.dst_addr_ch1_reg

  io.cfg_alu.math_alpha := reg_in.io.reg_alu_o.math_alpha_reg
  io.cfg_alu.math_beta  := reg_in.io.reg_alu_o.math_beta_reg
//  io.cfg_alu.act_alpha := reg_in.io.reg_alu_o.act_alpha_reg
//  io.cfg_alu.act_beta := reg_in.io.reg_alu_o.act_beta_reg
  io.cfg_alu.act_coefficient_a := reg_in.io.reg_alu_o.act_coefficient_a_reg
  io.cfg_alu.act_coefficient_b := reg_in.io.reg_alu_o.act_coefficient_b_reg
  io.cfg_alu.act_coefficient_c := reg_in.io.reg_alu_o.act_coefficient_c_reg
  io.cfg_alu.act_range         := reg_in.io.reg_alu_o.act_range_reg
  io.cfg_alu.innerprod_en      := reg_in.io.reg_alu_o.innerprod_ctrl_reg(0)

  //POOL CFG
  io.cfg_pool.en          := reg_in.io.reg_pool_o.pool_ctrl_reg(0)
  io.cfg_pool.op          := reg_in.io.reg_pool_o.pool_ctrl_reg(2, 1)
  io.cfg_pool.kernel_w    := reg_in.io.reg_pool_o.pool_ctrl_reg(4, 3)
  io.cfg_pool.kernel_h    := reg_in.io.reg_pool_o.pool_ctrl_reg(6, 5)
  io.cfg_pool.stride_w    := reg_in.io.reg_pool_o.pool_ctrl_reg(8, 7)
  io.cfg_pool.stride_h    := reg_in.io.reg_pool_o.pool_ctrl_reg(10, 9)
  io.cfg_pool.src_sel     := reg_in.io.reg_pool_o.pool_ctrl_reg(12, 11)
  io.cfg_pool.pool_bottom := reg_in.io.reg_pool_o.pool_ctrl_reg(17, 16)
  io.cfg_pool.pool_top    := reg_in.io.reg_pool_o.pool_ctrl_reg(19, 18)
  io.cfg_pool.pool_right  := reg_in.io.reg_pool_o.pool_ctrl_reg(21, 20)
  io.cfg_pool.pool_left   := reg_in.io.reg_pool_o.pool_ctrl_reg(23, 22)
  io.cfg_pool.pad_mode    := reg_in.io.reg_pool_o.pool_ctrl_reg(24)

  io.cfg_pool.ic     := reg_in.io.reg_pool_o.pool_shape_ic_reg(11, 0)
  io.cfg_pool.iw     := reg_in.io.reg_pool_o.pool_shape_iwh_reg(27, 16)
  io.cfg_pool.ih     := reg_in.io.reg_pool_o.pool_shape_iwh_reg(11, 0)
  io.cfg_pool.icstep := reg_in.io.reg_pool_o.pool_shape_icstep_reg(25, 0)
  io.cfg_pool.oc     := reg_in.io.reg_pool_o.pool_shape_oc_reg(11, 0)
  io.cfg_pool.ow     := reg_in.io.reg_pool_o.pool_shape_owh_reg(27, 16)
  io.cfg_pool.oh     := reg_in.io.reg_pool_o.pool_shape_owh_reg(11, 0)
  io.cfg_pool.ocstep := reg_in.io.reg_pool_o.pool_shape_ocstep_reg(25, 0)

  io.cfg_pool.ifm_addr  := reg_in.io.reg_pool_o.pool_ifm_addr_reg
  io.cfg_pool.ofm_addr  := reg_in.io.reg_pool_o.pool_ofm_addr_reg
  io.cfg_pool.pad_value := reg_in.io.reg_pool_o.pool_pad_value_reg

  //GEMM CFG
  io.cfg_gemm.en           := reg_in.io.reg_gemm_o.gemm_ctrl_reg(0)
  io.cfg_gemm.op           := reg_in.io.reg_gemm_o.gemm_ctrl_reg(2, 1)
  io.cfg_gemm.kernel       := reg_in.io.reg_gemm_o.gemm_ctrl_reg(5, 3)
  io.cfg_gemm.stride       := reg_in.io.reg_gemm_o.gemm_ctrl_reg(8, 6)
  io.cfg_gemm.pad_mode     := reg_in.io.reg_gemm_o.gemm_ctrl_reg(10, 9)
  io.cfg_gemm.pad_left     := reg_in.io.reg_gemm_o.gemm_ctrl_reg(12, 11)
  io.cfg_gemm.pad_right    := reg_in.io.reg_gemm_o.gemm_ctrl_reg(14, 13)
  io.cfg_gemm.pad_top      := reg_in.io.reg_gemm_o.gemm_ctrl_reg(16, 15)
  io.cfg_gemm.pad_bottom   := reg_in.io.reg_gemm_o.gemm_ctrl_reg(18, 17)
  io.cfg_gemm.bias_en      := reg_in.io.reg_gemm_o.gemm_ctrl_reg(19)
  io.cfg_gemm.requant_en   := reg_in.io.reg_gemm_o.gemm_ctrl_reg(20)
  io.cfg_gemm.layout_en    := reg_in.io.reg_gemm_o.gemm_ctrl_reg(21)
  io.cfg_gemm.oscale_en    := reg_in.io.reg_gemm_o.gemm_ctrl_reg(22)
  io.cfg_gemm.div_ifm_c_en := reg_in.io.reg_gemm_o.gemm_ctrl_reg(23)

  io.cfg_gemm.quant_data   := reg_in.io.reg_gemm_o.quant_data_reg
  io.cfg_gemm.requant_data := reg_in.io.reg_gemm_o.requant_data_reg
  io.cfg_gemm.dequant_addr := reg_in.io.reg_gemm_o.dequant_addr_reg
  io.cfg_gemm.bias_addr    := reg_in.io.reg_gemm_o.bias_addr_reg

  io.cfg_gemm.ic     := reg_in.io.reg_gemm_o.ifm_shape_c_reg(11, 0)
  io.cfg_gemm.iw     := reg_in.io.reg_gemm_o.ifm_shape_wh_reg(27, 16)
  io.cfg_gemm.ih     := reg_in.io.reg_gemm_o.ifm_shape_wh_reg(11, 0)
  io.cfg_gemm.icstep := reg_in.io.reg_gemm_o.ifm_shape_cstep_reg(25, 0)
  io.cfg_gemm.oc     := reg_in.io.reg_gemm_o.ofm_shape_c_reg(11, 0)
  io.cfg_gemm.ow     := reg_in.io.reg_gemm_o.ofm_shape_wh_reg(27, 16)
  io.cfg_gemm.oh     := reg_in.io.reg_gemm_o.ofm_shape_wh_reg(11, 0)
  io.cfg_gemm.ocstep := reg_in.io.reg_gemm_o.ofm_shape_cstep_reg(25, 0)

  io.cfg_gemm.wgt_len   := reg_in.io.reg_gemm_o.wgt_len_reg
  io.cfg_gemm.ifm_addr  := reg_in.io.reg_gemm_o.ifm_baseaddr_reg
  io.cfg_gemm.wgt_addr  := reg_in.io.reg_gemm_o.wgt_baseaddr_reg
  io.cfg_gemm.ofm_addr  := reg_in.io.reg_gemm_o.ofm_baseaddr_reg
  io.cfg_gemm.div_ifm_c := reg_in.io.reg_gemm_o.div_ifm_c_reg(11, 0)

  //SOFT RESET
  io.cfg_reset.soft_reset := reg_in.io.reg_reset_o.reset_reg(0)

}
