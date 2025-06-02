import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import util_function._
import chisel3.util._
import axi_fun._

class tcp extends Module with hw_config {
  val io = IO(new Bundle() {
    //intr
    val alu_task_done = Output(Bool())
    val pool_task_done = Output(Bool())
    val gemm_task_done = Output(Bool())
    //axi_lite
    val accel_axi_lite = if (!SIM_MODE) Some(new axi_lite_io(ACCEL_AXI_LITE_DATA_WIDTH, ACCEL_AXI_LITE_ADDR_WIDTH)) else None
    //axi
    val axi_ch0 = if (!SIM_MODE) Some(new axiIO) else None
    val axi_ch1 = if (!SIM_MODE) Some(new axiIO) else None
    //axi_sim
    val axi_ch0_w = if (SIM_MODE) Some(new axi_w_io) else None
    val axi_ch1_w = if (SIM_MODE) Some(new axi_w_io) else None
    val axi_ch0_r = if (SIM_MODE) Some(new axi_r_io) else None
    val axi_ch1_r = if (SIM_MODE) Some(new axi_r_io) else None
    val sim_alu_reg = if (SIM_MODE) Some(Input(new reg_alu_io)) else None
    val sim_alu_o_reg = if (SIM_MODE) Some(Output(UInt(32.W))) else None
    val sim_pool_reg = if (SIM_MODE) Some(Input(new reg_pool_io)) else None
    val sim_gemm_reg = if (SIM_MODE) Some(Input(new reg_gemm_io)) else None
    val sim_reset_reg = if (SIM_MODE) Some(Input(new reg_reset_io)) else None
    //use to eval performance
    val ifmbuf_task_done = if (SIM_MODE) Some(Output(Bool())) else None
    val ofmbuf_congested = if (SIM_MODE) Some(Output(Bool())) else None
    val axisendbuf_congested = if (SIM_MODE) Some(Output(Bool())) else None
    //IfmBuffer_sim
    val ifm_mem_read_port0 = if (SIM_IFMBUF_IO) Some(new ifm_r_io(IFM_BUFFER_DEPTH, IFM_BUFFER_WIDTH)) else None
    val ifm_mem_read_port1 = if (SIM_IFMBUF_IO) Some(new ifm_r_io(IFM_BUFFER_DEPTH, IFM_BUFFER_WIDTH)) else None
    //wgtBuf_sim
    val wgt_odata = if (SIM_WGTBUF_IO) Some(Decoupled(Vec(32, UInt(32.W)))) else None
    val wgt_task_done = if (SIM_WGTBUF_IO) Some(Output(Bool())) else None
    //ifmbufctl_sim
    val ifmctl_odata = if (SIM_IFMBUFCTL_IO) Some(Decoupled(Vec(mesh_rows, UInt(pe_data_w.W)))) else None
    val ifmctl_task_done = if (SIM_IFMBUFCTL_IO) Some(Output(Bool())) else None
    //accmem_sim
    val accmem_out = if (SIM_ACCMEM_IO) Some(Vec(mesh_columns, Valid(new acc_data))) else None
    //ofmbuf_sim osacleAndBias_sim conv_act_sim
    val gemm_odata_ch0 = if (SIM_CONV_ACT_IO | SIM_OFMBUF_IO | SIM_OSCALE_BIAS_IO) Some(Output(new data_gp(4, 32))) else None
    val gemm_odata_ch1 = if (SIM_CONV_ACT_IO | SIM_OFMBUF_IO | SIM_OSCALE_BIAS_IO) Some(Output(new data_gp(4, 32))) else None
  })

  io.alu_task_done := false.B
  io.pool_task_done := false.B
  io.gemm_task_done := false.B

  if (SIM_IFMBUF || SIM_WGTBUF || SIM_IFMBUFCTL || SIM_MESH || SIM_ACCMEM || SIM_OFMBUF || SIM_OCALE_BIAS || SIM_CONV_ACT) {
    if (SIM_MODE) {
      w_ch_zero(io.axi_ch0_w.get)
      w_ch_zero(io.axi_ch1_w.get)
    } else {
      w_ch_zero(io.axi_ch0.get)
      w_ch_zero(io.axi_ch1.get)
    }
  }

  //********************************************* axi-lite ***********************************************
  val regMap = Module(new regMap)
  if (!SIM_MODE) {
    val axi_lite_accel = Module(new axi_lite_accel)
    axi_lite_accel.io.s <> io.accel_axi_lite.get
    axi_lite_accel.io.s_axi_aclk <> clock
    axi_lite_accel.io.s_axi_aresetn <> ~ (reset.asBool || regMap.io.cfg_reset.soft_reset)
    regMap.io.axi_lite.get <> axi_lite_accel.io.o_slv_reg
    if (ALU_EN) {
      axi_lite_accel.io.i_slv_reg_34 <> alu.get.io.axi_lite_innerprod_o
    }
    else {
      axi_lite_accel.io.i_slv_reg_34 <> 0.U
    }

  } else {
    regMap.io.reg_alu.get <> io.sim_alu_reg.get
    regMap.io.reg_pool.get <> io.sim_pool_reg.get
    regMap.io.reg_gemm.get <> io.sim_gemm_reg.get
    regMap.io.reg_reset.get <> io.sim_reset_reg.get
  }
  //********************************************* axi ***********************************************
  val axi_ch0_ctrl = Module(new axi_r_ctrl)
  val axi_ch1_ctrl = Module(new axi_r_ctrl)
  val axi_ch0_cfg = Module(new axi_r_en_cfg(AXI_R_CH0_EN_CFG))
  val axi_ch1_cfg = Module(new axi_r_en_cfg(AXI_R_CH1_EN_CFG))

  axi_ch0_ctrl.reset <> (reset.asBool || regMap.io.cfg_reset.soft_reset)
  axi_ch1_ctrl.reset <> (reset.asBool || regMap.io.cfg_reset.soft_reset)

  axi_ch0_cfg.reset <> (reset.asBool || regMap.io.cfg_reset.soft_reset)
  axi_ch1_cfg.reset <> (reset.asBool || regMap.io.cfg_reset.soft_reset)

  axi_ch0_ctrl.io.dma <> axi_ch0_cfg.io.axiR
  axi_ch1_ctrl.io.dma <> axi_ch1_cfg.io.axiR
  if (SIM_MODE) {
    axi_ch0_ctrl.io.axi <> io.axi_ch0_r.get
    axi_ch1_ctrl.io.axi <> io.axi_ch1_r.get
  } else {
    connect_axi_r(axi_ch0_ctrl.io.axi, io.axi_ch0.get)
    connect_axi_r(axi_ch1_ctrl.io.axi, io.axi_ch1.get)
  }

  //  axi_ch0_ctrl.io.axi_busy <> axi_ch0_cfg.io.axi_busy
  //  axi_ch1_ctrl.io.axi_busy <> axi_ch1_cfg.io.axi_busy
  if (!ALU_EN) {
    axi_ch0_cfg.io.axir_alu.get.areq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    axi_ch1_cfg.io.axir_alu.get.areq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }
  if (!(GEMM_EN | (SIM_IFMBUF & SIM_MODE))) {
    axi_ch0_cfg.io.axir_gemmIfm.get.areq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    axi_ch1_cfg.io.axir_gemmIfm.get.areq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }
  if (!(GEMM_EN | (SIM_WGTBUF & SIM_MODE))) {
    axi_ch0_cfg.io.axir_gemmWgt.get.areq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    axi_ch1_cfg.io.axir_gemmWgt.get.areq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }
  if (!POOL_EN) {
    axi_ch0_cfg.io.axir_pool.get.areq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    axi_ch1_cfg.io.axir_pool.get.areq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }

  //********************************************* IfmBuffer ***********************************************
  val ifmbuffer = if (GEMM_EN || (SIM_IFMBUF & SIM_MODE)) Some(Module(new IfmBuffer)) else None
  if (GEMM_EN || (SIM_IFMBUF & SIM_MODE)) {
    if (GEMM_EN || (SIM_WGTBUF & SIM_MODE)) {
      ifmbuffer.get.reset <> (!RegNext(regMap.io.cfg_gemm.en,0.B) || !wgtBuf.get.io.w_axi_finish || reset.asBool || RegNext(regMap.io.cfg_reset.soft_reset,0.B))
    } else {
      ifmbuffer.get.reset <> (!RegNext(regMap.io.cfg_gemm.en,0.B) || reset.asBool || RegNext(regMap.io.cfg_reset.soft_reset,0.B))
    }
    ifmbuffer.get.io.axi_ch0 <> axi_ch0_cfg.io.axir_gemmIfm.get
    ifmbuffer.get.io.axi_ch1 <> axi_ch1_cfg.io.axir_gemmIfm.get
    ifmbuffer.get.io.cfg_gemm <> regMap.io.cfg_gemm

    if (SIM_IFMBUF_IO & SIM_MODE) {
      io.ifm_mem_read_port0.get <> ifmbuffer.get.io.ifm_r_ch0
      io.ifm_mem_read_port1.get <> ifmbuffer.get.io.ifm_r_ch1
    }
  }

  /** ******************************************* weight_buffer ***********************************************
   */
  lazy val wgtBuf = if (GEMM_EN || (SIM_WGTBUF & SIM_MODE)) Some(Module(new weightBuffer)) else None
  if (GEMM_EN || (SIM_WGTBUF & SIM_MODE)) {
    wgtBuf.get.reset <> (!regMap.io.cfg_gemm.en || reset.asBool || regMap.io.cfg_reset.soft_reset)

    wgtBuf.get.io.wgt_en := (if (GEMM_EN | (SIM_MODE & SIM_OCALE_BIAS)) oscaleBias.get.io.ready
    else RegNext(regMap.io.cfg_gemm.en,0.B))
    wgtBuf.get.io.cfg_gemm := regMap.io.cfg_gemm

    wgtBuf.get.io.axi_ch0 <> axi_ch0_cfg.io.axir_gemmWgt.get
    wgtBuf.get.io.axi_ch1 <> axi_ch1_cfg.io.axir_gemmWgt.get

    if (SIM_WGTBUF_IO) {
      io.wgt_odata.get <> wgtBuf.get.io.o_data
      io.wgt_task_done.get := wgtBuf.get.io.finish
    }
  }

  /** ******************************************* ifm_buffer ***********************************************
   */
  val ifmbufferctl = if (GEMM_EN || (SIM_IFMBUFCTL & SIM_MODE)) Some(Module(new IfmBufferCtl)) else None
  if (GEMM_EN || (SIM_IFMBUFCTL & SIM_MODE)) {
    ifmbufferctl.get.reset <> (!regMap.io.cfg_gemm.en || reset.asBool || regMap.io.cfg_reset.soft_reset)
    ifmbufferctl.get.io.cfg_gemm := regMap.io.cfg_gemm
    ifmbufferctl.get.io.ifm_read_port0 <> ifmbuffer.get.io.ifm_r_ch0
    ifmbufferctl.get.io.ifm_read_port1 <> ifmbuffer.get.io.ifm_r_ch1
    if (GEMM_EN | (SIM_OCALE_BIAS & SIM_MODE)) {
      ifmbufferctl.get.io.task_done := ifmbuffer.get.io.task_done & oscaleBias.get.io.ready
    } else {
      ifmbufferctl.get.io.task_done := ifmbuffer.get.io.task_done
    }

    if (SIM_IFMBUFCTL_IO) {
      io.ifmctl_odata.get <> ifmbufferctl.get.io.ifm
      io.ifmctl_task_done.get := ShiftRegister(wgtBuf.get.io.finish, 32, 0.U, 1.B)
      wgtBuf.get.io.o_data.ready := 1.B
    }

  }

  /** ******************************************* mesh accmem ***********************************************
   */
  lazy val mesh = if (GEMM_EN || (SIM_MESH & SIM_MODE)) Some(Module(new Mesh)) else None
  val accmem = if (GEMM_EN || (SIM_ACCMEM & SIM_MODE)) Some(Module(new AccMem)) else None
  if (GEMM_EN || (SIM_MESH & SIM_ACCMEM & SIM_MODE)) {
    mesh.get.reset <> (!regMap.io.cfg_gemm.en || reset.asBool || regMap.io.cfg_reset.soft_reset)
    mesh.get.io.w <> wgtBuf.get.io.o_data
    mesh.get.io.ifm <> ifmbufferctl.get.io.ifm
    mesh.get.io.w_finish <> wgtBuf.get.io.finish
    mesh.get.io.last_in <> ifmbufferctl.get.io.last_in
    mesh.get.io.cfg_gemm := regMap.io.cfg_gemm

    accmem.get.reset <> (!regMap.io.cfg_gemm.en || reset.asBool || regMap.io.cfg_reset.soft_reset)
    accmem.get.io.cfg_gemm := regMap.io.cfg_gemm

    if (GEMM_EN | (SIM_OFMBUF & SIM_MODE)) {
      accmem.get.io.ofmbuf_stop <> ofmBuf.get.io.ofmBuffer_congested
      mesh.get.io.ofmbuf_stop <> ofmBuf.get.io.ofmBuffer_congested
    } else {
      accmem.get.io.ofmbuf_stop := 0.B
      mesh.get.io.ofmbuf_stop := 0.B
    }
    accmem.get.io.ofm <> mesh.get.io.ofm
    if (SIM_ACCMEM_IO) {
      io.accmem_out.get <> accmem.get.io.out
    }
  }

  /** ******************************************* ofm buffer ***********************************************
   */
  lazy val ofmBuf = if (GEMM_EN | (SIM_OFMBUF & SIM_MODE)) Some(Module(new ofmBuffer)) else None
  if (GEMM_EN | (SIM_OFMBUF & SIM_MODE)) {
    ofmBuf.get.reset <> (!regMap.io.cfg_gemm.en || reset.asBool || regMap.io.cfg_reset.soft_reset)
    ofmBuf.get.io.i_data <> accmem.get.io.out
  }
  if (GEMM_EN) {
    ofmBuf.get.io.axiSend_congested := axi_send.get.io.congested
  } else if (SIM_OFMBUF & SIM_MODE) {
    ofmBuf.get.io.axiSend_congested := false.B
  }

  /** ******************************************* oscale and bias ***********************************************
   */
  lazy val oscaleBias = if (GEMM_EN | (SIM_OCALE_BIAS & SIM_MODE)) Some(Module(new oscaleAndBias)) else None
  if (GEMM_EN | (SIM_OCALE_BIAS & SIM_MODE)) {
    oscaleBias.get.reset <> (!regMap.io.cfg_gemm.en || reset.asBool || regMap.io.cfg_reset.soft_reset)
    oscaleBias.get.io.cfg_gemm <> regMap.io.cfg_gemm
    oscaleBias.get.io.axi <> axi_ch1_cfg.io.axir_opfusion.get
    oscaleBias.get.io.ofm_data_ch0 <> ofmBuf.get.io.o_data_ch0
    oscaleBias.get.io.ofm_data_ch1 <> ofmBuf.get.io.o_data_ch1
  }

  /** ******************************************* alu (math and act) ***********************************************
   */
  lazy val alu = if (GEMM_EN | ALU_EN | (SIM_MODE & SIM_CONV_ACT)) Some(Module(new aluTop)) else None
  if (GEMM_EN | ALU_EN | (SIM_MODE & SIM_CONV_ACT)) {
    alu.get.reset <> ( reset.asBool || regMap.io.cfg_reset.soft_reset)
    alu.get.io.cfg <> regMap.io.cfg_alu
    alu.get.io.bias_ch0 <> oscaleBias.get.io.bias_ch0
    alu.get.io.bias_ch1 <> oscaleBias.get.io.bias_ch1
    alu.get.io.axi_r_ch0 <> axi_ch0_cfg.io.axir_alu.get
    alu.get.io.axi_r_ch1 <> axi_ch1_cfg.io.axir_alu.get
    io.alu_task_done := alu.get.io.task_done & axi_send.get.io.empty
    alu.get.io.axi_send_congested_ch0 := axi_send.get.io.congested
    alu.get.io.axi_send_congested_ch1 := axi_send.get.io.congested
  }
  if (SIM_MODE) {
    io.sim_alu_o_reg.get <> (if (ALU_EN) alu.get.io.axi_lite_innerprod_o else 0.U)
  }

  //********************************************* Pool ***********************************************
  lazy val pool = if (POOL_EN) Some(Module(new pool)) else None
  if (POOL_EN) {
    pool.get.reset <> (!regMap.io.cfg_pool.en || reset.asBool || regMap.io.cfg_reset.soft_reset)
    pool.get.io.pool_io <> regMap.io.cfg_pool
    pool.get.io.pool_io <> regMap.io.cfg_pool

    io.pool_task_done := pool.get.io.task_done & axi_send.get.io.empty

    // axi read of pool ??
    pool.get.io.axi_ch0 <> axi_ch0_cfg.io.axir_pool.get
    pool.get.io.axi_ch1 <> axi_ch1_cfg.io.axir_pool.get
    pool.get.io.axiSend_congested <> axi_send.get.io.congested
  }

  /** ******************************************* opfusionTop ***********************************************
   */
  val opfusion = if (GEMM_EN) Some(Module(new opfusionTop)) else None
  if (GEMM_EN) {
    opfusion.get.reset <> ( reset.asBool || regMap.io.cfg_reset.soft_reset)
    opfusion.get.io.cfg_gemm <> regMap.io.cfg_gemm
    opfusion.get.io.act_data_ch0 <> alu.get.io.act_ch0
    opfusion.get.io.act_data_ch1 <> alu.get.io.act_ch1
    io.gemm_task_done := opfusion.get.io.gemm_task_done & axi_send.get.io.empty
  }


  /** ******************************************* axi send buffer ***********************************************
   */
  lazy val axi_send = if (GEMM_EN | ALU_EN | POOL_EN | SIM_MODE & SIM_CONV_ACT) Some(Module(new axiSendBuffer)) else None
  if (GEMM_EN | ALU_EN | POOL_EN | SIM_MODE & SIM_CONV_ACT) {
    axi_send.get.reset <> ( reset.asBool || regMap.io.cfg_reset.soft_reset)
    if (ALU_EN) {
      axi_send.get.io.alu_send_ch0 <> alu.get.io.axi_send_ch0
      axi_send.get.io.alu_send_ch1 <> alu.get.io.axi_send_ch1
    } else {
      axi_send.get.io.alu_send_ch0 <> 0.U.asTypeOf(axi_send.get.io.alu_send_ch0)
      axi_send.get.io.alu_send_ch1 <> 0.U.asTypeOf(axi_send.get.io.alu_send_ch1)
    }
    if (POOL_EN) {
      axi_send.get.io.pool_send_ch0 <> pool.get.io.axi_send_ch0
      axi_send.get.io.pool_send_ch1 <> pool.get.io.axi_send_ch1
    } else {
      axi_send.get.io.pool_send_ch0 <> 0.U.asTypeOf(axi_send.get.io.pool_send_ch0)
      axi_send.get.io.pool_send_ch1 <> 0.U.asTypeOf(axi_send.get.io.pool_send_ch1)
    }

    if (SIM_MODE) {
      axi_send.get.io.axi_ch0 <> io.axi_ch0_w.get
      axi_send.get.io.axi_ch1 <> io.axi_ch1_w.get
    } else {
      connect_axi_w(axi_send.get.io.axi_ch0, io.axi_ch0.get)
      connect_axi_w(axi_send.get.io.axi_ch1, io.axi_ch1.get)
    }
    axi_send.get.io.cfg_gemm <> regMap.io.cfg_gemm
    axi_send.get.io.cfg_alu <> regMap.io.cfg_alu
    axi_send.get.io.cfg_pool <> regMap.io.cfg_pool
    

    if (SIM_MODE & SIM_CONV_ACT | !GEMM_EN) {
      axi_send.get.io.opfusion_send_ch0 <> 0.U.asTypeOf(new data_axiSend)
      axi_send.get.io.opfusion_send_ch1 <> 0.U.asTypeOf(new data_axiSend)
    } else {
      axi_send.get.io.opfusion_send_ch0 <> opfusion.get.io.axi_send_ch0
      axi_send.get.io.opfusion_send_ch1 <> opfusion.get.io.axi_send_ch1
    }
  }

  if (SIM_OFMBUF_IO) {
    io.gemm_odata_ch0.get <> ofmBuf.get.io.o_data_ch0
    io.gemm_odata_ch1.get <> ofmBuf.get.io.o_data_ch1
  } else if (SIM_OSCALE_BIAS_IO) {
    io.gemm_odata_ch0.get <> oscaleBias.get.io.bias_ch0
    io.gemm_odata_ch1.get <> oscaleBias.get.io.bias_ch1
  } else if (SIM_CONV_ACT_IO) {
    io.gemm_odata_ch0.get <> alu.get.io.act_ch0
    io.gemm_odata_ch1.get <> alu.get.io.act_ch1
  }

  if (SIM_MODE && GEMM_EN) {
    io.ofmbuf_congested.get := ofmBuf.get.io.ofmBuffer_congested
    io.ifmbuf_task_done.get := ifmbuffer.get.io.task_done
    io.axisendbuf_congested.get := axi_send.get.io.congested
  } else if (SIM_MODE) {
    io.ofmbuf_congested.get := 0.U
    io.ifmbuf_task_done.get := 0.U
    io.axisendbuf_congested.get := 0.U
  }

  if (SIM_IFMBUF_IO & SIM_MODE) {
    io.ifmbuf_task_done.get := ifmbuffer.get.io.task_done
  }
}

object accel_gen extends App {
  new(chisel3.stage.ChiselStage)
    .execute(Array("--target-dir", "./verilog/accel"), Seq(ChiselGeneratorAnnotation(() => new tcp)))
}
