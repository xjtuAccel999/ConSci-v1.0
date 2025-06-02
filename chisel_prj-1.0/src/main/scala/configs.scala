import chisel3._
import chisel3.util.{log2Ceil, log2Up}

trait axi_config {
  val SIM_MODE         = false
  val AXI_ADDR_WIDTH   = if (SIM_MODE) 64 else 32
  val AXI_SIZE_WIDTH   = 32
  val AXI_DATA_WIDTH   = 128
  val RID              = Map("alu" -> 1, "gemm_ifm" -> 2, "gemm_wgt" -> 3, "pool" -> 4, "opfusion" -> 5)
  val AXI_R_CH0_EN_CFG = Map("alu" -> true, "gemm_ifm" -> true, "gemm_wgt" -> true, "pool" -> true, "opfusion" -> false)
  val AXI_R_CH1_EN_CFG = Map("alu" -> true, "gemm_ifm" -> true, "gemm_wgt" -> true, "pool" -> true, "opfusion" -> true)
  val AXI_SEND_ID      = Map("alu" -> 1, "gemm_ofm" -> 2, "pool" -> 3)
  val AXI_ID = 0
  val AXI_W_OUTSTANDING = 16
  val AXI_R_OUTSTANDING = 16
}

trait buffer_config {
  val ALU_SRC_DEPTH            = 128
  val IFM_BUFFER_DEPTH         = 65536 / 4 //512KB
  val IFM_BUFFER_WIDTH         = 256 //64Bytes
  val WGT_BUFFER_WIDTH         = 8 * 32
  val WGT_BUFFER_DEPTH         = 9 * 512 //144Kb*2
  val OSCALE_BIAS_BUFFER_DEPTH = 256
  val OFM_BUFFER_DEPTH         = 256
  val AXI_SEND_BUFFER_DEPTH    = 1024
  val dparam_size              = 1024 //2048
  val tpram_size               = 2048
}

trait hw_config extends axi_config with buffer_config with mesh_config {
  val ACCEL_AXI_LITE_DATA_WIDTH = 32
  val ACCEL_AXI_LITE_ADDR_WIDTH = 9

  val FPGA_MODE = false

  val ALU_EN    = true
  val POOL_EN   = true
  val GEMM_EN   = true

  val SIM_IFMBUF     = false
  val SIM_WGTBUF     = false
  val SIM_IFMBUFCTL  = false
  val SIM_MESH       = false
  val SIM_ACCMEM     = false
  val SIM_OFMBUF     = false
  val SIM_OCALE_BIAS = false
  val SIM_CONV_ACT   = false

  //when simulation gemm_en, the follow SIM_* must be false
  if (GEMM_EN)
    assert(!(SIM_IFMBUF | SIM_WGTBUF | SIM_IFMBUFCTL | SIM_MESH | SIM_ACCMEM | SIM_OFMBUF | SIM_OCALE_BIAS | SIM_CONV_ACT))
  if (SIM_OFMBUF)
    assert(SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL & SIM_MESH & SIM_ACCMEM)
  //when simulation accmem, if im2col_sim = wgtbuf_sim =  ifmbuf_sim = 1
  if (SIM_ACCMEM)
    assert(SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL & SIM_MESH)
  if (SIM_MESH)
    assert(SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL)
  if (SIM_IFMBUFCTL)
    assert(SIM_IFMBUF & SIM_WGTBUF)
  //when only simulation ifmbuf or wgtbuf, if imfbuf_sim ^ wgtbuf_sim = 1
  if (!(SIM_IFMBUFCTL | SIM_MESH | SIM_ACCMEM) & (SIM_IFMBUF | SIM_WGTBUF))
    assert(SIM_IFMBUF ^ SIM_WGTBUF)

  if (SIM_OCALE_BIAS)
    assert(SIM_IFMBUF & SIM_WGTBUF & SIM_ACCMEM & SIM_OFMBUF)
  if (SIM_CONV_ACT)
    assert(SIM_IFMBUF & SIM_WGTBUF & SIM_ACCMEM & SIM_OFMBUF & SIM_OCALE_BIAS)

  val SIM_IFMBUF_IO    = SIM_MODE & SIM_IFMBUF & !SIM_WGTBUF & !SIM_IFMBUFCTL & !SIM_MESH & !SIM_ACCMEM & !SIM_OFMBUF & !SIM_OCALE_BIAS & !SIM_CONV_ACT
  val SIM_WGTBUF_IO    = SIM_MODE & !SIM_IFMBUF & SIM_WGTBUF & !SIM_IFMBUFCTL & !SIM_MESH & !SIM_ACCMEM & !SIM_OFMBUF & !SIM_OCALE_BIAS & !SIM_CONV_ACT
  val SIM_IFMBUFCTL_IO = SIM_MODE & SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL & !SIM_MESH & !SIM_ACCMEM & !SIM_OFMBUF & !SIM_OCALE_BIAS & !SIM_CONV_ACT
//  val SIM_MESH_IO        = SIM_MODE & SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL & SIM_MESH & !SIM_ACCMEM & !SIM_OFMBUF & !SIM_OCALE_BIAS & !SIM_CONV_ACT
  val SIM_ACCMEM_IO      = SIM_MODE & SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL & SIM_MESH & SIM_ACCMEM & !SIM_OFMBUF & !SIM_OCALE_BIAS & !SIM_CONV_ACT
  val SIM_OFMBUF_IO      = SIM_MODE & SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL & SIM_MESH & SIM_ACCMEM & SIM_OFMBUF & !SIM_OCALE_BIAS & !SIM_CONV_ACT
  val SIM_OSCALE_BIAS_IO = SIM_MODE & SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL & SIM_MESH & SIM_ACCMEM & SIM_OFMBUF & SIM_OCALE_BIAS & !SIM_CONV_ACT
  val SIM_CONV_ACT_IO    = SIM_MODE & SIM_IFMBUF & SIM_WGTBUF & SIM_IFMBUFCTL & SIM_MESH & SIM_ACCMEM & SIM_OFMBUF & SIM_OCALE_BIAS & SIM_CONV_ACT
}

trait pe_config extends buffer_config {
  val wgt_data_w = 8
  val pe_data_w  = 2 * wgt_data_w + log2Ceil(WGT_BUFFER_DEPTH)
  assert(pe_data_w >= 24)
}

trait mesh_config extends pe_config {
  val mesh_size         = 32
  val mesh_rows         = mesh_size
  val mesh_columns      = mesh_size
  val ofm_buffer_addr_w = log2Up(mesh_rows)
  val ifm_delay         = 6 // ifm.valid && ifm.ready -> io.ifm.bits
  val wgt_delay         = 1
}

trait cal_cell_params {
  val USE_HARDFLOAT         = false
  val FP32_TO_SINT_CYCLES   = if (USE_HARDFLOAT) 2 else 2
  val FP32_ADD_CYCLES       = if (USE_HARDFLOAT) 2 else 5
  val FP32_MUL_CYCLES       = if (USE_HARDFLOAT) 2 else 2
  val SINT_TO_FLOAT_CYCYLES = if (USE_HARDFLOAT) 1 else 2
//  val ACTIVATION_FUNC_CYCLES = FP32_MUL_CYCLES * 3 - 3 + FP32_ADD_CYCLES * 3
  val ACTIVATION_FUNC_CYCLES =
    if (FP32_MUL_CYCLES * 2 > FP32_ADD_CYCLES + FP32_MUL_CYCLES + 1) FP32_MUL_CYCLES * 2 + FP32_ADD_CYCLES * 2
    else FP32_ADD_CYCLES * 3 + FP32_MUL_CYCLES + 1
}

trait mathFunc_config {
  val ABS_ID       = 1.U
  val THRESHOLD_ID = 2.U
  val EQUAL_ID     = 3.U
}

trait activationFunc_config {
  val NONE_ID      = 0.U
  val RELU_ID      = 1.U
  val LEAKYRELU_ID = 2.U
  val CLIP_ID      = 3.U
  val SIGMOID_ID   = 4.U
  val MISH_ID      = 5.U
  val HARDSWISH_ID = 6.U
  val SWISH_ID     = 7.U
}

trait alu_config extends mathFunc_config with activationFunc_config with axi_config with buffer_config with cal_cell_params

