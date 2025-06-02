import chisel3._
import chisel3.util._
import util_function._

class axi_lite_io(C_S_AXI_DATA_WIDTH: Int, C_S_AXI_ADDR_WIDTH: Int) extends Bundle {
  val axi_awaddr  = Input(UInt(C_S_AXI_ADDR_WIDTH.W))
  val axi_awprot  = Input(UInt(3.W))
  val axi_awvalid = Input(Bool())
  val axi_awready = Output(Bool())
  val axi_wdata   = Input(UInt(32.W))
  val axi_wstrb   = Input(UInt((C_S_AXI_DATA_WIDTH / 8).W))
  val axi_wvalid  = Input(Bool())
  val axi_wready  = Output(Bool())
  val axi_bresp   = Output(UInt(2.W))
  val axi_bvalid  = Output(Bool())
  val axi_bready  = Input(Bool())
  val axi_araddr  = Input(UInt(C_S_AXI_ADDR_WIDTH.W))
  val axi_arprot  = Input(UInt(3.W))
  val axi_arvalid = Input(Bool())
  val axi_arready = Output(Bool())
  val axi_rdata   = Output(UInt(C_S_AXI_DATA_WIDTH.W))
  val axi_rresp   = Output(UInt(2.W))
  val axi_rvalid  = Output(Bool())
  val axi_rready  = Input(Bool())
}

class axi_lite_accel extends BlackBox with HasBlackBoxPath with hw_config {
  val io = IO(new Bundle() {
    val s_axi_aclk    = Input(Clock())
    val s_axi_aresetn = Input(Bool())
    val s             = new axi_lite_io(ACCEL_AXI_LITE_DATA_WIDTH, ACCEL_AXI_LITE_ADDR_WIDTH)
    val o_slv_reg     = Output(Vec(128, UInt(ACCEL_AXI_LITE_DATA_WIDTH.W)))
    val i_slv_reg_34  = Input(UInt(ACCEL_AXI_LITE_DATA_WIDTH.W))
  })
  addPath("./src/main/hdl/axi_lite_accel.v")
}
