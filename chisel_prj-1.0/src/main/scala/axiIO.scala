import chisel3.{Bool, Bundle, Input, Output, UInt, _}
import chisel3.{Bool, Bundle, Input, Output, UInt, _}
import chisel3.util.log2Up

class axi_r_io extends Bundle with axi_config {
  val m_arid = Output(UInt(log2Up(AXI_ID + 1).W))
  val m_araddr = Output(UInt(AXI_ADDR_WIDTH.W))
  val m_arlen = Output(UInt(8.W))
  val m_arsize = Output(UInt(3.W))
  val m_arburst = Output(UInt(2.W))
  val m_arlock = Output(Bool())
  val m_arcache = Output(UInt(4.W))
  val m_arprot = Output(UInt(3.W))
  val m_arqos = Output(UInt(4.W))
  val m_arvalid = Output(Bool())
  val m_arready = Input(Bool())
  val m_rid = Input(UInt(log2Up(AXI_ID + 1).W))
  val m_rdata = Input(UInt(AXI_DATA_WIDTH.W))
  val m_rresp = Input(UInt(2.W))
  val m_rlast = Input(Bool())
  val m_rvalid = Input(Bool())
  val m_rready = Output(Bool())
}

class axi_w_io extends Bundle with axi_config {
  val m_awid = Output(UInt(log2Up(AXI_ID + 1).W))
  val m_awaddr = Output(UInt(AXI_ADDR_WIDTH.W))
  val m_awlen = Output(UInt(8.W))
  val m_awsize = Output(UInt(3.W))
  val m_awburst = Output(UInt(2.W))
  val m_awlock = Output(Bool())
  val m_awcache = Output(UInt(4.W))
  val m_awprot = Output(UInt(3.W))
  val m_awqos = Output(UInt(4.W))
  val m_awvalid = Output(Bool())
  val m_awready = Input(Bool())
  val m_wid = Output(UInt(log2Up(AXI_ID + 1).W))
  val m_wdata = Output(UInt(AXI_DATA_WIDTH.W))
  val m_wstrb = Output(UInt((AXI_DATA_WIDTH / 8).W))
  val m_wlast = Output(Bool())
  val m_wvalid = Output(Bool())
  val m_wready = Input(Bool())
  val m_bid = Input(UInt(log2Up(AXI_ID + 1).W))
  val m_bresp = Input(UInt(2.W))
  val m_bvalid = Input(Bool())
  val m_bready = Output(Bool())
}
class axiIO extends Bundle with axi_config {
  val m_arid = Output(UInt(log2Up(AXI_ID + 1).W))
  val m_araddr = Output(UInt(AXI_ADDR_WIDTH.W))
  val m_arlen = Output(UInt(8.W))
  val m_arsize = Output(UInt(3.W))
  val m_arburst = Output(UInt(2.W))
  val m_arlock = Output(Bool())
  val m_arcache = Output(UInt(4.W))
  val m_arprot = Output(UInt(3.W))
  val m_arqos = Output(UInt(4.W))
  val m_arvalid = Output(Bool())
  val m_arready = Input(Bool())
  val m_rid = Input(UInt(log2Up(AXI_ID + 1).W))
  val m_rdata = Input(UInt(AXI_DATA_WIDTH.W))
  val m_rresp = Input(UInt(2.W))
  val m_rlast = Input(Bool())
  val m_rvalid = Input(Bool())
  val m_rready = Output(Bool())

  val m_awid = Output(UInt(log2Up(AXI_ID + 1).W))
  val m_awaddr = Output(UInt(AXI_ADDR_WIDTH.W))
  val m_awlen = Output(UInt(8.W))
  val m_awsize = Output(UInt(3.W))
  val m_awburst = Output(UInt(2.W))
  val m_awlock = Output(Bool())
  val m_awcache = Output(UInt(4.W))
  val m_awprot = Output(UInt(3.W))
  val m_awqos = Output(UInt(4.W))
  val m_awvalid = Output(Bool())
  val m_awready = Input(Bool())
  val m_wid = Output(UInt(log2Up(AXI_ID + 1).W))
  val m_wdata = Output(UInt(AXI_DATA_WIDTH.W))
  val m_wstrb = Output(UInt((AXI_DATA_WIDTH / 8).W))
  val m_wlast = Output(Bool())
  val m_wvalid = Output(Bool())
  val m_wready = Input(Bool())
  val m_bid = Input(UInt(log2Up(AXI_ID + 1).W))
  val m_bresp = Input(UInt(2.W))
  val m_bvalid = Input(Bool())
  val m_bready = Output(Bool())
}
object axi_fun {

  def w_ch_zero(io: axiIO): Unit = {
    io.m_awid := 0.U
    io.m_awaddr := 0.U
    io.m_awlen := 0.U
    io.m_awsize := 0.U
    io.m_awburst := 0.U
    io.m_awlock := 0.U
    io.m_awcache := 0.U
    io.m_awprot := 0.U
    io.m_awqos := 0.U
    io.m_awvalid := 0.U
    io.m_wid := 0.U
    io.m_wdata := 0.U
    io.m_wstrb := 0.U
    io.m_wlast := 0.U
    io.m_wvalid := 0.U
    io.m_bready := 0.U
  }

  def w_ch_zero(io: axi_w_io): Unit = {
    io.m_awid := 0.U
    io.m_awaddr := 0.U
    io.m_awlen := 0.U
    io.m_awsize := 0.U
    io.m_awburst := 0.U
    io.m_awlock := 0.U
    io.m_awcache := 0.U
    io.m_awprot := 0.U
    io.m_awqos := 0.U
    io.m_awvalid := 0.U
    io.m_wid := 0.U
    io.m_wdata := 0.U
    io.m_wstrb := 0.U
    io.m_wlast := 0.U
    io.m_wvalid := 0.U
    io.m_bready := 0.U
  }

  def connect_axi_w(w_ch: axi_w_io, axi_io: axiIO): Unit = {
    axi_io.m_awid <> w_ch.m_awid
    axi_io.m_awaddr <> w_ch.m_awaddr
    axi_io.m_awlen <> w_ch.m_awlen
    axi_io.m_awsize <> w_ch.m_awsize
    axi_io.m_awburst <> w_ch.m_awburst
    axi_io.m_awlock <> w_ch.m_awlock
    axi_io.m_awcache <> w_ch.m_awcache
    axi_io.m_awprot <> w_ch.m_awprot
    axi_io.m_awqos <> w_ch.m_awqos
    axi_io.m_awvalid <> w_ch.m_awvalid
    axi_io.m_awready <> w_ch.m_awready
    axi_io.m_wid <> w_ch.m_wid
    axi_io.m_wdata <> w_ch.m_wdata
    axi_io.m_wstrb <> w_ch.m_wstrb
    axi_io.m_wlast <> w_ch.m_wlast
    axi_io.m_wvalid <> w_ch.m_wvalid
    axi_io.m_wready <> w_ch.m_wready
    axi_io.m_bid <> w_ch.m_bid
    axi_io.m_bresp <> w_ch.m_bresp
    axi_io.m_bvalid <> w_ch.m_bvalid
    axi_io.m_bready <> w_ch.m_bready
  }


  def connect_axi_r(r_ch: axi_r_io, axi_io: axiIO): Unit = {
    axi_io.m_arid <> r_ch.m_arid
    axi_io.m_araddr <> r_ch.m_araddr
    axi_io.m_arlen <> r_ch.m_arlen
    axi_io.m_arsize <> r_ch.m_arsize
    axi_io.m_arburst <> r_ch.m_arburst
    axi_io.m_arlock <> r_ch.m_arlock
    axi_io.m_arcache <> r_ch.m_arcache
    axi_io.m_arprot <> r_ch.m_arprot
    axi_io.m_arqos <> r_ch.m_arqos
    axi_io.m_arvalid <> r_ch.m_arvalid
    axi_io.m_arready <> r_ch.m_arready
    axi_io.m_rid <> r_ch.m_rid
    axi_io.m_rdata <> r_ch.m_rdata
    axi_io.m_rresp <> r_ch.m_rresp
    axi_io.m_rlast <> r_ch.m_rlast
    axi_io.m_rvalid <> r_ch.m_rvalid
    axi_io.m_rready <> r_ch.m_rready
  }
}