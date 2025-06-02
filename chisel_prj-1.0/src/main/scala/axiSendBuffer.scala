import chisel3.{util, _}
import chisel3.util._
import util_function._

class data_axiSend extends Bundle with axi_config {
  val data       = Input(UInt(AXI_DATA_WIDTH.W))
  val size       = Input(UInt(AXI_SIZE_WIDTH.W))
  val addr       = Input(UInt(AXI_ADDR_WIDTH.W))
  val data_valid = Input(Bool())
  val size_valid = Input(Bool())
  val addr_valid = Input(Bool())
}



class axiSendBufferCell extends Module with axi_config with buffer_config {
  val io = IO(new Bundle() {
    //input
    val opfusion_send = new data_axiSend
    val alu_send      = new data_axiSend
    val pool_send     = new data_axiSend
    val input_sel     = Input(UInt(3.W))
    //output
    val axi       = new axi_w_io
    val empty     = Output(Bool())
    val congested = Output(Bool())
  })

//  val data_fifo = Module(new Queue(UInt(AXI_DATA_WIDTH.W), entries = AXI_SEND_BUFFER_DEPTH))
//  val addr_fifo = Module(new Queue(UInt(AXI_ADDR_WIDTH.W), entries = AXI_SEND_BUFFER_DEPTH))
//  val size_fifo = Module(new Queue(UInt(AXI_SIZE_WIDTH.W), entries = AXI_SEND_BUFFER_DEPTH))
  val data_fifo = Module(new fwft_fifo(AXI_DATA_WIDTH, AXI_SEND_BUFFER_DEPTH, "block", true))
  val addr_fifo = Module(new fwft_fifo(AXI_ADDR_WIDTH, AXI_SEND_BUFFER_DEPTH, "block", true))
  val size_fifo = Module(new fwft_fifo(AXI_SIZE_WIDTH, AXI_SEND_BUFFER_DEPTH, "block", true))

  io.empty := data_fifo.io.count === 0.U

  val congested_top_limit  = data_fifo.io.count > (AXI_SEND_BUFFER_DEPTH - ALU_SRC_DEPTH - 1).U
  val congested_down_limit = data_fifo.io.count < 33.U
  val congested            = RegInit(false.B)
  congested    := Mux(congested_top_limit, true.B, Mux(congested_down_limit, false.B, congested))
  io.congested := congested

  val data_fifo_bits  = RegInit(0.U(AXI_DATA_WIDTH.W))
  val addr_fifo_bits  = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val size_fifo_bits  = RegInit(0.U(AXI_SIZE_WIDTH.W))
  val data_fifo_valid = RegInit(false.B)
  val addr_fifo_valid = RegInit(false.B)
  val size_fifo_valid = RegInit(false.B)

  data_fifo.io.enq.bits  := data_fifo_bits
  data_fifo.io.enq.valid := data_fifo_valid
  addr_fifo.io.enq.bits  := addr_fifo_bits
  addr_fifo.io.enq.valid := addr_fifo_valid
  size_fifo.io.enq.bits  := size_fifo_bits
  size_fifo.io.enq.valid := size_fifo_valid

  switch(io.input_sel) {
    is(AXI_SEND_ID("alu").U) {
      data_fifo_bits  := io.alu_send.data
      data_fifo_valid := io.alu_send.data_valid
      addr_fifo_bits  := io.alu_send.addr
      addr_fifo_valid := io.alu_send.addr_valid
      size_fifo_bits  := io.alu_send.size
      size_fifo_valid := io.alu_send.size_valid
    }
    is(AXI_SEND_ID("gemm_ofm").U) {
      data_fifo_bits  := io.opfusion_send.data
      data_fifo_valid := io.opfusion_send.data_valid
      addr_fifo_bits  := io.opfusion_send.addr
      addr_fifo_valid := io.opfusion_send.addr_valid
      size_fifo_bits  := io.opfusion_send.size
      size_fifo_valid := io.opfusion_send.size_valid
    }
    is(AXI_SEND_ID("pool").U) {
      data_fifo_bits  := io.pool_send.data
      data_fifo_valid := io.pool_send.data_valid
      addr_fifo_bits  := io.pool_send.addr
      addr_fifo_valid := io.pool_send.addr_valid
      size_fifo_bits  := io.pool_send.size
      size_fifo_valid := io.pool_send.size_valid
    }

  }

  val outstanding_cnt = RegInit(0.U(log2Ceil(AXI_W_OUTSTANDING+1).W))
  switch(Cat(io.axi.m_awvalid&&io.axi.m_awready, io.axi.m_wlast&&io.axi.m_wvalid&&io.axi.m_wready)){
    is("b00".U){
      outstanding_cnt := outstanding_cnt
    }
    is("b01".U){
      outstanding_cnt := outstanding_cnt - 1.U
    }
    is("b10".U){
      outstanding_cnt := outstanding_cnt + 1.U
    }
    is("b11".U){
      outstanding_cnt := outstanding_cnt
    }
  }

  val sIdle :: sCheck4K :: sAxiSend :: Nil = Enum(3)

  val state = RegInit(sIdle)
  val fifo_size_left = RegInit(0.U(AXI_SIZE_WIDTH.W))
  val fifo_addr_left = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val size_gt_256 = fifo_size_left >= 256.U
  val size_limit = Mux(size_gt_256, 256.U, fifo_size_left(8,0))
  val addr_end = fifo_addr_left + Cat(size_limit, 0.U(log2Ceil(AXI_DATA_WIDTH/8).W))
  val addr_4K = Cat(addr_end(AXI_ADDR_WIDTH-1, 12), 0.U(12.W))
  val cross_4K = addr_end(AXI_ADDR_WIDTH-1, 12) > fifo_addr_left(AXI_ADDR_WIDTH-1, 12) && addr_end(11,0) =/= 0.U
  val cross_4K_size = Wire(UInt(9.W))
  cross_4K_size := (addr_4K - fifo_addr_left)(AXI_ADDR_WIDTH-1, log2Ceil(AXI_DATA_WIDTH / 8))
  val send_addr = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val send_size = RegInit(0.U(9.W))
  size_fifo.io.deq.ready := false.B
  addr_fifo.io.deq.ready := false.B

  val size_send_fifo = Module(new Queue(UInt(8.W), entries = AXI_W_OUTSTANDING))
//  val size_send_fifo = Module(new standard_fifo(8, AXI_W_OUTSTANDING, "block"))

  size_send_fifo.io.enq.valid := false.B
  size_send_fifo.io.enq.bits := 0.U
  val size_send_fifo_odata = Mux(size_send_fifo.io.count === 0.U, 0.U, size_send_fifo.io.deq.bits)

  val awvalid = RegInit(false.B)

  switch(state){
    is(sIdle) {
      when(addr_fifo.io.deq.valid && fifo_size_left === 0.U && outstanding_cnt < (AXI_W_OUTSTANDING - 1).U) {
        size_fifo.io.deq.ready := true.B
        addr_fifo.io.deq.ready := true.B
        fifo_size_left := size_fifo.io.deq.bits
        fifo_addr_left := addr_fifo.io.deq.bits
        state := sCheck4K
      }.otherwise {
        fifo_size_left := 0.U
        state := sIdle
      }
    }
    is(sCheck4K){
      size_fifo.io.deq.ready := false.B
      addr_fifo.io.deq.ready := false.B
      when(size_send_fifo.io.enq.ready){
        size_send_fifo.io.enq.valid := true.B
        switch(Cat(cross_4K,size_gt_256)){
          is("b00".U){
            send_addr := fifo_addr_left
            send_size := fifo_size_left
            fifo_size_left := 0.U
            size_send_fifo.io.enq.bits := fifo_size_left - 1.U
          }
          is("b01".U){
            send_addr := fifo_addr_left
            send_size := 256.U
            fifo_addr_left := fifo_addr_left + 4096.U
            fifo_size_left := fifo_size_left - 256.U
            size_send_fifo.io.enq.bits := 255.U
          }
          is("b10".U){
            send_addr := fifo_addr_left
            send_size := cross_4K_size
            fifo_addr_left := addr_4K
            fifo_size_left := fifo_size_left - cross_4K_size
            size_send_fifo.io.enq.bits := cross_4K_size - 1.U
          }
          is("b11".U){
            send_addr := fifo_addr_left
            send_size := cross_4K_size
            fifo_addr_left := addr_4K
            fifo_size_left := fifo_size_left - cross_4K_size
            size_send_fifo.io.enq.bits := cross_4K_size - 1.U
          }
        }
        state := sAxiSend
        awvalid := true.B
      }.otherwise{
        state := sCheck4K
      }
    }
    is(sAxiSend){
      when(io.axi.m_awvalid && io.axi.m_awready){
        awvalid := false.B
        when(fifo_size_left === 0.U){
          when(addr_fifo.io.deq.valid && outstanding_cnt < (AXI_W_OUTSTANDING - 1).U) {
            size_fifo.io.deq.ready := true.B
            addr_fifo.io.deq.ready := true.B
            fifo_size_left := size_fifo.io.deq.bits
            fifo_addr_left := addr_fifo.io.deq.bits
            state := sCheck4K
          }.otherwise{
            state := sIdle
          }
        }.otherwise{
          state := sCheck4K
        }
      }.otherwise{
        state := sAxiSend
      }
    }
  }

  val last_cnt = RegInit(0.U(8.W))
  val send_len = RegInit(0.U(8.W))

  dontTouch(last_cnt)

  io.axi.m_wvalid := false.B

  when(io.axi.m_wlast){
    last_cnt := 0.U
  }.elsewhen(io.axi.m_wvalid && io.axi.m_wready){
    last_cnt := last_cnt + 1.U
  }

  val sBegin :: sWaitLast :: Nil = Enum(2)

  val wlast_state = RegInit(sBegin)
  size_send_fifo.io.deq.ready := false.B

  switch(wlast_state){
    is(sBegin){
      when(size_send_fifo.io.deq.valid){
        size_send_fifo.io.deq.ready := true.B
        wlast_state := sWaitLast
      }.otherwise{
        size_send_fifo.io.deq.ready := false.B
        wlast_state := sBegin
      }
      send_len := size_send_fifo_odata
    }
    is(sWaitLast){
      io.axi.m_wvalid := data_fifo.io.deq.valid
      size_send_fifo.io.deq.ready := false.B
      when(io.axi.m_wlast){
        when(size_send_fifo.io.deq.valid){
          send_len := size_send_fifo_odata
          size_send_fifo.io.deq.ready := true.B
          wlast_state := sWaitLast
        }.otherwise{
          wlast_state := sBegin
        }
      }.otherwise{
        size_send_fifo.io.deq.ready := false.B
        wlast_state := sWaitLast
      }
    }
  }

  dontTouch(size_send_fifo.io.count)

  io.axi.m_awid := AXI_ID.U
  io.axi.m_awaddr := send_addr
  io.axi.m_awlen := send_size - 1.U
  io.axi.m_awsize := log2Ceil(AXI_DATA_WIDTH/8).U
  io.axi.m_awburst := "b01".U
  io.axi.m_awlock := false.B
  io.axi.m_awcache := "b0010".U
  io.axi.m_awprot := 0.U
  io.axi.m_awqos := 0.U
  io.axi.m_awvalid := awvalid
  io.axi.m_wid := AXI_ID.U
  io.axi.m_wdata := data_fifo.io.deq.bits
  io.axi.m_wstrb := -1.S((AXI_DATA_WIDTH/8).W).asUInt
  io.axi.m_wlast := last_cnt === send_len && io.axi.m_wvalid && io.axi.m_wready
  data_fifo.io.deq.ready := io.axi.m_wvalid && io.axi.m_wready
  io.axi.m_bready := true.B
}

class axiSendBuffer extends Module with axi_config {
  val io = IO(new Bundle() {
    //input
    val opfusion_send_ch0 = new data_axiSend
    val opfusion_send_ch1 = new data_axiSend
    val alu_send_ch0      = new data_axiSend
    val alu_send_ch1      = new data_axiSend
    val pool_send_ch0     = new data_axiSend
    val pool_send_ch1     = new data_axiSend
    val cfg_gemm          = Input(new cfg_gemm_io)
    val cfg_pool          = Input(new cfg_pool_io)
    val cfg_alu           = Input(new cfg_alu_io)
    //output
    val axi_ch0   = new axi_w_io
    val axi_ch1   = new axi_w_io
    val empty     = Output(Bool())
    val congested = Output(Bool())
  })

  val input_sel = RegInit(0.U(3.W))
  when(io.cfg_alu.math_en | (io.cfg_alu.act_en & io.cfg_alu.act_src_sel(0))) {
    input_sel := AXI_SEND_ID("alu").U
  }.elsewhen(io.cfg_gemm.en) {
    input_sel := AXI_SEND_ID("gemm_ofm").U
  }.elsewhen(io.cfg_pool.en) {
    input_sel := AXI_SEND_ID("pool").U
  }.otherwise {
    input_sel := 0.U
  }

  val axi_send_cell0 = Module(new axiSendBufferCell)
  val axi_send_cell1 = Module(new axiSendBufferCell)

  axi_send_cell0.io.axi           <> io.axi_ch0
  axi_send_cell0.io.opfusion_send <> io.opfusion_send_ch0
  axi_send_cell0.io.alu_send      <> io.alu_send_ch0
  axi_send_cell0.io.pool_send     <> io.pool_send_ch0
  axi_send_cell0.io.input_sel     <> input_sel

  axi_send_cell1.io.axi           <> io.axi_ch1
  axi_send_cell1.io.opfusion_send <> io.opfusion_send_ch1
  axi_send_cell1.io.alu_send      <> io.alu_send_ch1
  axi_send_cell1.io.pool_send     <> io.pool_send_ch1
  axi_send_cell1.io.input_sel     <> input_sel

  io.congested := axi_send_cell0.io.congested || axi_send_cell1.io.congested

  io.empty := axi_send_cell0.io.empty & axi_send_cell1.io.empty
}
