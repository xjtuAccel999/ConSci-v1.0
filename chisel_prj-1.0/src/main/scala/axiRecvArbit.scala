import chisel3._
import chisel3.util._
import util_function._

class axiCtrl_io(AXI_SIZE_WIDTH: Int, AXI_ADDR_WIDTH: Int) extends Bundle {
  val axiEn   = Output(Bool())
  val axiAreq = Output(Bool())
  val axiSize = Output(UInt(AXI_SIZE_WIDTH.W))
  val axiAddr = Output(UInt(AXI_ADDR_WIDTH.W))
}

class axiRData_io(AXI_DATA_WIDTH: Int) extends Bundle {
  val valid = Input(Bool())
  val data  = Input(UInt(AXI_DATA_WIDTH.W))
  val last  = Input(Bool())
}

class axiOutstanding extends Bundle {
  val outstanding_almost_full = Input(Bool())
  val outstanding_empty = Input(Bool())
}

class axi_r extends Bundle with axi_config {
  val id   = Input(UInt(log2Ceil(RID.values.toList.max + 1).W))
  val outstanding = new axiOutstanding
  val areq = new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH)
  val data = new axiRData_io(AXI_DATA_WIDTH)
}

class axiDma_io(AXI_ADDR_WIDTH: Int, AXI_DATA_WIDTH: Int) extends Bundle {
  val areq  = Output(Bool())
  val addr  = Output(UInt(AXI_ADDR_WIDTH.W))
  val size  = Output(UInt(32.W))
  val outstanding_almost_full  = Input(Bool())
  val outstanding_empty  = Input(Bool())
  val data  = Input(UInt(AXI_DATA_WIDTH.W))
  val valid = Input(Bool())
  val ready = Output(Bool())
  val last  = Input(Bool())
}

class axiRAreqAbrit extends Module with axi_config {
  val io = IO(new Bundle() {
//    val axi_busy = Input(Bool())
    val aluAreq      = Flipped(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    val gemmIfmAreq  = Flipped(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    val gemmWgtAreq  = Flipped(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    val poolAreq     = Flipped(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    val opfusionAreq = Flipped(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
    //...
    val sel   = Output(UInt(log2Ceil(RID.values.toList.max + 1).W))
    val toaxi = new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH)
  })

  val sChose :: sKeep :: sFinish :: Nil = Enum(3)

  val state = RegInit(sChose)
  val sel   = RegInit(0.U)
  val addr  = RegInit(0.U)
  val size  = RegInit(0.U)
  val en    = RegInit(0.U)
  val areq  = RegInit(0.U)

  //axiEn must be high during select
  switch(state) {
    is(sChose) {
      sel   := 0.U
      state := sChose
        when(io.aluAreq.axiEn) {
          sel   := RID("alu").U
//          state := Mux(io.axi_busy, sKeep, sChose)
          state := sKeep
        }.elsewhen(io.gemmIfmAreq.axiEn) {
          sel   := RID("gemm_ifm").U
//          state := Mux(io.axi_busy, sKeep, sChose)
          state := sKeep
        }.elsewhen(io.gemmWgtAreq.axiEn) {
          sel   := RID("gemm_wgt").U
//          state := Mux(io.axi_busy, sKeep, sChose)
          state := sKeep
        }.elsewhen(io.poolAreq.axiEn) {
          sel   := RID("pool").U
//          state := Mux(io.axi_busy, sKeep, sChose)
          state := sKeep
        }.elsewhen(io.opfusionAreq.axiEn) {
          sel   := RID("opfusion").U
//          state := Mux(io.axi_busy, sKeep, sChose)
          state := sKeep
        }
        //...
      }
//    is(sKeep) { state := Mux(!io.axi_busy, sFinish, sKeep) }
    is(sKeep) { state := Mux(fallEdge(io.toaxi.axiEn), sFinish, sKeep) }
    is(sFinish) { state := sChose }
  }

  //allocate data based on sel
  switch(sel) {
    is(0.U) { //no effect
      en := false.B
    }
    is(RID("alu").U) { //axi_alu
      en   := io.aluAreq.axiEn
      addr := io.aluAreq.axiAddr
      size := io.aluAreq.axiSize
      areq := io.aluAreq.axiAreq
    }
    is(RID("gemm_ifm").U) { //axi_gemm_ifm
      en   := io.gemmIfmAreq.axiEn
      addr := io.gemmIfmAreq.axiAddr
      size := io.gemmIfmAreq.axiSize
      areq := io.gemmIfmAreq.axiAreq
    }
    is(RID("gemm_wgt").U) { //axi_gemm_wgt
      en   := io.gemmWgtAreq.axiEn
      addr := io.gemmWgtAreq.axiAddr
      size := io.gemmWgtAreq.axiSize
      areq := io.gemmWgtAreq.axiAreq
    }
    is(RID("pool").U) { //axi_pool
      en   := io.poolAreq.axiEn
      addr := io.poolAreq.axiAddr
      size := io.poolAreq.axiSize
      areq := io.poolAreq.axiAreq
    }
    is(RID("opfusion").U) { //axi_opfusion
      en   := io.opfusionAreq.axiEn
      addr := io.opfusionAreq.axiAddr
      size := io.opfusionAreq.axiSize
      areq := io.opfusionAreq.axiAreq
    }
    
    //...
  }
  io.sel           := sel;
  io.toaxi.axiAddr := addr
  io.toaxi.axiSize := size
  io.toaxi.axiEn   := en
  io.toaxi.axiAreq := areq
}

//----------------------read channel----------------------//
class axiRBufSel extends Module with axi_config {
  val io = IO(new Bundle() {
    val sel          = Input(UInt(log2Ceil(RID.values.toList.max + 1).W))
    val dataIn       = new axiRData_io(AXI_DATA_WIDTH)
    val aluData      = Flipped(new axiRData_io(AXI_DATA_WIDTH))
    val gemmIfmData  = Flipped(new axiRData_io(AXI_DATA_WIDTH))
    val gemmWgtData  = Flipped(new axiRData_io(AXI_DATA_WIDTH))
    val poolData     = Flipped(new axiRData_io(AXI_DATA_WIDTH))
    val opfusionData = Flipped(new axiRData_io(AXI_DATA_WIDTH))
    //...
  })
  io.aluData.data       := Mux(io.sel === RID("alu").U, io.dataIn.data, 0.U)
  io.aluData.valid      := Mux(io.sel === RID("alu").U, io.dataIn.valid, 0.U)
  io.aluData.last       := Mux(io.sel === RID("alu").U, io.dataIn.last, 0.U)

  io.poolData.data      := Mux(io.sel === RID("pool").U, io.dataIn.data, 0.U)
  io.poolData.valid     := Mux(io.sel === RID("pool").U, io.dataIn.valid, 0.U)
  io.poolData.last      := Mux(io.sel === RID("pool").U, io.dataIn.last, 0.U)

  io.gemmIfmData.data   := Mux(io.sel === RID("gemm_ifm").U, io.dataIn.data, 0.U)
  io.gemmIfmData.valid  := Mux(io.sel === RID("gemm_ifm").U, io.dataIn.valid, 0.U)
  io.gemmIfmData.last   := Mux(io.sel === RID("gemm_ifm").U, io.dataIn.last, 0.U)

  io.gemmWgtData.data   := Mux(io.sel === RID("gemm_wgt").U, io.dataIn.data, 0.U)
  io.gemmWgtData.valid  := Mux(io.sel === RID("gemm_wgt").U, io.dataIn.valid, 0.U)
  io.gemmWgtData.last   := Mux(io.sel === RID("gemm_wgt").U, io.dataIn.last, 0.U)

  io.opfusionData.data  := Mux(io.sel === RID("opfusion").U, io.dataIn.data, 0.U)
  io.opfusionData.valid := Mux(io.sel === RID("opfusion").U, io.dataIn.valid, 0.U)
  io.opfusionData.last  := Mux(io.sel === RID("opfusion").U, io.dataIn.last, 0.U)

  //...
}

class axiR extends Module with axi_config {
  val io = IO(new Bundle() {
    val axi_r    = new axiDma_io(AXI_ADDR_WIDTH, AXI_DATA_WIDTH)
    val dataOut  = Flipped(new axiRData_io(AXI_DATA_WIDTH))
    val axiCtrl  = Flipped(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH)) //ctrl input
    val outstanding = Flipped(new axiOutstanding)
  })

  io.axi_r.ready := true.B
  io.dataOut.data := RegNext(io.axi_r.data,0.U)
  io.dataOut.valid := RegNext(io.axi_r.valid,0.U)
  io.dataOut.last := RegNext(io.axi_r.last,0.U)
  io.axi_r.areq := io.axiCtrl.axiAreq
  io.axi_r.addr := io.axiCtrl.axiAddr
  io.axi_r.size := io.axiCtrl.axiSize
  io.outstanding.outstanding_almost_full := io.axi_r.outstanding_almost_full
  io.outstanding.outstanding_empty := io.axi_r.outstanding_empty

}

class axi_r_en_cfg(r_en_cfg: Map[String, Boolean]) extends Module with hw_config {
  val io = IO(new Bundle() {
//    val axi_busy = Input(Bool())
    val axiR = new axiDma_io(AXI_ADDR_WIDTH, AXI_DATA_WIDTH)
    //----------------------read channel----------------------//
    val axir_alu      = if (r_en_cfg("alu")) Some(Flipped(new axi_r)) else None
    val axir_pool     = if (r_en_cfg("pool")) Some(Flipped(new axi_r)) else None
    val axir_gemmIfm  = if (r_en_cfg("gemm_ifm")) Some(Flipped(new axi_r)) else None
    val axir_gemmWgt  = if (r_en_cfg("gemm_wgt")) Some(Flipped(new axi_r)) else None
    val axir_opfusion = if (r_en_cfg("opfusion")) Some(Flipped(new axi_r)) else None
  })

  //----------------------read channel----------------------//
  val axiR          = Module(new axiR)
  val axiRBufSel    = Module(new axiRBufSel)
  val axiRAreqAbrit = Module(new axiRAreqAbrit)
  axiR.io.axi_r        <> io.axiR
  axiR.io.axiCtrl      <> axiRAreqAbrit.io.toaxi
  axiRBufSel.io.sel    := axiRAreqAbrit.io.sel
  axiRBufSel.io.dataIn <> axiR.io.dataOut
//  io.axi_busy <> axiRAreqAbrit.io.axi_busy

  if (r_en_cfg("alu")) {
    io.axir_alu.get.data     <> axiRBufSel.io.aluData
    axiRAreqAbrit.io.aluAreq <> io.axir_alu.get.areq
    io.axir_alu.get.id       <> axiRAreqAbrit.io.sel
    io.axir_alu.get.outstanding <> axiR.io.outstanding
  } else {
    axiRAreqAbrit.io.aluAreq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }
  if (r_en_cfg("pool")) {
    io.axir_pool.get.data     <> axiRBufSel.io.poolData
    axiRAreqAbrit.io.poolAreq <> io.axir_pool.get.areq
    io.axir_pool.get.id       <> axiRAreqAbrit.io.sel
    io.axir_pool.get.outstanding <> axiR.io.outstanding
  } else {
    axiRAreqAbrit.io.poolAreq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }
  if (r_en_cfg("gemm_ifm")) {
    io.axir_gemmIfm.get.data     <> axiRBufSel.io.gemmIfmData
    axiRAreqAbrit.io.gemmIfmAreq <> io.axir_gemmIfm.get.areq
    io.axir_gemmIfm.get.id       <> axiRAreqAbrit.io.sel
    io.axir_gemmIfm.get.outstanding <> axiR.io.outstanding
  } else {
    axiRAreqAbrit.io.gemmIfmAreq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }
  if (r_en_cfg("gemm_wgt")) {
    io.axir_gemmWgt.get.data     <> axiRBufSel.io.gemmWgtData
    axiRAreqAbrit.io.gemmWgtAreq <> io.axir_gemmWgt.get.areq
    io.axir_gemmWgt.get.id       <> axiRAreqAbrit.io.sel
    io.axir_gemmWgt.get.outstanding <> axiR.io.outstanding
  } else {
    axiRAreqAbrit.io.gemmWgtAreq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }
  if (r_en_cfg("opfusion")) {
    io.axir_opfusion.get.data     <> axiRBufSel.io.opfusionData
    axiRAreqAbrit.io.opfusionAreq <> io.axir_opfusion.get.areq
    io.axir_opfusion.get.id       <> axiRAreqAbrit.io.sel
    io.axir_opfusion.get.outstanding <> axiR.io.outstanding
  } else {
    axiRAreqAbrit.io.opfusionAreq <> 0.U.asTypeOf(new axiCtrl_io(AXI_SIZE_WIDTH, AXI_ADDR_WIDTH))
  }
  //...

}



class axi_r_ctrl extends Module with axi_config {
  val io = IO(new Bundle() {
    val axi = new axi_r_io
    val dma = Flipped(new axiDma_io(AXI_ADDR_WIDTH, AXI_DATA_WIDTH))
//    val axi_busy = Output(Bool())
  })

  val addr_fifo = Module(new fwft_fifo(AXI_ADDR_WIDTH, AXI_R_OUTSTANDING, "block", true))
  val size_fifo = Module(new fwft_fifo(AXI_SIZE_WIDTH+1, AXI_R_OUTSTANDING, "block", true)) //add extern bit -> align flag
  val size_send_fifo = Module(new fwft_fifo(AXI_SIZE_WIDTH + 1, AXI_R_OUTSTANDING, "block", true)) //use for align cnt
  val strb_send_fifo = Module(new fwft_fifo(AXI_DATA_WIDTH/32, AXI_R_OUTSTANDING, "block", true)) //use for align cnt

  io.dma.outstanding_almost_full := addr_fifo.io.count >= (AXI_R_OUTSTANDING-4).U
  io.dma.outstanding_empty := addr_fifo.io.count === 0.U

  val fifo_enq_areq = io.dma.areq && addr_fifo.io.enq.ready
  val addr_align_flag = io.dma.addr(3, 0) === 0.U
  val addr_fifo_enq_bits = RegInit(0.U(AXI_ADDR_WIDTH.W))
  val size_fifo_enq_bits = RegInit(0.U(AXI_SIZE_WIDTH.W))
  val size_send_fifo_enq_bits = RegInit(0.U((AXI_SIZE_WIDTH+1).W))
  val strb_send_fifo_enq_bits = RegInit(0.U(4.W))
  addr_fifo.io.enq.valid := RegNext(fifo_enq_areq,0.U)
  size_fifo.io.enq.valid := addr_fifo.io.enq.valid
  addr_fifo.io.enq.bits := addr_fifo_enq_bits
  size_fifo.io.enq.bits := size_fifo_enq_bits
  size_send_fifo.io.enq.valid := size_fifo.io.enq.valid
  size_send_fifo.io.enq.bits := size_send_fifo_enq_bits
  strb_send_fifo.io.enq.valid := addr_fifo.io.enq.valid
  strb_send_fifo.io.enq.bits := strb_send_fifo_enq_bits
  when(fifo_enq_areq) {
    addr_fifo_enq_bits := Mux(addr_align_flag, io.dma.addr, Cat(io.dma.addr(AXI_ADDR_WIDTH - 1, 4), 0.U(4.W)))
    size_fifo_enq_bits := Mux(addr_align_flag, io.dma.size, Cat(1.U(1.W), io.dma.size + 1.U))
    size_send_fifo_enq_bits := Mux(addr_align_flag, io.dma.size-1.U, Cat(1.U(1.W), io.dma.size))
    strb_send_fifo_enq_bits := io.dma.addr(3, 0)
  }

  val outstanding_cnt = RegInit(0.U(log2Ceil(AXI_R_OUTSTANDING+1).W))
  switch(Cat(io.axi.m_arvalid&&io.axi.m_arready, io.axi.m_rlast&&io.axi.m_rvalid&&io.axi.m_rready)){
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
  
//  val busy_cnt = RegInit(0.U(log2Ceil(AXI_R_OUTSTANDING+1).W))
//  switch(Cat(io.axi.m_arvalid&&io.axi.m_arready, io.dma.last)){
//    is("b00".U){
//      busy_cnt := busy_cnt
//    }
//    is("b01".U){
//      busy_cnt := busy_cnt - 1.U
//    }
//    is("b10".U){
//      busy_cnt := busy_cnt + 1.U
//    }
//    is("b11".U){
//      busy_cnt := busy_cnt
//    }
//  }
//  io.axi_busy := busy_cnt =/= 0.U

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
  val size_send_fifo_odata = Mux(size_send_fifo.io.count === 0.U, 0.U, size_send_fifo.io.deq.bits)
  val arvalid = RegInit(false.B)

  switch(state){
    is(sIdle) {
      when(addr_fifo.io.deq.valid && fifo_size_left === 0.U && outstanding_cnt < (AXI_R_OUTSTANDING - 1).U) {
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
      switch(Cat(cross_4K,size_gt_256)){
        is("b00".U){
          send_addr := fifo_addr_left
          send_size := fifo_size_left
          fifo_size_left := 0.U
        }
        is("b01".U){
          send_addr := fifo_addr_left
          send_size := 256.U
          fifo_addr_left := fifo_addr_left + 4096.U
          fifo_size_left := fifo_size_left - 256.U
        }
        is("b10".U){
          send_addr := fifo_addr_left
          send_size := cross_4K_size
          fifo_addr_left := addr_4K
          fifo_size_left := fifo_size_left - cross_4K_size
        }
        is("b11".U){
          send_addr := fifo_addr_left
          send_size := cross_4K_size
          fifo_addr_left := addr_4K
          fifo_size_left := fifo_size_left - cross_4K_size
        }
      }
      state := sAxiSend
      arvalid := true.B
    }
    is(sAxiSend){
      when(io.axi.m_arvalid && io.axi.m_arready){
        arvalid := false.B
        when(fifo_size_left === 0.U){
          when(addr_fifo.io.deq.valid && outstanding_cnt < (AXI_R_OUTSTANDING - 1).U) {
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

  val last_cnt = RegInit(0.U(AXI_SIZE_WIDTH.W))
  val send_len = RegInit(0.U(AXI_SIZE_WIDTH.W))
  val strb = RegInit(0.U(4.W))
  val valid_mask_en = RegInit(false.B)
  when(io.axi.m_rvalid && io.axi.m_rready){
    when(last_cnt === send_len){
      last_cnt := 0.U
    }.otherwise {
      last_cnt := last_cnt + 1.U
    }
  }

  val sBegin :: sWait :: sFinish :: Nil = Enum(3)

  val rlast_state = RegInit(sBegin)
  size_send_fifo.io.deq.ready := false.B
  strb_send_fifo.io.deq.ready := size_send_fifo.io.deq.ready
  strb := strb_send_fifo.io.deq.bits

  switch(rlast_state){
    is(sBegin){
      when(size_send_fifo.io.deq.valid) {
        size_send_fifo.io.deq.ready := true.B
        valid_mask_en := size_send_fifo_odata(AXI_SIZE_WIDTH)
        send_len := size_send_fifo_odata(AXI_SIZE_WIDTH-2,0)
        when(size_send_fifo_odata(AXI_SIZE_WIDTH)) { // not align
          rlast_state := sWait
        }.otherwise {
          rlast_state := sBegin
        }
      }.otherwise{
        size_send_fifo.io.deq.ready := false.B
        rlast_state := sBegin
      }
    }
    is(sWait){
      size_send_fifo.io.deq.ready := false.B
      when(last_cnt === send_len){
        when(size_send_fifo.io.deq.valid){
          size_send_fifo.io.deq.ready := true.B
          valid_mask_en := size_send_fifo_odata(AXI_SIZE_WIDTH)
          rlast_state := sFinish
          send_len := size_send_fifo_odata(AXI_SIZE_WIDTH-2,0)
        }.otherwise{
          rlast_state := sBegin
        }
      }.otherwise{
        rlast_state := sWait
      }
    }
    is(sFinish) {
      rlast_state := sBegin
      size_send_fifo.io.deq.ready := false.B
    }
  }

  val axi_data_t = RegEnable(io.axi.m_rdata, 0.U, io.axi.m_rvalid && io.axi.m_rready)
  val rvalid_mask = valid_mask_en && last_cnt === 0.U
  val dma_valid = !rvalid_mask && io.axi.m_rvalid && io.axi.m_rready
  val dma_data = Wire(UInt(AXI_DATA_WIDTH.W))

  dontTouch(dma_data)
  dontTouch(axi_data_t)

  dma_data  := io.axi.m_rdata
  switch(strb) {
    is(0x4.U) {
      dma_data := Cat(io.axi.m_rdata(31, 0), axi_data_t(127, 32))
    }
    is(0x8.U) {
      dma_data := Cat(io.axi.m_rdata(63, 0), axi_data_t(127, 64))
    }
    is(0xc.U) {
      dma_data := Cat(io.axi.m_rdata(95, 0), axi_data_t(127, 96))
    }
  }

  io.dma.valid := RegNext(dma_valid,0.U)
  io.dma.data := RegNext(dma_data,0.U)
  io.dma.last := RegNext(send_len === last_cnt && dma_valid,0.U)

  io.axi.m_arid := AXI_ID.U
  io.axi.m_araddr := send_addr
  io.axi.m_arlen := send_size - 1.U
  io.axi.m_arsize := log2Ceil(AXI_DATA_WIDTH/8).U
  io.axi.m_arburst := "b01".U
  io.axi.m_arlock := false.B
  io.axi.m_arcache := "b0010".U
  io.axi.m_arprot := 0.U
  io.axi.m_arqos := 0.U
  io.axi.m_arvalid := arvalid
  io.axi.m_rready := io.dma.ready
}
