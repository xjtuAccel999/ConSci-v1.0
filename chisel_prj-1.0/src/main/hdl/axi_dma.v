module axi_dma#
(
parameter  integer        M_AXI_ID_WIDTH     	    = 1				,
parameter  integer        M_AXI_ID			        = 0				,
parameter  integer        M_AXI_ADDR_WIDTH			= 32			,
parameter  integer        M_AXI_DATA_WIDTH			= 128
)
(
input   wire [M_AXI_ADDR_WIDTH-1 : 0]      fdma_w_addr          ,
input                                       fdma_w_areq          ,
input   wire [31 : 0]                      fdma_w_size          ,
output                                      fdma_w_busy          ,

input   wire [M_AXI_DATA_WIDTH-1 :0]       fdma_w_data			,
input   wire [M_AXI_DATA_WIDTH/8-1 :0]       fdma_w_strb			,
output  wire                               fdma_w_valid         ,
input	wire                               fdma_w_ready			,

input   wire [M_AXI_ADDR_WIDTH-1 : 0]     fdma_r_addr          ,
input                                       fdma_r_areq          ,
input   wire [31 : 0]                      fdma_r_size          ,
output                                      fdma_r_busy          ,

output  wire [M_AXI_DATA_WIDTH-1 :0]       fdma_r_data			,
output  wire                               fdma_r_valid         ,
input	wire                               fdma_r_ready			,

input 	wire  								m_axi_aclk			,
input 	wire  								m_axi_aresetn		,
output 	wire [M_AXI_ID_WIDTH-1 : 0]		    m_axi_awid			,
output 	wire [M_AXI_ADDR_WIDTH-1 : 0] 	    m_axi_awaddr		,
output 	wire [7 : 0]						m_axi_awlen			,
output 	wire [2 : 0] 						m_axi_awsize		,
output 	wire [1 : 0] 						m_axi_awburst		,
output 	wire  								m_axi_awlock		,
output 	wire [3 : 0] 						m_axi_awcache		,
output 	wire [2 : 0] 						m_axi_awprot		,
output 	wire [3 : 0] 						m_axi_awqos			,
output 	wire  								m_axi_awvalid		,
input	wire  								m_axi_awready		,
output  wire [M_AXI_ID_WIDTH-1 : 0] 		m_axi_wid			,
output  wire [M_AXI_DATA_WIDTH-1 : 0] 	    m_axi_wdata			,
output  wire [M_AXI_DATA_WIDTH/8-1 : 0] 	m_axi_wstrb			,
output  wire  								m_axi_wlast			,
output  wire  								m_axi_wvalid		,
input   wire  								m_axi_wready		,
input   wire [M_AXI_ID_WIDTH-1 : 0] 		m_axi_bid			,
input   wire [1 : 0] 						m_axi_bresp			,
input   wire  								m_axi_bvalid		,
output  wire  								m_axi_bready		,
output  wire [M_AXI_ID_WIDTH-1 : 0] 		m_axi_arid			,

output  wire [M_AXI_ADDR_WIDTH-1 : 0] 	    m_axi_araddr		,
output  wire [7 : 0] 						m_axi_arlen			,
output  wire [2 : 0] 						m_axi_arsize		,
output  wire [1 : 0] 						m_axi_arburst		,
output  wire  								m_axi_arlock		,
output  wire [3 : 0] 						m_axi_arcache		,
output  wire [2 : 0] 						m_axi_arprot		,
output  wire [3 : 0] 						m_axi_arqos			,
output  wire  								m_axi_arvalid		,
input   wire  								m_axi_arready		,
input   wire [M_AXI_ID_WIDTH-1 : 0] 		m_axi_rid			,
input   wire [M_AXI_DATA_WIDTH-1 : 0] 	    m_axi_rdata			,
input   wire [1 : 0] 						m_axi_rresp			,
input   wire  								m_axi_rlast			,
input   wire  								m_axi_rvalid		,
output  wire  								m_axi_rready

	);

localparam AXI_BYTES =  M_AXI_DATA_WIDTH/8;
localparam SHIFT_AXI_BYTES = clogb2(AXI_BYTES)-1'b1;

function integer clogb2 (input integer bit_depth);
begin
	 for(clogb2=0; bit_depth>0; clogb2=clogb2+1)
	 bit_depth = bit_depth >> 1;
end
endfunction
//fdma axi write----------------------------------------------
reg 	[M_AXI_ADDR_WIDTH-1 : 0] 	axi_awaddr	=0;
reg  						 		axi_awvalid	= 1'b0;
wire 	[M_AXI_DATA_WIDTH-1 : 0] 	axi_wdata	;
wire								axi_wlast	;
reg  								axi_wvalid	= 1'b0;
wire                               w_next      = m_axi_wvalid ;
reg   [8 :0]                       wburst_len  = 1  ;
reg   [7 :0]                       wburst_cnt  = 0  ;
reg   [31:0]                       wfdma_cnt   = 0  ;
reg                                axi_wstart_locked  =0;
// wire  [15:0] axi_wburst_size   =   wburst_len * AXI_BYTES;
wire  [15:0] axi_wburst_size   =   wburst_len << SHIFT_AXI_BYTES;
reg   [31 : 0]                     fdma_w_size_reg;


assign m_axi_awid		= M_AXI_ID;
assign m_axi_awaddr		= axi_awaddr;
assign m_axi_awlen		= wburst_len - 1;
assign m_axi_awsize		= clogb2(AXI_BYTES-1);
assign m_axi_awburst	= 2'b01;
assign m_axi_awlock		= 1'b0;
assign m_axi_awcache	= 4'b0010;
assign m_axi_awprot		= 3'h0;
assign m_axi_awqos		= 4'h0;
assign m_axi_awvalid	= axi_awvalid;
assign m_axi_wdata		= axi_wdata;
// assign m_axi_wstrb		= {(AXI_BYTES){1'b1}};
assign m_axi_wstrb		= fdma_w_strb;
assign m_axi_wlast		= axi_wlast;
assign m_axi_wvalid		= axi_wvalid & fdma_w_ready;
assign m_axi_bready		= 1'b1;
//----------------------------------------------------------------------------
//AXI4 FULL Write
assign  axi_wdata        = fdma_w_data;
assign  fdma_w_valid      = w_next;
reg     fdma_wstart_locked = 1'b0;
wire    fdma_wend;
wire    fdma_wstart;
assign   fdma_w_busy = fdma_wstart_locked ;

always @(posedge m_axi_aclk)
	if(m_axi_aresetn == 1'b0 || fdma_wend == 1'b1 )
		fdma_wstart_locked <= 1'b0;
	else if(fdma_wstart)
		fdma_wstart_locked <= 1'b1;

assign fdma_wstart = (fdma_wstart_locked == 1'b0 && fdma_w_areq == 1'b1);
//AXI4 write burst lenth busrt addr ------------------------------
always @(posedge m_axi_aclk)
    if(fdma_wstart)
        axi_awaddr <= fdma_w_addr;
    else if(axi_wlast == 1'b1)
        axi_awaddr <= axi_awaddr + axi_wburst_size ;
//AXI4 write cycle -----------------------------------------------
reg axi_wstart_locked_r1 = 1'b0, axi_wstart_locked_r2 = 1'b0;
always @(posedge m_axi_aclk)begin
    axi_wstart_locked_r1 <= axi_wstart_locked;
    axi_wstart_locked_r2 <= axi_wstart_locked_r1;
end
always @(posedge m_axi_aclk)
	if((fdma_wstart_locked == 1'b1) &&  axi_wstart_locked == 1'b0)
	    axi_wstart_locked <= 1'b1;
	else if(axi_wlast == 1'b1 || fdma_wstart == 1'b1)
	    axi_wstart_locked <= 1'b0;

//AXI4 addr valid and write addr-----------------------------------
always @(posedge m_axi_aclk)
     if((axi_wstart_locked_r1 == 1'b1) &&  axi_wstart_locked_r2 == 1'b0)
         axi_awvalid <= 1'b1;
     else if((axi_wstart_locked == 1'b1 && m_axi_awready == 1'b1)|| axi_wstart_locked == 1'b0)
         axi_awvalid <= 1'b0;
//AXI4 write data---------------------------------------------------
always @(posedge m_axi_aclk)
    if(axi_wlast == 1'b1)
        axi_wvalid <= 1'b0;
    else
		axi_wvalid <= m_axi_wready;//
//AXI4 write data burst len counter----------------------------------
always @(posedge m_axi_aclk)
	if(axi_wstart_locked == 1'b0)
		wburst_cnt <= 'd0;
	else if(w_next)
		wburst_cnt <= wburst_cnt + 1'b1;

assign axi_wlast = (w_next == 1'b1) && (wburst_cnt == m_axi_awlen);
//fdma write data burst len counter----------------------------------
reg wburst_len_req = 1'b0;
reg [31:0] fdma_wleft_cnt =32'd0;

always @(posedge m_axi_aclk)
        wburst_len_req <= fdma_wstart|axi_wlast;

always @(posedge m_axi_aclk)
	if( fdma_wstart )begin
		wfdma_cnt <= 1'd0;
		fdma_wleft_cnt <= fdma_w_size;
		fdma_w_size_reg <= fdma_w_size - 1'b1;
	end
	else if(w_next)begin
		wfdma_cnt <= wfdma_cnt + 1'b1;
	    fdma_wleft_cnt <= fdma_w_size_reg - wfdma_cnt;
    end

assign  fdma_wend = w_next && (fdma_wleft_cnt == 1 );

wire 	[M_AXI_ADDR_WIDTH-1 : 0] 	axi_awaddr_next_power_4k = {{axi_awaddr[M_AXI_ADDR_WIDTH-1:12] + 1'b1},{12'b0}};
wire    [M_AXI_ADDR_WIDTH-1 : 0]    num_wdata_next_4k_t      = axi_awaddr_next_power_4k-axi_awaddr;
wire    [8 : 0]                     num_wdata_next_4k        = num_wdata_next_4k_t[clogb2(M_AXI_DATA_WIDTH/8)+7:clogb2(M_AXI_DATA_WIDTH/8)-1];

always @(posedge m_axi_aclk)
    begin
     if(wburst_len_req)
        begin
            if(fdma_wleft_cnt > 256)
                begin
                    if(num_wdata_next_4k<256)
                        begin
                            wburst_len <= num_wdata_next_4k;
                        end
                    else
                        begin
                            wburst_len <= 256;
                        end

                end
            else
                begin
                    if(num_wdata_next_4k<fdma_wleft_cnt[8:0])
                        begin
                            wburst_len <= num_wdata_next_4k;
                        end
                    else
                        begin
                            wburst_len <= fdma_wleft_cnt[8:0];
                        end
                end
        end
     else
        begin
            wburst_len <= wburst_len;
        end
    end

//fdma axi read----------------------------------------------
reg 	[M_AXI_ADDR_WIDTH-1 : 0] 	axi_araddr =0	;
reg  						 		axi_arvalid	 =1'b0;
wire								axi_rlast	;
reg  								axi_rready	= 1'b0;
wire                               r_next      = (m_axi_rvalid && m_axi_rready);
reg   [8 :0]                       rburst_len  = 1  ;
reg   [7 :0]                       rburst_cnt  = 0  ;
reg   [31:0]                       rfdma_cnt   = 0  ;
reg                                axi_rstart_locked =0;
// wire  [15:0] axi_rburst_size   =   rburst_len * AXI_BYTES;
wire  [15:0] axi_rburst_size   =   rburst_len << SHIFT_AXI_BYTES ;
reg   [31:0]                       fdma_r_size_reg ;

assign m_axi_arid		= M_AXI_ID;
assign m_axi_araddr		= axi_araddr;
assign m_axi_arlen		= rburst_len - 1;
assign m_axi_arsize		= clogb2((AXI_BYTES)-1);
assign m_axi_arburst	= 2'b01;
assign m_axi_arlock		= 1'b0;
assign m_axi_arcache	= 4'b0010;
assign m_axi_arprot		= 3'h0;
assign m_axi_arqos		= 4'h0;
assign m_axi_arvalid	= axi_arvalid;
assign m_axi_rready		= axi_rready&&fdma_r_ready;
assign fdma_r_data       = m_axi_rdata;
assign fdma_r_valid      = r_next;

//AXI4 FULL Read-----------------------------------------

reg     fdma_rstart_locked = 1'b0;
wire    fdma_rend;
wire    fdma_rstart;
assign   fdma_r_busy = fdma_rstart_locked ;

always @(posedge m_axi_aclk)
	if(m_axi_aresetn == 1'b0 || fdma_rend == 1'b1)
		fdma_rstart_locked <= 1'b0;
	else if(fdma_rstart)
		fdma_rstart_locked <= 1'b1;

assign fdma_rstart = (fdma_rstart_locked == 1'b0 && fdma_r_areq == 1'b1);
//AXI4 read burst lenth busrt addr ------------------------------
always @(posedge m_axi_aclk)
    if(fdma_rstart == 1'b1)
        axi_araddr <= fdma_r_addr;
    else if(axi_rlast == 1'b1)
        axi_araddr <= axi_araddr + axi_rburst_size ;
//AXI4 r_cycle_flag-------------------------------------
reg axi_rstart_locked_r1 = 1'b0, axi_rstart_locked_r2 = 1'b0;
always @(posedge m_axi_aclk)begin
    axi_rstart_locked_r1 <= axi_rstart_locked;
    axi_rstart_locked_r2 <= axi_rstart_locked_r1;
end
always @(posedge m_axi_aclk)
	if((fdma_rstart_locked == 1'b1) &&  axi_rstart_locked == 1'b0)
	    axi_rstart_locked <= 1'b1;
	else if(axi_rlast == 1'b1 || fdma_rstart == 1'b1)
	    axi_rstart_locked <= 1'b0;

//AXI4 addr valid and read addr-----------------------------------
always @(posedge m_axi_aclk)
     if((axi_rstart_locked_r1 == 1'b1) &&  axi_rstart_locked_r2 == 1'b0)
         axi_arvalid <= 1'b1;
     else if((axi_rstart_locked == 1'b1 && m_axi_arready == 1'b1)|| axi_rstart_locked == 1'b0)
         axi_arvalid <= 1'b0;
//AXI4 read data---------------------------------------------------
always @(posedge m_axi_aclk)
	if((axi_rstart_locked_r1 == 1'b1) &&  axi_rstart_locked_r2 == 1'b0)
		axi_rready <= 1'b1;
	else if(axi_rlast == 1'b1 || axi_rstart_locked == 1'b0)
		axi_rready <= 1'b0;//

//AXI4 read data burst len counter----------------------------------
always @(posedge m_axi_aclk)
	if(axi_rstart_locked == 1'b0)
		rburst_cnt <= 'd0;
	else if(r_next)
		rburst_cnt <= rburst_cnt + 1'b1;
assign axi_rlast = (r_next == 1'b1) && (rburst_cnt == m_axi_arlen);
//fdma read data burst len counter----------------------------------
reg rburst_len_req = 1'b0;
reg [31:0] fdma_rleft_cnt =32'd0;

always @(posedge m_axi_aclk)
	    rburst_len_req <= fdma_rstart | axi_rlast;

always @(posedge m_axi_aclk)
	if(fdma_rstart )begin
		rfdma_cnt <= 1'd0;
	    fdma_rleft_cnt <= fdma_r_size;
		fdma_r_size_reg<= fdma_r_size - 1'b1;
	end
	else if(r_next)begin
		rfdma_cnt <= rfdma_cnt + 1'b1;
		fdma_rleft_cnt <= fdma_r_size_reg  - rfdma_cnt;
    end

assign  fdma_rend = r_next && (fdma_rleft_cnt == 1 );
//axi auto burst len caculate-----------------------------------------
wire 	[M_AXI_ADDR_WIDTH-1 : 0] 	axi_araddr_next_power_4k = {{axi_araddr[M_AXI_ADDR_WIDTH-1:12] + 1'b1},{12'b0}};
wire    [M_AXI_ADDR_WIDTH-1 : 0]    num_rdata_next_4k_t      = axi_araddr_next_power_4k-axi_araddr;
wire    [8 : 0]                     num_rdata_next_4k        = num_rdata_next_4k_t[clogb2(M_AXI_DATA_WIDTH/8)+7:clogb2(M_AXI_DATA_WIDTH/8)-1];

always @(posedge m_axi_aclk)
    begin
     if(rburst_len_req)
        begin
            if(fdma_rleft_cnt > 256)
                begin
                    if(num_rdata_next_4k<256)
                        begin
                            rburst_len <= num_rdata_next_4k;
                        end
                    else
                        begin
                            rburst_len <= 256;
                        end

                end
            else
                begin
                    if(num_rdata_next_4k<fdma_rleft_cnt[8:0])
                        begin
                            rburst_len <= num_rdata_next_4k;
                        end
                    else
                        begin
                            rburst_len <= fdma_rleft_cnt[8:0];
                        end
                end
        end
     else
        begin
            rburst_len <= rburst_len;
        end
    end

endmodule

