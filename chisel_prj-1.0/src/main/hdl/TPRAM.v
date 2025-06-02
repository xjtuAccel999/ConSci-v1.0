module TPRAM
#(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 1024,
    parameter RAM_STYLE_VAL = "block"
)
(
    input CLKA,
    input CLKB,
    input CENB,
    input CENA,
    input [$clog2(DEPTH)-1:0] AB,
    input [$clog2(DEPTH)-1:0] AA,
    input [DATA_WIDTH-1:0] DB,
    output reg [DATA_WIDTH-1:0] QA
);

(*ram_style = RAM_STYLE_VAL*) reg [DATA_WIDTH-1:0] mem[DEPTH-1:0];

always @(posedge CLKB) begin
    if(!CENB)
        mem[AB] <= DB;
end

always @(posedge CLKA) begin
    if(!CENA)
        QA <= mem[AA];
    else
        QA <= {(DATA_WIDTH/32+1){$random}};
end


endmodule