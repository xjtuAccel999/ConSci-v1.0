module DPRAM
#(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 1024,
    parameter RAM_STYLE_VAL = "block"
)
(
    input                               CLKA    ,
    input                               CLKB    ,
    input                               WENA    ,
    input                               WENB    ,
    input                               CENA    ,
    input                               CENB    ,
    input       [$clog2(DEPTH)-1:0]     AA      ,
    input       [$clog2(DEPTH)-1:0]     AB      ,
    input       [DATA_WIDTH-1:0]        DA      ,
    input       [DATA_WIDTH-1:0]        DB      ,
    output reg  [DATA_WIDTH-1:0]        QA      ,
    output reg  [DATA_WIDTH-1:0]        QB
);

(*ram_style = RAM_STYLE_VAL*) reg [DATA_WIDTH-1:0] mem[DEPTH-1:0];

always @(posedge CLKA) begin
    if(!CENA && !WENA)
        mem[AA] <= DA;
end

always @(posedge CLKB) begin
    if(!CENB && !WENB)
        mem[AB] <= DB;
end

always @(posedge CLKA) begin
    if(!CENA && WENA)
        QA <= mem[AA];
    else
        QA <= {(DATA_WIDTH/32+1){$random}};
end

always @(posedge CLKB) begin
    if(!CENB && WENB)
        QB <= mem[AB];
    else
        QB <= {(DATA_WIDTH/32+1){$random}};
end

endmodule