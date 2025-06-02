module SPRAM
#(
    parameter DATA_WIDTH = 32,
    parameter DEPTH = 1024,
    parameter RAM_STYLE_VAL = "block"
)
(
    input CLK,
    input CEN,
    input WEN,
    input [$clog2(DEPTH)-1:0] A,
    input [DATA_WIDTH-1:0] D,
    output reg [DATA_WIDTH-1:0] Q
);

(*ram_style = RAM_STYLE_VAL*) reg [DATA_WIDTH-1:0] mem[DEPTH-1:0];

always @(posedge CLK) begin
    if(!CEN && !WEN)
        mem[A] <= D;
end

always @(posedge CLK) begin
    if(!CEN && WEN)
        Q <= mem[A];
    else
        Q <= {(DATA_WIDTH/32+1){$random}};
end

endmodule