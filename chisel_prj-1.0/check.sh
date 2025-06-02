#!/usr/bin/env bash

DUT_V=$(find ./verilog -name "*.v")
#echo $DUT_V
DUT_SV=$(find ./verilog -name "*.sv")

# 删除initial、宏定义、行首顶格注释
sed -i '/initial begin/,/end \/\/ initial/d'  $DUT_V
sed -i '/`ifdef/,/`endif/d'  $DUT_V
sed -i '/^\/\//d'  $DUT_V
sed -i '/`ifndef/,/`endif/d'  $DUT_V

for file in $DUT_V; do python3 clear_comments.py $file; done
for file in $DUT_SV; do python3 clear_comments.py $file; done

for file in $DUT_V; do python3 split_modules.py $file; done
for file in $DUT_SV; do python3 split_modules.py $file; done
#python3 split_modules.py $file

#DUT_V=$(find ./verilog -name "*.v")
#DUT_SV=$(find ./verilog -name "*.sv")
#LINT_FLAGS="-Wall -Wno-DECLFILENAME"
## lint without UNUSED
#verilator --lint-only --top-module tcp $LINT_FLAGS -Wno-UNUSED $DUT_V $DUT_SV > lint.log  2>&1
## lint with UNUSED
#verilator --lint-only --top-module tcp $LINT_FLAGS $DUT_V $DUT_SV > lint_unused.log 2>&1