
TARGET = tcp
VSRC = $(shell find ./vsrc -name *.v) 
CSRC = $(shell find ./csrc -name *.cpp)  \
       $(shell find ./csrc/ncnn -name *.cpp) \
       $(shell find ./csrc/layer -name *.cpp)
INC = -I./csrc/ -I./csrc/ncnn -I./csrc/layer
IDEA_V = ../chisel_prj/verilog/accel/*.v
IDEA_SV = ../chisel_prj/verilog/accel/*.sv


parallel_sim:
	make sim -j$(shell nproc)

veri: $(CSRC) $(VSRC)
	rm -rf log.txt
	verilator -Wno-fatal $^ $(INC)   --top-module $(TARGET) --cc --exe --build --trace --threads 12
	@# verilator -Wno-fatal $^ $(INC)   --top-module $(TARGET) --cc --exe --build --trace --CFLAGS -DOPENMP_THREADS=$(shell nproc)  --CFLAGS -fopenmp --threads $(shell nproc)
	@# verilator -Wno-fatal $^ $(INC)   --top-module $(TARGET) --cc --exe --build --trace
	@# verilator -Wno-fatal $^ $(INC)   --top-module $(TARGET) --cc --exe --build --trace --debug --CFLAGS -g --LDFLAGS -g
	@# verilator -Wno-fatal $^ $(INC)   --top-module $(TARGET) --cc --exe --build --trace --CFLAGS -O2 --threads $(shell nproc)
	@# verilator  -O2 -Wno-fatal $^ $(INC)   --top-module $(TARGET) --cc --exe --build --trace --CFLAGS -O2 --threads 4
	@# verilator -O3 -Wno-fatal $^ $(INC)   --top-module $(TARGET) --cc --exe --build --trace --CFLAGS -O3
	@# verilator -Wno-fatal $^ $(INC)   --top-module $(TARGET) --cc --exe --build --trace --LDFLAGS -fsanitize=address --CFLAGS -fsanitize=address

sim: 
	make dir
	
	@echo "[ log ]: Compiling Verilator..."
	@date '+%Y-%m-%d %H:%M:%S'
	@time -f "[ log ]: Compilation time: %E" \
	make veri

	@echo "[ log ]: Running simulation..."
	@date '+%Y-%m-%d %H:%M:%S'
	@time -f "[ log ]: simulation time: %E" \
	./obj_dir/Vtcp
	# ./obj_dir/Vtcp 2>&1 | tee log.txt

dir:
	rm -rf ./log/*
	
	if [ ! -d "./log/ifmbuf" ]; then mkdir -p ./log/ifmbuf; fi
	if [ ! -d "./log/wgtbuf" ]; then mkdir -p ./log/wgtbuf; fi
	if [ ! -d "./log/ifmbufctl" ]; then mkdir -p ./log/ifmbufctl; fi
	if [ ! -d "./log/ofmbuf" ]; then mkdir -p ./log/ofmbuf; fi
	if [ ! -d "./log/accmem" ]; then mkdir -p ./log/accmem; fi
	if [ ! -d "./log/opfusion" ]; then mkdir ./log/opfusion; fi
	if [ ! -d "./log/pool" ]; then mkdir ./log/pool; fi

	if [ ! -d "./vsrc/" ]; then mkdir -p ./vsrc/; fi


copy:
	if [ ! -d "./vsrc" ]; then mkdir  ./vsrc; fi
	rm -rf ./vsrc/*
	cp $(IDEA_V) ./vsrc

gtk:
	gtkwave wave.vcd

gtk_fst:
	@size=`du -k wave.vcd | awk '{print $$1}'`; \
	if [ $$size -lt 524288 ]; then \
		make gtk; \
	else \
		echo "wave.vcd is too large to open, converting to wave.fst"; \
		vcd2fst wave.vcd wave.fst; rm wave.vcd;\
		gtkwave wave.fst; \
	fi

vcd2fst:
	vcd2fst wave.vcd wave.fst
	rm wave.vcd

clean:
	rm -rf ./obj_dir/*
	rm -rf ./wave.vcd
	rm -rf ./log/*

hw_resim:
	cd ../chisel_prj/ && sbt "test:runMain accel_gen"
	make resim

resim: clean copy parallel_sim

config_acc:
	sed -i "s/^\/\/ #define FORWARD_ON_CPU/#define FORWARD_ON_CPU/g" csrc/config.h
	sed -i "s/^#define FORWARD_ON_NPU/\/\/ #define FORWARD_ON_NPU/g" csrc/config.h
	sed -i "s/^\/\/ #define TEST_ACCMEM/#define TEST_ACCMEM/g" csrc/config.h

config_acc_hw: config_acc
	sed -i "s/val gemm_en = true/val gemm_en = false/g" ../chisel_prj/src/main/scala/configs.scala
	sed -i "s/val im2col_sim = false/val im2col_sim = true/g" ../chisel_prj/src/main/scala/configs.scala
	sed -i "s/val wgtbuf_sim = false/val wgtbuf_sim = true/g" ../chisel_prj/src/main/scala/configs.scala
	sed -i "s/val ifmbuf_sim = false/val ifmbuf_sim = true/g" ../chisel_prj/src/main/scala/configs.scala
	sed -i "s/val accmem_sim = false/val accmem_sim = true/g" ../chisel_prj/src/main/scala/configs.scala

	
.PHONY:  sim copy gtk clean 