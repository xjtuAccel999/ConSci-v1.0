
cmake_minimum_required (VERSION 3.3)
project (tools)

include_directories(../csrc/ncnn)
include_directories(../csrc/layer)
include_directories(../csrc/utils)
include_directories(../csrc)
include_directories(.)

aux_source_directory(../csrc/ncnn  ncnn_csrc)
aux_source_directory(../csrc/layer layer_csrc)

add_definitions(-DNCNN_TOOLS)

add_executable(u_ncnnoptimize u_ncnnoptimize.cpp u_weightorder.cpp ${ncnn_csrc} ${layer_csrc})
add_executable(u_ncnn2int8    u_ncnn2int8.cpp    u_weightorder.cpp ${ncnn_csrc} ${layer_csrc})
