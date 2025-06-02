import os
import shutil
# file_path_src = r'/media/qin/教程/datasets/coco/images/train2017'   #源目录
# file_path_tar = r'/home/qin/ncnnAccel/dataset/coco'   #目标目录
file_path_src = r'/media/qin/教程/datasets/VOC/images/train2007'   #源目录
file_path_tar = r'/home/qin/ncnnAccel/dataset/voc'   #目标目录
cnt = 0
for files in os.walk(file_path_src):
    sorted_files = sorted(files[2])
    for file in sorted_files:
        print(file + "-->star")
        mv_files = os.path.join(file_path_src, file)
        # shutil.move(mv_files, file_path_tar)  #剪切，复制的话换成shutil.copy即可
        shutil.copy(mv_files, file_path_tar)  #剪切，复制的话换成shutil.copy即可
        cnt += 1
        if cnt == 1000:  #复制的文件个数
            break