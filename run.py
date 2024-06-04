import os
import sys
import time

subject = "zbt"

# start = time.time()
# os.system("cd /mnt/sda/wangyichen/Server/process/fast_process && python process_video.py --subject %s" %subject)
# end = time.time()
# print(" ")
# print("----------------------------------------------------------------------------------------------------------------------------Analysis Video Time:{:.2f} s".format(end-start))
# 440.97.s 589.96s for me

# start = time.time()
# os.system("cd /mnt/sda/wangyichen/Server/process/face-parsing.PyTorch && python test.py --actor %s" %subject)
# # os.system("cd /mnt/sda/wangyichen/Server/process/face-parsing.PyTorch && python parsing.py --subject %s" %subject)
# end = time.time()
# print("----------------------------------------------------------------------------------------------------------------------------Face Parsing Time:{:.2f} s".format(end-start))
# # 250.30s 110.35s 48.09s 75.29s for me

# start = time.time()
# SUBJECT_DIR = os.path.join(BASE_DIR, subject)
# input_source = os.path.join(SUBJECT_DIR, "images")
# output_alpha = os.path.join(SUBJECT_DIR, "RVM/alpha")
# os.system("cd /mnt/sda/wangyichen/Server/process/RobustVideoMatting && python inference.py --variant resnet50 --checkpoint rvm_resnet50.pth --device cuda --input-source %s --output-type png_sequence --output-alpha %s --num-workers 12" %(input_source, output_alpha))
# os.system("cd /mnt/sda/wangyichen/Server/process && python postprocess.py -i %s" %SUBJECT_DIR)
# end = time.time()
# print("-----------------------------------------------------------------------------------------------------------------------------Video Matting Time:{:.2f} s".format(end-start))
# # 183.69s 61.64s 85.86 for me

start = time.time()
os.system("cd /data2/wangyichen/FlashAvatar && proxychains python train.py --seed 0 --idname %s" %subject)
end = time.time()
print("------------------------------------------------------------------------------------------------------------------------------Training Time:{:.2f} s".format(end-start))
# 1851.00s for 5w iter 1267.69s for 4w iter