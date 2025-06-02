#-------------------------------------------------------------------------------------------
#     net     |    size   |  sram_ifm  |  time
#-------------------------------------------------------------------------------------------
# yolov3-tiny |  416x416  |    2MB     | FREQ = 200MHz, CYCLES = 6442157, TIME = 32.210785ms
# yolov3-tiny |  416x416  |    1MB     | FREQ = 200MHz, CYCLES = 6477671, TIME = 32.388355ms
# yolov3-tiny |  416x416  |   512KB    | FREQ = 200MHz, CYCLES = 6544419, TIME = 32.722095ms
# yolov3-tiny |  416x416  |   256KB    | FREQ = 200MHz, CYCLES = 6699272, TIME = 33.496361ms
# yolov3-tiny |  416x416  |   128KB    | FREQ = 200MHz, CYCLES = 7193499, TIME = 35.967495ms
#-------------------------------------------------------------------------------------------
# [error]: when set sram_ifm to 128KB, the CHECK_ALIGN can't PASS!
# OFM_SRAM = 256KB

import matplotlib.pyplot as plt

x_values = [0,   0.125,    0.25,     0.5,       1,       2]
y_values = [0, 7193499, 6699272, 6544419, 6477671, 6442157]

plt.plot(x_values, y_values)
plt.scatter(0.125, 7193499, color='red', marker='o')
plt.text(0.125, 7193499, "error", color='black', ha='right', va='bottom')

plt.xlabel('sram_size / MB')
plt.ylabel('speed / cycles')
plt.title('ifmbuf_sram_cal')

plt.show()
