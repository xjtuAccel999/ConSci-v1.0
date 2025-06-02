#-------------------------------------------------------------------------------------------
#     net     |    size   |  sram_ifm  |  depth  |  time
#-------------------------------------------------------------------------------------------
# yolov3-tiny |  416x416  |   512KB    |   512   | FREQ = 200MHz, CYCLES = 6544419, TIME = 32.722095ms

# yolov3-tiny |  416x416  |   256KB    |   256   | FREQ = 200MHz, CYCLES = 6544419, TIME = 32.722095ms
# yolov3-tiny |  416x416  |   128KB    |   128   | FREQ = 200MHz, CYCLES = 6544419, TIME = 32.722095ms
# yolov3-tiny |  416x416  |    64KB    |    64   | FREQ = 200MHz, CYCLES = 6544419, TIME = 32.722095ms
#-------------------------------------------------------------------------------------------
# FIFO_WIDTH = 128
# IFM_SRAM = 512KB

import matplotlib.pyplot as plt

x_values = [0,      64,    128,     256,     512]
y_values = [0, 6544419, 6544419, 6544419,      0]

plt.plot(x_values, y_values)

plt.xlabel('sram_size / MB')
plt.ylabel('speed / cycles')
plt.title('ifmbuf_sram_cal')

plt.show()
