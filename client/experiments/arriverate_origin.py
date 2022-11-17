# coding=utf-8
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

name_list = ['hm_0', 'hm_1', 'mds_0', 'mds_1']
num_list = [0.464906684, 0.190486908, 0.0472115, 0.498776467]

ax = plt.axes()

plt.bar(range(len(num_list)), num_list, color='y', tick_label=name_list)

plt.minorticks_on()

ax.xaxis.set_major_locator(FixedLocator([0, 2]))
ax.xaxis.set_minor_locator(FixedLocator([1.1, 1.2]))

# config major/minor ticks
# method 1, use ax.xaxis
ax.xaxis.set_tick_params(which='major', length=20, width=5, color='r')
ax.xaxis.set_tick_params(which='minor', length=5, width=1, color='b')
# method 2, use ax.tick_params with 'x'
# ax.tick_params('x', which='major', length=20, width=5, color='r')
# ax.tick_params('x', which='minor', length=5, width=1, color='b')
# method 3, use plt.tick_params with 'x'
# plt.tick_params('x', which='major', length=20, width=5, color='r')
# plt.tick_params('x', which='minor', length=5, width=1, color='b')

plt.show()
