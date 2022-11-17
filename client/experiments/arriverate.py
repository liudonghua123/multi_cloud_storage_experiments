# coding=utf-8
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

name_list = ['hm_0', 'hm_1', 'mds_0', 'mds_1']
num_list = [0.464906684, 0.190486908, 0.0472115, 0.498776467]

fig, ax = plt.subplots()

ax.bar(range(len(num_list)), num_list, color='y')

ax.set_xticklabels(name_list, rotation=45)

ax.minorticks_on()
ax.xaxis.set_major_locator(FixedLocator([0, 1, 2, 3]))
ax.xaxis.set_minor_locator(FixedLocator([0.1, 0.4, 0.75, 1.5, 2.5]))
ax.tick_params('x', which='major', length=20, width=5, color='r')
ax.tick_params('x', which='minor', length=5, width=1, color='b')

plt.show()
