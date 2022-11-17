import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

x1 = np.random.rand(10)
x2 = np.random.rand(10)
x3 = np.random.rand(10)
x4 = np.random.rand(10)
y1 = np.random.rand(10)
y2 = np.random.rand(10)
y3 = np.random.rand(10)
y4 = np.random.rand(10)

# # plt.subplot: Add an Axes to the current figure or retrieve an existing Axes.
# # This is a wrapper of .Figure.add_subplot which provides additional behavior when working with the implicit API (see the notes section).
# axis: Axes = plt.subplot(3,2)
# # plt.subplots: Create a figure and a set of subplots.
# axis: tuple[Figure, list[list[Axes]]] = plt.subplots(3,2)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.plot(x1, y1)
ax1.set_title('ax1')
ax2.plot(x2, y2)
ax2.set_title('ax2')
ax3.plot(x3, y3)
ax3.set_title('ax3')
ax4.plot(x4, y4)
ax4.set_title('ax4')

# fig = plt.figure(2, figsize=(1,1))
new_plot = fig.add_subplot(325)
new_plot.plot(np.random.rand(10), np.random.rand(10))
new_plot.set_title('new_plot')


plt.show()
