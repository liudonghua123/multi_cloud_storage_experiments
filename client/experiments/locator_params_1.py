# importing libraries
import matplotlib.pyplot as plt

# Y-axis Values
y = [-1, 4, 9, 16, 25]

# X-axis Values
x = [1, 2, 3, 4, 5]

plt.locator_params(axis='x', nbins=5)

# adding grid to the plot
axes = plt.axes()
axes.grid()

# defining the plot
plt.plot(x, y, 'mx', color='green')

# range of y-axis in the plot
plt.ylim(ymin=-1.2, ymax=30)

# Set the margins
plt.margins(0.2)

# printing the plot
plt.show()
