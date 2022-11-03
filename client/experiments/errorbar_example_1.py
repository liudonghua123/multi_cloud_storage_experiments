import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')


aluminum = np.array([6.4e-5 , 3.01e-5 , 2.36e-5, 3.0e-5, 7.0e-5, 4.5e-5, 3.8e-5,
                     4.2e-5, 2.62e-5, 3.6e-5])
copper = np.array([4.5e-5 , 1.97e-5 , 1.6e-5, 1.97e-5, 4.0e-5, 2.4e-5, 1.9e-5, 
                   2.41e-5 , 1.85e-5, 3.3e-5 ])
steel = np.array([3.3e-5 , 1.2e-5 , 0.9e-5, 1.2e-5, 1.3e-5, 1.6e-5, 1.4e-5, 
                  1.58e-5, 1.32e-5 , 2.1e-5])

x_values = np.array(range(len(aluminum)))
plt.scatter(x_values,aluminum,label="Aluminium")
plt.scatter(x_values,copper,label="Copper")
plt.scatter(x_values,steel,label="Steel")
plt.title("Initial Data Visualization")
plt.legend()
plt.show()

aluminum_mean = np.mean(aluminum)
copper_mean = np.mean(copper)
steel_mean = np.mean(steel)

aluminum_std = np.std(aluminum)
copper_std = np.std(copper)
steel_std = np.std(steel)

labels = ['Aluminum', 'Copper', 'Steel']
x_pos = np.arange(len(labels))
CTEs = [aluminum_mean, copper_mean, steel_mean]
error = [aluminum_std, copper_std, steel_std]

plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.2,
       color='green',
       ecolor='red',
       capsize=10)

ax.set_ylabel('Coefficient of Thermal Expansion')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
ax.yaxis.grid(True)
plt.show()


x = np.linspace(0,5.5,5)
y = np.exp(-x)

xerr = np.random.random_sample(5)
yerr = np.random.random_sample(5)
fig, ax = plt.subplots()

ax.errorbar(x, y,
            xerr=xerr,
            yerr=yerr,
            fmt='-o',
           color='yellow',
           ecolor='green')

ax.set_xlabel('x-axis')
ax.set_ylabel('y-axis')
ax.set_title('Line plot with error bars')
plt.show()