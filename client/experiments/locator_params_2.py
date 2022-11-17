#!/usr/bin/python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

x = np.linspace(0, 10, 11)
n = 30
tights = [False, True]
nbins = [5, 10, 20]

for i, (n, tight) in enumerate(product(nbins, tights)):
    ax = plt.subplot(3, 2, 1 + i)
    ax.plot(x, x)
    # print(ax == plt.gca())
    ax.text(0.0, 1.05, f'tight={tight},nbins={n}',
            fontsize=15, transform=ax.transAxes)
    ax.locator_params("y", tight=tight, nbins=n)
    ax.locator_params("x", tight=False)
plt.show()
