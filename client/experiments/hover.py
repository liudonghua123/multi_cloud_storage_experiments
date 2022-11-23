import matplotlib.pyplot as plt
import mplcursors
import numpy as np

labels = ["a", "b", "c", "d", "e"]
x = np.array([0, 1, 2, 3, 4])

fig, ax = plt.subplots()
line, = ax.plot(x, x, "ro")
# mplcursors.cursor(hover=True)
mplcursors.cursor(ax, hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(f"x: {sel.target_[0]} y: {sel.target_[1]}"))

plt.show()
