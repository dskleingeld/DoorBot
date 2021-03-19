import numpy as np
from matplotlib import pyplot as plt


# x = np.loadtxt("x.txt")
# y = np.loadtxt("y.txt")

ranges = np.loadtxt("ranges.txt")
angles = np.linspace(-.75*np.pi, .75*np.pi, num=270)

angles = np.arange(0,270)

sin = np.sin(angles)
cos = np.cos(angles)

x = ranges*sin
y = ranges*cos
txt = np.arange(0, len(x))

start = 77
stop = 107

x = x[start:stop]
y = y[start:stop]
txt = txt[start:stop]

fig, ax = plt.subplots()
ax.scatter(x, y)

for x, y, txt in zip(x, y, txt):
    ax.annotate(txt, (x, y))

plt.show()
