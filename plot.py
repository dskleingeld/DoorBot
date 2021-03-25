from matplotlib import pyplot as plt
from collections import deque
import numpy as np
from typing import List
import sys

from analysis import Door
from actions import Action


def report_status(doors: List[Door]):
    # sys.stdout.write('\r')
    # sys.stdout.flush()
    if len(doors) > 0:
        b = doors[0]
        # sys.stdout.write(
        print(
            f"doors: #{len(doors)}, best at [{b.center().x:.2},"
            f"{b.center().y:.2}] {b.center().angle():4.1f}° "
            f"score: {b.score():.2}", end="\r")
    else:
        print("no doors found", end="\r")
        # sys.stdout.write("no doors found")


class Plot:
    # from: https://stackoverflow.com/questions/
    # 40126176/fast-live-plotting-in-matplotlib-pyplot
    def __init__(self):
        fig, (ax, ax2) = plt.subplots(figsize=(15, 15),
                                      nrows=2,
                                      gridspec_kw={'height_ratios': [2, 1]})
        self.lidar, = ax.plot([], [], linestyle='None',
                              marker="o", color="blue", alpha=0.2)
        self.closest, = ax.plot([], [], linestyle='None',
                                marker="o", color="red", alpha=1)
        self.controls, = ax2.plot([])
        self.control_data = deque(maxlen=500)
        self.fig, self.ax, self.ax2 = fig, ax, ax2
        self.show()

    def show(self):
        self.ax.scatter((0,), (0,), color="black", linewidth=6)
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-2, 8)
        self.ax2.set_xlim(0, 500)
        self.ax2.set_ylim(-2.2, 1.2)
        self.fig.canvas.draw()
        plt.show(block=False)

    def redraw(self):
        self.ax.draw_artist(self.lidar)
        self.ax.draw_artist(self.closest)
        self.ax.draw_artist(self.controls)
        plt.pause(0.01)

    def update(self, x: np.ndarray,
               y: np.ndarray, idx: int, action: Action):
        self.lidar.set_data(x, y)
        self.closest.set_data(x[idx], y[idx])
        self.control_data.append(action.plot_y())
        x = range(len(self.control_data))
        y = list(self.control_data)
        self.controls.set_data(x, y)
        self.redraw()
