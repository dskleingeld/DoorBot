from matplotlib import pyplot as plt
from collections import deque
import numpy as np
from typing import List

from analysis import Door
from actions import Action


class Plot:
    # from: https://stackoverflow.com/questions/
    # 40126176/fast-live-plotting-in-matplotlib-pyplot
    def __init__(self):
        fig, (ax, ax2) = plt.subplots(
            nrows=2, gridspec_kw={'height_ratios': [2, 1]})
        self.lidar, = ax.plot([], [], linestyle='None',
                              marker="o", color="blue", alpha=0.2)
        self.doors, = ax.plot([], [], linestyle='None',
                              marker="o", color="green", markersize=10)
        self.controls, = ax2.plot([])
        self.control_data = deque(maxlen=500)
        self.fig, self.ax, self.ax2 = fig, ax, ax2
        self.show()

    def show(self):
        self.ax.scatter((0,), (0,), color="red")
        self.ax.set_xlim(-8, 8)
        self.ax.set_ylim(-2, 8)
        self.ax2.set_xlim(0, 500)
        self.ax2.set_ylim(-2.2, 1.2)
        self.fig.canvas.draw()
        # self.bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        plt.show(block=False)

    def redraw(self):
        # self.fig.canvas.restore_region(self.bg)
        self.ax.draw_artist(self.lidar)
        self.ax.draw_artist(self.doors)
        self.ax.draw_artist(self.controls)
        # self.fig.canvas.blit(self.ax.bbox)
        # self.fig.canvas.flush_events()
        plt.pause(0.01)

    def update(self, doors: List[Door], x: np.ndarray,
               y: np.ndarray, action: Action):
        self.lidar.set_data(x, y)
        self.control_data.append(action.value)
        x = range(len(self.control_data))
        y = list(self.control_data)
        self.controls.set_data(x, y)
        x = list(map(lambda d: d.coord.x, doors))
        y = list(map(lambda d: d.coord.y, doors))
        self.doors.set_data(x, y)
        self.redraw()
