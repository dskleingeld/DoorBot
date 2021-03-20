from src.agents import Pioneer
from enum import Enum
from typing import List, Tuple
from loguru import logger
import numpy as np
from analysis import find_doors, Door
import remote
from matplotlib import pyplot as plt

ANGLES = np.linspace(-.75*np.pi, .75*np.pi, num=270)
SIN = np.sin(ANGLES)
COS = np.cos(ANGLES)


class Plot:
    # from: https://stackoverflow.com/questions/
    # 40126176/fast-live-plotting-in-matplotlib-pyplot
    def __init__(self):
        fig, ax = plt.subplots()
        self.lidar, = ax.plot([], [], marker="o", color="blue", alpha=0.2)
        self.doors, = ax.plot([], [], marker="o", color="green")
        ax.scatter((0,), (0,), color="red")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-2, 8)
        fig.canvas.draw()
        self.bg = fig.canvas.copy_from_bbox(ax.bbox)
        self.fig, self.ax = fig, ax
        plt.show(block=False)

    def redraw(self):
        self.fig.canvas.restore_region(self.bg)
        self.ax.draw_artist(self.lidar)
        self.ax.draw_artist(self.doors)
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def update(self, doors: List[Door], x: np.ndarray, y: np.ndarray):
        self.lidar.set_data(x, y)
        x = list(map(lambda d: d.coord.x, doors))
        y = list(map(lambda d: d.coord.y, doors))
        self.doors.set_data(x, y)
        self.redraw()


class BotState(Enum):
    Started = 1


class State:
    current: BotState = BotState.Started
    plot: Plot = Plot()
    # control = remote.Control()


def convert(data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    ranges = np.array(data)
    x = -1*ranges*SIN
    y = ranges*COS

    data = np.zeros((2, 270))
    return x, y


def forward(agent: Pioneer): return agent.change_velocity([0.1, 0.1])
def rot_left(agent: Pioneer): return agent.change_velocity([0.1, -0.1])
def rot_right(agent: Pioneer): return agent.change_velocity([-0.1, 0.1])
def rot_away(agent: Pioneer, angle: float): return rot_towards(agent, -angle)


def rot_towards(agent: Pioneer, angle: float):
    if angle < 0:
        rot_left(agent)
    else:
        rot_right(agent)


def track_door(agent, target):
    if abs(target.angle) < 5.0 and target.coord.y < 0.3:
        forward(agent)
    elif target.coord.x < 0.3:
        rot_towards(agent, target.angle)
    elif 60 > abs(target.angle) < 90:
        forward(agent)
    elif abs(target.angle) < 45:
        rot_away(agent, target.angle)


def loop(agent: Pioneer, state: State):
    ranges = agent.read_lidars()
    data = convert(ranges)
    # key = state.control.apply(agent)
    # if key == "p":
    #     np.savetxt("data.txt", ranges)
    #     np.savetxt("ranges.txt", ranges)
    #     print("saved scene")

    doors = find_doors(data, ranges)
    state.plot.update(doors, *data)

    if len(doors) > 0:
        track_door(agent, doors[0])
    else:
        forward(agent)
