from src.agents import Pioneer
from enum import Enum
from typing import List
from loguru import logger
import numpy as np
from analysis import find_doors
import remote
from matplotlib import pyplot as plt
import seq_ransac

ANGLES = np.linspace(-.75*np.pi, .75*np.pi, num=270)
SIN = np.sin(ANGLES)
COS = np.cos(ANGLES)


plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter([], [])
line = ax.plot([], [])
ax.scatter([0], [0], color="red")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.draw()


class Current(Enum):
    Started = 1


class State:
    current: Current = Current.Started
    control = remote.Control()


def convert(data: List[float]) -> np.ndarray:
    ranges = np.array(data)
    np.savetxt("ranges.txt", ranges)
    x = ranges*SIN
    y = ranges*COS
    np.savetxt("x.txt", x)
    np.savetxt("y.txt", y)
    while True:
        continue

    lines = seq_ransac.find_lines(x, y)
    print(lines)

    sc.set_offsets(np.c_[x, y])

    for line in lines:
        x = np.arange(line.start[0], 10)
        y = line.model.predict_y(x)
        ax.plot(x, y)

    fig.canvas.draw_idle()
    plt.pause(0.1)
    while True:
        continue

    data = np.zeros((2, 270))
    return data


def loop(agent: Pioneer, state: State):
    data = agent.read_lidars()
    data = convert(data)
    state.control.apply(agent)
    # doors = find_doors(data)
    # logger.info("started")
