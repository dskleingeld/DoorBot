from matplotlib import pyplot as plt

import math
import numpy as np
from scipy import signal
from functools import partial
from typing import List, Optional
from doors import Door, Point

PLOT = True

if PLOT:
    fig, ax = plt.subplots()
    ax.set_xlim(-8, 8)
    ax.set_ylim(-2, 8)
    ax.plot((0,), (0,), marker="o", linestyle='None', color="red")
    line_lidar, = ax.plot([], [], marker="o", linestyle='None',
                          color="blue", alpha=0.2)
    line_median, = ax.plot([], [], marker="o", linestyle='None',
                           color="yellow", alpha=0.2)
    line_doors, = ax.plot([], [], marker="o", linestyle='None', color="green")
    line_waypoints, = ax.plot(
        [], [], marker="o", linestyle='None', color="purple")
    fig.canvas.draw()
    plt.show(block=False)


class Bin:
    def __init__(self, index: int, coord: np.ndarray, range: float):
        self.indices = [index]
        self.ranges = [range]
        self.coords = coord

    def center(self) -> np.ndarray:
        if len(self.coords.shape) > 1:
            return self.coords.mean(axis=0)
        else:
            return self.coords

    def range(self) -> float:
        return sum(self.ranges)/len(self.ranges)

    def idx(self) -> int:
        return int(sum(self.indices)/len(self.indices))

    def add(self, i, coord, range):
        self.coords = np.row_stack((self.coords, coord))
        self.ranges.append(range)
        self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __contains__(self, coord: np.ndarray) -> bool:
        # FIXME clustering distance should dep on range or
        # if points are on a line
        return np.linalg.norm(coord - self.center()) < 4


class Clusters:
    def __init__(self):
        self.bins = []

    def update(self, i: int, coord: np.ndarray, range: float):
        for bin in self.bins:
            if coord in bin:
                return bin.add(i, coord, range)
        return self.bins.append(Bin(i, coord, range))

    def to_openings(self):
        return [bin for bin in self.bins if len(bin) > 6]


def build_door(x: np.ndarray, y: np.ndarray,
               ranges: np.ndarray, bin: Bin) -> Optional[Door]:

    def find_door_post(bin: Bin, hay) -> Optional[int]:
        i = min(bin.indices) - 1
        for i in hay:
            if bin.range() - ranges[i] > 1.5:
                return i
        return None

    left = find_door_post(bin, range(min(bin.indices), 0, -1))
    right = find_door_post(bin, range(max(bin.indices), len(ranges)))

    if left is None or right is None:
        return None

    left = Point(x[left], y[left])
    right = Point(x[right], y[right])

    return Door(5, left, right)


def find(x: np.ndarray, y: np.ndarray,
         ranges: np.ndarray) -> List[Door]:

    clusters = Clusters()
    median = signal.medfilt(ranges, 31)
    for i, _ in enumerate(ranges):
        if ranges[i] > median[i] + 1:
            coord = np.array([x[i], y[i]])
            clusters.update(i, coord, ranges[i])

    openings = clusters.to_openings()
    openings = [o for o in openings if o.range() > median[o.idx()] + 1.5]
    to_door = partial(build_door, x, y, ranges)
    doors = list(filter(lambda o: o is not None, map(to_door, openings)))

    update_plot(x, y, median, doors)

    return doors


def update_plot(x, y, median, doors):
    if PLOT:
        line_lidar.set_data(x, y)
        ANGLES = np.linspace(-.75*np.pi, .75*np.pi, num=270)
        SIN = np.sin(ANGLES)
        COS = np.cos(ANGLES)
        x = -1*SIN*median
        y = COS*median
        line_median.set_data(x, y)

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        if len(doors) > 0:
            x = [door.center().x for door in doors]
            y = [door.center().y for door in doors]
            line_doors.set_data(x, y)
            x = [door.waypoint().x for door in doors]
            y = [door.waypoint().y for door in doors]
            line_waypoints.set_data(x, y)
            ax.draw_artist(line_doors)
            ax.draw_artist(line_waypoints)
        ax.draw_artist(line_lidar)
        ax.draw_artist(line_median)
        plt.pause(0.001)


if __name__ == "__main__":
    ranges = np.loadtxt("ranges.txt")
    data = np.loadtxt("data.txt")
    x, y = data[0], data[1]
    print(data)

    find(x, y, ranges)
    input("test")
