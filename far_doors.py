from matplotlib import pyplot as plt
from dataclasses import dataclass

import numpy as np
from scipy import signal
from typing import Tuple, List

ANGLES = np.linspace(-.75*np.pi, .75*np.pi, num=270)


def find_doors(data: Tuple[np.ndarray, np.ndarray], ranges: np.ndarray):
    far = far_doors(*data, ranges)


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
        return np.linalg.norm(coord - self.center()) < 2


class Clusters:
    def __init__(self):
        self.bins = []

    def update(self, i: int, coord: np.ndarray, range: float):
        for bin in self.bins:
            if coord in bin:
                return bin.add(i, coord, range)
        return self.bins.append(Bin(i, coord, range))

    def openings(self):
        return [bin for bin in self.bins if len(bin) > 6]


def far_doors(x: np.ndarray, y: np.ndarray, ranges: np.ndarray) -> List[Bin]:

    clusters = Clusters()
    median = signal.medfilt(ranges, 31)
    for i, _ in enumerate(ranges):
        if ranges[i] > median[i] + 1:
            coord = np.array([x[i], y[i]])
            clusters.update(i, coord, ranges[i])

    options = clusters.openings()
    doors = [o for o in options if o.range() > median[o.idx()] + 0.5]

    return doors

    # fig, ax = plt.subplots()
    # ax.scatter(x, y, color="blue", alpha=0.2)
    # ax.scatter((0,), (0,), color="red")

    # ANGLES = np.linspace(-.75*np.pi, .75*np.pi, num=270)
    # SIN = np.sin(ANGLES)
    # COS = np.cos(ANGLES)
    # x = -1*SIN*median
    # y = COS*median
    # ax.scatter(x, y, color="yellow", alpha=0.2)

    # plt.xlim(-10, 10)
    # plt.ylim(-10, 10)

    # for door in doors:
    #     center = door.center()
    #     x, y = center[0], center[1]
    #     ax.scatter(x, y, color="green")

    # plt.show()


def plot(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="blue", alpha=0.5)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()


if __name__ == "__main__":
    ranges = np.loadtxt("ranges.txt")
    data = np.loadtxt("data.txt")
    x, y = -1*data[0], data[1]
    # plot(x, y)

    find_doors(data, ranges)
