import numpy as np
from dataclasses import dataclass
from doors import Door, Point
from typing import List, Optional
from scipy import signal
from skimage.measure import LineModelND, ransac
from matplotlib import pyplot as plt


PLOT = True

if PLOT:
    fig, ax = plt.subplots()
    ax.set_xlim(-8, 8)
    ax.set_ylim(-2, 8)
    ax.plot((0,), (0,), marker="o", linestyle='None', color="red")
    line_lidar, = ax.plot([], [], marker="o", linestyle='None',
                          color="blue", alpha=0.2)
    line_median, = ax.plot([], [], marker="o", linestyle='None',
                           color="yellow", alpha=0.4)
    line_doors, = ax.plot([], [], marker="o", linestyle='None', color="green")
    line_waypoints, = ax.plot(
        [], [], marker="o", linestyle='None', color="purple")
    fig.canvas.draw()
    plt.show(block=False)


@dataclass
class Gap:
    start: int
    stop: int

    def width_ok(self, x: np.ndarray, y: np.ndarray) -> bool:
        start = np.array([x[self.start], y[self.start]])
        stop = np.array([x[self.stop], y[self.stop]])
        width = np.linalg.norm(start-stop)
        return 1 < width < 2

    def fit(self, x: np.ndarray, y: np.ndarray) -> LineModelND:
        # fit the data some points away from the door
        data = np.zeros((20, 2))
        data[0:10, 0] = x[self.start-20:self.start-10]
        data[0:10, 1] = y[self.start-20:self.start-10]
        data[10:, 0] = x[self.stop+10:self.stop+20]
        data[10:, 1] = y[self.stop+10:self.stop+20]
        model, _ = ransac(data, LineModelND,
                          min_samples=12, residual_threshold=1)

        return model

    def to_door(self, x: np.ndarray, y: np.ndarray) -> Door:
        # fit a line through the gap
        model = self.fit(x, y)
        # correct the start and stop doorposts using
        # the model
        def point(i: int): return Point(x[i], y[i])
        start = closest_on_door(model, point(self.start))
        end = closest_on_door(model, point(self.stop))

        return Door(10, start, end)


def closest_on_door(model: LineModelND, point: Point) -> Point:
    (origin, direction) = model.params
    normal = (direction[1], -direction[0])
    a = normal[0]
    b = normal[1]
    c = -1*(a*origin[0] + b*origin[1])

    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation
    x = (b*(b*point.x-a*point.y) - a*c)/(a*a+b*b)
    y = (a*(-b*point.x+a*point.y) - b*c)/(a*a+b*b)

    return Point(x, y)


def add_gap(median, i: int, gaps: List[Gap]):
    # here i is the index of the start of the gap
    for j, curr in enumerate(median[i+1:]):
        if curr - median[i] < 2:
            gaps.append(Gap(i, i+1+j))
            break


def find(x: np.ndarray, y: np.ndarray,
         ranges: np.ndarray) -> List[Door]:

    # search for gaps
    gaps: List[Gap] = []
    median = signal.medfilt(ranges, 11)
    prev = median[0]
    for prev_idx, curr in enumerate(median[1:-1]):
        if curr - prev > 4:
            add_gap(median, prev_idx, gaps)
        prev = curr

    print(gaps)

    # convert to door if gap wide enough
    doors = [g.to_door(x, y) for g in gaps if g.width_ok(x, y)]
    print(len(doors))
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

    find(x, y, ranges)
    input("test")
