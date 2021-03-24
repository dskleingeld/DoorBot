import numpy as np
from dataclasses import dataclass
from doors import Door, Point
from typing import List, Optional
from scipy import signal
from skimage.measure import LineModelND, ransac
from matplotlib import pyplot as plt
import math


PLOT = False
if __name__ == "__main__":
    PLOT = True

if PLOT:
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlim(-8, 8)
    ax.set_ylim(-2, 8)
    # ax.plot((0,), (0,), marker="o", linestyle='None', color="red")
    line_lidar, = ax.plot([], [], marker="o", linestyle='None',
                          color="blue", alpha=0.2)
    line_median, = ax.plot([], [], marker="o", linestyle='None',
                           color="yellow", alpha=0.4)
    line_doors, = ax.plot([], [], marker="o", linestyle='None', color="green")
    line_waypoints, = ax.plot(
        [], [], marker="o", linestyle='None', color="purple")
    fig.canvas.draw()
    plt.show(block=False)


class LineParams:
    # ax + by + c = 0
    def __init__(self, model: LineModelND):
        (origin, direction) = model.params
        normal = (direction[1], -direction[0])
        self.a = normal[0]
        self.b = normal[1]
        self.c = -1*(self.a*origin[0] + self.b*origin[1])

    def distance_to(self, point: Point) -> float:
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation
        a, b, c = self.a, self.b, self.c
        distance = abs(a*point.x+b*point.y+c)/(math.sqrt(a*a+b*b))
        return distance

    def closest_on(self, point: Point) -> Point:
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation
        a, b, c = self.a, self.b, self.c
        x = (b*(b*point.x-a*point.y) - a*c)/(a*a+b*b)
        y = (a*(-b*point.x+a*point.y) - b*c)/(a*a+b*b)
        return Point(x, y)


@dataclass
class Gap:
    start: int
    stop: int
    N = 10
    D = 2

    def fit(self, x: np.ndarray, y: np.ndarray) -> LineModelND:
        # fit the data some points away from the door
        N = self.N  # number of samples on both sides
        D = self.D  # distance from doorpost to start sampling
        data = np.zeros((2*N, 2))

        plt.scatter(x[self.start], y[self.start])
        data[0:N, 0] = x[self.start-N-D:self.start-D]
        data[0:N, 1] = y[self.start-N-D:self.start-D]
        data[N:, 0] = x[self.stop+D:self.stop+N+D]
        data[N:, 1] = y[self.stop+D:self.stop+N+D]
        plt.scatter(data[:, 0], data[:, 1], color="green")

        model, inliers = ransac(data, LineModelND,
                                min_samples=8, residual_threshold=0.05)
        return model, inliers

    def find_post(self, line: LineParams, hay, x, y) -> int:
        start = Point(x[self.start], y[self.start])
        start = line.closest_on(start)

        dist_l = [line.distance_to(Point(x[i], y[i])) for i in hay]
        dist_p = [np.linalg.norm(start - Point(x[i], y[i])) for i in hay]
        dist_l /= np.max(dist_l)  # normalize
        dist_p /= np.max(dist_p)
        score = dist_p + dist_l

        return hay[0]+np.argmin(score)

    def to_door(self, x: np.ndarray, y: np.ndarray, ranges) -> Optional[Door]:
        N = self.N

        model, inliers = self.fit(x, y)
        if np.sum(inliers) < self.N:
            return None

        xl = np.linspace(x[self.start], x[self.stop], num=2)
        yl = model.predict_y(xl)
        plt.plot(xl, yl)

        # walk along door to the center to find the best estimate door posts
        start, stop = None, None
        line = LineParams(model)
        hay = range(self.start-N, self.start)
        i = self.find_post(line, hay, x, y)
        start = Point(x[i], y[i])

        hay = range(self.stop, self.stop+N)
        i = self.find_post(line, hay, x, y)
        stop = Point(x[i], y[i])

        if start is None or stop is None:
            return None
        return Door(10, start, stop)


def add_gap(median, i: int, start_dist: float, gaps: List[Gap]):
    # here i is the index of the start of the gap
    for j in range(i, len(median)):
        dist = median[j-1] - median[j]
        if dist > 0 and dist > start_dist*0.5:
            gaps.append(Gap(i, j))
            break


def find(x: np.ndarray, y: np.ndarray,
         ranges: np.ndarray) -> List[Door]:

    # search for gaps
    gaps: List[Gap] = []
    median = signal.medfilt(ranges, 11)
    prev = median[0]
    for start_idx, curr in enumerate(median[1:-1]):
        dist = curr - prev
        if dist > 3:
            add_gap(median, start_idx, dist, gaps)
        prev = curr

    doors = []
    for gap in gaps:
        # convert find x, y index closest to median
        if __name__ == "__main__":
            xp = [x[gap.start], x[gap.stop]]
            yp = [y[gap.start], y[gap.stop]]
            plt.scatter(xp, yp, color="red", marker="o", linewidths=4)

        door = gap.to_door(x, y, ranges)
        if door is None:
            continue

        if __name__ == "__main__":
            xp = [door.right.x, door.left.x]
            yp = [door.right.y, door.left.y]
            plt.scatter(xp, yp, color="purple", marker="o", linewidths=4)

        if 0.2 > door.width() > 2:
            doors.append(door)

    update_plot(x, y, median, doors)
    return doors


def update_plot(x, y, median, doors):
    if PLOT:
        line_lidar.set_data(x, y)
        # ANGLES = np.linspace(-.75*np.pi, .75*np.pi, num=270)
        ANGLES = np.loadtxt("angles.txt") / 180*np.pi
        SIN = np.sin(ANGLES)
        COS = np.cos(ANGLES)
        x = -1*SIN*median
        y = COS*median
        # for i, xy in enumerate(zip(x, y)):
            # plt.annotate(i, xy)
        # line_median.set_data(x, y)

        # plt.xlim(-10, 10)
        # plt.ylim(-10, 10)

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
        # plt.pause(0.001)


if __name__ == "__main__":
    ranges = np.loadtxt("ranges.txt")
    data = np.loadtxt("data.txt")
    x, y = data[0], data[1]

    find(x, y, ranges)
    plt.xlim(-1, 2)
    plt.ylim(-0.5, 3.0)
    plt.show()
