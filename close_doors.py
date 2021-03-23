import numpy as np
from dataclasses import dataclass
from doors import Door, Point
from typing import List, Optional
from scipy import signal
from skimage.measure import LineModelND, ransac
from matplotlib import pyplot as plt


PLOT = False
if __name__ == "__main__":
    PLOT = True

if PLOT:
    fig, ax = plt.subplots()
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


@dataclass
class Gap:
    start: int
    stop: int

    def width_ok(self, x: np.ndarray, y: np.ndarray) -> bool:
        start = np.array([x[self.start], y[self.start]])
        stop = np.array([x[self.stop], y[self.stop]])
        width = np.linalg.norm(start-stop)
        print(width)
        return 0.5 < width < 2.2

    def fit(self, x: np.ndarray, y: np.ndarray) -> LineModelND:
        # fit the data some points away from the door
        N = 10 # number of samples on both sides
        D = 2 # distance from doorpost to start sampling
        data = np.zeros((2*N, 2))

        data[0:N, 0] = x[self.start-N-D:self.start-D]
        data[0:N, 1] = y[self.start-N-D:self.start-D]
        data[N:, 0] = x[self.stop+D:self.stop+N+D]
        data[N:, 1] = y[self.stop+D:self.stop+N+D]

        model, inliers = ransac(data, LineModelND,
                          min_samples=12, residual_threshold=0.05)

        # print(f"inliers: {np.sum(inliers)}")
        # xl = np.linspace(x[self.start], x[self.stop], num=2)
        # yl = model.predict_y(xl)
        # plt.plot(xl, yl)
        # plt.scatter(data[inliers, 0], data[inliers, 1], color="black")

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

    def to_xy_idx(self, median: np.ndarray, ranges: np.ndarray, x, y):
        # find jump in data, doorpost is there
        dist = ranges[self.start] - ranges[self.start-6:self.start+6]
        self.start -= np.argmax(dist)

        dist = ranges[self.stop] - ranges[self.stop-6:self.stop+6]
        self.stop += np.argmax(dist)


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
        xp = [x[gap.start], x[gap.stop]]
        yp = [y[gap.start], y[gap.stop]]
        plt.scatter(xp, yp, color="red", marker="o", linewidths=4)

        gap.to_xy_idx(median, ranges, x, y)
        xp = [x[gap.start], x[gap.stop]]
        yp = [y[gap.start], y[gap.stop]]
        plt.scatter(xp, yp, color="purple", marker="o", linewidths=4)

        if gap.width_ok(x, y):
            doors.append(gap.to_door(x, y))

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
        # for i, xy in enumerate(zip(x, y)):
        #     plt.annotate(i, xy)
        # line_median.set_data(x, y)

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
    plt.show()
