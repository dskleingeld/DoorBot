import numpy as np
from dataclasses import dataclass
from doors import Door, Point
from typing import List, Optional
from scipy import signal
from skimage.measure import LineModelND, ransac


@dataclass
class Gap:
    start: int
    stop: int

    def width_ok(self, x: np.ndarray, y: np.ndarray) -> bool:
        start = np.array([x[self.start], y[self.start]])
        stop = np.array([x[self.stop], y[self.stop]])
        width = np.linalg.norm(start-stop)
        return 0.2 < width < 3

    def fit(self, x: np.ndarray, y: np.ndarray) -> LineModelND:
        data = np.zeros((2, 20))
        data[0:10] = x[self.start-10:self.start]
        data[10:] = x[self.stop:self.stop+10]
        model, _ = ransac(data, LineModelND)
        return model

    def to_door(self, x: np.ndarray, y: np.ndarrayy) -> Door:
        # fit a line through the gap
        model = self.fit(x, y)
        # correct the start and stop doorposts using
        # the model
        def point(i: int): return Point(x[i], y[i])
        start = closest_on_line(point(self.start))
        end = closest_on_line(point(self.start))
        return Door(5, start, end)


def to_ax_bx_c(origin: np.ndarray, direction: np.ndarray) -> (float, float, float):
    a_over_b = 


def closest_on_line(model: LineModelND, point: Point):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_an_equation
    a = model.params[0]
    a = model.params[0]
    


def add_gap(median, i: int, gaps: List[Gap]):
    # here i is the index of the start of the gap
    for j, curr in median[i:]:
        if curr - median[i] < 2:
            gaps.append(Gap(i, j))


def find(x: np.ndarray, y: np.ndarray,
         ranges: np.ndarray) -> List[Door]:

    # search for gaps
    gaps: List[Gap] = []
    median = signal.medfilt(ranges, 5)
    prev = median[0]
    for i, curr in enumerate(median[1::-1]):
        if curr - prev > 2:
            add_gap(median, i, gaps)
        prev = curr

    # check if gap wide enough
    gaps = [g for g in gaps if g.width_ok(x, y)]

    # using ransac

    # from the inliers of the line find the left and right door post

    return []
