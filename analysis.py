import numpy as np
from scipy import signal
from typing import Tuple, List
from dataclasses import dataclass
from collections import namedtuple

import far_doors
Point = namedtuple('Point', 'x y')


@dataclass
class Door:
    score: float
    angle: float
    coord: Point
    distance: float


def from_far(door: far_doors.Bin) -> Door:
    score = 5.0
    score -= door.range()
    angle = np.arctan(door.center()[1]/door.range())
    distance = door.range()
    coord = Point(door.center()[0], door.center()[1])

    return Door(score, angle, coord, distance)


def find_doors(data: Tuple[np.ndarray, np.ndarray],
               ranges: np.ndarray) -> List[Door]:
    far = far_doors.find(*data, ranges)
    doors = list(map(from_far, far))

    doors.sort(key=lambda k: k.score)
    return doors
