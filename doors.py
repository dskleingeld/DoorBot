import math
import numpy as np
from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    def distance(self) -> float:
        return np.linalg.norm([self.x, self.y])

    def angle(self) -> float:
        radians = np.arctan(self.y/self.distance())
        return 180/math.pi * radians


@ dataclass
class Door:
    score: float
    left: Point
    right: Point

    def center(self) -> Point:
        xlen = self.left.x - self.right.x
        ylen = self.left.y - self.right.y
        return Point(self.left.x-xlen/2, self.left.y-ylen/2)

    def distance(self) -> float:
        return np.linalg.norm(self.center())

    def angle_on(self) -> float:  # in degrees
        d = (self.right.x - self.left.x)/2
        c = (self.right.y - self.left.x)/2
        beta = 180/math.pi * np.arctan(d/c)
        gamma = self.center().angle()
        alpha = 90 - gamma - beta
        return alpha

    def waypoint(self) -> Point:
        return Point(0, 0)
