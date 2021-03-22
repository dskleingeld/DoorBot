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

    def __sub__(self, other) -> np.ndarray:
        return np.array([self.x-other.x, self.y-other.y])

    def np(self) -> np.ndarray:
        return np.array([self.x, self.y])


@ dataclass
class Door:
    base_score: float
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
        # orthogonalize center to origin
        # with left-right as first base vector
        v1 = self.left.np() - self.right.np()
        a2 = np.zeros((2)) - self.center().np()
        v2 = a2 - (a2@v1)/(v1@v1) * v1  # gram smith

        pos = self.center().np() + v2 * 1
        pos = Point(pos[0], pos[1])
        return pos

    def score(self) -> float:
        return self.base_score - self.distance()
