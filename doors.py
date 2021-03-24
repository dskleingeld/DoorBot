import math
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional, Deque


@dataclass
class Point:
    x: float
    y: float

    def distance(self) -> float:
        return np.linalg.norm([self.x, self.y])

    def angle(self) -> float:
        # angle in degrees
        radians = np.arctan(self.x/self.y)
        return 180/math.pi * radians

    def __sub__(self, other) -> np.ndarray:
        return np.array([self.x-other.x, self.y-other.y])

    def np(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @classmethod
    def from_np(cls, array):
        return Point(array[0], array[1])


@ dataclass
class Door:
    base_score: float
    left: Point
    right: Point

    def width(self) -> bool:
        width = np.linalg.norm(self.left.np()-self.right.np())
        return width

    def center(self) -> Point:
        xlen = self.left.x - self.right.x
        ylen = self.left.y - self.right.y
        return Point(self.left.x-xlen/2, self.left.y-ylen/2)

    def distance(self) -> float:
        return np.linalg.norm(self.center().np())

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

        pos = self.center().np() + v2 * 0.4
        pos = Point(pos[0], pos[1])
        return pos

    def score(self) -> float:
        # closer doors are better
        score = self.base_score - self.distance()
        # do not turn back and go through door from
        # whence the bot came
        score -= abs(self.center().angle()) / 10
        return self.base_score - self.distance()


class DoorHistory:
    """ use a number of past locations to build a mean door location
        and score"""
    M, N = 3, 8

    def __init__(self):
        self.doors: List[Tuple[np.ndarray, deque]] = []

    def mean_door(self, queue: Deque[Door]) -> Door:
        n = 0
        left, right = np.zeros(2), np.zeros(2)
        for i in range(0, min(len(queue), self.M)):
            door = queue[i]
            if door is None:
                continue
            left += door.left.np()
            right += door.right.np()
            n += 1
        left /= n
        right /= n
        return Door(0, Point.from_np(left), Point.from_np(right))

    def update_center(self, center, queue: Deque[Door]):
        center = np.zeros(2)
        for i in range(0, self.M):
            door = queue[0]
            if door is not None:
                center += door.center().np()
        center /= self.M

    def update(self, new: List[Door]):
        handled: List[int] = []
        to_remove: List[int] = []
        for j, (center, list) in enumerate(self.doors):
            for i, door in enumerate(new):
                if np.linalg.norm(center-door.center().np()) < 0.3:
                    list.append(door)
                    handled.append(i)
                    self.update_center(center, list)
                    break
            # no new door for this location, add a None
            if list[0] is None and list[1] is None:
                to_remove.append(j)
            else:
                list.append(None)

        # remove locations that can not provide an up to date center
        to_remove.sort(reverse=True)
        for i in to_remove:
            del self.doors[i]

        # add any doors that did not match an existing location
        for i, door in enumerate(new):
            if i in handled:
                continue
            list = deque([door], maxlen=self.N)
            center = door.center().np()
            self.doors.append((center, list))

    def score(self, queue: Deque[Door]) -> float:
        return sum([door.score() for door in queue if door is not None])

    def best_guss(self) -> Optional[Door]:
        best = None
        best_score = -9999.0
        for _, queue in self.doors:
            if len(queue) > 2:
                if queue[0] is None and queue[1] is None and queue[2] is None:
                    continue
            score = self.score(queue)
            if score > best_score:
                best_score = score
                best = self.mean_door(queue)
        return best
