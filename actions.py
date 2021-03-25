import enum
import sys
from typing import List, Tuple
from loguru import logger
import numpy as np
from src.agents import Pioneer
from dataclasses import dataclass


class Direction(enum.Enum):
    Forward = enum.auto()
    ForwardLeft = enum.auto()
    ForwardRight = enum.auto()
    Left = enum.auto()
    Right = enum.auto()
    Back = enum.auto()
    Stay = enum.auto()
    Save = enum.auto()  # signal save observations to disk


@dataclass
class Action:
    dir: Direction
    speed: float  # between zero and one
    # amount to deviate from forward speed for slightly turn
    # between zero and one, if one we make a full turn
    deviation: float

    @staticmethod
    def forward(speed: float):  # no typing (not yet supported)
        return Action(Direction.Forward, speed, 0)

    @staticmethod
    def left(speed: float):  # no typing (not yet supported)
        return Action(Direction.Left, speed, 0)

    @staticmethod
    def right(speed: float):  # no typing (not yet supported)
        return Action(Direction.Right, speed, 0)

    @staticmethod
    def back(speed: float):  # no typing (not yet supported)
        return Action(Direction.Back, speed, 0)

    @staticmethod
    def forward_left(forward: float, turn: float):  # typing not supported
        return Action(Direction.ForwardLeft, forward, turn)

    @staticmethod
    def forward_right(forward: float, turn: float):  # typing not supported
        return Action(Direction.ForwardRight, forward, turn)

    def perform(self, agent: Pioneer):
        v = self.speed
        if self.dir == Direction.Forward:
            agent.change_velocity([v, v])

        elif self.dir == Direction.Left:
            agent.change_velocity([-v, v])
        elif self.dir == Direction.ForwardLeft:
            agent.change_velocity([v*(1-self.deviation), v*(1+self.deviation)])

        elif self.dir == Direction.Right:
            agent.change_velocity([v, -v])
        elif self.dir == Direction.ForwardRight:
            agent.change_velocity([v*(1+self.deviation), v*(1-self.deviation)])

        elif self.dir == Direction.Stay:
            return
        elif self.dir == Direction.Save:
            return
        else:
            sys.exit(f"unknown action {self}")

    def plot_y(self) -> float:
        if self.dir == Direction.Forward:
            return 0

        elif self.dir == Direction.Left:
            return 1
        elif self.dir == Direction.ForwardLeft:
            return 0 + self.deviation

        elif self.dir == Direction.Right:
            return -1
        elif self.dir == Direction.ForwardRight:
            return 0 - self.deviation

        elif self.dir == Direction.Stay:
            return -2
        elif self.dir == Direction.Save:
            return -2
        else:
            sys.exit(f"unknown action {self}")


def rot_towards(angle: float) -> Action:
    speed = max(abs(angle)/90, 0.10)
    if angle > 0:
        return Action.left(speed)
    else:
        return Action.right(speed)


def rot_towards_moving(angle: float, speed) -> Action:
    turn = max(abs(angle)/90, 0.08)
    if angle > 0:
        return Action.forward_left(speed, turn)
    else:
        return Action.forward_right(speed, turn)
