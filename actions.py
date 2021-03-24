import enum
from typing import List, Tuple
from loguru import logger
import numpy as np
from src.agents import Pioneer


class Action(enum.Enum):
    Forward = enum.auto()
    ForwardLeft = enum.auto()
    ForwardRight = enum.auto()
    Left = enum.auto()
    Right = enum.auto()
    Backward = enum.auto()
    Stay = enum.auto()
    Save = enum.auto()  # signal save observations to disk

    def perform(self, agent: Pioneer):
        if self == self.Forward:
            agent.change_velocity([0.4, 0.4])

        elif self == self.Left:
            agent.change_velocity([-0.15, 0.15])
        elif self == self.ForwardLeft:
            agent.change_velocity([0.15, 0.3])

        elif self == self.Right:
            agent.change_velocity([0.15, -0.15])
        elif self == self.ForwardRight:
            agent.change_velocity([0.3, 0.15])

        elif self == self.Stay:
            return
        elif self == self.Save:
            return
        else:
            logger.error(f"unknown action {self}")


def rot_away(angle: float) -> Action:
    return rot_towards(-angle)


def rot_towards(angle: float) -> Action:
    if angle > 0:
        return Action.Left
    else:
        return Action.Right
