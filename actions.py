from enum import Enum
from typing import List, Tuple
from loguru import logger
import numpy as np
from src.agents import Pioneer


class Action(Enum):
    Forward = 0
    Left = -1
    Right = 1
    Backward = -2
    Stay = -3

    def perform(self, agent: Pioneer):
        if self == self.Forward:
            agent.change_velocity([0.15, 0.15])
        elif self == self.Left:
            agent.change_velocity([0.05, -0.05])
        elif self == self.Right:
            agent.change_velocity([-0.05, 0.05])
        else:
            logger.error(f"unknown action {self}")


def rot_away(angle: float) -> Action:
    return rot_towards(-angle)


def rot_towards(angle: float) -> Action:
    if angle > 0:
        return Action.Left
    else:
        return Action.Right
