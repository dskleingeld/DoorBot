from src.agents import Pioneer
from enum import Enum
from typing import List, Tuple
from loguru import logger
import numpy as np
from analysis import find_doors
# import remote
from plot import Plot
from actions import Action, rot_away, rot_towards


ANGLES = np.linspace(-.75*np.pi, .75*np.pi, num=270)
SIN = np.sin(ANGLES)
COS = np.cos(ANGLES)


def convert(data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    ranges = np.array(data)
    x = -1*ranges*SIN
    y = ranges*COS

    data = np.zeros((2, 270))
    return x, y


class BotState(Enum):
    Started = 1


class State:
    current: BotState = BotState.Started
    plot: Plot = Plot()
    # control = remote.Control()


def track_door(target) -> Action:
    if abs(target.angle) < 5.0 and target.coord.y < 0.3:
        return Action.Forward
    elif target.coord.x < 0.3:
        return rot_towards(target.angle)
    elif 60 > abs(target.angle) < 90:
        return Action.Forward
    elif abs(target.angle) < 45:
        return rot_away(target.angle)
    else:
        logger.warning("no direction to go")
        return Action.Stay


def loop(agent: Pioneer, state: State):
    ranges = agent.read_lidars()
    data = convert(ranges)
    # key = state.control.apply(agent)
    # if key == "p":
    #     np.savetxt("data.txt", ranges)
    #     np.savetxt("ranges.txt", ranges)
    #     print("saved scene")

    doors = find_doors(data, ranges)
    if len(doors) > 0:
        action = track_door(doors[0])
    else:
        action = Action.Forward

    state.plot.update(doors, *data, action)
    action.perform(agent)
