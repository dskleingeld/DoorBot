import sys
from src.agents import Pioneer
from typing import List, Tuple
from loguru import logger
import enum
import numpy as np
from analysis import find_doors, passing_door
import remote
from plot import Plot, report_status
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


class BotState(enum.Enum):
    NoDoor = enum.auto()
    TrackingDoor = enum.auto()
    AlignedOnDoor = enum.auto()
    PassingDoor = enum.auto()


class State:
    current: BotState = BotState.NoDoor
    plot: Plot = Plot()
    control = remote.Control()


def handle_tracking(state: State, data, ranges) -> Action:
    doors = find_doors(data, ranges)
    if len(doors) == 0:
        logger.critical("lost door, dumping data")
        np.savetxt("data.txt", data)
        np.savetxt("ranges.txt", ranges)
        sys.exit()
        state.current = BotState.NoDoor
        return Action.Forward

    door = doors[0]
    if abs(door.angle_on()) < 2:
        logger.debug(f"angle on door: {door.angle_on()}")
        if abs(door.center().angle()) < 3:
            logger.debug(f"driving towards door {door.center().angle()}")
            return Action.Forward
        else:
            logger.debug(f"turning towards door {door.center().angle()}")
            return rot_towards(door.center().angle())
    elif abs(door.waypoint().angle()) < 6:
        logger.debug(f"driving towards waypoint {door.waypoint().angle()}")
        return Action.Forward
    else:
        logger.debug(f"turning towards waypoint {door.waypoint().angle()}")
        return rot_towards(door.waypoint().angle())


def brain(state: State, data, ranges) -> Action:
    if state.current == BotState.NoDoor:
        doors = find_doors(data, ranges)
        if len(doors) > 0:
            logger.info("tracking door")
            state.current = BotState.TrackingDoor
    elif state.current == BotState.TrackingDoor:
        return handle_tracking(state, data, ranges)
    elif state.current == BotState.AlignedOnDoor:
        if passing_door(data, ranges):
            logger.info("passing door")
            state.current = BotState.PassingDoor
    elif state.current == BotState.PassingDoor:
        if not passing_door(data, ranges):
            logger.info("passed door")
            state.current = BotState.NoDoor
    else:
        logger.warning(f"INVALID STATE: {state.current}")
    return Action.Forward


def loop(agent: Pioneer, state: State):
    ranges = agent.read_lidars()
    ranges = np.array(ranges)
    data = convert(ranges)

    c = state.control.apply(agent)
    if c == "p":
        np.savetxt("data.txt", data)
        np.savetxt("ranges.txt", ranges)

    action = brain(state, data, ranges)

    doors = find_doors(data, ranges)
    state.plot.update(doors, *data, action)
    report_status(doors)
    action.perform(agent)
