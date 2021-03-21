from src.agents import Pioneer
from typing import List, Tuple
from loguru import logger
import enum
import numpy as np
from analysis import find_doors, passing_door
# import remote
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
    current: BotState = BotState.Started
    plot: Plot = Plot()
    # control = remote.Control()


def track_door(target) -> Action:
    angle = target.angle()
    coord = target.center()

    if abs(angle) < 5.0 and coord.y < 0.3:
        return Action.Forward
    elif coord.y < 1.5:
        return rot_towards(angle)
    elif 60 > abs(angle) < 90:
        return Action.Forward
    elif abs(angle) < 45:
        return rot_away(angle)
    else:
        logger.warning("no direction to go")
        return Action.Stay


def handle_tracking(state: State, data, ranges) -> Action:
    doors = find_doors(data, ranges)
    if len(doors) == 0:
        state.current = BotState.NoDoor
        return Action.Forward
    door = doors[0]
    if door.angle_on() < 3:
        if door.center().angle() < 3:
            return Action.Forward
        else:
            return rot_towards(door.center().angle())
    elif door.waypoint().angle() < 3:
        return Action.Forward
    else:
        return rot_towards(door.waypoint().angle())


def decide(state: State, data, ranges) -> Action:
    if state.current == BotState.NoDoor:
        doors = find_doors(data, ranges)
        if len(doors) > 0:
            state.current = BotState.TrackingDoor
    elif state.current == BotState.TrackingDoor:
        return handle_tracking(state, data, ranges)
    elif state.current == BotState.AlignedOnDoor:
        if passing_door(data, ranges):
            state.current = BotState.PassingDoor
    elif state.current == BotState.PassingDoor:
        if not passing_door(data, ranges):
            state.current = BotState.NoDoor
    else:
        logger.warning(f"INVALID STATE: {state.current}")
    return Action.Forward


def loop(agent: Pioneer, state: State):
    ranges = agent.read_lidars()
    data = convert(ranges)

    action = decide(state, data, ranges)

    doors = find_doors(data, ranges)
    state.plot.update(doors, *data, action)
    report_status(doors)
    action.perform(agent)
